import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy
from mathutils import Vector


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.operator import (  # noqa: E402
    MESH_OT_holey_tools_resize_hole,
    _update_diameter_estimator,
)


def create_ring_mesh(name, rings, segments=16):
    vertices = []
    for radius, z in rings:
        vertices.extend(
            (
                radius * math.cos(math.tau * index / segments),
                radius * math.sin(math.tau * index / segments),
                z,
            )
            for index in range(segments)
        )

    faces = []
    for ring_index in range(len(rings) - 1):
        start = ring_index * segments
        following_start = (ring_index + 1) * segments
        for index in range(segments):
            following = (index + 1) % segments
            faces.append(
                (
                    start + index,
                    start + following,
                    following_start + following,
                    following_start + index,
                )
            )

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    return obj, bm


# A through-hole wall followed by a larger chamfer must expose all three radius
# estimates without changing the fitted cylinder axis or center.
chamfer_object, chamfer_bmesh = create_ring_mesh(
    "diameter_estimator_test",
    ((1.0, -1.0), (1.0, 0.5), (1.5, 1.0)),
)
estimator = SimpleNamespace(_object=chamfer_object)
estimator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(estimator, face)
)
estimated_radii = {}
for mode in ('MINIMUM', 'AVERAGE', 'MAXIMUM'):
    estimator.diameter_estimator = mode
    fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
        estimator, list(chamfer_bmesh.faces)
    )
    estimated_radii[mode] = fit["radius"]

assert abs(estimated_radii['MINIMUM'] - 1.0) < 1.0e-5
assert 1.0 < estimated_radii['AVERAGE'] < 1.5
assert abs(estimated_radii['MAXIMUM'] - 1.5) < 1.0e-5

redo_estimator = SimpleNamespace(
    fit_valid=True,
    diameter_estimator='MINIMUM',
    fit_radius_minimum=estimated_radii['MINIMUM'],
    fit_radius_average=estimated_radii['AVERAGE'],
    fit_radius_maximum=estimated_radii['MAXIMUM'],
    fit_radius=estimated_radii['AVERAGE'],
    diameter=estimated_radii['AVERAGE'] * 2.0,
    _fit={"radius": estimated_radii['AVERAGE']},
)
_update_diameter_estimator(redo_estimator, None)
assert abs(redo_estimator.fit_radius - 1.0) < 1.0e-5
assert abs(redo_estimator.diameter - 2.0) < 1.0e-5
assert abs(redo_estimator._fit["radius"] - 1.0) < 1.0e-5


# Linear connected proportional editing with a radius of two should give an
# adjacent ring at distance one exactly half of the radial displacement.
bpy.ops.object.mode_set(mode='OBJECT')
chamfer_object.select_set(False)
prop_object, prop_bmesh = create_ring_mesh(
    "proportional_test",
    ((1.0, -1.0), (1.0, 1.0), (2.0, 1.0)),
    segments=8,
)
wall_faces = list(prop_bmesh.faces[:8])
operator = SimpleNamespace(
    _object=prop_object,
    _locked_indices={face.index for face in wall_faces},
    scale=1.0,
    diameter=4.0,
)
operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(operator, face)
)
operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(operator)
)
operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    operator, wall_faces
)
operator._end_planes = MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
    operator, wall_faces, operator._fit["axis"]
)

tool_settings = bpy.context.scene.tool_settings
tool_settings.use_proportional_edit = True
tool_settings.use_proportional_connected = True
tool_settings.proportional_size = 2.0
tool_settings.proportional_edit_falloff = 'LINEAR'
selected_vertices = {vert for face in wall_faces for vert in face.verts}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    operator, bpy.context, prop_bmesh, selected_vertices
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(operator, bpy.context)

prop_bmesh = bmesh.from_edit_mesh(prop_object.data)
inner_radius = math.hypot(prop_bmesh.verts[0].co.x, prop_bmesh.verts[0].co.y)
outer_radius = math.hypot(prop_bmesh.verts[16].co.x, prop_bmesh.verts[16].co.y)
assert abs(inner_radius - 2.0) < 1.0e-5
assert abs(outer_radius - 3.0) < 1.0e-5


# Depth is available for two planar boundary loops even when they are not
# parallel. The outer end stays fixed while the oblique free end moves.
bpy.ops.object.mode_set(mode='OBJECT')
prop_object.select_set(False)
depth_object, depth_bmesh = create_ring_mesh(
    "depth_adjustment_test",
    ((1.0, -2.0), (1.0, 2.0)),
)
for vert in depth_bmesh.verts:
    base = -2.0 if vert.index < 16 else 2.0
    vert.co.z = base if vert.index < 16 else base + 0.25 * vert.co.x
    if vert.index == 16:
        # Remain within the scale-aware plane epsilon while ensuring that an
        # unwanted best-plane reprojection would visibly change this vertex.
        vert.co.z += 5.0e-6
depth_bmesh.normal_update()

depth_faces = list(depth_bmesh.faces)
depth_operator = SimpleNamespace(
    _object=depth_object,
    _locked_indices={face.index for face in depth_faces},
    scale=1.0,
    diameter=2.0,
    depth=4.0,
)
depth_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(depth_operator, face)
)
depth_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(depth_operator)
)
depth_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(depth_operator)
)
depth_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    depth_operator, depth_faces
)
depth_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        depth_operator, depth_faces, depth_operator._fit["axis"]
    )
)
depth_operator._depth_fit = (
    MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
        depth_operator, depth_operator._end_planes, depth_operator._fit
    )
)
assert depth_operator._depth_fit is not None, depth_operator._end_planes
assert abs(depth_operator._depth_fit["initial"] - 4.0) < 1.0e-5
assert not depth_operator._depth_fit["parallel"]
assert depth_operator._fit["is_stud"]
assert depth_operator._depth_fit["outer_plane_index"] == 0

# The same adjacent normals classify the exposed hole rim and the model-side
# stud base oppositely, matching the direction of their wall normals.
classified_planes = [dict(plane) for plane in depth_operator._end_planes]
classified_planes[0]["adjacent_normal"] = Vector((0.0, 0.0, 1.0))
classified_planes[1]["adjacent_normal"] = Vector((0.0, 0.0, 1.0))
hole_fit = dict(depth_operator._fit, is_stud=False)
hole_depth = MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    depth_operator, classified_planes, hole_fit
)
stud_depth = MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    depth_operator, classified_planes, depth_operator._fit
)
assert hole_depth["outer_plane_index"] == 1
assert stud_depth["outer_plane_index"] == 0

through_planes = [dict(plane) for plane in classified_planes]
through_planes[0]["adjacent_normal"] = Vector((0.0, 0.0, -1.0))
through_planes[1]["adjacent_normal"] = Vector((0.0, 0.0, 1.0))
assert MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    depth_operator, through_planes, hole_fit
) is None
assert MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    depth_operator, through_planes, depth_operator._fit
) is not None

single_adjacent_planes = [dict(plane) for plane in classified_planes]
single_adjacent_planes[1]["adjacent_normal"] = None
assert MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    depth_operator, single_adjacent_planes, hole_fit
)["outer_plane_index"] == 1

nonplanar_planes = [dict(plane) for plane in depth_operator._end_planes]
nonplanar_planes[0]["max_error"] = nonplanar_planes[0]["tolerance"] * 2.0
assert MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    depth_operator, nonplanar_planes, depth_operator._fit
) is None

nonparallel_planes = [dict(plane) for plane in depth_operator._end_planes]
nonparallel_planes[1]["normal"] = Vector((0.02, 0.0, 1.0)).normalized()
nonparallel_depth = MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    depth_operator, nonparallel_planes, depth_operator._fit
)
assert nonparallel_depth is not None
assert not nonparallel_depth["parallel"]

depth_vertices = {vert for face in depth_faces for vert in face.verts}
tool_settings.use_proportional_edit = False
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    depth_operator, bpy.context, depth_bmesh, depth_vertices
)
identity_coordinates = {
    vert.index: vert.co.copy() for vert in depth_bmesh.verts
}
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    depth_operator, bpy.context
)
depth_bmesh = bmesh.from_edit_mesh(depth_object.data)
for vert in depth_bmesh.verts:
    assert vert.co == identity_coordinates[vert.index], (
        vert.index, vert.co, identity_coordinates[vert.index]
    )

depth_operator.diameter = 3.0
depth_operator.depth = 6.0
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    depth_operator, bpy.context
)

depth_bmesh = bmesh.from_edit_mesh(depth_object.data)
for vert in depth_bmesh.verts:
    assert abs(math.hypot(vert.co.x, vert.co.y) - 1.5) < 1.0e-5, vert.co
    end_height = -2.0 if vert.index < 16 else 4.0
    expected_z = end_height if vert.index < 16 else end_height + 0.25 * vert.co.x
    assert abs(vert.co.z - expected_z) < 1.0e-5, vert.co


# Connected proportional vertices on the model side of a stud base must not
# move axially when depth changes, even though they are inside the falloff radius.
bpy.ops.object.mode_set(mode='OBJECT')
depth_object.select_set(False)
anchored_object, anchored_bmesh = create_ring_mesh(
    "anchored_stud_depth_test",
    ((2.0, -3.0), (1.0, -2.0), (1.0, 2.0)),
)
anchored_faces = list(anchored_bmesh.faces[16:])
anchored_operator = SimpleNamespace(
    _object=anchored_object,
    _locked_indices={face.index for face in anchored_faces},
    scale=1.0,
    diameter=2.0,
    depth=6.0,
)
anchored_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(anchored_operator, face)
)
anchored_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(anchored_operator)
)
anchored_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(anchored_operator)
)
anchored_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    anchored_operator, anchored_faces
)
anchored_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        anchored_operator, anchored_faces, anchored_operator._fit["axis"]
    )
)
anchored_operator._depth_fit = (
    MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
        anchored_operator,
        anchored_operator._end_planes,
        anchored_operator._fit,
    )
)
assert anchored_operator._depth_fit["outer_plane_index"] == 0

tool_settings.use_proportional_edit = True
tool_settings.use_proportional_connected = True
tool_settings.proportional_size = 10.0
tool_settings.proportional_edit_falloff = 'LINEAR'
anchored_vertices = {vert for face in anchored_faces for vert in face.verts}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    anchored_operator, bpy.context, anchored_bmesh, anchored_vertices
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    anchored_operator, bpy.context
)

anchored_bmesh = bmesh.from_edit_mesh(anchored_object.data)
for vert in anchored_bmesh.verts[:16]:
    assert abs(vert.co.z + 3.0) < 1.0e-6, vert.co
for vert in anchored_bmesh.verts[16:32]:
    assert abs(vert.co.z + 2.0) < 1.0e-6, vert.co
for vert in anchored_bmesh.verts[32:]:
    assert abs(vert.co.z - 4.0) < 1.0e-6, vert.co


# Mirror the Geometry Nodes circular-cone fit on an incomplete frustum. Side
# defect edges from omitted faces must not be mistaken for end boundaries.
bpy.ops.object.mode_set(mode='OBJECT')
anchored_object.select_set(False)
cone_object, cone_bmesh = create_ring_mesh(
    "incomplete_cone_test",
    ((2.0, -2.0), (1.0, 2.0)),
    segments=32,
)
missing_faces = [
    cone_bmesh.faces[index] for index in (1, 7, 13, 22)
]
bmesh.ops.delete(cone_bmesh, geom=missing_faces, context='FACES_ONLY')
cone_bmesh.faces.index_update()
cone_bmesh.faces.ensure_lookup_table()
cone_faces = list(cone_bmesh.faces)
cone_operator = SimpleNamespace(
    _object=cone_object,
    _locked_indices={face.index for face in cone_faces},
    diameter_estimator='AVERAGE',
    scale=1.0,
)
cone_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(cone_operator, face)
)
cone_operator.flood_angle = math.radians(60.0)
cone_flood = MESH_OT_holey_tools_resize_hole._flood_fill(
    cone_operator, cone_faces[0]
)
assert len(cone_flood) == len(cone_faces), (len(cone_flood), len(cone_faces))
cone_faces = list(cone_flood)
cone_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    cone_operator, cone_faces
)
assert cone_operator._fit["is_cone"], cone_operator._fit
assert abs(abs(cone_operator._fit["slope"]) - 0.25) < 1.0e-5
assert abs(
    cone_operator._fit["half_angle"] - math.atan(0.25)
) < 1.0e-5
cone_operator._end_planes = MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
    cone_operator,
    cone_faces,
    cone_operator._fit["axis"],
    cone_operator._fit,
)
assert len(cone_operator._end_planes) == 2, cone_operator._end_planes
cone_operator._depth_fit = MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    cone_operator, cone_operator._end_planes, cone_operator._fit
)
assert cone_operator._depth_fit is not None
assert abs(cone_operator._depth_fit["initial"] - 4.0) < 1.0e-5
outer_coordinate = cone_operator._depth_fit["outer_coordinate"]
assert abs(
    cone_operator._fit["radius"]
    + cone_operator._fit["slope"] * outer_coordinate
    - 2.0
) < 1.0e-5

cone_operator._fit["radius_coordinate"] = outer_coordinate
cone_operator.diameter = 4.0
cone_operator.depth = 4.0
cone_operator.conic_angle = cone_operator._fit["half_angle"]
cone_operator.angle_valid = True
cone_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(cone_operator)
)
cone_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(cone_operator)
)
tool_settings.use_proportional_edit = False
cone_vertices = {vert for face in cone_faces for vert in face.verts}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    cone_operator, bpy.context, cone_bmesh, cone_vertices
)

# Increasing depth alone follows the original cone generators. Changing the
# half-angle afterward pivots the taper about the wider top/model-side circle.
cone_operator.depth = 5.0
cone_operator.conic_angle = math.atan(0.1)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    cone_operator, bpy.context
)
cone_bmesh = bmesh.from_edit_mesh(cone_object.data)
outer_points = [vert.co for vert in cone_bmesh.verts[:32]]
inner_points = [vert.co for vert in cone_bmesh.verts[32:]]
assert max(abs(point.z + 2.0) for point in outer_points) < 1.0e-5
assert max(abs(math.hypot(point.x, point.y) - 2.0) for point in outer_points) < 1.0e-5
assert max(abs(point.z - 3.0) for point in inner_points) < 1.0e-5
assert max(abs(math.hypot(point.x, point.y) - 1.5) for point in inner_points) < 1.0e-5

print("HOLEY_TOOLS_MODE_TESTS_OK")
