import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.operator import MESH_OT_holey_tools_resize_hole  # noqa: E402


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

print("HOLEY_TOOLS_MODE_TESTS_OK")
