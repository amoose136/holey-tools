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


def create_split_top_ring_mesh(name, segments=12):
    """Create cylinder faces with a dissolvable vertex in every top edge."""
    vertices = []
    for z in (-1.0, 1.0):
        vertices.extend(
            (
                math.cos(math.tau * index / segments),
                math.sin(math.tau * index / segments),
                z,
            )
            for index in range(segments)
        )
    top_midpoint_start = len(vertices)
    for index in range(segments):
        following = (index + 1) % segments
        angle = math.tau * index / segments
        following_angle = math.tau * following / segments
        vertices.append((
            (math.cos(angle) + math.cos(following_angle)) * 0.5,
            (math.sin(angle) + math.sin(following_angle)) * 0.5,
            1.0,
        ))

    faces = []
    for index in range(segments):
        following = (index + 1) % segments
        faces.append((
            index,
            segments + index,
            top_midpoint_start + index,
            segments + following,
            following,
        ))

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


def create_saddle_end_mesh(name, segments=16):
    """Create a cylindrical wall meeting a non-analytic saddle annulus."""
    vertices = []
    for radius, offset in ((1.0, -1.0), (1.0, 0.0), (2.0, 0.0)):
        for index in range(segments):
            angle = math.tau * index / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vertices.append((x, y, 0.2 * (x * x - y * y) + offset))
    faces = []
    for ring in range(2):
        first = ring * segments
        second = (ring + 1) * segments
        for index in range(segments):
            following = (index + 1) % segments
            faces.append((
                first + index,
                first + following,
                second + following,
                second + index,
            ))
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


def create_closed_ring_solid(name, top_outer_z=1.0, segments=32):
    """Create a watertight annular solid with an inward-facing hole wall."""
    rings = (
        (1.0, -1.0),
        (1.0, 0.0),
        (2.0, top_outer_z),
        (2.0, -1.0),
    )
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
    for index in range(segments):
        following = (index + 1) % segments
        inner_bottom = index
        inner_top = segments + index
        outer_top = segments * 2 + index
        outer_bottom = segments * 3 + index
        next_inner_bottom = following
        next_inner_top = segments + following
        next_outer_top = segments * 2 + following
        next_outer_bottom = segments * 3 + following
        faces.extend((
            (
                inner_bottom,
                inner_top,
                next_inner_top,
                next_inner_bottom,
            ),
            (inner_top, outer_top, next_outer_top, next_inner_top),
            (
                outer_top,
                outer_bottom,
                next_outer_bottom,
                next_outer_top,
            ),
            (
                outer_bottom,
                inner_bottom,
                next_inner_bottom,
                next_outer_bottom,
            ),
        ))
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


def create_planar_blind_hole(name, segments=32):
    """Create a closed solid with separate planar exterior and blind cap rims."""
    rings = (
        (1.0, -1.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, -2.0),
    )
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
    blind_center = len(vertices)
    vertices.append((0.0, 0.0, -1.0))
    base_center = len(vertices)
    vertices.append((0.0, 0.0, -2.0))

    inner_wall = []
    exterior = []
    outer_wall = []
    blind_cap = []
    base = []
    for index in range(segments):
        following = (index + 1) % segments
        inner_bottom = index
        inner_top = segments + index
        outer_top = segments * 2 + index
        outer_bottom = segments * 3 + index
        next_inner_bottom = following
        next_inner_top = segments + following
        next_outer_top = segments * 2 + following
        next_outer_bottom = segments * 3 + following
        inner_wall.append((
            inner_bottom,
            inner_top,
            next_inner_top,
            next_inner_bottom,
        ))
        exterior.append((
            inner_top,
            outer_top,
            next_outer_top,
            next_inner_top,
        ))
        outer_wall.append((
            outer_top,
            outer_bottom,
            next_outer_bottom,
            next_outer_top,
        ))
        blind_cap.append((inner_bottom, next_inner_bottom, blind_center))
        base.append((outer_bottom, base_center, next_outer_bottom))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        vertices,
        [],
        inner_wall + exterior + outer_wall + blind_cap + base,
    )
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    return obj, bm


def create_planar_to_sloped_blind_hole(name, segments=32):
    """Create a planar hole collar that becomes sloped beyond radius 1.25."""
    rings = (
        (1.0, -1.0),
        (1.0, 0.0),
        (1.25, 0.0),
        (2.0, 0.45),
        (2.0, -2.0),
    )
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
    blind_center = len(vertices)
    vertices.append((0.0, 0.0, -1.0))
    base_center = len(vertices)
    vertices.append((0.0, 0.0, -2.0))

    faces = []
    for ring in range(len(rings) - 1):
        first = ring * segments
        following_ring = (ring + 1) * segments
        for index in range(segments):
            following = (index + 1) % segments
            faces.append((
                first + index,
                following_ring + index,
                following_ring + following,
                first + following,
            ))
    for index in range(segments):
        following = (index + 1) % segments
        faces.append((index, following, blind_center))
        outer_bottom = (len(rings) - 1) * segments
        faces.append((
            outer_bottom + index,
            base_center,
            outer_bottom + following,
        ))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    return obj, bm


def create_blind_spherical_hole(
    name,
    segments=32,
    blind_center_z=None,
    blind_cap_rings=0,
):
    """Create a closed spherical cap with a blind cylindrical hole."""
    sphere_radius = 3.0
    hole_radius = 1.0
    support_radius = 1.01
    outer_radius = 2.0
    hole_bottom = 1.5

    def sphere_height(radius):
        return math.sqrt(sphere_radius * sphere_radius - radius * radius)

    rings = (
        (hole_radius, hole_bottom),
        (hole_radius, sphere_height(hole_radius)),
        (support_radius, sphere_height(support_radius)),
        (outer_radius, sphere_height(outer_radius)),
        (outer_radius, 0.0),
    )
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
    cap_center_z = hole_bottom if blind_center_z is None else blind_center_z
    cap_ring_starts = [0]
    for ring_index in range(1, blind_cap_rings + 1):
        fraction = ring_index / (blind_cap_rings + 1)
        radius = hole_radius * (1.0 - fraction)
        z = hole_bottom + (cap_center_z - hole_bottom) * fraction
        cap_ring_starts.append(len(vertices))
        vertices.extend(
            (
                radius * math.cos(math.tau * index / segments),
                radius * math.sin(math.tau * index / segments),
                z,
            )
            for index in range(segments)
        )
    blind_center = len(vertices)
    vertices.append((
        0.0,
        0.0,
        cap_center_z,
    ))
    base_center = len(vertices)
    vertices.append((0.0, 0.0, 0.0))

    inner_wall = []
    rim_support = []
    spherical_shell = []
    outer_wall = []
    blind_cap = []
    base = []
    for index in range(segments):
        following = (index + 1) % segments
        inner_bottom = index
        inner_top = segments + index
        support = segments * 2 + index
        outer_top = segments * 3 + index
        outer_bottom = segments * 4 + index
        next_inner_bottom = following
        next_inner_top = segments + following
        next_support = segments * 2 + following
        next_outer_top = segments * 3 + following
        next_outer_bottom = segments * 4 + following
        inner_wall.append((
            inner_bottom,
            inner_top,
            next_inner_top,
            next_inner_bottom,
        ))
        rim_support.append((
            inner_top,
            support,
            next_support,
            next_inner_top,
        ))
        spherical_shell.append((
            support,
            outer_top,
            next_outer_top,
            next_support,
        ))
        outer_wall.append((
            outer_top,
            outer_bottom,
            next_outer_bottom,
            next_outer_top,
        ))
        base.append((outer_bottom, base_center, next_outer_bottom))

    for ring_index in range(len(cap_ring_starts) - 1):
        outer_start = cap_ring_starts[ring_index]
        inner_start = cap_ring_starts[ring_index + 1]
        for index in range(segments):
            following = (index + 1) % segments
            blind_cap.append((
                outer_start + index,
                outer_start + following,
                inner_start + following,
                inner_start + index,
            ))
    cap_inner_start = cap_ring_starts[-1]
    for index in range(segments):
        following = (index + 1) % segments
        blind_cap.append((
            cap_inner_start + index,
            cap_inner_start + following,
            blind_center,
        ))

    # Keep the inner wall first so tests can select it without depending on
    # Blender's later face-index assignment for unrelated surface categories.
    faces = (
        inner_wall
        + rim_support
        + spherical_shell
        + outer_wall
        + blind_cap
        + base
    )
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=False)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    return obj, bm


def create_blind_cube_edge_hole(name, segments=32):
    """Boolean a blind cylinder through the edge shared by two cube faces."""
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (2.0, 2.0, 2.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    entrance = Vector((2.0, 0.0, 2.0))
    axis = Vector((-1.0, 0.0, -1.0)).normalized()
    cutter_depth = 2.4
    cutter_center = entrance + axis * 1.0
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=segments,
        radius=0.5,
        depth=cutter_depth,
        end_fill_type='NGON',
        location=cutter_center,
        rotation=axis.to_track_quat('Z', 'Y').to_euler(),
    )
    cutter = bpy.context.active_object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    cutter.select_set(False)
    modifier = obj.modifiers.new(name="blind edge test", type='BOOLEAN')
    modifier.operation = 'DIFFERENCE'
    modifier.solver = 'EXACT'
    modifier.object = cutter
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    bpy.data.objects.remove(cutter, do_unlink=True)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    wall_faces = []
    for face in bm.faces:
        point = face.calc_center_median()
        offset = point - entrance
        coordinate = offset.dot(axis)
        radial = offset - axis * coordinate
        if (
            -0.05 <= coordinate <= cutter_depth
            and abs(radial.length - 0.5) <= 0.03
            and radial.length_squared > 1.0e-12
            and face.normal.dot(radial.normalized()) < -0.8
        ):
            wall_faces.append(face)
    return obj, bm, wall_faces, entrance, axis


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


# Growing the radius projects the moved top rim along the hole axis onto the
# undeformed base mesh instead of reusing the old rim plane. The sloped annulus
# rises one unit while its radius grows from one to two.
bpy.ops.object.mode_set(mode='OBJECT')
prop_object.select_set(False)
growth_object, growth_bmesh = create_ring_mesh(
    "base_surface_projection_test",
    ((1.0, -1.0), (1.0, 0.0), (2.0, 1.0)),
)
growth_faces = list(growth_bmesh.faces[:16])
growth_operator = SimpleNamespace(
    _object=growth_object,
    _locked_indices={face.index for face in growth_faces},
    scale=1.0,
    diameter=3.0,
    angle_valid=False,
)
growth_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(growth_operator, face)
)
growth_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(growth_operator)
)
growth_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    growth_operator, growth_faces
)
growth_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        growth_operator,
        growth_faces,
        growth_operator._fit["axis"],
        growth_operator._fit,
    )
)
tool_settings.use_proportional_edit = False
growth_vertices = {vert for face in growth_faces for vert in face.verts}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    growth_operator, bpy.context, growth_bmesh, growth_vertices
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    growth_operator, bpy.context
)
growth_bmesh = bmesh.from_edit_mesh(growth_object.data)
assert max(
    abs(math.hypot(vert.co.x, vert.co.y) - 1.5)
    for vert in growth_bmesh.verts[16:32]
) < 1.0e-5
assert max(abs(vert.co.z - 0.5) for vert in growth_bmesh.verts[16:32]) < 1.0e-5


# Shrinking first moves the rim radially inward, then intersects the virtual
# fairing model. The same annulus fits a cone z = radius - 1, which extrapolates
# the new half-radius rim to z = -0.5 without constructing a real cap.
bpy.ops.object.mode_set(mode='OBJECT')
growth_object.select_set(False)
shrink_object, shrink_bmesh = create_ring_mesh(
    "fair_surface_projection_test",
    ((1.0, -1.0), (1.0, 0.0), (2.0, 1.0)),
)
shrink_faces = list(shrink_bmesh.faces[:16])
shrink_operator = SimpleNamespace(
    _object=shrink_object,
    _locked_indices={face.index for face in shrink_faces},
    scale=1.0,
    diameter=1.0,
    angle_valid=False,
)
shrink_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(shrink_operator, face)
)
shrink_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(shrink_operator)
)
shrink_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    shrink_operator, shrink_faces
)
shrink_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        shrink_operator,
        shrink_faces,
        shrink_operator._fit["axis"],
        shrink_operator._fit,
    )
)
shrink_vertices = {vert for face in shrink_faces for vert in face.verts}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    shrink_operator, bpy.context, shrink_bmesh, shrink_vertices
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    shrink_operator, bpy.context
)
assert (
    shrink_operator._resize_surfaces_by_vertex[16]["hypothesis"]["method"]
    == "CONIC"
)
shrink_bmesh = bmesh.from_edit_mesh(shrink_object.data)
assert max(
    abs(math.hypot(vert.co.x, vert.co.y) - 0.5)
    for vert in shrink_bmesh.verts[16:32]
) < 1.0e-5
assert max(abs(vert.co.z + 0.5) for vert in shrink_bmesh.verts[16:32]) < 1.0e-4

# Curved analytic endcaps are intersected directly. At radius 0.5, an axis ray
# through a radius-three sphere reaches its upper sheet at sqrt(9 - 0.25).
sphere_projection = (
    MESH_OT_holey_tools_resize_hole._project_hypothesis_along_axis(
        {
            "method": "SPHERE",
            "center": Vector((0.0, 0.0, 0.0)),
            "radius": 3.0,
        },
        Vector((0.5, 0.0, 0.0)),
        Vector((1.0, 0.0, 3.0)),
        Vector((0.0, 0.0, 1.0)),
    )
)
assert sphere_projection is not None
assert abs(sphere_projection.z - math.sqrt(8.75)) < 1.0e-6


# A saddle annulus rejects the analytic hypotheses. Shrinking therefore rays
# against an organically faired patch built on a temporary wall-free copy.
bpy.ops.object.mode_set(mode='OBJECT')
shrink_object.select_set(False)
organic_object, organic_bmesh = create_saddle_end_mesh(
    "organic_resize_projection_test"
)
organic_faces = list(organic_bmesh.faces[:16])
organic_operator = SimpleNamespace(
    _object=organic_object,
    _locked_indices={face.index for face in organic_faces},
    scale=1.0,
    diameter=1.0,
    angle_valid=False,
)
organic_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(organic_operator, face)
)
organic_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(organic_operator)
)
organic_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    organic_operator, organic_faces
)
organic_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        organic_operator,
        organic_faces,
        organic_operator._fit["axis"],
        organic_operator._fit,
    )
)
organic_vertices = {vert for face in organic_faces for vert in face.verts}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    organic_operator, bpy.context, organic_bmesh, organic_vertices
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    organic_operator, bpy.context
)
organic_surface = organic_operator._resize_surfaces_by_vertex[16]
assert organic_surface["hypothesis"]["method"] == "ORGANIC", organic_surface
assert organic_surface.get("organic_bvh") is not None
organic_bmesh = bmesh.from_edit_mesh(organic_object.data)
organic_top = list(organic_bmesh.verts[16:32])
assert max(
    abs(math.hypot(vert.co.x, vert.co.y) - 0.5)
    for vert in organic_top
) < 1.0e-5
assert max(vert.co.z for vert in organic_top) - min(
    vert.co.z for vert in organic_top
) > 0.02, (min(vert.co.z for vert in organic_top), max(vert.co.z for vert in organic_top))


# An unchanged Boolean blind base is a single plane. Remove the triangulated
# center without dissolving its circular perimeter into the adjacent wall.
bpy.ops.object.mode_set(mode='OBJECT')
organic_object.select_set(False)
base_cleanup_object, base_cleanup_bmesh = create_planar_blind_hole(
    "unchanged_boolean_base_cleanup_test"
)
base_cleanup_wall = list(base_cleanup_bmesh.faces[:32])
base_cleanup_operator = SimpleNamespace(_object=base_cleanup_object)
base_cleanup_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(
        base_cleanup_operator, face
    )
)
base_cleanup_operator._fit = (
    MESH_OT_holey_tools_resize_hole._fit_cylinder(
        base_cleanup_operator, base_cleanup_wall
    )
)
base_cleanup_cutter = []
for z_coordinate in (0.25, -1.0):
    base_cleanup_cutter.extend(
        Vector((
            math.cos(math.tau * index / 32),
            math.sin(math.tau * index / 32),
            z_coordinate,
        ))
        for index in range(32)
    )
MESH_OT_holey_tools_resize_hole._cleanup_unchanged_boolean_base(
    base_cleanup_operator,
    base_cleanup_object,
    base_cleanup_cutter,
)
base_cleanup_bmesh = bmesh.from_edit_mesh(base_cleanup_object.data)
base_cleanup_faces = [
    face
    for face in base_cleanup_bmesh.faces
    if all(abs(vert.co.z + 1.0) < 1.0e-6 for vert in face.verts)
    and max(math.hypot(vert.co.x, vert.co.y) for vert in face.verts)
    <= 1.000001
]
assert len(base_cleanup_faces) == 1, base_cleanup_faces
assert len(base_cleanup_faces[0].verts) == 32


# A true hole bounded by one plane on each exterior side remains in the direct
# deformation path; it does not need reconstruction or a Boolean.
bpy.ops.object.mode_set(mode='OBJECT')
base_cleanup_object.select_set(False)
planar_hole_object, planar_hole_bmesh = create_closed_ring_solid(
    "single_plane_resize_test", top_outer_z=0.0
)
planar_hole_faces = [
    planar_hole_bmesh.faces[index * 4] for index in range(32)
]
planar_hole_operator = SimpleNamespace(
    _object=planar_hole_object,
    _locked_indices={face.index for face in planar_hole_faces},
    scale=1.0,
    diameter=3.0,
    angle_valid=False,
)
planar_hole_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(planar_hole_operator, face)
)
planar_hole_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(planar_hole_operator)
)
planar_hole_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    planar_hole_operator, planar_hole_faces
)
assert not planar_hole_operator._fit["is_stud"]
planar_hole_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        planar_hole_operator,
        planar_hole_faces,
        planar_hole_operator._fit["axis"],
        planar_hole_operator._fit,
    )
)
planar_hole_vertices = {
    vert for face in planar_hole_faces for vert in face.verts
}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    planar_hole_operator,
    bpy.context,
    planar_hole_bmesh,
    planar_hole_vertices,
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    planar_hole_operator, bpy.context
)
assert not planar_hole_operator._resize_requires_boolean
planar_hole_bmesh = bmesh.from_edit_mesh(planar_hole_object.data)
assert max(
    abs(math.hypot(vert.co.x, vert.co.y) - 1.5)
    for vert in planar_hole_bmesh.verts[:64]
) < 1.0e-5

# Finalizing a smaller direct resize selects the inset wall, expands that face
# region once, and applies the same 0.1-degree limited-dissolve cleanup.
planar_hole_operator.diameter = 1.0
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    planar_hole_operator, bpy.context, commit_boolean=True
)
planar_hole_bmesh = bmesh.from_edit_mesh(planar_hole_object.data)
assert any(face.select for face in planar_hole_bmesh.faces)
assert min(
    math.hypot(vert.co.x, vert.co.y)
    for vert in planar_hole_bmesh.verts
) < 0.50001


# A one-plane blind hole remains a direct resize, but its two rims belong to
# different planes. Enlargement projects the exterior rim onto the surrounding
# mesh and the back rim onto its independently fitted blind-cap plane.
bpy.ops.object.mode_set(mode='OBJECT')
planar_hole_object.select_set(False)
planar_blind_object, planar_blind_bmesh = create_planar_blind_hole(
    "single_plane_blind_resize_test"
)
planar_blind_faces = list(planar_blind_bmesh.faces[:32])
planar_blind_operator = SimpleNamespace(
    _object=planar_blind_object,
    _locked_indices={face.index for face in planar_blind_faces},
    scale=1.0,
    diameter=3.0,
    angle_valid=False,
    _depth_fit=None,
)
planar_blind_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(
        planar_blind_operator, face
    )
)
planar_blind_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(planar_blind_operator)
)
planar_blind_operator._fit = (
    MESH_OT_holey_tools_resize_hole._fit_cylinder(
        planar_blind_operator, planar_blind_faces
    )
)
assert not planar_blind_operator._fit["is_stud"]
planar_blind_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        planar_blind_operator,
        planar_blind_faces,
        planar_blind_operator._fit["axis"],
        planar_blind_operator._fit,
    )
)
planar_blind_vertices = {
    vert for face in planar_blind_faces for vert in face.verts
}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    planar_blind_operator,
    bpy.context,
    planar_blind_bmesh,
    planar_blind_vertices,
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    planar_blind_operator, bpy.context
)
assert not planar_blind_operator._resize_requires_boolean
blind_bottom_surface = (
    planar_blind_operator._resize_surfaces_by_vertex[0]
)
blind_top_surface = (
    planar_blind_operator._resize_surfaces_by_vertex[32]
)
assert not blind_bottom_surface["exterior"]
assert blind_top_surface["exterior"]
assert blind_bottom_surface is not blind_top_surface
assert blind_bottom_surface["hypothesis"]["method"] == "PLANES"
assert blind_top_surface["hypothesis"]["method"] == "PLANES"
planar_blind_bmesh = bmesh.from_edit_mesh(planar_blind_object.data)
for vert in planar_blind_bmesh.verts[:32]:
    assert abs(math.hypot(vert.co.x, vert.co.y) - 1.5) < 1.0e-5
    assert abs(vert.co.z + 1.0) < 1.0e-5
for vert in planar_blind_bmesh.verts[32:64]:
    assert abs(math.hypot(vert.co.x, vert.co.y) - 1.5) < 1.0e-5
    assert abs(vert.co.z) < 1.0e-5


# Exact reconstruction of a square opening on a single plane must retain the
# already accepted plane instead of automatically refitting the reduced rim.
# The temporary cap and the resulting Boolean top surface stay exactly flat.
bpy.ops.object.mode_set(mode='OBJECT')
planar_blind_object.select_set(False)
square_object, square_bmesh = create_planar_blind_hole(
    "square_planar_boolean_resize_test", segments=4
)
square_faces = list(square_bmesh.faces[:4])
square_operator = SimpleNamespace(
    _object=square_object,
    _locked_indices={face.index for face in square_faces},
    scale=1.0,
    diameter=2.5,
    angle_valid=False,
    _depth_fit=None,
    fairing_method='PLANES',
    fill_style='CIRCLE',
    circle_loops=1,
    grid_angle=0.0,
    final_simplify=False,
)
square_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(square_operator, face)
)
square_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(square_operator)
)
square_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    square_operator, square_faces
)
square_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        square_operator,
        square_faces,
        square_operator._fit["axis"],
        square_operator._fit,
    )
)
square_vertices = {vert for face in square_faces for vert in face.verts}
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    square_operator, square_bmesh, square_faces
)
square_surfaces = {
    id(surface): surface
    for surface in square_operator._resize_surfaces_by_vertex.values()
}.values()
square_exterior = [surface for surface in square_surfaces if surface["exterior"]]
assert len(square_exterior) == 1
assert square_exterior[0]["hypothesis"]["method"] == "PLANES"
assert len(square_exterior[0]["hypothesis"]["planes"]) == 1
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    square_operator, bpy.context, square_bmesh, square_vertices
)
square_operator._resize_dynamic_requires_boolean = True
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    square_operator, bpy.context, commit_boolean=True
)
square_bmesh = bmesh.from_edit_mesh(square_object.data)
assert all(len(edge.link_faces) == 2 for edge in square_bmesh.edges)
square_top_faces = [
    face
    for face in square_bmesh.faces
    if face.normal.z > 0.9 and face.calc_center_median().z > -0.1
]
assert square_top_faces
assert max(
    abs(vert.co.z)
    for face in square_top_faces
    for vert in face.verts
) < 1.0e-6


# A rim can begin on a true plane but grow beyond that collar onto a sloped
# cap. The target-radius BVH probe must switch to reconstruction before direct
# projection deforms any source vertices.
bpy.ops.object.mode_set(mode='OBJECT')
square_object.select_set(False)
transition_object, transition_bmesh = create_planar_to_sloped_blind_hole(
    "planar_to_nonplanar_resize_test"
)
transition_faces = list(transition_bmesh.faces[:32])
transition_operator = SimpleNamespace(
    _object=transition_object,
    _locked_indices={face.index for face in transition_faces},
    scale=1.0,
    diameter=3.0,
    angle_valid=False,
    _depth_fit=None,
)
transition_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(
        transition_operator, face
    )
)
transition_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(transition_operator)
)
transition_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    transition_operator, transition_faces
)
transition_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        transition_operator,
        transition_faces,
        transition_operator._fit["axis"],
        transition_operator._fit,
    )
)
transition_vertices = {
    vert for face in transition_faces for vert in face.verts
}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    transition_operator,
    bpy.context,
    transition_bmesh,
    transition_vertices,
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    transition_operator, bpy.context
)
assert transition_operator._resize_requires_boolean
assert not transition_operator._resize_base_requires_boolean
transition_bmesh = bmesh.from_edit_mesh(transition_object.data)
assert all(len(edge.link_faces) == 2 for edge in transition_bmesh.edges)
transition_preview_angles = {
    (
        round(radial.dot(transition_operator._fit["basis_x"])
              / radial.length, 5),
        round(radial.dot(transition_operator._fit["basis_y"])
              / radial.length, 5),
    )
    for vert in transition_bmesh.verts
    for offset in (transition_object.matrix_world @ vert.co
                   - transition_operator._fit["center"],)
    for radial in (offset - transition_operator._fit["axis"]
                   * offset.dot(transition_operator._fit["axis"]),)
    if abs(radial.length - 1.5) < 1.0e-4
}
assert len(transition_preview_angles) == 16, transition_preview_angles
assert transition_operator._boolean_preview_source is not None
# Once a planar collar has crossed into reconstruction, keep that decision for
# the rest of the interaction. This prevents tiny pointer/numeric changes near
# the transition radius from alternating between direct and Boolean previews.
transition_operator.diameter = 2.2
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    transition_operator, bpy.context
)
assert transition_operator._resize_requires_boolean
assert transition_operator._resize_dynamic_requires_boolean
transition_operator.diameter = 3.0
MESH_OT_holey_tools_resize_hole._replace_boolean_preview_with_source(
    transition_operator
)
transition_bmesh = bmesh.from_edit_mesh(transition_object.data)
transition_bmesh.faces.ensure_lookup_table()
transition_faces = [
    transition_bmesh.faces[index]
    for index in sorted(transition_operator._locked_indices)
]
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    transition_operator, transition_bmesh, transition_faces
)
transition_operator._resize_dynamic_requires_boolean = True
transition_surfaces = list({
    id(surface): surface
    for surface in transition_operator._resize_surfaces_by_vertex.values()
}.values())
transition_blind_coordinates = (
    MESH_OT_holey_tools_resize_hole._preserved_blind_coordinates(
        transition_operator, transition_surfaces
    )
)
transition_cutter_vertices, _transition_cutter_faces = (
    MESH_OT_holey_tools_resize_hole._boolean_cutter_geometry(
        transition_operator,
        transition_object,
        transition_blind_coordinates,
    )
)
transition_outer, transition_inner = transition_blind_coordinates
transition_direction = 1.0 if transition_inner >= transition_outer else -1.0
transition_axis = transition_operator._fit["axis"]
transition_cutter_outer = (
    transition_object.matrix_world @ transition_cutter_vertices[0]
    - transition_operator._fit["center"]
).dot(transition_axis)
assert (
    (transition_outer - transition_cutter_outer) * transition_direction
    > transition_operator.diameter * 0.1
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    transition_operator, bpy.context, commit_boolean=True
)
transition_bmesh = bmesh.from_edit_mesh(transition_object.data)
assert all(len(edge.link_faces) == 2 for edge in transition_bmesh.edges)


# Angular resolution comes from wall-face normals, not boundary vertex angles.
# A dissolvable midpoint in every top edge doubles the apparent vertex samples
# while the cylinder still has only twelve actual angular facets.
bpy.ops.object.mode_set(mode='OBJECT')
transition_object.select_set(False)
split_top_object, split_top_bmesh = create_split_top_ring_mesh(
    "split_top_segment_estimate", segments=12
)
split_top_faces = list(split_top_bmesh.faces)
split_top_operator = SimpleNamespace(
    _object=split_top_object,
    _locked_indices={face.index for face in split_top_faces},
    diameter_estimator='AVERAGE',
)
split_top_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(split_top_operator, face)
)
split_top_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    split_top_operator, split_top_faces
)
split_vertex_angles = {
    round(math.atan2(vert.co.y, vert.co.x) % math.tau, 6)
    for vert in split_top_bmesh.verts
}
assert len(split_vertex_angles) == 24
assert MESH_OT_holey_tools_resize_hole._estimate_angular_segments(
    split_top_operator, split_top_faces
) == 12
assert MESH_OT_holey_tools_resize_hole._estimate_angular_segments(
    split_top_operator, split_top_faces[2:]
) == 12


# A real hole whose exterior support is conical gets a low-resolution live
# Boolean result, then a source-resolution reconstruction at confirmation. Both
# remain closed manifolds with the requested 1.5-unit through-hole radius.
bpy.ops.object.mode_set(mode='OBJECT')
split_top_object.select_set(False)
boolean_object, boolean_bmesh = create_closed_ring_solid(
    "boolean_resize_test"
)
boolean_faces = [boolean_bmesh.faces[index * 4] for index in range(32)]
boolean_operator = SimpleNamespace(
    _object=boolean_object,
    _locked_indices={face.index for face in boolean_faces},
    scale=1.0,
    diameter=3.0,
    angle_valid=False,
)
boolean_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(boolean_operator, face)
)
boolean_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(boolean_operator)
)
boolean_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    boolean_operator, boolean_faces
)
assert not boolean_operator._fit["is_stud"], boolean_operator._fit
boolean_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        boolean_operator,
        boolean_faces,
        boolean_operator._fit["axis"],
        boolean_operator._fit,
    )
)
boolean_operator._depth_fit = (
    MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
        boolean_operator,
        boolean_operator._end_planes,
        boolean_operator._fit,
    )
)
boolean_vertices = {vert for face in boolean_faces for vert in face.verts}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    boolean_operator, bpy.context, boolean_bmesh, boolean_vertices
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    boolean_operator, bpy.context
)
assert boolean_operator._resize_requires_boolean
assert boolean_operator._resize_angular_segments == 32
boolean_bmesh = bmesh.from_edit_mesh(boolean_object.data)
assert all(len(edge.link_faces) == 2 for edge in boolean_bmesh.edges)
preview_angles = {
    (
        round(vert.co.x / math.hypot(vert.co.x, vert.co.y), 5),
        round(vert.co.y / math.hypot(vert.co.x, vert.co.y), 5),
    )
    for vert in boolean_bmesh.verts
    if abs(math.hypot(vert.co.x, vert.co.y) - 1.5) < 1.0e-4
}
assert len(preview_angles) == 16, preview_angles
# Pointer moves inside the half-second window retain the previous Boolean mesh.
# Ending the drag performs one trailing rebuild at the latest requested value.
preview_mesh_pointer = boolean_object.data.as_pointer()
boolean_operator._drag_active = True
boolean_operator._boolean_preview_last_update = 1.0e30
boolean_operator.diameter = 2.8
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    boolean_operator, bpy.context
)
assert boolean_object.data.as_pointer() == preview_mesh_pointer
assert boolean_operator._boolean_preview_pending
boolean_operator._drag_active = False
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    boolean_operator, bpy.context
)
assert boolean_object.data.as_pointer() != preview_mesh_pointer
assert not boolean_operator._boolean_preview_pending
boolean_operator.diameter = 3.0
boolean_operator.boolean_segments = 12
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    boolean_operator, bpy.context, commit_boolean=True
)
boolean_bmesh = bmesh.from_edit_mesh(boolean_object.data)
boolean_bmesh.edges.ensure_lookup_table()
assert all(len(edge.link_faces) == 2 for edge in boolean_bmesh.edges)
minimum_radius = min(
    math.hypot(vert.co.x, vert.co.y) for vert in boolean_bmesh.verts
)
expected_apothem = 1.5 * math.cos(math.pi / boolean_operator.boolean_segments)
assert abs(minimum_radius - expected_apothem) < 1.0e-4, minimum_radius
final_angles = {
    (
        round(vert.co.x / math.hypot(vert.co.x, vert.co.y), 5),
        round(vert.co.y / math.hypot(vert.co.x, vert.co.y), 5),
    )
    for vert in boolean_bmesh.verts
    if abs(math.hypot(vert.co.x, vert.co.y) - 1.5) < 1.0e-4
}
assert len(final_angles) == 12, final_angles
assert boolean_operator._boolean_preview_source is None
assert not any(
    obj.name.startswith("Holey Tools Boolean Cutter")
    for obj in bpy.data.objects
)

# Shrinking the same non-planar hole also uses reconstruction. The retained
# conic patch continues to z = radius - 1 before the smaller cutter is applied.
bpy.ops.object.mode_set(mode='OBJECT')
boolean_object.select_set(False)
boolean_shrink_object, boolean_shrink_bmesh = create_closed_ring_solid(
    "boolean_shrink_test"
)
boolean_shrink_faces = [
    boolean_shrink_bmesh.faces[index * 4] for index in range(32)
]
boolean_shrink_operator = SimpleNamespace(
    _object=boolean_shrink_object,
    _locked_indices={face.index for face in boolean_shrink_faces},
    scale=1.0,
    diameter=1.0,
    angle_valid=False,
)
boolean_shrink_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(
        boolean_shrink_operator, face
    )
)
boolean_shrink_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(boolean_shrink_operator)
)
boolean_shrink_operator._fit = (
    MESH_OT_holey_tools_resize_hole._fit_cylinder(
        boolean_shrink_operator, boolean_shrink_faces
    )
)
boolean_shrink_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        boolean_shrink_operator,
        boolean_shrink_faces,
        boolean_shrink_operator._fit["axis"],
        boolean_shrink_operator._fit,
    )
)
boolean_shrink_operator._depth_fit = None
boolean_shrink_vertices = {
    vert for face in boolean_shrink_faces for vert in face.verts
}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    boolean_shrink_operator,
    bpy.context,
    boolean_shrink_bmesh,
    boolean_shrink_vertices,
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    boolean_shrink_operator, bpy.context, commit_boolean=True
)
assert boolean_shrink_operator._resize_requires_boolean
boolean_shrink_bmesh = bmesh.from_edit_mesh(boolean_shrink_object.data)
assert all(len(edge.link_faces) == 2 for edge in boolean_shrink_bmesh.edges)
minimum_radius = min(
    math.hypot(vert.co.x, vert.co.y)
    for vert in boolean_shrink_bmesh.verts
)
assert abs(minimum_radius - 0.5) < 1.0e-4, minimum_radius
assert any(face.select for face in boolean_shrink_bmesh.faces)
new_rim = [
    vert for vert in boolean_shrink_bmesh.verts
    if abs(math.hypot(vert.co.x, vert.co.y) - 0.5) < 1.0e-4
]
assert abs(max(vert.co.z for vert in new_rim) + 0.5) < 5.0e-3, (
    min(vert.co.z for vert in new_rim),
    max(vert.co.z for vert in new_rim),
)


# A spherical exterior can curve inward enough that the centers of the first
# support faces fall inside the old cylindrical cavity envelope. Reconstruction
# must preserve those accepted support faces, remove the blind cap, fair the
# old rim, and only then apply the smaller Boolean cutter.
bpy.ops.object.mode_set(mode='OBJECT')
boolean_shrink_object.select_set(False)
blind_sphere_object, blind_sphere_bmesh = create_blind_spherical_hole(
    "blind_spherical_resize_test"
)
blind_sphere_faces = list(blind_sphere_bmesh.faces[:32])
blind_sphere_operator = SimpleNamespace(
    _object=blind_sphere_object,
    _locked_indices={face.index for face in blind_sphere_faces},
    scale=1.0,
    diameter=0.7,
    angle_valid=False,
)
blind_sphere_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(
        blind_sphere_operator, face
    )
)
blind_sphere_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(blind_sphere_operator)
)
blind_sphere_operator._fit = (
    MESH_OT_holey_tools_resize_hole._fit_cylinder(
        blind_sphere_operator, blind_sphere_faces
    )
)
assert not blind_sphere_operator._fit["is_stud"]
blind_sphere_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        blind_sphere_operator,
        blind_sphere_faces,
        blind_sphere_operator._fit["axis"],
        blind_sphere_operator._fit,
    )
)
# Match the reported blind-hole case, whose second boundary does not provide a
# sufficiently planar/dependable depth fit.
blind_sphere_operator._depth_fit = None
blind_sphere_vertices = {
    vert for face in blind_sphere_faces for vert in face.verts
}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    blind_sphere_operator,
    bpy.context,
    blind_sphere_bmesh,
    blind_sphere_vertices,
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    blind_sphere_operator, bpy.context
)
blind_sphere_preview = bmesh.from_edit_mesh(blind_sphere_object.data)
assert all(len(edge.link_faces) == 2 for edge in blind_sphere_preview.edges)
MESH_OT_holey_tools_resize_hole._replace_boolean_preview_with_source(
    blind_sphere_operator
)
blind_sphere_bmesh = bmesh.from_edit_mesh(blind_sphere_object.data)
blind_sphere_bmesh.faces.ensure_lookup_table()
blind_sphere_faces = [
    blind_sphere_bmesh.faces[index]
    for index in sorted(blind_sphere_operator._locked_indices)
]
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    blind_sphere_operator, blind_sphere_bmesh, blind_sphere_faces
)
blind_sphere_operator._resize_dynamic_requires_boolean = True
blind_surfaces = {
    id(surface): surface
    for surface in blind_sphere_operator._resize_surfaces_by_vertex.values()
}.values()
blind_exterior = [
    surface for surface in blind_surfaces if surface["exterior"]
]
blind_interior = [
    surface for surface in blind_surfaces if not surface["exterior"]
]
assert len(blind_exterior) == 1, blind_exterior
assert len(blind_interior) == 1, blind_interior
assert blind_sphere_operator._resize_requires_boolean
assert all(
    math.hypot(
        face.calc_center_median().x,
        face.calc_center_median().y,
    ) <= 1.02
    for face in blind_exterior[0]["support_faces"]
)
MESH_OT_holey_tools_resize_hole._store_fit_for_redo(
    blind_sphere_operator
)
assert blind_sphere_operator._depth_fit is not None, (
    blind_exterior[0]["hypothesis"],
    blind_interior[0]["hypothesis"],
)
assert blind_sphere_operator._depth_fit["surface_depth"]
assert blind_sphere_operator.depth_valid
assert blind_sphere_operator.fit_depth_outer_plane == -1
blind_sphere_operator.diameter = 0.7
blind_sphere_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(
        blind_sphere_operator
    )
)
blind_outer_coordinate = blind_sphere_operator._depth_fit["outer_coordinate"]
blind_inner_coordinate = blind_sphere_operator._depth_fit["inner_coordinate"]
assert blind_sphere_operator.depth > 1.0
# Reconstruction must rediscover the post-delete boundary from the protected
# support patch rather than relying on pre-delete BMesh edge objects.
blind_exterior[0]["loop_verts"] = []
blind_exterior[0]["loop_edges"] = []
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    blind_sphere_operator, bpy.context, commit_boolean=True
)
blind_sphere_bmesh = bmesh.from_edit_mesh(blind_sphere_object.data)
blind_sphere_bmesh.edges.ensure_lookup_table()
assert all(len(edge.link_faces) == 2 for edge in blind_sphere_bmesh.edges)
target_radius = blind_sphere_operator.diameter * 0.5
blind_radius_tolerance = max(target_radius * 0.06, 1.0e-3)
cut_vertices = [
    vert for vert in blind_sphere_bmesh.verts
    if vert.co.z > 1.49
    and abs(
        math.hypot(vert.co.x, vert.co.y) - target_radius
    ) < blind_radius_tolerance
]
assert cut_vertices
blind_direction = (
    1.0 if blind_inner_coordinate >= blind_outer_coordinate else -1.0
)
blind_depth = abs(blind_inner_coordinate - blind_outer_coordinate)
blind_wall_progress = []
for vert in blind_sphere_bmesh.verts:
    point = blind_sphere_object.matrix_world @ vert.co
    offset = point - blind_sphere_operator._fit["center"]
    coordinate = offset.dot(blind_sphere_operator._fit["axis"])
    radial = offset - blind_sphere_operator._fit["axis"] * coordinate
    if abs(radial.length - target_radius) < blind_radius_tolerance:
        blind_wall_progress.append(
            (coordinate - blind_outer_coordinate) * blind_direction
        )
assert blind_wall_progress
assert max(blind_wall_progress) <= blind_depth + 1.0e-3
assert min(
    abs(progress - blind_depth) for progress in blind_wall_progress
) < 1.0e-3
assert not any(
    obj.name.startswith("Holey Tools Boolean Cutter")
    for obj in bpy.data.objects
)


# Redo restores the original blind mesh before executing again. When the
# diameter shrinks, remove every ring of a dense dished cap rather than leaving
# its out-of-envelope inner rings as one oversized floating bottom.
bpy.ops.object.mode_set(mode='OBJECT')
blind_sphere_object.select_set(False)
redo_through_object, redo_through_bmesh = create_blind_spherical_hole(
    "redo_dense_cap_shrink_test",
    blind_center_z=1.0,
    blind_cap_rings=5,
)
redo_through_faces = list(redo_through_bmesh.faces[:32])
redo_through_operator = SimpleNamespace(
    _object=redo_through_object,
    _locked_indices={face.index for face in redo_through_faces},
    scale=1.0,
    diameter=2.0,
    angle_valid=False,
)
redo_through_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(
        redo_through_operator, face
    )
)
redo_through_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(
        redo_through_operator
    )
)
redo_through_operator._fit = (
    MESH_OT_holey_tools_resize_hole._fit_cylinder(
        redo_through_operator, redo_through_faces
    )
)
redo_through_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        redo_through_operator,
        redo_through_faces,
        redo_through_operator._fit["axis"],
        redo_through_operator._fit,
    )
)
redo_through_vertices = {
    vert for face in redo_through_faces for vert in face.verts
}
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    redo_through_operator,
    redo_through_bmesh,
    redo_through_faces,
)
assert redo_through_operator._resize_requires_boolean
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    redo_through_operator,
    bpy.context,
    redo_through_bmesh,
    redo_through_vertices,
)
MESH_OT_holey_tools_resize_hole._store_fit_for_redo(
    redo_through_operator
)
redo_surfaces = list({
    id(surface): surface
    for surface in redo_through_operator._resize_surfaces_by_vertex.values()
}.values())
redo_exterior = next(surface for surface in redo_surfaces if surface["exterior"])
redo_interior = next(surface for surface in redo_surfaces if not surface["exterior"])
redo_axis = redo_through_operator._fit["axis"]
redo_center = redo_through_operator._fit["center"]
redo_outer_coordinate = sum(
    (point - redo_center).dot(redo_axis)
    for point in redo_exterior["loop_world_points"]
) / len(redo_exterior["loop_world_points"])
redo_inner_coordinate = sum(
    (point - redo_center).dot(redo_axis)
    for point in redo_interior["loop_world_points"]
) / len(redo_interior["loop_world_points"])
redo_through_operator.depth_valid = True
redo_through_operator.fit_depth_initial = abs(
    redo_inner_coordinate - redo_outer_coordinate
)
redo_through_operator.fit_depth_outer = redo_outer_coordinate
redo_through_operator.fit_depth_inner = redo_inner_coordinate
redo_through_operator.fit_depth_outer_plane = -1
redo_through_operator.fit_radius_coordinate = redo_outer_coordinate

# Recreate the transient-state loss that occurs between Adjust Last Operation
# executions while retaining the serialized RNA-style properties and edits.
redo_through_operator.diameter = 0.7
redo_through_operator.depth = redo_through_operator.fit_depth_initial
redo_through_operator._fit = None
redo_through_operator._locked_indices = set()
redo_through_operator._original_vertex_coords = None
redo_through_operator._proportional_weights = None
redo_through_operator._resize_projection_ready = False
redo_through_operator._resize_requires_boolean = False
redo_through_operator._resize_dynamic_requires_boolean = False
redo_through_operator._resize_surfaces_by_vertex = {}
redo_through_operator._resize_base_bvh = None
redo_through_operator._resize_base_span = 0.0
assert MESH_OT_holey_tools_resize_hole._restore_fit_for_redo(
    redo_through_operator
)
redo_through_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(
        redo_through_operator
    )
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    redo_through_operator,
    bpy.context,
    commit_boolean=True,
)
redo_through_bmesh = bmesh.from_edit_mesh(redo_through_object.data)
assert all(len(edge.link_faces) == 2 for edge in redo_through_bmesh.edges)
floating_bottom = []
for face in redo_through_bmesh.faces:
    center = face.calc_center_median()
    radial = math.hypot(center.x, center.y)
    if 0.95 < center.z < 1.55 and 0.42 < radial < 1.01:
        floating_bottom.append(face)
assert not floating_bottom, floating_bottom


# A blind hole entering through a cube edge has one exterior rim split across
# two planes. It must use reconstruction instead of independently projecting
# the rim vertices onto whichever cube face happens to win a ray test.
bpy.ops.object.mode_set(mode='OBJECT')
redo_through_object.select_set(False)
(
    cube_edge_object,
    cube_edge_bmesh,
    cube_edge_faces,
    cube_edge_entrance,
    cube_edge_axis,
) = create_blind_cube_edge_hole("blind_cube_edge_resize_test")
assert len(cube_edge_faces) >= 24, len(cube_edge_faces)
cube_edge_operator = SimpleNamespace(
    _object=cube_edge_object,
    _locked_indices={face.index for face in cube_edge_faces},
    scale=1.0,
    diameter=0.6,
    angle_valid=False,
)
cube_edge_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(cube_edge_operator, face)
)
cube_edge_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(cube_edge_operator)
)
cube_edge_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    cube_edge_operator, cube_edge_faces
)
assert not cube_edge_operator._fit["is_stud"]
cube_edge_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        cube_edge_operator,
        cube_edge_faces,
        cube_edge_operator._fit["axis"],
        cube_edge_operator._fit,
    )
)
cube_edge_operator._depth_fit = None
cube_edge_vertices = {
    vert for face in cube_edge_faces for vert in face.verts
}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    cube_edge_operator,
    bpy.context,
    cube_edge_bmesh,
    cube_edge_vertices,
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    cube_edge_operator, bpy.context
)
cube_edge_preview = bmesh.from_edit_mesh(cube_edge_object.data)
assert all(len(edge.link_faces) == 2 for edge in cube_edge_preview.edges)
MESH_OT_holey_tools_resize_hole._replace_boolean_preview_with_source(
    cube_edge_operator
)
cube_edge_bmesh = bmesh.from_edit_mesh(cube_edge_object.data)
cube_edge_bmesh.faces.ensure_lookup_table()
cube_edge_faces = [
    cube_edge_bmesh.faces[index]
    for index in sorted(cube_edge_operator._locked_indices)
]
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    cube_edge_operator, cube_edge_bmesh, cube_edge_faces
)
cube_edge_operator._resize_dynamic_requires_boolean = True
cube_edge_surfaces = {
    id(surface): surface
    for surface in cube_edge_operator._resize_surfaces_by_vertex.values()
}.values()
cube_edge_exterior = [
    surface for surface in cube_edge_surfaces if surface["exterior"]
]
cube_edge_interior = [
    surface for surface in cube_edge_surfaces if not surface["exterior"]
]
assert len(cube_edge_exterior) == 1, cube_edge_exterior
assert len(cube_edge_interior) == 1, cube_edge_interior
assert cube_edge_exterior[0]["hypothesis"]["method"] == "PLANES", (
    cube_edge_exterior[0]["hypothesis"]
)
assert len(cube_edge_exterior[0]["hypothesis"]["planes"]) == 2
assert cube_edge_operator._resize_requires_boolean
MESH_OT_holey_tools_resize_hole._store_fit_for_redo(
    cube_edge_operator
)
assert (
    cube_edge_operator.fit_mesh_vertex_count,
    cube_edge_operator.fit_mesh_edge_count,
    cube_edge_operator.fit_mesh_face_count,
) == (
    len(cube_edge_bmesh.verts),
    len(cube_edge_bmesh.edges),
    len(cube_edge_bmesh.faces),
)
assert len(cube_edge_operator.fit_mesh_topology_signature) == 64
assert set(MESH_OT_holey_tools_resize_hole._resolve_resize_wall_faces(
    cube_edge_operator, cube_edge_bmesh
)) == set(cube_edge_faces)
assert cube_edge_operator._depth_fit is not None
assert cube_edge_operator._depth_fit["surface_depth"]
assert cube_edge_operator.depth_valid
assert cube_edge_operator.fit_depth_outer_plane == -1
cube_edge_operator.diameter = 0.6
cube_edge_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(cube_edge_operator)
)
cube_outer_coordinate = cube_edge_operator._depth_fit["outer_coordinate"]
cube_inner_coordinate = cube_edge_operator._depth_fit["inner_coordinate"]
assert abs(cube_edge_operator.depth - 2.2) < 2.0e-2
cube_initial_depth = cube_edge_operator.depth
cube_edge_operator.depth = cube_initial_depth * 0.75
cube_edge_exterior[0]["loop_verts"] = []
cube_edge_exterior[0]["loop_edges"] = []
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    cube_edge_operator, bpy.context, commit_boolean=True
)
cube_edge_bmesh = bmesh.from_edit_mesh(cube_edge_object.data)
cube_edge_bmesh.edges.ensure_lookup_table()
assert all(len(edge.link_faces) == 2 for edge in cube_edge_bmesh.edges)
cube_edge_cut = []
for vert in cube_edge_bmesh.verts:
    offset = vert.co - cube_edge_entrance
    coordinate = offset.dot(cube_edge_axis)
    radial = offset - cube_edge_axis * coordinate
    if (
        -0.05 <= coordinate <= 2.4
        and abs(radial.length - 0.3) < 1.0e-4
    ):
        cube_edge_cut.append(vert)
assert cube_edge_cut
cube_direction = 1.0 if cube_inner_coordinate >= cube_outer_coordinate else -1.0
cube_depth = cube_edge_operator.depth
cube_wall_progress = []
for vert in cube_edge_bmesh.verts:
    point = cube_edge_object.matrix_world @ vert.co
    offset = point - cube_edge_operator._fit["center"]
    coordinate = offset.dot(cube_edge_operator._fit["axis"])
    radial = offset - cube_edge_operator._fit["axis"] * coordinate
    if abs(radial.length - 0.3) < 1.0e-4:
        cube_wall_progress.append(
            (coordinate - cube_outer_coordinate) * cube_direction
        )
assert cube_wall_progress
assert max(cube_wall_progress) <= cube_depth + 1.0e-3
assert min(abs(progress - cube_depth) for progress in cube_wall_progress) < 1.0e-3

# If Blender presents the preceding Boolean result instead of the original
# indexed wall, rediscover the connected coaxial wall rather than treating the
# now-unrelated serialized face indices as the hole selection.
serialized_cube_indices = {
    int(index)
    for index in cube_edge_operator.fit_face_indices.split(",")
    if index
}
cube_edge_operator.diameter = 0.55
MESH_OT_holey_tools_resize_hole._reset_execute_transients(
    cube_edge_operator
)
assert MESH_OT_holey_tools_resize_hole._restore_fit_for_redo(
    cube_edge_operator
)
# A Boolean result may coincidentally have exactly the source element counts.
# Connectivity, rather than counts alone, must still identify it as stale.
cube_edge_operator.fit_mesh_vertex_count = len(cube_edge_bmesh.verts)
cube_edge_operator.fit_mesh_edge_count = len(cube_edge_bmesh.edges)
cube_edge_operator.fit_mesh_face_count = len(cube_edge_bmesh.faces)
resolved_cube_faces = (
    MESH_OT_holey_tools_resize_hole._resolve_resize_wall_faces(
        cube_edge_operator, cube_edge_bmesh
    )
)
assert cube_edge_operator._locked_indices != serialized_cube_indices
assert len(resolved_cube_faces) >= 8
assert all(face.is_valid for face in resolved_cube_faces)

signature_a = MESH_OT_holey_tools_resize_hole._canonical_topology_signature(
    4, 5, ((0, 1, 2), (0, 2, 3))
)
signature_b = MESH_OT_holey_tools_resize_hole._canonical_topology_signature(
    4, 5, ((0, 1, 3), (1, 2, 3))
)
assert signature_a != signature_b
assert signature_a == (
    MESH_OT_holey_tools_resize_hole._canonical_topology_signature(
        4, 5, ((3, 2, 0), (2, 1, 0))
    )
)


# Depth is available for two planar boundary loops even when they are not
# parallel. The outer end stays fixed while the oblique free end moves.
bpy.ops.object.mode_set(mode='OBJECT')
cube_edge_object.select_set(False)
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

near_parallel_planes = [dict(plane) for plane in classified_planes]
near_parallel_planes[1]["normal"] = Vector((
    math.sin(math.radians(0.39)),
    0.0,
    math.cos(math.radians(0.39)),
))
near_parallel_depth = MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    depth_operator, near_parallel_planes, depth_operator._fit
)
assert near_parallel_depth is not None
assert near_parallel_depth["parallel"]

distinct_planes = [dict(plane) for plane in classified_planes]
distinct_planes[1]["normal"] = Vector((
    math.sin(math.radians(0.41)),
    0.0,
    math.cos(math.radians(0.41)),
))
distinct_depth = MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    depth_operator, distinct_planes, depth_operator._fit
)
assert distinct_depth is not None
assert not distinct_depth["parallel"]

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


# Adjust Last Operation can retain a Python wrapper after Blender removes the
# Object RNA it referenced. Rebinding must not dereference that stale object and
# must discard every transient mesh/BVH reference before redo reconstruction.
bpy.ops.object.mode_set(mode='OBJECT')
cone_object.select_set(False)
stale_mesh = bpy.data.meshes.new("stale_redo_object_mesh")
stale_object = bpy.data.objects.new("stale_redo_object", stale_mesh)
bpy.context.collection.objects.link(stale_object)
stale_reference = stale_object
bpy.data.objects.remove(stale_object, do_unlink=True)
replacement_mesh = bpy.data.meshes.new("redo_replacement_mesh")
replacement_mesh.from_pydata(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (),
    ((0, 1, 2),),
)
replacement_object = bpy.data.objects.new(
    "redo_replacement", replacement_mesh
)
bpy.context.collection.objects.link(replacement_object)
bpy.context.view_layer.objects.active = replacement_object
replacement_object.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
redo_binding = SimpleNamespace(
    _object=stale_reference,
    _fit={"stale": True},
    _locked_indices={1},
    _original_vertex_coords={0: Vector((0.0, 0.0, 0.0))},
    _proportional_weights={0: 1.0},
    _resize_projection_ready=True,
    _resize_requires_boolean=True,
    _resize_base_requires_boolean=True,
    _resize_dynamic_requires_boolean=True,
    _resize_surfaces_by_vertex={0: object()},
    _resize_base_bvh=object(),
    _resize_base_span=1.0,
)
assert MESH_OT_holey_tools_resize_hole._bind_execute_object(
    redo_binding, replacement_object
)
assert redo_binding._object is replacement_object
assert redo_binding._fit is None
assert not redo_binding._locked_indices
assert not redo_binding._resize_projection_ready
assert not redo_binding._resize_requires_boolean
assert not redo_binding._resize_base_requires_boolean
assert not redo_binding._resize_dynamic_requires_boolean

# Even when Blender retains the same Object wrapper, every redo execution must
# discard mesh-derived state before restoring the serialized fit.
redo_binding._fit = {"stale": True}
redo_binding._locked_indices = {2}
redo_binding._original_vertex_coords = {0: Vector((1.0, 2.0, 3.0))}
redo_binding._resize_projection_ready = True
redo_binding._resize_surfaces_by_vertex = {0: object()}
redo_binding._resize_base_bvh = object()
redo_binding._resize_base_requires_boolean = True
MESH_OT_holey_tools_resize_hole._reset_execute_transients(redo_binding)
assert redo_binding._fit is None
assert not redo_binding._locked_indices
assert redo_binding._original_vertex_coords is None
assert not redo_binding._resize_projection_ready
assert not redo_binding._resize_surfaces_by_vertex
assert redo_binding._resize_base_bvh is None
assert not redo_binding._resize_base_requires_boolean

print("HOLEY_TOOLS_MODE_TESTS_OK")
