import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bmesh
import bpy
from mathutils import Vector


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import GeometryMixin  # noqa: E402
from src.operator import (  # noqa: E402
    MESH_OT_holey_tools_resize_hole,
    _PersistentPreview,
    _conic_depth_preserving_bottom,
    _update_conic_angle,
    _update_conic_depth,
    _update_conic_truncated_depth,
    _update_diameter_estimator,
)


def evaluated_vertex_coordinates(obj):
    """Return object-local coordinates after viewport modifiers."""
    obj.update_from_editmode()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return [vertex.co.copy() for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()


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


def create_countersunk_through_solid(name, segments=32, cone_rows=1):
    """Create a closed cylinder with a countersink feeding a through-hole."""
    rings = (
        (5.0, 0.0),
        (2.0, 0.0),
        *(
            (2.0 - row / cone_rows, -row / cone_rows)
            for row in range(1, cone_rows + 1)
        ),
        (1.0, -3.0),
        (5.0, -3.0),
    )
    vertices = [
        (
            radius * math.cos(math.tau * index / segments),
            radius * math.sin(math.tau * index / segments),
            z,
        )
        for radius, z in rings
        for index in range(segments)
    ]
    faces = []

    def bridge(first_ring, second_ring, reverse=False):
        for index in range(segments):
            following = (index + 1) % segments
            face = (
                first_ring + index,
                first_ring + following,
                second_ring + following,
                second_ring + index,
            )
            faces.append(tuple(reversed(face)) if reverse else face)

    bridge(0, segments)
    bridge(0, segments * (cone_rows + 3), reverse=True)
    bridge(segments * (cone_rows + 3), segments * (cone_rows + 2), reverse=True)
    for row in range(1, cone_rows + 1):
        bridge(segments * row, segments * (row + 1))
    bridge(segments * (cone_rows + 1), segments * (cone_rows + 2))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, (), faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
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

# Scaling simplification is independently optional and defaults off. A direct
# shrink without it changes coordinates without growing or replacing selection.
planar_hole_operator.scaling_final_simplify = False
planar_hole_operator.diameter = 1.2
unsimplified_counts = (
    len(planar_hole_bmesh.verts),
    len(planar_hole_bmesh.edges),
    len(planar_hole_bmesh.faces),
)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    planar_hole_operator, bpy.context, commit_boolean=True
)
planar_hole_bmesh = bmesh.from_edit_mesh(planar_hole_object.data)
assert (
    len(planar_hole_bmesh.verts),
    len(planar_hole_bmesh.edges),
    len(planar_hole_bmesh.faces),
) == unsimplified_counts

# Enabling it selects the inset wall, expands that face region once, and applies
# the same 0.1-degree Limited Dissolve cleanup used by the prior direct path.
planar_hole_operator.scaling_final_simplify = True
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
square_segments = 32
square_object, square_bmesh = create_planar_blind_hole(
    "triangular_boolean_simplify_test", segments=square_segments
)
square_faces = list(square_bmesh.faces[:square_segments])
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
    final_simplify=True,
    boolean_segments=3,
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
square_material_slots = len(square_object.data.materials)
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
# A deliberately coarse triangular cutter makes residual fairing topology easy
# to spot. Split loop normals from the Boolean must not preserve the circle-fill
# rings/spokes; only the few bridges required to represent a hole may remain.
square_top_set = set(square_top_faces)
square_coplanar_edges = {
    edge
    for edge in square_bmesh.edges
    if (
        len(edge.link_faces) == 2
        and all(face in square_top_set for face in edge.link_faces)
        and edge.calc_face_angle(0.0) <= math.radians(0.1)
    )
}
assert len(square_coplanar_edges) <= 6, len(square_coplanar_edges)
assert len(square_object.data.materials) == square_material_slots


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
# The redo override can deliberately retain the direct scaling path even after
# the enlarged rim leaves its initial planar collar.
transition_operator.resize_method = 'SCALE'
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    transition_operator, bpy.context
)
assert not transition_operator._resize_requires_boolean
assert getattr(transition_operator, "_boolean_preview_source", None) is None
forced_scaling_coordinates = [
    vert.co.copy()
    for vert in bmesh.from_edit_mesh(transition_object.data).verts
]

# Returning to Auto re-evaluates the same target against the support and takes
# the reconstruction path.
transition_operator.resize_method = 'AUTO'
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
    for coordinate in evaluated_vertex_coordinates(
        transition_operator._boolean_preview_display_object
    )
    for offset in (transition_object.matrix_world @ coordinate
                   - transition_operator._fit["center"],)
    for radial in (offset - transition_operator._fit["axis"]
                   * offset.dot(transition_operator._fit["axis"]),)
    if abs(radial.length - 1.5) < 1.0e-4
}
assert len(transition_preview_angles) == 32, transition_preview_angles
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
# A manual Scaling choice must retire the retained Auto Boolean even during
# the debounce interval. Compare against the original direct-scaling result,
# including unchanged topology, rather than only checking the mode label.
transition_operator.resize_method = 'SCALE'
transition_operator._drag_active = True
transition_operator._boolean_preview_last_update = time.monotonic()
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    transition_operator, bpy.context
)
assert transition_operator.resize_method == 'SCALE'
assert not transition_operator._resize_requires_boolean
assert not transition_operator._resize_dynamic_requires_boolean
assert transition_operator._boolean_preview_source is None
assert transition_operator._boolean_reconstruction_source is None
assert transition_operator._boolean_preview_display_object is None
assert transition_operator._boolean_preview_cutter_object is None
assert not transition_operator._boolean_preview_pending
assert not transition_object.hide_get()
scaled_vertices = list(bmesh.from_edit_mesh(transition_object.data).verts)
assert len(scaled_vertices) == len(forced_scaling_coordinates)
assert all(
    (vert.co - expected).length < 1.0e-6
    for vert, expected in zip(scaled_vertices, forced_scaling_coordinates)
)
# Repeated forced evaluations stay in Scaling; Auto is reconsidered only when
# the user explicitly selects it again.
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    transition_operator, bpy.context, commit_boolean=True
)
assert transition_operator.resize_method == 'SCALE'
assert not transition_operator._resize_requires_boolean
transition_operator._drag_active = False
transition_operator.resize_method = 'AUTO'
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    transition_operator, bpy.context
)
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


# With Exact Boolean forced on a hole with conical support, its 32 source sides
# are used live, then a redo segment override is accepted at confirmation. Both
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
    resize_method='BOOLEAN',
)
boolean_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(boolean_operator, face)
)
boolean_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(boolean_operator)
)
boolean_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(boolean_operator)
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
boolean_source_signature = (
    MESH_OT_holey_tools_resize_hole._object_topology_signature(boolean_object)
)
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    boolean_operator, boolean_bmesh, boolean_faces
)
MESH_OT_holey_tools_resize_hole._store_fit_for_redo(boolean_operator)
boolean_operator.diameter = 3.0
boolean_redo_source = object.__new__(_PersistentPreview)
boolean_redo_source._draw_handle = None
boolean_redo_source._source_object_name = boolean_object.name
(
    boolean_redo_source._source_mesh_name,
    boolean_redo_source._source_mesh,
    boolean_redo_source._source_bmesh,
    boolean_redo_source._source_materials,
) = _PersistentPreview._capture_canonical_source(boolean_operator)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    boolean_operator, bpy.context
)
assert boolean_operator._resize_requires_boolean
assert boolean_operator._resize_angular_segments == 32
assert boolean_object.hide_get()
assert boolean_operator._boolean_preview_display_object is not None
assert not boolean_operator._boolean_preview_display_object.hide_get()
assert len(boolean_operator._boolean_preview_display_object.modifiers) == 1
assert (
    boolean_object.data.as_pointer()
    != boolean_operator._boolean_preview_display_object.data.as_pointer()
)
assert (
    MESH_OT_holey_tools_resize_hole._object_topology_signature(boolean_object)
    == boolean_source_signature
)
boolean_bmesh = bmesh.from_edit_mesh(boolean_object.data)
assert all(len(edge.link_faces) == 2 for edge in boolean_bmesh.edges)
preview_angles = {
    (
        round(coordinate.x / math.hypot(coordinate.x, coordinate.y), 5),
        round(coordinate.y / math.hypot(coordinate.x, coordinate.y), 5),
    )
    for coordinate in evaluated_vertex_coordinates(
        boolean_operator._boolean_preview_display_object
    )
    if abs(math.hypot(coordinate.x, coordinate.y) - 1.5) < 1.0e-4
}
assert len(preview_angles) == 32, preview_angles
assert boolean_operator._boolean_reconstruction_source is not None
cached_reconstruction_pointer = (
    boolean_operator._boolean_reconstruction_source.as_pointer()
)
assert MESH_OT_holey_tools_resize_hole._live_boolean_segments(
    SimpleNamespace(boolean_segments=32, _resize_angular_segments=0)
) == 32
assert MESH_OT_holey_tools_resize_hole._live_boolean_segments(
    SimpleNamespace(boolean_segments=33, _resize_angular_segments=0)
) == 32
assert MESH_OT_holey_tools_resize_hole._live_boolean_segments(
    SimpleNamespace(boolean_segments=12, _resize_angular_segments=0)
) == 12
# Pointer moves inside the half-second window retain the previous Boolean mesh.
# Ending the drag performs one trailing cutter update at the latest value.
preview_mesh_pointer = boolean_object.data.as_pointer()
preview_cutter_pointer = (
    boolean_operator._boolean_preview_cutter_object.as_pointer()
)
preview_cutter_mesh_pointer = (
    boolean_operator._boolean_preview_cutter_object.data.as_pointer()
)
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
assert boolean_object.data.as_pointer() == preview_mesh_pointer
assert (
    boolean_operator._boolean_preview_cutter_object.as_pointer()
    == preview_cutter_pointer
)
assert (
    boolean_operator._boolean_preview_cutter_object.data.as_pointer()
    == preview_cutter_mesh_pointer
)
assert not boolean_operator._boolean_preview_pending
assert (
    boolean_operator._boolean_reconstruction_source.as_pointer()
    == cached_reconstruction_pointer
)
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
assert boolean_operator._boolean_reconstruction_source is None
assert not boolean_object.hide_get()
assert not any(
    obj.name.startswith("Holey Tools Boolean Cutter")
    for obj in bpy.data.objects
)
assert not any(
    obj.name.startswith("Holey Tools Boolean Preview")
    for obj in bpy.data.objects
)
assert not any(
    modifier.name.startswith("Holey Tools Recreate Hole")
    for modifier in boolean_object.modifiers
)

# Repeat the full redo execution from its canonical source, including a case
# whose conical exterior Auto now handles directly. Explicit choices persist
# across mesh replacement, fit restoration, and repeated property changes.
for requested_method in ('SCALE', 'SCALE', 'BOOLEAN', 'SCALE'):
    assert boolean_redo_source.restore_source(boolean_object)
    boolean_operator.resize_method = requested_method
    if requested_method == 'SCALE':
        with patch.object(
            GeometryMixin,
            "_enlargement_reaches_nonplanar_support",
            side_effect=AssertionError("Explicit Scaling ran Auto mode selection"),
        ):
            result = MESH_OT_holey_tools_resize_hole.execute(
                boolean_operator, bpy.context
            )
    else:
        result = MESH_OT_holey_tools_resize_hole.execute(
            boolean_operator, bpy.context
        )
    assert result == {'FINISHED'}, (requested_method, result)
    assert boolean_operator.resize_method == requested_method
    assert not boolean_operator._resize_base_requires_boolean
    assert boolean_operator._resize_requires_boolean == (
        requested_method == 'BOOLEAN'
    )
    if requested_method == 'SCALE':
        assert (
            MESH_OT_holey_tools_resize_hole._object_topology_signature(
                boolean_object
            ) == boolean_source_signature
        )
boolean_redo_source.remove()

# Explicit Exact Boolean also reconstructs a shrink of this conic-supported hole.
# The retained
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
    final_simplify=True,
    resize_method='BOOLEAN',
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
shrink_select_mode = (False, True, False)
bpy.context.tool_settings.mesh_select_mode = shrink_select_mode
boolean_shrink_vertices = {
    vert for face in boolean_shrink_faces for vert in face.verts
}
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    boolean_shrink_operator,
    bpy.context,
    boolean_shrink_bmesh,
    boolean_shrink_vertices,
)
shrink_material_slots = len(boolean_shrink_object.data.materials)
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
assert tuple(bpy.context.tool_settings.mesh_select_mode) == shrink_select_mode
selected_shrink_faces = [
    face for face in boolean_shrink_bmesh.faces if face.select
]
assert selected_shrink_faces
# Final simplification starts from every surviving fair-surface face, expands
# exactly once toward the original support, then converts that full region to
# edge selection. Boolean-generated wall/cap faces must be excluded even if
# Blender interpolates the temporary fair tag onto them.
assert any(abs(face.normal.z) > 0.1 for face in selected_shrink_faces)
assert len(boolean_shrink_object.data.materials) == shrink_material_slots
# Dissolving the expanded fair region must not bridge the newly cut opening
# with a polygon whose perimeter is the new rim.
rogue_fair_caps = []
for face in boolean_shrink_bmesh.faces:
    radial_coordinates = [
        math.hypot(vert.co.x, vert.co.y) for vert in face.verts
    ]
    center = face.calc_center_median()
    if (
        radial_coordinates
        and min(radial_coordinates) > 0.49
        and math.hypot(center.x, center.y) < 0.4
        and all(
            abs(vert.co.z - (math.hypot(vert.co.x, vert.co.y) - 1.0))
            < 1.0e-4
            for vert in face.verts
        )
    ):
        rogue_fair_caps.append(face)
assert not rogue_fair_caps
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
# Keep this reconstruction/depth fixture's vertex radius at .3; Diameter now
# describes the smaller circle which fits between the cutter's flat sides.
cube_edge_operator.diameter = 0.6 * math.cos(math.pi / cube_edge_operator.boolean_segments)
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

# Restoring the modal source while switching into Exact Boolean can leave the
# live operator holding face indices from an intermediate topology. The preview
# path must rediscover the fitted wall instead of indexing the BMesh blindly.
cube_edge_operator._locked_indices = {len(cube_edge_bmesh.faces) + 12}
resolved_out_of_range_faces = (
    MESH_OT_holey_tools_resize_hole._resolve_resize_wall_faces(
        cube_edge_operator, cube_edge_bmesh
    )
)
assert len(resolved_out_of_range_faces) >= 8
assert all(face.is_valid for face in resolved_out_of_range_faces)
assert all(
    0 <= index < len(cube_edge_bmesh.faces)
    for index in cube_edge_operator._locked_indices
)

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
cone_operator.flood_angle = math.radians(25.0)
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
# Boolean reconstruction can leave a slightly corrugated cone-to-bore seam.
# Conic depth uses the boundary rings' axial coordinates and must remain
# adjustable even when a generic planar-end test would reject that seam.
rough_conic_planes = [dict(plane) for plane in cone_operator._end_planes]
rough_conic_planes[1]["max_error"] = 0.01
rough_conic_planes[1]["tolerance"] = 1.0e-6
rough_conic_depth = MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    cone_operator, rough_conic_planes, cone_operator._fit
)
assert rough_conic_depth is not None
assert abs(rough_conic_depth["initial"] - 4.0) < 1.0e-5
MESH_OT_holey_tools_resize_hole._store_fit_for_redo(cone_operator)
assert abs(cone_operator.depth - 8.0) < 1.0e-5
assert abs(cone_operator.truncated_depth - 4.0) < 1.0e-5
assert cone_operator.conic_depth_previous == cone_operator.depth
assert (
    cone_operator.conic_truncated_depth_previous
    == cone_operator.truncated_depth
)
outer_coordinate = cone_operator._depth_fit["outer_coordinate"]
assert abs(
    cone_operator._fit["radius"]
    + cone_operator._fit["slope"] * outer_coordinate
    - 2.0
) < 1.0e-5

cone_operator._fit["radius_coordinate"] = outer_coordinate
cone_operator.diameter = 4.0
cone_operator.depth = 8.0
cone_operator.truncated_depth = 4.0
cone_operator.conic_angle = cone_operator._fit["half_angle"] * 2.0
cone_operator.angle_valid = True
cone_operator.depth_valid = True
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

# Editing only the included angle pivots the conic around its physical bottom
# circle. The truncation plane and radius remain fixed while apex depth changes.
old_angle = cone_operator.conic_angle
old_bottom_radius = (
    cone_operator.depth - cone_operator.truncated_depth
) * math.tan(old_angle * 0.5)
new_angle = math.atan(0.5) * 2.0
preserved_depth = _conic_depth_preserving_bottom(
    cone_operator.depth,
    cone_operator.truncated_depth,
    old_angle,
    new_angle,
)
assert abs(
    (preserved_depth - cone_operator.truncated_depth)
    * math.tan(new_angle * 0.5)
    - old_bottom_radius
) < 1.0e-9
angle_update_operator = SimpleNamespace(
    conic_angle=new_angle,
    conic_angle_previous=old_angle,
    depth=cone_operator.depth,
    truncated_depth=cone_operator.truncated_depth,
    fit_valid=True,
    fit_is_cone=True,
    depth_valid=True,
    angle_valid=True,
)
_update_conic_angle(angle_update_operator, None)
assert abs(angle_update_operator.depth - preserved_depth) < 1.0e-9
assert angle_update_operator.conic_angle_previous == new_angle

# Either axial control translates the truncation circle along the cone while
# preserving its radius. The two depths consequently change by equal amounts.
dimension_update_operator = SimpleNamespace(
    depth=8.0,
    truncated_depth=4.0,
    conic_depth_previous=8.0,
    conic_truncated_depth_previous=4.0,
    fit_valid=True,
    fit_is_cone=True,
    depth_valid=True,
    angle_valid=True,
)
dimension_update_operator.depth = 9.5
_update_conic_depth(dimension_update_operator, None)
assert abs(dimension_update_operator.truncated_depth - 5.5) < 1.0e-9
assert abs(
    dimension_update_operator.depth
    - dimension_update_operator.truncated_depth
    - 4.0
) < 1.0e-9
dimension_update_operator.truncated_depth = 3.25
_update_conic_truncated_depth(dimension_update_operator, None)
assert abs(dimension_update_operator.depth - 7.25) < 1.0e-9
assert abs(
    dimension_update_operator.depth
    - dimension_update_operator.truncated_depth
    - 4.0
) < 1.0e-9
cone_operator.depth = preserved_depth
cone_operator.conic_angle = new_angle
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    cone_operator, bpy.context
)
cone_bmesh = bmesh.from_edit_mesh(cone_object.data)
pivot_outer_points = [vert.co for vert in cone_bmesh.verts[:32]]
pivot_inner_points = [vert.co for vert in cone_bmesh.verts[32:]]
assert max(abs(point.z - 2.0) for point in pivot_inner_points) < 1.0e-5
assert max(
    abs(math.hypot(point.x, point.y) - old_bottom_radius)
    for point in pivot_inner_points
) < 1.0e-5
assert max(abs(point.z + 2.0) for point in pivot_outer_points) < 1.0e-5
assert max(
    abs(
        math.hypot(point.x, point.y)
        - (old_bottom_radius + 4.0 * math.tan(new_angle * 0.5))
    )
    for point in pivot_outer_points
) < 1.0e-5

# Truncated Depth must remain authoritative when Blender evaluates redo before
# its update callback has copied the delta into Depth. The fixed small-circle
# radius is retained while the circle moves one unit farther from the mouth.
cone_operator.depth = 8.0
cone_operator.truncated_depth = 5.0
cone_operator.conic_angle = math.atan(0.25) * 2.0
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    cone_operator, bpy.context
)
cone_bmesh = bmesh.from_edit_mesh(cone_object.data)
outer_points = [vert.co for vert in cone_bmesh.verts[:32]]
inner_points = [vert.co for vert in cone_bmesh.verts[32:]]
assert max(abs(point.z + 2.0) for point in outer_points) < 1.0e-5
assert max(abs(math.hypot(point.x, point.y) - 2.25) for point in outer_points) < 1.0e-5
assert max(abs(point.z - 3.0) for point in inner_points) < 1.0e-5
assert max(abs(math.hypot(point.x, point.y) - 1.0) for point in inner_points) < 1.0e-5


# A countersink can terminate at a circle that continues into a normal bore,
# with no flat center face. The conic dimensions remain valid, and an Exact
# Boolean cutter extends that truncation radius through the remaining bore
# instead of capping it and inventing a blind bottom.
bpy.ops.object.mode_set(mode='OBJECT')
cone_object.select_set(False)
countersink_object, countersink_bmesh = create_ring_mesh(
    "countersink_with_center_bore_test",
    ((2.0, -2.0), (1.0, 2.0), (1.0, 5.0)),
    segments=32,
)
bmesh.ops.reverse_faces(
    countersink_bmesh, faces=list(countersink_bmesh.faces)
)
countersink_bmesh.normal_update()
countersink_faces = list(countersink_bmesh.faces[:32])
countersink_operator = SimpleNamespace(
    _object=countersink_object,
    _locked_indices={face.index for face in countersink_faces},
    diameter_estimator='AVERAGE',
    scale=1.0,
    fit_valid=False,
    angle_valid=True,
    depth_valid=True,
)
countersink_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(
        countersink_operator, face
    )
)
countersink_operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(
    countersink_operator, countersink_faces
)
countersink_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        countersink_operator,
        countersink_faces,
        countersink_operator._fit["axis"],
        countersink_operator._fit,
    )
)
countersink_operator._depth_fit = (
    MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
        countersink_operator,
        countersink_operator._end_planes,
        countersink_operator._fit,
    )
)
assert countersink_operator._depth_fit is not None
countersink_operator._fit["radius_coordinate"] = (
    countersink_operator._depth_fit["outer_coordinate"]
)
countersink_operator.depth = 8.0
countersink_operator.truncated_depth = 4.0
countersink_operator.conic_angle = math.atan(0.25) * 2.0
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    countersink_operator, countersink_bmesh, countersink_faces
)
countersink_surfaces = {
    id(surface): surface
    for surface in countersink_operator._resize_surfaces_by_vertex.values()
}.values()
assert any(
    surface.get("bore_continuation") and not surface["exterior"]
    for surface in countersink_surfaces
)
# Bore classification performs its own coaxial-cylinder test rather than
# trusting the general fair-surface winner. A locally cylindrical continuation
# must therefore survive even if that competition happened to report Sphere.
synthetic_bore_surface = next(
    surface for surface in countersink_surfaces
    if surface.get("bore_continuation")
)
original_bore_hypothesis = synthetic_bore_surface["hypothesis"]
synthetic_bore_surface["hypothesis"] = {"method": "SPHERE"}
assert MESH_OT_holey_tools_resize_hole._is_cylindrical_bore_continuation(
    countersink_operator, synthetic_bore_surface
)
assert synthetic_bore_surface.get("bore_fit") is not None
synthetic_bore_surface["hypothesis"] = original_bore_hypothesis
assert set(range(64, 96)).issubset(
    countersink_operator._resize_bore_vertex_indices
)
# This isolated wall fixture has no exterior annulus to identify the model side,
# so orient its already fitted endpoints by the cone radii as a real countersink
# support would.
if MESH_OT_holey_tools_resize_hole._fitted_radius_at(
    countersink_operator,
    countersink_operator._depth_fit["outer_coordinate"],
) < MESH_OT_holey_tools_resize_hole._fitted_radius_at(
    countersink_operator,
    countersink_operator._depth_fit["inner_coordinate"],
):
    countersink_operator._depth_fit["outer_coordinate"], countersink_operator._depth_fit["inner_coordinate"] = (
        countersink_operator._depth_fit["inner_coordinate"],
        countersink_operator._depth_fit["outer_coordinate"],
    )
assert MESH_OT_holey_tools_resize_hole._conic_continues_as_cylindrical_bore(
    countersink_operator
)
# Live Boolean preview discards the transient support classifications after it
# caches the reconstructed base. The latched continuation must still produce a
# compound three-ring cutter instead of capping the countersink at its small
# circle on the next drag update.
cached_countersink_surfaces = countersink_operator._resize_surfaces_by_vertex
countersink_operator._resize_surfaces_by_vertex = {}
assert MESH_OT_holey_tools_resize_hole._conic_continues_as_cylindrical_bore(
    countersink_operator
)
cached_vertices, cached_faces = (
    MESH_OT_holey_tools_resize_hole._boolean_cutter_geometry(
        countersink_operator,
        countersink_object,
        segment_override=8,
    )
)
assert len(cached_vertices) == 24
assert len(cached_faces) == 18
countersink_operator._resize_surfaces_by_vertex = cached_countersink_surfaces
countersink_bottom_radius = (
    MESH_OT_holey_tools_resize_hole._fitted_radius_at(
        countersink_operator,
        countersink_operator._depth_fit["inner_coordinate"],
    )
)
# Geometry retains the fitted continuation-bore radius independently of update
# callback ordering. Each of the three conic controls can therefore rebuild a
# cutter without changing the cylindrical through-hole it joins.
conic_control_values = (
    (12.0, 4.0, math.atan(0.25) * 2.0),
    (8.0, 5.0, math.atan(0.25) * 2.0),
    (8.0, 4.0, math.atan(0.4) * 2.0),
)
for depth, truncated_depth, angle in conic_control_values:
    countersink_operator.depth = depth
    countersink_operator.truncated_depth = truncated_depth
    countersink_operator.conic_angle = angle
    target_inner = (
        MESH_OT_holey_tools_resize_hole._target_conic_inner_coordinate(
            countersink_operator
        )
    )
    assert abs(
        MESH_OT_holey_tools_resize_hole._target_radius_at(
            countersink_operator, target_inner
        )
        - countersink_bottom_radius
    ) < 1.0e-9
# Depth is authoritative for Boolean geometry even if a redo callback has not
# yet copied its delta into Truncated Depth.
countersink_operator.depth = 8.5
countersink_operator.truncated_depth = 4.0
countersink_operator.conic_angle = math.atan(0.25) * 2.0
depth_only_span = (
    MESH_OT_holey_tools_resize_hole._target_conic_truncated_depth(
        countersink_operator
    )
)
assert abs(depth_only_span - 4.5) < 1.0e-5, depth_only_span
countersink_operator.depth = 8.0
countersink_operator.truncated_depth = 4.0
countersink_operator.conic_angle = math.atan(0.25) * 2.0
countersink_operator.diameter = 4.0
countersink_operator.resize_method = 'SCALE'
countersink_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(countersink_operator)
)
countersink_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(
        countersink_operator
    )
)
countersink_deformation_vertices = {
    vert for face in countersink_faces for vert in face.verts
}
countersink_deformation_vertices.update(
    countersink_bmesh.verts[index]
    for index in countersink_operator._resize_bore_vertex_indices
)
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    countersink_operator,
    bpy.context,
    countersink_bmesh,
    countersink_deformation_vertices,
)
countersink_operator.depth = 8.5
countersink_operator.truncated_depth = 4.5
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    countersink_operator, bpy.context
)
countersink_bmesh = bmesh.from_edit_mesh(countersink_object.data)
assert max(
    abs(
        math.hypot(vert.co.x, vert.co.y)
        - countersink_bottom_radius
    )
    for vert in countersink_bmesh.verts[64:96]
) < 1.0e-5
countersink_vertices, countersink_cutter_faces = (
    MESH_OT_holey_tools_resize_hole._boolean_cutter_geometry(
        countersink_operator,
        countersink_object,
        segment_override=8,
    )
)
assert len(countersink_vertices) == 24
assert len(countersink_cutter_faces) == 18
countersink_world = [
    countersink_object.matrix_world @ point
    for point in countersink_vertices
]
axis = countersink_operator._fit["axis"].normalized()
outer_coordinate = countersink_operator._depth_fit["outer_coordinate"]
inner_coordinate = countersink_operator._depth_fit["inner_coordinate"]
direction = 1.0 if inner_coordinate >= outer_coordinate else -1.0
cutter_outer_coordinate = sum(
    (point - countersink_operator._fit["center"]).dot(axis)
    for point in countersink_world[:8]
) / 8.0
# Exact Boolean must not cap the large cone ring coplanar with the reconstructed
# exterior. Its first ring deliberately extends outside the model.
assert (
    (outer_coordinate - cutter_outer_coordinate) * direction
    > 1.0e-6
)
for index in range(8):
    inner = countersink_world[8 + index] - countersink_operator._fit["center"]
    end = countersink_world[16 + index] - countersink_operator._fit["center"]
    inner -= countersink_operator._fit["axis"] * inner.dot(
        countersink_operator._fit["axis"]
    )
    end -= countersink_operator._fit["axis"] * end.dot(
        countersink_operator._fit["axis"]
    )
    assert abs(inner.length - end.length) < 1.0e-6
    assert inner.length > countersink_bottom_radius


# Forced Exact Boolean must use apex Depth itself, not wait for a secondary RNA
# property to drive its compound cutter. Rebuild a closed countersunk solid,
# deepen it by 0.5, and verify both the new 2.5-unit mouth and manifold result.
bpy.ops.object.mode_set(mode='OBJECT')
countersink_object.select_set(False)
boolean_countersink_object, boolean_countersink_bmesh = (
    create_countersunk_through_solid("boolean_countersink_depth_test")
)
boolean_countersink_faces = list(boolean_countersink_bmesh.faces[96:128])
boolean_countersink_operator = SimpleNamespace(
    _object=boolean_countersink_object,
    flood_angle=math.radians(20.0),
    diameter_estimator='AVERAGE',
    scale=1.0,
    diameter=0.0,
    angle_valid=False,
    depth_valid=False,
    _depth_fit=None,
    _locked_indices={face.index for face in boolean_countersink_faces},
    _resize_projection_ready=False,
    _resize_dynamic_requires_boolean=False,
    _original_vertex_coords=None,
    _proportional_weights=None,
    _boolean_preview_source=None,
    _boolean_reconstruction_source=None,
    resize_method='BOOLEAN',
    boolean_segments=32,
    fairing_method='PLANES',
    fill_style='PLANE',
    circle_loops=5,
    grid_match_curvature=True,
    final_simplify=False,
    scaling_final_simplify=False,
    fit_valid=False,
)
boolean_countersink_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(
        boolean_countersink_operator, face
    )
)
boolean_countersink_operator._fit = (
    MESH_OT_holey_tools_resize_hole._fit_cylinder(
        boolean_countersink_operator, boolean_countersink_faces
    )
)
boolean_countersink_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        boolean_countersink_operator,
        boolean_countersink_faces,
        boolean_countersink_operator._fit["axis"],
        boolean_countersink_operator._fit,
    )
)
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    boolean_countersink_operator,
    boolean_countersink_bmesh,
    boolean_countersink_faces,
)
boolean_deformation_vertices = {
    vert for face in boolean_countersink_faces for vert in face.verts
}
boolean_countersink_bmesh.verts.ensure_lookup_table()
boolean_deformation_vertices.update(
    boolean_countersink_bmesh.verts[index]
    for index in boolean_countersink_operator._resize_bore_vertex_indices
)
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    boolean_countersink_operator,
    bpy.context,
    boolean_countersink_bmesh,
    boolean_deformation_vertices,
)
MESH_OT_holey_tools_resize_hole._store_fit_for_redo(
    boolean_countersink_operator
)
boolean_countersink_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(
        boolean_countersink_operator
    )
)
boolean_countersink_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(
        boolean_countersink_operator
    )
)
boolean_countersink_operator.depth += 0.5
_update_conic_depth(boolean_countersink_operator, None)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    boolean_countersink_operator,
    bpy.context,
    commit_boolean=True,
)
boolean_countersink_bmesh = bmesh.from_edit_mesh(
    boolean_countersink_object.data
)
assert all(edge.is_manifold for edge in boolean_countersink_bmesh.edges)
boolean_top_radii = [
    math.hypot(vert.co.x, vert.co.y)
    for vert in boolean_countersink_bmesh.verts
    if abs(vert.co.z) < 1.0e-4
    and math.hypot(vert.co.x, vert.co.y) < 4.0
]
assert boolean_top_radii
assert max(abs(radius - 2.5) for radius in boolean_top_radii) < 1.0e-4
boolean_bore_radii = [
    math.hypot(vert.co.x, vert.co.y)
    for vert in boolean_countersink_bmesh.verts
    if vert.co.z < -1.1
    and math.hypot(vert.co.x, vert.co.y) < 2.0
]
assert boolean_bore_radii
boolean_bore_error = max(abs(radius - 1.0) for radius in boolean_bore_radii)
assert boolean_bore_error < 1.0e-4, (
    min(boolean_bore_radii), max(boolean_bore_radii), boolean_bore_error
)

# A fresh invocation on the Boolean result must be able to resize the same
# conic again. The clearance-wall projection used to collapse solver-split
# vertices onto identical coordinates without welding them, leaving duplicate
# boundary edges and a zero-length edge on this second pass.
boolean_countersink_bmesh.faces.ensure_lookup_table()
second_countersink_faces = [
    face
    for face in boolean_countersink_bmesh.faces
    if (
        0.2 < abs(face.normal.z) < 0.95
        and -2.0 < face.calc_center_median().z < 0.1
        and math.hypot(
            face.calc_center_median().x,
            face.calc_center_median().y,
        ) < 4.0
    )
]
assert len(second_countersink_faces) == 32
second_countersink_operator = SimpleNamespace(
    _object=boolean_countersink_object,
    flood_angle=math.radians(20.0),
    diameter_estimator='AVERAGE',
    scale=1.0,
    diameter=0.0,
    angle_valid=False,
    depth_valid=False,
    _depth_fit=None,
    _locked_indices={face.index for face in second_countersink_faces},
    _resize_projection_ready=False,
    _resize_dynamic_requires_boolean=False,
    _original_vertex_coords=None,
    _proportional_weights=None,
    _boolean_preview_source=None,
    _boolean_reconstruction_source=None,
    resize_method='BOOLEAN',
    boolean_segments=32,
    fairing_method='AUTO',
    fill_style='AUTO',
    circle_loops=5,
    grid_match_curvature=True,
    final_simplify=False,
    scaling_final_simplify=False,
    fit_valid=False,
)
second_countersink_operator._world_normal = lambda face: (
    MESH_OT_holey_tools_resize_hole._world_normal(
        second_countersink_operator, face
    )
)
second_countersink_operator._fit = (
    MESH_OT_holey_tools_resize_hole._fit_cylinder(
        second_countersink_operator, second_countersink_faces
    )
)
second_countersink_operator._end_planes = (
    MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
        second_countersink_operator,
        second_countersink_faces,
        second_countersink_operator._fit["axis"],
        second_countersink_operator._fit,
    )
)
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    second_countersink_operator,
    boolean_countersink_bmesh,
    second_countersink_faces,
)
second_deformation_vertices = {
    vert for face in second_countersink_faces for vert in face.verts
}
boolean_countersink_bmesh.verts.ensure_lookup_table()
second_deformation_vertices.update(
    boolean_countersink_bmesh.verts[index]
    for index in second_countersink_operator._resize_bore_vertex_indices
)
MESH_OT_holey_tools_resize_hole._capture_deformation_state(
    second_countersink_operator,
    bpy.context,
    boolean_countersink_bmesh,
    second_deformation_vertices,
)
MESH_OT_holey_tools_resize_hole._store_fit_for_redo(
    second_countersink_operator
)
second_countersink_operator._target_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_scale(
        second_countersink_operator
    )
)
second_countersink_operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(
        second_countersink_operator
    )
)
second_countersink_operator.depth += 0.25
_update_conic_depth(second_countersink_operator, None)
MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
    second_countersink_operator,
    bpy.context,
    commit_boolean=True,
)
boolean_countersink_bmesh = bmesh.from_edit_mesh(
    boolean_countersink_object.data
)
assert all(edge.is_manifold for edge in boolean_countersink_bmesh.edges)
assert not any(
    (edge.verts[0].co - edge.verts[1].co).length < 1.0e-8
    for edge in boolean_countersink_bmesh.edges
)


# Adjust Last Operation can retain a Python wrapper after Blender removes the
# Object RNA it referenced. Rebinding must not dereference that stale object and
# must discard every transient mesh/BVH reference before redo reconstruction.
bpy.ops.object.mode_set(mode='OBJECT')
boolean_countersink_object.select_set(False)
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

# The modal direct-resize path has already moved edit-mesh vertices when the
# persistent Redo overlay is created. Its retained baseline must normalize those
# vertices back to the coordinates captured before the preview, and it must be
# able to put that canonical topology back before a later execution.
replacement_bmesh = bmesh.from_edit_mesh(replacement_object.data)
replacement_bmesh.verts.ensure_lookup_table()
redo_original = replacement_bmesh.verts[0].co.copy()
replacement_bmesh.verts[0].co += Vector((7.0, -3.0, 2.0))
bmesh.update_edit_mesh(replacement_object.data, destructive=False)
redo_source_operator = SimpleNamespace(
    _object=replacement_object,
    _original_vertex_coords={0: redo_original},
)
(
    redo_source_name,
    redo_source_mesh,
    redo_source_bmesh,
    redo_source_materials,
) = _PersistentPreview._capture_canonical_source(redo_source_operator)
assert (
    redo_source_mesh.vertices[0].co - redo_original
).length < 1.0e-9
redo_source_snapshot = object.__new__(_PersistentPreview)
redo_source_snapshot._draw_handle = None
redo_source_snapshot._source_object_name = replacement_object.name
redo_source_snapshot._source_mesh_name = redo_source_name
redo_source_snapshot._source_mesh = redo_source_mesh
redo_source_snapshot._source_bmesh = redo_source_bmesh
redo_source_snapshot._source_materials = redo_source_materials
# The real source is hidden while an Exact Boolean display object owns the live
# preview. Confirmation must still restore it and return it to Edit Mode.
replacement_object.hide_set(True)
assert redo_source_snapshot.restore_source(replacement_object)
assert not replacement_object.hide_get()
assert replacement_object.mode == 'EDIT'
replacement_bmesh = bmesh.from_edit_mesh(replacement_object.data)
replacement_bmesh.verts.ensure_lookup_table()
assert (
    replacement_bmesh.verts[0].co - redo_original
).length < 1.0e-9
redo_source_snapshot.remove()


# Auto keeps a plane/cone/cylinder countersink in Scaling. Exercise every conic
# control from the same redo source and retain the continuing bore's diameter.
bpy.ops.object.mode_set(mode='OBJECT')
replacement_object.select_set(False)
auto_cone_object, auto_cone_bm = create_countersunk_through_solid(
    "auto_countersink_scaling", cone_rows=3
)
auto_cone_faces = [face for face in auto_cone_bm.faces if 0.2 < abs(face.normal.z) < 0.95]


def prepare_auto_resize(obj, bm, faces):
    op = SimpleNamespace(
        _object=obj, _locked_indices={face.index for face in faces},
        diameter_estimator='AVERAGE', resize_method='AUTO', scale=1.0,
        fit_valid=False, scaling_final_simplify=False,
    )
    op._world_normal = lambda face: Resize._world_normal(op, face)
    op._target_scale = lambda: Resize._target_scale(op)
    op._target_depth_scale = lambda: Resize._target_depth_scale(op)
    op._fit = Resize._fit_cylinder(op, faces)
    op._end_planes = Resize._fit_boundary_planes(op, faces, op._fit["axis"], op._fit)
    Resize._build_resize_projection_state(op, bm, faces)
    Resize._capture_deformation_state(op, bpy.context, bm, {vert for face in faces for vert in face.verts})
    Resize._store_fit_for_redo(op)
    snapshot = object.__new__(_PersistentPreview)
    snapshot._draw_handle = None
    snapshot._source_object_name = obj.name
    (snapshot._source_mesh_name, snapshot._source_mesh,
     snapshot._source_bmesh, snapshot._source_materials) = _PersistentPreview._capture_canonical_source(op)
    return op, snapshot


Resize = MESH_OT_holey_tools_resize_hole
auto_cone_op, auto_cone_source = prepare_auto_resize(auto_cone_object, auto_cone_bm, auto_cone_faces)
auto_cone_signature = Resize._object_topology_signature(auto_cone_object)
assert auto_cone_op._conic_bore_continuation
assert not auto_cone_op._resize_requires_boolean

# Use a registered RNA operator to exercise real conic callbacks and the
# selection-click lifecycle. Snap and tiny pointer motion must still dismiss
# without executing or changing a single source vertex or mesh face.
def check_untouched_cone(untouched_cone, context):
    untouched_cone._object = auto_cone_object
    untouched_cone._candidate_indices = {face.index for face in auto_cone_faces}
    untouched_cone._initial_selection = {face.index for face in auto_cone_bm.faces if face.select}
    untouched_cone._set_status_text = lambda _context: None
    untouched_cone._finish_modal = lambda _context: None
    untouched_cone._dimension_gizmo_at_event = lambda _event: None
    original_points = [vert.co.copy() for vert in auto_cone_bm.verts]
    assert untouched_cone._lock_selection_and_fit(context, build_preview=False)
    assert untouched_cone._active_dimension == 'DEPTH'
    saved_snap = context.scene.tool_settings.use_snap
    context.scene.tool_settings.use_snap = True
    untouched_cone._begin_dimension_drag(
        context, SimpleNamespace(mouse_x=10, mouse_y=20),
        untouched_cone._active_dimension,
    )
    for x, y in ((10, 20), (11, 21)):
        untouched_cone._update_dimension_drag(
            context, SimpleNamespace(mouse_x=x, mouse_y=y, ctrl=False, shift=False)
        )
    untouched_cone._drag_active = False
    assert untouched_cone.modal(
        SimpleNamespace(area=SimpleNamespace(type='VIEW_3D')),
        SimpleNamespace(type='LEFTMOUSE', value='PRESS'),
    ) == {'CANCELLED'}
    assert Resize._object_topology_signature(auto_cone_object) == auto_cone_signature
    assert all(vert.co == expected for vert, expected in zip(auto_cone_bm.verts, original_points))
    # RNA coupling must permit closure, then recover the fitted bottom radius
    # and physical span when a positive apex depth is entered again.
    initial_depth = untouched_cone.depth
    initial_truncated = untouched_cone.truncated_depth
    untouched_cone.depth = 0.0
    assert untouched_cone.depth == 0.0
    assert untouched_cone.truncated_depth == 0.0
    untouched_cone.depth = initial_depth
    assert abs(untouched_cone.truncated_depth - initial_truncated) < 1.0e-6
    initial_angle = untouched_cone.conic_angle
    initial_diameter = untouched_cone.diameter
    untouched_cone.diameter = 0.0
    untouched_cone.conic_angle = 0.0
    assert not Resize._fair_resize_requested(untouched_cone)
    untouched_cone.conic_angle = math.radians(60)
    assert abs(untouched_cone.depth - (
        untouched_cone.truncated_depth
        + untouched_cone.fit_conic_bottom_radius / math.tan(math.radians(30))
    )) < 1.0e-6
    untouched_cone.conic_angle = initial_angle
    untouched_cone.diameter = initial_diameter
    context.scene.tool_settings.use_snap = saved_snap
    return {'CANCELLED'}


with patch.object(Resize, "execute", check_untouched_cone), patch.object(
    Resize, "poll", classmethod(lambda _cls, _context: True)
):
    bpy.utils.register_class(Resize)
    try:
        assert bpy.ops.mesh.holey_tools_resize_hole('EXEC_DEFAULT') == {'CANCELLED'}
    finally:
        bpy.utils.unregister_class(Resize)

for depth, truncated, angle in (
    (2.3, 1.3, math.pi / 2.0),
    (1.6, 0.6, math.pi / 2.0),
    (1.5, 1.0, 2.0 * math.atan(2.0)),
):
    assert auto_cone_source.restore_source(auto_cone_object)
    auto_cone_op.depth = depth
    auto_cone_op.truncated_depth = truncated
    auto_cone_op.conic_angle = angle
    assert Resize.execute(auto_cone_op, bpy.context) == {'FINISHED'}
    assert not auto_cone_op._resize_requires_boolean
    assert Resize._object_topology_signature(auto_cone_object) == auto_cone_signature
    bm = bmesh.from_edit_mesh(auto_cone_object.data)
    bm.verts.ensure_lookup_table()
    assert all(edge.is_manifold for edge in bm.edges)
    # Shared lower circle follows truncation; the remote bore mouth stays fixed.
    for vert in bm.verts[128:192]:
        assert abs(math.hypot(vert.co.x, vert.co.y) - 1.0) < 1.0e-5
    assert max(abs(vert.co.z + truncated) for vert in bm.verts[128:160]) < 1.0e-5
    assert max(abs(vert.co.z + 3.0) for vert in bm.verts[160:192]) < 1.0e-5
auto_cone_source.remove()

# A real triangulated countersink can have a slightly biased fitted axis even
# though its surrounding top is perfectly planar. Depth translation must not
# be reapplied after projecting that exterior rim: doing so tilts individual
# support triangles, and the next invocation mistakes the top for a sphere.
for dimension, values, callback in (
    ("depth", (1.7, 1.5), _update_conic_depth),
    ("truncated_depth", (0.7, 0.5), _update_conic_truncated_depth),
    ("conic_angle", (math.radians(80), math.radians(70)), _update_conic_angle),
):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.object.select_set(False)
    obj, bm = create_countersunk_through_solid(
        "repeat_imperfect_countersink_" + dimension, cone_rows=3
    )
    bm.verts.ensure_lookup_table()
    bm.verts[64].co.x += 0.002
    bm.normal_update()
    cone_faces = [face for face in bm.faces if 0.2 < abs(face.normal.z) < 0.95]
    bmesh.ops.triangulate(bm, faces=cone_faces)
    bm.faces.index_update()
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
    signature = Resize._object_topology_signature(obj)
    for value in values:
        # Deliberately refit the current mesh, as a new invocation does, rather
        # than restoring a redo snapshot or reusing the previous fit.
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        cone_faces = [face for face in bm.faces if 0.2 < abs(face.normal.z) < 0.95]
        op, snapshot = prepare_auto_resize(obj, bm, cone_faces)
        try:
            assert op._fit["is_cone"] and op._conic_bore_continuation
            assert not op._resize_requires_boolean, (dimension, value)
            assert op._fit["axis"].cross(Vector((0, 0, 1))).length > 1.0e-6
            setattr(op, dimension, value)
            callback(op, None)
            assert Resize.execute(op, bpy.context) == {'FINISHED'}
            assert not op._resize_requires_boolean
            assert Resize._object_topology_signature(obj) == signature
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            assert max(abs(vert.co.z) for vert in bm.verts[:64]) < 1.0e-6
            assert all(edge.is_manifold for edge in bm.edges)
            assert max(abs(vert.co.z + 3.0) for vert in bm.verts[160:192]) < 1.0e-6
            assert max(abs(math.hypot(vert.co.x, vert.co.y) - 1.0) for vert in bm.verts[128:192]) < 1.0e-3
        finally:
            snapshot.remove()

# Resizing the cylinder instead slides the shared cone circle along the bore
# axis. Multiple cone rows redistribute on the original generators; their
# angle and the large mouth stay fixed for both shrinking and enlargement.
bpy.ops.object.mode_set(mode='OBJECT')
auto_cone_object.select_set(False)
auto_bore_object, auto_bore_bm = create_countersunk_through_solid(
    "auto_bore_with_subdivided_cone", cone_rows=3
)
auto_bore_faces = [
    face for face in auto_bore_bm.faces
    if abs(face.normal.z) < 0.1 and math.hypot(face.calc_center_median().x, face.calc_center_median().y) < 1.1
]
auto_bore_op, auto_bore_source = prepare_auto_resize(auto_bore_object, auto_bore_bm, auto_bore_faces)
auto_bore_signature = Resize._object_topology_signature(auto_bore_object)
assert auto_bore_op._resize_conic_surfaces_by_vertex
assert not auto_bore_op._resize_requires_boolean
for diameter in (0.8, 2.4, 3.4, 2.0):
    assert auto_bore_source.restore_source(auto_bore_object)
    auto_bore_op.diameter = diameter
    assert Resize.execute(auto_bore_op, bpy.context) == {'FINISHED'}
    assert not auto_bore_op._resize_requires_boolean
    assert Resize._object_topology_signature(auto_bore_object) == auto_bore_signature
    bm = bmesh.from_edit_mesh(auto_bore_object.data)
    bm.verts.ensure_lookup_table()
    assert all(edge.is_manifold for edge in bm.edges)
    assert all(face.calc_area() > 1.0e-8 for face in bm.faces)
    for vert in bm.verts[32:160]:
        assert abs(math.hypot(vert.co.x, vert.co.y) - (2.0 + vert.co.z)) < 1.0e-5
    assert max(abs(vert.co.z) for vert in bm.verts[32:64]) < 1.0e-5
    for vert in bm.verts[128:192]:
        assert abs(math.hypot(vert.co.x, vert.co.y) - diameter / 2.0) < 1.0e-5
    assert max(abs(vert.co.z + 3.0) for vert in bm.verts[160:192]) < 1.0e-5
auto_bore_source.remove()

# Zero closes the complete cavity, even with Scaling explicitly chosen, and
# redo can subsequently return to a nonzero size from the unchanged source.
for fixture in (create_planar_blind_hole, lambda name: create_closed_ring_solid(name, top_outer_z=0.0)):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.active_object.select_set(False)
    obj, bm = fixture("zero_closes_hole")
    bm.normal_update()
    faces = [face for face in bm.faces if abs(face.normal.z) < .1 and math.hypot(face.calc_center_median().x, face.calc_center_median().y) < 1.1]
    op, source = prepare_auto_resize(obj, bm, faces)
    op.report = lambda _level, message: (_ for _ in ()).throw(AssertionError(message))
    for dimension in ('diameter', 'depth') if op.depth_valid else ('diameter',):
        for mode in ('SCALE', 'BOOLEAN'):
            assert source.restore_source(obj)
            op.diameter = op.fit_radius * 2.0
            op.depth = op.fit_depth_initial
            setattr(op, dimension, 0.0)
            op.resize_method = mode
            assert Resize.execute(op, bpy.context) == {'FINISHED'}
            closed = bmesh.from_edit_mesh(obj.data)
            assert all(edge.is_manifold for edge in closed.edges)
            assert all(face.calc_area() > 1.0e-8 for face in closed.faces)
            assert not any(abs(face.normal.z) < 0.1 and math.hypot(face.calc_center_median().x, face.calc_center_median().y) < 1.1 for face in closed.faces)
    # Modal fair-only display is reversible and never substitutes its filled
    # mesh for the edit-mode source consumed by the next nonzero preview.
    assert source.restore_source(obj)
    Resize._reset_execute_transients(op)
    Resize._restore_fit_for_redo(op)
    op.diameter = 0.0
    Resize._apply_diameter_to_mesh(op, bpy.context)
    assert op._resize_fair_only_preview
    assert op._boolean_preview_display_object is not None
    op.diameter = 1.5
    op.resize_method = 'SCALE'
    if op.depth_valid:
        op.depth = op.fit_depth_initial
    Resize._apply_diameter_to_mesh(op, bpy.context, commit_boolean=True)
    assert not op._resize_fair_only_preview
    assert all(edge.is_manifold for edge in bmesh.from_edit_mesh(obj.data).edges)
    source.remove()

# A bare through-cone has two exterior mouths, not a blind cap or a preexisting
# bore. Shortening it retains the small exit and inserts a constant-radius bore.
bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.active_object.select_set(False)
segments = 32
vertices = [
    (radius * math.cos(math.tau * i / segments), radius * math.sin(math.tau * i / segments), z)
    for radius, z in ((5.0, 0.0), (2.0, 0.0), (1.0, -2.0), (5.0, -2.0))
    for i in range(segments)
]
faces = []
for i in range(segments):
    j = (i + 1) % segments
    faces.extend(((i, j, segments+j, segments+i),
                  (segments+i, segments+j, 2*segments+j, 2*segments+i),
                  (2*segments+i, 2*segments+j, 3*segments+j, 3*segments+i),
                  (3*segments+i, 3*segments+j, j, i)))
mesh = bpy.data.meshes.new("through_cone")
mesh.from_pydata(vertices, [], faces)
pristine_cone_mesh = mesh.copy()
obj = bpy.data.objects.new("through_cone", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(mesh)
bm.normal_update()
bm.faces.ensure_lookup_table()
cone_faces = [face for face in bm.faces if .1 < abs(face.normal.z) < .95]
op, source = prepare_auto_resize(obj, bm, cone_faces)
op.report = lambda _level, message: (_ for _ in ()).throw(AssertionError(message))
assert op._conic_exit_coordinates is not None
assert op.depth_valid and op.angle_valid
for mode in ('AUTO', 'BOOLEAN', 'SCALE'):
    for truncated in (1.2, .8, 0.0):
        assert source.restore_source(obj)
        op.resize_method = mode
        op.truncated_depth = truncated
        op.depth = truncated + 2.0
        assert Resize.execute(op, bpy.context) == {'FINISHED'}
        bm = bmesh.from_edit_mesh(obj.data)
        bm.normal_update()
        assert all(edge.is_manifold for edge in bm.edges)
        assert all(face.calc_area() > 1.0e-8 for face in bm.faces)
        exit_verts = [vert for vert in bm.verts if abs(vert.co.z + 2.0) < 1.0e-5 and math.hypot(vert.co.x, vert.co.y) < 1.1]
        assert len(exit_verts) >= 32
        assert all(abs(math.hypot(vert.co.x, vert.co.y) - 1.0) < 1.0e-5 for vert in exit_verts)
        cylindrical = [face for face in bm.faces if abs(face.normal.z) < .01 and math.hypot(face.calc_center_median().x, face.calc_center_median().y) < 1.1]
        assert len(cylindrical) >= 32
        from mathutils.bvhtree import BVHTree
        assert BVHTree.FromBMesh(bm).ray_cast(Vector((0, 0, 1)), Vector((0, 0, -1)), 5)[0] is None
        assert op._resize_requires_boolean == (mode != 'SCALE')
source.remove()

# Explicit Scaling previews can repeatedly change truncation without stacking
# connector rings, and returning to the original dimensions restores topology.
# Capture a fresh source from the pristine fixture, not the previous result.
bpy.ops.object.mode_set(mode='OBJECT')
obj.data = pristine_cone_mesh.copy()
bpy.data.meshes.remove(pristine_cone_mesh)
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.normal_update()
bm.faces.ensure_lookup_table()
op, source = prepare_auto_resize(obj, bm, [face for face in bm.faces if .1 < abs(face.normal.z) < .95])
signature = Resize._object_topology_signature(obj)
op.resize_method = 'SCALE'
for truncated in (1.2, .8, 1.5):
    op.truncated_depth = truncated
    op.depth = truncated + 2.0
    Resize._apply_diameter_to_mesh(op, bpy.context)
    assert op._resize_scaled_exit_preview
    bm = bmesh.from_edit_mesh(obj.data)
    assert len(bm.verts) == 160
    assert all(edge.is_manifold for edge in bm.edges)
op.truncated_depth = op.fit_truncated_depth_initial
op.depth = op.fit_conic_depth_initial
Resize._apply_diameter_to_mesh(op, bpy.context, commit_boolean=True)
assert Resize._object_topology_signature(obj) == signature
source.remove()

# Closing either part of a countersink removes its entire connected cavity.
for select_cone in (True, False):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.active_object.select_set(False)
    obj, bm = create_countersunk_through_solid("close_countersink", cone_rows=3)
    bm.normal_update()
    faces = [face for face in bm.faces if (
        .1 < abs(face.normal.z) < .95 if select_cone else
        abs(face.normal.z) < .1 and math.hypot(face.calc_center_median().x, face.calc_center_median().y) < 1.1
    )]
    op, source = prepare_auto_resize(obj, bm, faces)
    if select_cone:
        # Conic controls expose Depth, not the hidden Diameter baseline.
        op.depth = 0.0
        _update_conic_depth(op, None)
    else:
        op.diameter = 0.0
    op.report = lambda _level, message: (_ for _ in ()).throw(AssertionError(message))
    assert Resize.execute(op, bpy.context) == {'FINISHED'}
    bm = bmesh.from_edit_mesh(obj.data)
    assert all(edge.is_manifold for edge in bm.edges)
    assert all(face.calc_area() > 1.0e-8 for face in bm.faces)
    assert all(abs(face.normal.z) > .95 or math.hypot(face.calc_center_median().x, face.calc_center_median().y) > 4.9 for face in bm.faces)
    source.remove()

dimension_update_operator.depth = 0.0
_update_conic_depth(dimension_update_operator, None)
assert dimension_update_operator.depth == 0.0
assert dimension_update_operator.truncated_depth == 0.0

# Exact Boolean on a countersink must not re-tessellate its unchanged cylinder.
# Rotate the source samples off the fit basis and include axial subdivisions:
# matching only segment count, but not angular phase, used to split every row.
def bore_polygons(bm):
    return {
        tuple(sorted(tuple(round(c, 5) for c in vert.co) for vert in face.verts))
        for face in bm.faces
        if abs(face.normal.z) < .01
        and all(abs(math.hypot(vert.co.x, vert.co.y) - 1.0) < 1.0e-5 for vert in face.verts)
    }


bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.active_object.select_set(False)
obj, bm = create_countersunk_through_solid("preserve_boolean_bore", cone_rows=3)
for vert in bm.verts:
    x, y = vert.co.x, vert.co.y
    vert.co.x = x * math.cos(.173) - y * math.sin(.173)
    vert.co.y = x * math.sin(.173) + y * math.cos(.173)
longitudinal = [
    edge for edge in bm.edges
    if abs(edge.verts[0].co.z - edge.verts[1].co.z) > 1.5
    and all(abs(math.hypot(v.co.x, v.co.y) - 1.0) < 1.0e-6 for v in edge.verts)
]
bmesh.ops.subdivide_edges(bm, edges=longitudinal, cuts=2, use_grid_fill=True)
bm.normal_update()
bm.verts.index_update()
bm.faces.index_update()
bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
cone_faces = [face for face in bm.faces if .2 < abs(face.normal.z) < .95]
op, source = prepare_auto_resize(obj, bm, cone_faces)
op.report = lambda _level, message: (_ for _ in ()).throw(AssertionError(message))
original_bore = bore_polygons(bm)
assert len(original_bore) == 96


def expected_preserved_bore(join):
    return {
        tuple(sorted((x, y, round(join, 5) if abs(z + 1.0) < 1.0e-5 else z)
                     for x, y, z in polygon))
        for polygon in original_bore
    }


for method, truncated, angle, simplify in (
    ('SCALE', 1.2, math.pi / 2, False),
    ('BOOLEAN', 1.2, math.pi / 2, False),
    ('BOOLEAN', .6, math.pi / 2, True),
    ('BOOLEAN', 1.0, math.radians(110), True),
):
    assert source.restore_source(obj)
    op.resize_method = method
    op.truncated_depth = truncated
    op.conic_angle = angle
    op.depth = truncated + 1.0 / math.tan(angle / 2)
    op.final_simplify = simplify
    assert Resize.execute(op, bpy.context) == {'FINISHED'}
    current = bmesh.from_edit_mesh(obj.data)
    assert all(edge.is_manifold for edge in current.edges)
    assert all(face.calc_area() > 1.0e-8 for face in current.faces)
    assert bore_polygons(current) == expected_preserved_bore(-truncated), (method, truncated, len(bore_polygons(current)))

# A changed cutter resolution transitions only at the junction, while every
# original cylinder face/edge further down the bore remains unchanged.
for segments in (3, 16, 47):
    assert source.restore_source(obj)
    op.resize_method = 'BOOLEAN'
    op.boolean_segments = segments
    op.truncated_depth = .8
    op.conic_angle = math.pi / 2
    op.depth = 1.8
    assert Resize.execute(op, bpy.context) == {'FINISHED'}
    current = bmesh.from_edit_mesh(obj.data)
    assert all(edge.is_manifold for edge in current.edges), segments
    assert all(face.calc_area() > 1.0e-8 for face in current.faces), segments
    assert expected_preserved_bore(-.802) <= bore_polygons(current), (segments, len(bore_polygons(current)))

# Cached live updates move the retained junction as well as the cutter. They
# must not keep the cylinder frozen at the first preview's truncation depth.
assert source.restore_source(obj)
Resize._reset_execute_transients(op)
assert Resize._restore_fit_for_redo(op)
op.boolean_segments = 32
op.truncated_depth = 1.2
op.depth = 2.2
Resize._apply_diameter_to_mesh(op, bpy.context)
assert op._boolean_preserved_bore is not None
display = op._boolean_preview_display_object
for truncated, segments in ((.7, 32), (1.4, 16), (1.1, 32)):
    op.truncated_depth = truncated
    op.depth = truncated + 1.0
    op.boolean_segments = segments
    Resize._apply_diameter_to_mesh(op, bpy.context)
    assert op._boolean_preview_display_object is display
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = display.evaluated_get(depsgraph)
    evaluated_mesh = evaluated.to_mesh()
    current = bmesh.new()
    try:
        current.from_mesh(evaluated_mesh)
        current.normal_update()
        assert all(edge.is_manifold for edge in current.edges)
        join = -truncated - (.002 if segments != 32 else 0.0)
        assert expected_preserved_bore(join) <= bore_polygons(current), (truncated, segments)
    finally:
        current.free()
        evaluated.to_mesh_clear()
Resize._replace_boolean_preview_with_source(op, consume=True)
Resize._discard_boolean_preview_source(op)

# Moving the junction across a source row must use the general reconstruction
# rather than folding the preserved first strip through that row.
assert source.restore_source(obj)
op.truncated_depth = 1.8
op.depth = 2.8
op.boolean_segments = 32
with patch.object(GeometryMixin, "_cap_preserved_boolean_bore", side_effect=AssertionError("Consumed source row must not be preserved")):
    assert Resize.execute(op, bpy.context) == {'FINISHED'}
current = bmesh.from_edit_mesh(obj.data)
assert all(edge.is_manifold for edge in current.edges)
assert all(face.calc_area() > 1.0e-8 for face in current.faces)

# A fresh invocation after a Boolean must preserve the bore just as redo does.
assert source.restore_source(obj)
op.boolean_segments = 32
op.truncated_depth = 1.2
op.depth = 2.2
assert Resize.execute(op, bpy.context) == {'FINISHED'}
for truncated in (1.4, .9):
    current = bmesh.from_edit_mesh(obj.data)
    current.normal_update()
    cone_faces = [face for face in current.faces if .2 < abs(face.normal.z) < .95]
    next_op, next_source = prepare_auto_resize(obj, current, cone_faces)
    next_op.report = op.report
    next_op.resize_method = 'BOOLEAN'
    next_op.truncated_depth = truncated
    next_op.depth = truncated + 1.0
    assert Resize.execute(next_op, bpy.context) == {'FINISHED'}
    current = bmesh.from_edit_mesh(obj.data)
    assert bore_polygons(current) == expected_preserved_bore(-truncated)
    assert all(edge.is_manifold for edge in current.edges)
    next_source.remove()
source.remove()
assert not [material for material in bpy.data.materials if material.name.startswith("Holey Tools Fair Surface") and material.users == 0]

# The preserved junction is expressed in world-space fit coordinates, while
# source mesh vertices remain local, including for rotated/scaled objects.
bpy.ops.object.mode_set(mode='OBJECT')
obj.select_set(False)
obj, bm = create_countersunk_through_solid("preserve_bore_transformed")
from mathutils import Matrix
obj.matrix_world = Matrix.Translation((3, -4, 2)) @ Matrix.Rotation(.41, 4, 'X') @ Matrix.Scale(1.5, 4)
cone_faces = list(bm.faces[96:128])
op, source = prepare_auto_resize(obj, bm, cone_faces)
op.resize_method = 'BOOLEAN'
op.truncated_depth = 1.8
op.depth = 3.3
op.final_simplify = True
op.report = lambda _level, message: (_ for _ in ()).throw(AssertionError(message))
assert Resize.execute(op, bpy.context) == {'FINISHED'}
current = bmesh.from_edit_mesh(obj.data)
assert all(edge.is_manifold for edge in current.edges)
cylinder = bore_polygons(current)
assert len(cylinder) == 32
assert {z for polygon in cylinder for _x, _y, z in polygon} == {-3.0, -1.2}
assert len(obj.data.materials) == 0
source.remove()

# Zero included angle removes a countersink, not its connected through-hole.
# Test the visible modal result and redo, including a hidden zero Diameter.
bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.object.select_set(False)
obj, bm = create_countersunk_through_solid("zero_angle_keeps_through_bore", cone_rows=3)
cone_faces = [face for face in bm.faces if .2 < abs(face.normal.z) < .95]
op, source = prepare_auto_resize(obj, bm, cone_faces)
op.report = lambda _level, message: (_ for _ in ()).throw(AssertionError(message))
from mathutils.bvhtree import BVHTree


def check_zero_angle_bore(current):
    current.normal_update()
    assert all(edge.is_manifold for edge in current.edges), (
        method, [(tuple(edge.verts[0].co), tuple(edge.verts[1].co), len(edge.link_faces))
                 for edge in current.edges if not edge.is_manifold]
    )
    assert all(face.calc_area() > 1.0e-8 for face in current.faces)
    assert BVHTree.FromBMesh(current).ray_cast(Vector((0, 0, 1)), Vector((0, 0, -1)), 5)[0] is None
    # A ray just outside the original bore must hit the restored top surface.
    hit = BVHTree.FromBMesh(current).ray_cast(Vector((1.4, 0, 1)), Vector((0, 0, -1)), 5)[0]
    assert hit is not None and abs(hit.z) < 1.0e-5
    wall_verts = [vert for vert in current.verts if math.hypot(vert.co.x, vert.co.y) < 1.1]
    assert len(wall_verts) >= 64
    assert max(abs(math.hypot(vert.co.x, vert.co.y) - 1.0) for vert in wall_verts) < 1.0e-5


for method in ('AUTO', 'SCALE', 'BOOLEAN'):
    assert source.restore_source(obj)
    Resize._reset_execute_transients(op)
    assert Resize._restore_fit_for_redo(op)
    op.resize_method = method
    op.diameter = 0.0
    op.conic_angle = math.radians(60)
    _update_conic_angle(op, None)
    Resize._apply_diameter_to_mesh(op, bpy.context)
    positive_display = getattr(op, "_boolean_preview_display_object", None)
    operand_points = (
        [vert.co.copy() for vert in positive_display.data.vertices]
        if positive_display is not None else None
    )
    op.conic_angle = 0.0
    _update_conic_angle(op, None)
    assert not Resize._fair_resize_requested(op)
    Resize._apply_diameter_to_mesh(op, bpy.context)
    display = getattr(op, "_boolean_preview_display_object", None)
    if display is None:
        check_zero_angle_bore(bmesh.from_edit_mesh(obj.data))
    else:
        assert display is positive_display
        assert all(vert.co == point for vert, point in zip(display.data.vertices, operand_points))
        evaluated = display.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        current = bmesh.new()
        try:
            current.from_mesh(mesh)
            check_zero_angle_bore(current)
        finally:
            current.free()
            evaluated.to_mesh_clear()
    Resize._replace_boolean_preview_with_source(op, consume=True)
    Resize._discard_boolean_preview_source(op)
    assert source.restore_source(obj)
    assert Resize.execute(op, bpy.context) == {'FINISHED'}
    check_zero_angle_bore(bmesh.from_edit_mesh(obj.data))
    # Reopening the same redo operation must recover a real cone and apex.
    op.conic_angle = math.radians(60)
    _update_conic_angle(op, None)
    assert abs(op.depth - (op.truncated_depth + 1.0 / math.tan(math.radians(30)))) < 1.0e-5
    assert source.restore_source(obj)
    assert Resize.execute(op, bpy.context) == {'FINISHED'}
    current = bmesh.from_edit_mesh(obj.data)
    current.normal_update()
    assert all(edge.is_manifold for edge in current.edges)
    assert BVHTree.FromBMesh(current).ray_cast(Vector((0, 0, 1)), Vector((0, 0, -1)), 5)[0] is None
source.remove()

# Fresh selection after removing a countersink must flood the whole straight
# bore, not split it at the former cone/bore seam using stale face normals.
bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.object.select_set(False)
obj, bm = create_countersunk_through_solid("resize_after_zero_angle", cone_rows=3)
cone_faces = list(bm.faces[96:192])
op, source = prepare_auto_resize(obj, bm, cone_faces)
op.conic_angle = 0.0
_update_conic_angle(op, None)
Resize._apply_diameter_to_mesh(op, bpy.context)
# Cancellation restores normals as well as coordinates.
Resize._restore_original_geometry(op)
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()
assert .2 < abs(bm.faces[96].normal.z) < .95
Resize._apply_diameter_to_mesh(op, bpy.context)
assert source.restore_source(obj)
assert Resize.execute(op, bpy.context) == {'FINISHED'}
source.remove()
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()
probe = SimpleNamespace(_object=obj, flood_angle=math.radians(20))
probe._world_normal = lambda face: Resize._world_normal(probe, face)
wall = Resize._flood_fill(probe, bm.faces[192])
assert len(wall) == 128, len(wall)
assert all(abs(face.normal.z) < 1.0e-5 for face in wall)
op, source = prepare_auto_resize(obj, bm, wall)
assert not op._fit["is_cone"] and not op._resize_requires_boolean
op.diameter = 1.6
Resize._apply_diameter_to_mesh(op, bpy.context)
assert source.restore_source(obj)
assert Resize.execute(op, bpy.context) == {'FINISHED'}
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()
bm.verts.ensure_lookup_table()
assert all(abs(bm.faces[i].normal.z) < 1.0e-5 for i in range(96, 224))
assert max(abs(math.hypot(v.co.x, v.co.y) - .8) for v in bm.verts[32:192]) < 1.0e-5
assert all(edge.is_manifold for edge in bm.edges)
source.remove()

# Boolean preview snapshots must also preserve preceding edits which exist
# only in BMesh, rather than resurrecting the stale mesh datablock coordinates.
op, source = prepare_auto_resize(obj, bm, list(bm.faces[96:224]))
op.resize_method = 'BOOLEAN'
op.diameter = 1.5
bm.verts[0].co.x += .125
expected = bm.verts[0].co.copy()
bm.normal_update()
bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
Resize._apply_diameter_to_mesh(op, bpy.context)
assert op._boolean_preview_source is not None
assert op._boolean_preview_source.vertices[0].co == expected
Resize._replace_boolean_preview_with_source(op, consume=True)
Resize._discard_boolean_preview_source(op)
bm = bmesh.from_edit_mesh(obj.data)
bm.verts.ensure_lookup_table()
assert bm.verts[0].co == expected
source.remove()

print("HOLEY_TOOLS_MODE_TESTS_OK")
