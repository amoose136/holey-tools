import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy
from mathutils import Matrix, Vector


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src  # noqa: E402
from src.fair_hole import (  # noqa: E402
    MESH_OT_holey_tools_fair_hole,
    _limited_dissolve_analytic_boundary,
    _limited_dissolve_cylinder_patch,
)
from src.fairing import (  # noqa: E402
    FairingError,
    RELATIVE_RESIDUAL_TOLERANCE,
    _limited_dissolve_analytic_boundary as shared_analytic_dissolve,
    _limited_dissolve_cylinder_patch as shared_cylinder_dissolve,
    boundary_component,
    choose_hypothesis,
    fair_boundary_loop,
    fit_sphere_hypothesis,
    limited_dissolve_boundary,
    limited_dissolve_patch,
    ordered_boundary_loop,
    _organic_direction_samples,
    _organic_grid_frame,
    _organic_grid_layout,
    _plane_intersection,
    _simplify_organic_boundary,
    _support_planes,
)
from src.geometry import GeometryMixin  # noqa: E402


assert _limited_dissolve_analytic_boundary is shared_analytic_dissolve
assert _limited_dissolve_cylinder_patch is shared_cylinder_dissolve

parallel_reference = Vector((0.0, 0.0, 0.0))
parallel_plane = {
    "point": parallel_reference,
    "normal": Vector((0.0, 0.0, 1.0)),
}
near_parallel_plane = {
    "point": parallel_reference,
    "normal": Vector((
        math.sin(math.radians(0.39)),
        0.0,
        math.cos(math.radians(0.39)),
    )),
}
distinct_plane = {
    "point": parallel_reference,
    "normal": Vector((
        math.sin(math.radians(0.41)),
        0.0,
        math.cos(math.radians(0.41)),
    )),
}
assert _plane_intersection(
    [parallel_plane, near_parallel_plane], parallel_reference, 1.0e-6
) is None
assert _plane_intersection(
    [parallel_plane, distinct_plane], parallel_reference, 1.0e-6
) is not None


def edit_mesh(name, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.normal_update()
    return obj, bm


def leave_edit_mode(obj):
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)


def grid_mesh(name, coordinates, remove_cells):
    rows = len(coordinates)
    columns = len(coordinates[0])
    vertices = [coordinate for row in coordinates for coordinate in row]
    faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            if (row, column) in remove_cells:
                continue
            first = row * columns + column
            faces.append(
                (
                    first,
                    first + 1,
                    first + columns + 1,
                    first + columns,
                )
            )
    return edit_mesh(name, vertices, faces)


def inner_boundary_seed(bm):
    components = []
    remaining = {edge for edge in bm.edges if len(edge.link_faces) == 1}
    while remaining:
        seed = next(iter(remaining))
        component = boundary_component(seed)
        components.append(component)
        remaining.difference_update(component)
    return min(components, key=len)


def assert_closed_rim(edges):
    for edge in edges:
        assert len(edge.link_faces) == 2
        first, second = edge.link_faces
        assert first.normal.dot(second.normal) > 0.0


def support_face(points, start_index):
    return SimpleNamespace(
        verts=[
            SimpleNamespace(index=start_index + index, co=Vector(point))
            for index, point in enumerate(points)
        ]
    )


# Non-adjacent support runs can still be parts of one geometric plane. They
# must be merged before applying the three-distinct-plane limit.
separate_plane_regions = [
    support_face(((1, 1, 0), (2, 1, 0), (1, 2, 0)), 0),
    support_face(((0, 1, 1), (0, 2, 1), (0, 1, 2)), 3),
    support_face(((-1, -1, 0), (-2, -1, 0), (-1, -2, 0)), 6),
    support_face(((1, 0, 1), (2, 0, 1), (1, 0, 2)), 9),
]
merged_support_planes = _support_planes(
    [None] * len(separate_plane_regions),
    Matrix.Identity(4),
    1.0e-7,
    separate_plane_regions,
)
assert merged_support_planes is not None
assert len(merged_support_planes) == 3, merged_support_planes
assert sorted(len(plane["points"]) for plane in merged_support_planes) == [3, 3, 6]


# A planar annulus should take the first hypothesis and close only its inner rim.
planar_vertices = [
    (-2.0, -2.0, 0.0),
    (2.0, -2.0, 0.0),
    (2.0, 2.0, 0.0),
    (-2.0, 2.0, 0.0),
    (-1.0, -1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, -1.0, 0.0),
]
planar_faces = [
    (0, 1, 7, 4),
    (1, 2, 6, 7),
    (2, 3, 5, 6),
    (3, 0, 4, 5),
]
planar_object, planar_bm = edit_mesh(
    "planar_hole_fairing", planar_vertices, planar_faces
)
planar_components = []
planar_remaining = {edge for edge in planar_bm.edges if len(edge.link_faces) == 1}
while planar_remaining:
    planar_seed = next(iter(planar_remaining))
    planar_component_candidate = boundary_component(planar_seed)
    planar_components.append(planar_component_candidate)
    planar_remaining.difference_update(planar_component_candidate)
planar_component = min(
    planar_components,
    key=lambda component: sum(
        vert.co.length for edge in component for vert in edge.verts
    ),
)
planar_loop, planar_edges = ordered_boundary_loop(edges=planar_component)
planar_adapter = SimpleNamespace(_object=planar_object)
planar_adapter._world_normal = lambda face: GeometryMixin._world_normal(
    planar_adapter, face
)
planar_hypothesis = choose_hypothesis(
    planar_loop,
    planar_edges,
    planar_object.matrix_world,
    lambda faces: GeometryMixin._fit_cylinder(planar_adapter, faces),
    planar_adapter._world_normal,
)
assert planar_hypothesis["method"] == "PLANES", planar_hypothesis
planar_result = fair_boundary_loop(
    planar_bm,
    planar_loop,
    planar_edges,
    planar_object.matrix_world,
    planar_hypothesis,
)
assert len(planar_result["verts"]) == 1
assert len(planar_result["faces"]) == 4
assert_closed_rim(planar_edges)


# Fair Hole normalizes redundant collinear rim vertices before fitting, then
# limited-dissolves the completed coplanar patch and its former seam.
leave_edit_mode(planar_object)
dissolve_inner = (
    (-1.0, -1.0, 0.0),
    (0.0, -1.0, 0.0),
    (1.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (-1.0, 0.0, 0.0),
)
dissolve_outer = tuple(
    (x * 2.0, y * 2.0, z) for x, y, z in dissolve_inner
)
dissolve_faces = [
    (index, (index + 1) % 8, 8 + (index + 1) % 8, 8 + index)
    for index in range(8)
]
dissolve_object, dissolve_bm = edit_mesh(
    "limited_dissolve_fairing",
    dissolve_outer + dissolve_inner,
    dissolve_faces,
)
dissolve_component = min(
    (
        boundary_component(edge)
        for edge in dissolve_bm.edges
        if len(edge.link_faces) == 1
    ),
    key=lambda component: sum(
        vert.co.length for edge in component for vert in edge.verts
    ),
)
dissolve_loop, dissolve_edges = ordered_boundary_loop(
    edges=dissolve_component
)
dissolve_loop, dissolve_edges = limited_dissolve_boundary(
    dissolve_bm, dissolve_loop, dissolve_edges
)
assert len(dissolve_loop) == 4, len(dissolve_loop)
dissolve_adapter = SimpleNamespace(_object=dissolve_object)
dissolve_adapter._world_normal = lambda face: GeometryMixin._world_normal(
    dissolve_adapter, face
)
dissolve_hypothesis = choose_hypothesis(
    dissolve_loop,
    dissolve_edges,
    dissolve_object.matrix_world,
    lambda faces: GeometryMixin._fit_cylinder(dissolve_adapter, faces),
    dissolve_adapter._world_normal,
)
dissolve_result = fair_boundary_loop(
    dissolve_bm,
    dissolve_loop,
    dissolve_edges,
    dissolve_object.matrix_world,
    dissolve_hypothesis,
)
dissolve_result.update(limited_dissolve_patch(
    dissolve_bm,
    dissolve_loop,
    dissolve_edges,
    dissolve_result["verts"],
    dissolve_result["faces"],
))
assert dissolve_result["faces"]
assert all(face.is_valid for face in dissolve_result["faces"])
assert all(vert.is_valid for vert in dissolve_result["verts"])
dissolved_boundary = {
    edge for edge in dissolve_bm.edges if len(edge.link_faces) == 1
}
assert len(dissolved_boundary) >= 3
assert boundary_component(next(iter(dissolved_boundary))) == dissolved_boundary

# Exercise the redo toggle through the operator. The seam is still welded, but
# disabling final simplification must leave only the generated patch selected.
leave_edit_mode(dissolve_object)
operator_object, operator_bm = edit_mesh(
    "limited_dissolve_operator",
    dissolve_outer + dissolve_inner,
    dissolve_faces,
)
operator_component = min(
    (
        boundary_component(edge)
        for edge in operator_bm.edges
        if len(edge.link_faces) == 1
    ),
    key=lambda component: sum(
        vert.co.length for edge in component for vert in edge.verts
    ),
)
fair_operator = SimpleNamespace(
    edge_indices=",".join(
        str(edge.index) for edge in sorted(
            operator_component, key=lambda edge: edge.index
        )
    ),
    result_method="",
    result_fill_style="",
    result_residual=0.0,
    fairing_method='AUTO',
    fill_style='AUTO',
    final_simplify=False,
    _object=None,
    report=lambda *_args: None,
)
assert MESH_OT_holey_tools_fair_hole.execute(
    fair_operator, bpy.context
) == {'FINISHED'}
operator_bm = bmesh.from_edit_mesh(operator_object.data)
assert fair_operator.result_method == "PLANES"
assert fair_operator.result_fill_style == "PLANE"
operator_boundary = {
    edge for edge in operator_bm.edges if len(edge.link_faces) == 1
}
assert len(operator_boundary) >= 3
assert boundary_component(next(iter(operator_boundary))) == operator_boundary
selected_operator_faces = [face for face in operator_bm.faces if face.select]
assert selected_operator_faces
assert len(selected_operator_faces) < len(operator_bm.faces)

# A normally tessellated circular rim must survive the 0.1-degree pre-pass.
# The former 5-degree batch classified every vertex on this 128-sided circle as
# removable and could erase the selected boundary entirely.
leave_edit_mode(operator_object)
dense_segments = 128
dense_vertices = [
    (
        radius * math.cos(math.tau * index / dense_segments),
        radius * math.sin(math.tau * index / dense_segments),
        0.0,
    )
    for radius in (2.0, 1.0)
    for index in range(dense_segments)
]
dense_faces = [
    (
        index,
        (index + 1) % dense_segments,
        dense_segments + (index + 1) % dense_segments,
        dense_segments + index,
    )
    for index in range(dense_segments)
]
dense_object, dense_bm = edit_mesh(
    "dense_limited_dissolve_ring", dense_vertices, dense_faces
)
dense_component = min(
    (
        boundary_component(edge)
        for edge in dense_bm.edges
        if len(edge.link_faces) == 1
    ),
    key=lambda component: sum(
        vert.co.length for edge in component for vert in edge.verts
    ),
)
dense_loop, dense_edges = ordered_boundary_loop(edges=dense_component)
dense_loop, dense_edges = limited_dissolve_boundary(
    dense_bm, dense_loop, dense_edges
)
assert len(dense_loop) == dense_segments


# A two-by-two missing patch on a cylindrical grid has four distinct support
# planes, so it must fall through to the shared circular-cone/cylinder fit.
leave_edit_mode(dense_object)
segments = 12
z_values = (-2.0, -1.0, 0.0, 1.0, 2.0)
cylinder_vertices = [
    (
        math.cos(math.tau * segment / segments),
        math.sin(math.tau * segment / segments),
        z,
    )
    for z in z_values
    for segment in range(segments)
]
cylinder_faces = []
for band in range(len(z_values) - 1):
    for segment in range(segments):
        if band in {1, 2} and segment in {0, 1}:
            continue
        following = (segment + 1) % segments
        lower = band * segments
        upper = (band + 1) * segments
        cylinder_faces.append(
            (lower + segment, lower + following, upper + following, upper + segment)
        )
cylinder_object, cylinder_bm = edit_mesh(
    "cylinder_hole_fairing", cylinder_vertices, cylinder_faces
)
cylinder_components = []
remaining = {edge for edge in cylinder_bm.edges if len(edge.link_faces) == 1}
while remaining:
    seed = next(iter(remaining))
    component = boundary_component(seed)
    cylinder_components.append(component)
    remaining.difference_update(component)
cylinder_component = min(
    (component for component in cylinder_components if len(component) > 4),
    key=lambda component: len(component),
)
cylinder_loop, cylinder_edges = ordered_boundary_loop(edges=cylinder_component)
cylinder_loop_count = len(cylinder_loop)
cylinder_bm, cylinder_loop, cylinder_edges = (
    _limited_dissolve_analytic_boundary(
        cylinder_object,
        cylinder_bm,
        cylinder_loop,
        cylinder_edges,
    )
)
assert len(cylinder_loop) < cylinder_loop_count
cylinder_adapter = SimpleNamespace(_object=cylinder_object, diameter_estimator='AVERAGE')
cylinder_adapter._world_normal = lambda face: GeometryMixin._world_normal(
    cylinder_adapter, face
)
cylinder_hypothesis = choose_hypothesis(
    cylinder_loop,
    cylinder_edges,
    cylinder_object.matrix_world,
    lambda faces: GeometryMixin._fit_cylinder(cylinder_adapter, faces),
    cylinder_adapter._world_normal,
)
assert cylinder_hypothesis["method"] == "CONIC", cylinder_hypothesis
assert abs(cylinder_hypothesis["fit"]["axis"].z) > 0.9, cylinder_hypothesis
cylinder_result = fair_boundary_loop(
    cylinder_bm,
    cylinder_loop,
    cylinder_edges,
    cylinder_object.matrix_world,
    cylinder_hypothesis,
)
assert 0 < len(cylinder_result["verts"]) < len(cylinder_loop), (
    len(cylinder_result["verts"]),
    len(cylinder_loop),
    [tuple(vert.co) for vert in cylinder_result["verts"]],
    len(cylinder_result["faces"]),
)
assert len(cylinder_result["faces"]) >= 2
axis = cylinder_hypothesis["fit"]["axis"]
axial_coordinates = [
    (cylinder_object.matrix_world @ vert.co).dot(axis)
    for vert in cylinder_result["verts"]
]
assert max(axial_coordinates) - min(axial_coordinates) < 1.0e-6
cylinder_bm, cylinder_cleanup = _limited_dissolve_cylinder_patch(
    cylinder_object,
    cylinder_bm,
    cylinder_result["faces"],
)
assert cylinder_cleanup["faces"]
cylinder_boundaries = []
cylinder_remaining = {
    edge for edge in cylinder_bm.edges if len(edge.link_faces) == 1
}
while cylinder_remaining:
    component = boundary_component(next(iter(cylinder_remaining)))
    cylinder_boundaries.append(component)
    cylinder_remaining.difference_update(component)
assert len(cylinder_boundaries) == 2, [
    len(component) for component in cylinder_boundaries
]

# Exercise the complete operator sequence for the cylinder-specific Blender
# operations: boundary dissolve, axial collapse, fixed weld, grow, dissolve.
leave_edit_mode(cylinder_object)
cylinder_operator_object, cylinder_operator_bm = edit_mesh(
    "cylinder_operator_fairing", cylinder_vertices, cylinder_faces
)
cylinder_operator_components = []
cylinder_operator_remaining = {
    edge for edge in cylinder_operator_bm.edges if len(edge.link_faces) == 1
}
while cylinder_operator_remaining:
    component = boundary_component(next(iter(cylinder_operator_remaining)))
    cylinder_operator_components.append(component)
    cylinder_operator_remaining.difference_update(component)
cylinder_operator_component = min(
    (component for component in cylinder_operator_components if len(component) > 4),
    key=len,
)
cylinder_operator = SimpleNamespace(
    edge_indices=",".join(
        str(edge.index) for edge in sorted(
            cylinder_operator_component, key=lambda edge: edge.index
        )
    ),
    fairing_method='AUTO',
    fill_style='AUTO',
    final_simplify=True,
    result_method="",
    result_fill_style="",
    result_residual=0.0,
    diameter_estimator='AVERAGE',
    _object=None,
    report=lambda *_args: None,
)
cylinder_operator._world_normal = lambda face: GeometryMixin._world_normal(
    cylinder_operator, face
)
assert MESH_OT_holey_tools_fair_hole.execute(
    cylinder_operator, bpy.context
) == {'FINISHED'}
assert cylinder_operator.result_method == "CONIC"
assert cylinder_operator.result_fill_style == "CYLINDER"
cylinder_operator_bm = bmesh.from_edit_mesh(cylinder_operator_object.data)
cylinder_operator_remaining = {
    edge for edge in cylinder_operator_bm.edges if len(edge.link_faces) == 1
}
cylinder_operator_boundaries = []
while cylinder_operator_remaining:
    component = boundary_component(next(iter(cylinder_operator_remaining)))
    cylinder_operator_boundaries.append(component)
    cylinder_operator_remaining.difference_update(component)
assert len(cylinder_operator_boundaries) == 2


# Fit a sphere from only the faces touching a missing curved grid patch.
leave_edit_mode(cylinder_operator_object)
angular = [-0.6, -0.3, 0.0, 0.3, 0.6]
sphere_coordinates = [
    [
        (
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        )
        for longitude in angular
    ]
    for latitude in angular
]
sphere_object, sphere_bm = grid_mesh(
    "sphere_hole_fairing",
    sphere_coordinates,
    {(1, 1), (1, 2), (2, 1), (2, 2)},
)
sphere_component = inner_boundary_seed(sphere_bm)
sphere_loop, sphere_edges = ordered_boundary_loop(edges=sphere_component)
sphere_loop_count = len(sphere_loop)
sphere_bm, sphere_loop, sphere_edges = _limited_dissolve_analytic_boundary(
    sphere_object, sphere_bm, sphere_loop, sphere_edges
)
assert len(sphere_loop) <= sphere_loop_count
sphere_adapter = SimpleNamespace(_object=sphere_object)
sphere_adapter._world_normal = lambda face: GeometryMixin._world_normal(
    sphere_adapter, face
)
sphere_fit = fit_sphere_hypothesis(
    sphere_edges, sphere_object.matrix_world, sphere_adapter._world_normal
)
assert sphere_fit is not None
assert abs(sphere_fit["radius"] - 1.0) < 0.08, sphere_fit

# A passing conic fit must not short-circuit the sphere. Both are compared and
# the smaller normalized residual wins.
fake_conic_residual = (
    sphere_fit["residual"] + RELATIVE_RESIDUAL_TOLERANCE
) * 0.5
competing_hypothesis = choose_hypothesis(
    sphere_loop,
    sphere_edges,
    sphere_object.matrix_world,
    lambda _faces: {
        "fit_rank": 2,
        "fit_relative_error": fake_conic_residual,
    },
    sphere_adapter._world_normal,
)
assert competing_hypothesis["method"] == "SPHERE", competing_hypothesis

# Redo overrides bypass only the automatic residual preference, not the
# numerical rank checks. A forced cylinder also removes any fitted taper.
forced_cylinder = choose_hypothesis(
    sphere_loop,
    sphere_edges,
    sphere_object.matrix_world,
    lambda _faces: {
        "fit_rank": 2,
        "fit_relative_error": 1.0,
        "axis": Vector((0.0, 0.0, 1.0)),
        "basis_x": Vector((1.0, 0.0, 0.0)),
        "basis_y": Vector((0.0, 1.0, 0.0)),
        "center": Vector((0.0, 0.0, 0.0)),
        "radius": 1.0,
        "slope": 0.5,
        "half_angle": math.atan(0.5),
        "is_cone": True,
    },
    sphere_adapter._world_normal,
    forced_method='CYLINDER',
)
assert forced_cylinder["method"] == "CONIC"
assert forced_cylinder["fit"]["is_cone"] is False
assert forced_cylinder["fit"]["slope"] == 0.0
assert choose_hypothesis(
    sphere_loop,
    sphere_edges,
    sphere_object.matrix_world,
    lambda _faces: forced_cylinder["fit"],
    sphere_adapter._world_normal,
    forced_method='ORGANIC',
)["method"] == "ORGANIC"

# Surface choice and topology choice are independent. A sphere can use the
# source-aligned grid while every generated point remains on the fitted sphere.
sphere_grid_result = fair_boundary_loop(
    sphere_bm,
    sphere_loop,
    sphere_edges,
    sphere_object.matrix_world,
    sphere_fit,
    'GRID',
)
assert sphere_grid_result["fill_style"] == "GRID"
assert max(
    abs((sphere_object.matrix_world @ vert.co - sphere_fit["center"]).length
        - sphere_fit["radius"])
    for vert in sphere_grid_result["verts"]
) < 1.0e-5


# The final organic path simplifies its boundary at 10 degrees, creates an
# SVD-aligned 2D grid, and solves only its fitted-axis depths.
leave_edit_mode(sphere_object)
grid_values = (-2.0, -1.0, 0.0, 1.0, 2.0)
organic_coordinates = [
    [(x, y, 0.35 * x * y) for x in grid_values]
    for y in grid_values
]
organic_object, organic_bm = grid_mesh(
    "organic_hole_fairing",
    organic_coordinates,
    {(1, 1), (1, 2), (2, 1), (2, 2)},
)
organic_component = inner_boundary_seed(organic_bm)
organic_loop, organic_edges = ordered_boundary_loop(edges=organic_component)
organic_result = fair_boundary_loop(
    organic_bm,
    organic_loop,
    organic_edges,
    Matrix.Identity(4),
    {"method": "ORGANIC", "residual": 0.0},
)
assert len(organic_result["boundary_verts"]) <= len(organic_loop)
assert len(organic_result["verts"]) >= 1
assert_closed_rim(organic_result["boundary_edges"])
assert all(math.isfinite(component) for vert in organic_result["verts"] for component in vert.co)
organic_depth_errors = [
    abs(vert.co.z - 0.35 * vert.co.x * vert.co.y)
    for vert in organic_result["verts"]
]
assert max(organic_depth_errors) < 0.08, max(organic_depth_errors)

# The same organic surface continuation can be used with concentric-ring
# topology when a clipped grid is undesirable.
leave_edit_mode(organic_object)
organic_circle_object, organic_circle_bm = grid_mesh(
    "organic_circle_hole_fairing",
    organic_coordinates,
    {(1, 1), (1, 2), (2, 1), (2, 2)},
)
organic_circle_component = inner_boundary_seed(organic_circle_bm)
organic_circle_loop, organic_circle_edges = ordered_boundary_loop(
    edges=organic_circle_component
)
organic_circle_result = fair_boundary_loop(
    organic_circle_bm,
    organic_circle_loop,
    organic_circle_edges,
    Matrix.Identity(4),
    {"method": "ORGANIC", "residual": 0.0},
    'CIRCLE',
    circle_loops=4,
)
assert organic_circle_result["fill_style"] == "CIRCLE"
assert len(organic_circle_result["verts"]) == (
    4 * len(organic_circle_result["boundary_verts"]) + 1
)


# Organic boundary cleanup removes only unsupported intermediate circle/rim
# vertices. The four source-grid spokes and every exterior edge must survive.
leave_edit_mode(organic_circle_object)
targeted_outer = (
    (-2.0, -2.0, 0.0),
    (2.0, -2.0, 0.0),
    (2.0, 2.0, 0.0),
    (-2.0, 2.0, 0.0),
)
targeted_faces = [
    (
        index,
        (index + 1) % 4,
        4 + (2 * index + 2) % 8,
        4 + 2 * index + 1,
        4 + 2 * index,
    )
    for index in range(4)
]
targeted_object, targeted_bm = edit_mesh(
    "organic_targeted_boundary_dissolve",
    targeted_outer + dissolve_inner,
    targeted_faces,
)
targeted_component = min(
    (
        boundary_component(edge)
        for edge in targeted_bm.edges
        if len(edge.link_faces) == 1
    ),
    key=lambda component: sum(
        vert.co.length for edge in component for vert in edge.verts
    ),
)
targeted_loop, targeted_edges = ordered_boundary_loop(
    edges=targeted_component
)
targeted_samples = _organic_direction_samples(
    targeted_loop, targeted_edges, Matrix.Identity(4)
)
assert len(targeted_loop) == 8
assert len(targeted_samples) == 4
targeted_loop, targeted_edges = _simplify_organic_boundary(
    targeted_bm, targeted_loop, targeted_edges
)
assert len(targeted_loop) == 4
assert len(_organic_direction_samples(
    targeted_loop, targeted_edges, Matrix.Identity(4)
)) == 4
targeted_boundary_components = []
targeted_remaining = {
    edge for edge in targeted_bm.edges if len(edge.link_faces) == 1
}
while targeted_remaining:
    component = boundary_component(next(iter(targeted_remaining)))
    targeted_boundary_components.append(component)
    targeted_remaining.difference_update(component)
assert sorted(len(component) for component in targeted_boundary_components) == [4, 4]


# A twelve-vertex annulus receives a 3-by-3 interval budget because the grid
# perimeter is sized from the boundary count. Clipping those SVD-aligned lines
# must insert both boundary splits and interior grid intersections.
leave_edit_mode(targeted_object)
net_segments = 12
net_vertices = [
    (
        radius * math.cos(math.tau * index / net_segments),
        radius * math.sin(math.tau * index / net_segments),
        0.0,
    )
    for radius in (2.0, 1.0)
    for index in range(net_segments)
]
net_faces = [
    (
        index,
        (index + 1) % net_segments,
        net_segments + (index + 1) % net_segments,
        net_segments + index,
    )
    for index in range(net_segments)
]
net_object, net_bm = edit_mesh("organic_edge_extension_net", net_vertices, net_faces)
net_component = min(
    (
        boundary_component(edge)
        for edge in net_bm.edges
        if len(edge.link_faces) == 1
    ),
    key=lambda component: sum(
        vert.co.length for edge in component for vert in edge.verts
    ),
)
net_loop, net_edges = ordered_boundary_loop(edges=net_component)
net_layout = _organic_grid_layout(net_loop, net_edges, Matrix.Identity(4))
assert sum(net_layout["grid_shape"]) == round(len(net_loop) * 0.5)
assert len(net_layout["input_coords"]) >= len(net_loop)
assert any(not source for source in net_layout["orig_verts"])
net_result = fair_boundary_loop(
    net_bm,
    net_loop,
    net_edges,
    Matrix.Identity(4),
    {"method": "ORGANIC", "residual": 0.0},
)
assert len(net_result["boundary_verts"]) >= len(net_loop)
assert len(net_result["verts"]) >= 1
assert_closed_rim(net_result["boundary_edges"])


# Exterior edges, rather than the boundary's bounding box, establish grid
# orientation. Two equally populated orthogonal direction families exercise the
# otherwise ambiguous equal-singular-value case.
leave_edit_mode(net_object)
alignment_angle = 0.37
alignment_direction = Vector((
    math.cos(alignment_angle), math.sin(alignment_angle), 0.0
))
alignment_perpendicular = Vector((
    -math.sin(alignment_angle), math.cos(alignment_angle), 0.0
))
alignment_inner = (
    (-1.0, -1.0, 0.0),
    (1.0, -1.0, 0.0),
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
)
alignment_outer = tuple(
    tuple(
        Vector(point)
        - (
            alignment_direction
            if index % 2 == 0
            else alignment_perpendicular
        ) * 1.5
    )
    for index, point in enumerate(alignment_inner)
)
alignment_faces = [
    (index, (index + 1) % 4, 4 + (index + 1) % 4, 4 + index)
    for index in range(4)
]
alignment_object, alignment_bm = edit_mesh(
    "organic_svd_grid_alignment",
    alignment_outer + alignment_inner,
    alignment_faces,
)
alignment_component = next(
    component
    for edge in alignment_bm.edges
    if len(edge.link_faces) == 1
    for component in (boundary_component(edge),)
    if all(vert.index >= 4 for edge in component for vert in edge.verts)
)
alignment_loop, alignment_edges = ordered_boundary_loop(
    edges=alignment_component
)
alignment_samples = _organic_direction_samples(
    alignment_loop, alignment_edges, Matrix.Identity(4)
)
assert len(alignment_samples) == 4
_center, _axis, grid_x, grid_y, _points, _values = _organic_grid_frame(
    alignment_loop,
    alignment_edges,
    Matrix.Identity(4),
    alignment_samples,
)
assert abs(grid_x.dot(alignment_direction)) > 0.999
grid_angle_offset = 0.23
_center, _axis, rotated_grid_x, _grid_y, _points, _values = (
    _organic_grid_frame(
        alignment_loop,
        alignment_edges,
        Matrix.Identity(4),
        alignment_samples,
        grid_angle_offset,
    )
)
expected_grid_x = (
    grid_x * math.cos(grid_angle_offset)
    + grid_y * math.sin(grid_angle_offset)
).normalized()
assert abs(rotated_grid_x.dot(expected_grid_x)) > 0.999

# A vertex-connected pair of faces produces a branched boundary flood and must
# be rejected before any topology is created.
leave_edit_mode(alignment_object)
branched_object, branched_bm = edit_mesh(
    "branched_boundary",
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
    ],
    [(0, 1, 2), (0, 3, 4)],
)
branched_component = boundary_component(branched_bm.edges[0])
assert len(branched_component) == 6
try:
    ordered_boundary_loop(edges=branched_component)
except FairingError:
    pass
else:
    raise AssertionError("A branched boundary component was accepted")

src.register()
assert hasattr(bpy.types, "MESH_OT_holey_tools_fair_hole")
src.unregister()

print("HOLEY_TOOLS_FAIR_HOLE_TESTS_OK")
