import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src
import src.fair_hole as fair_hole_module
import src.grid_patcher as grid_patcher
from src.fair_hole import MESH_OT_holey_tools_fair_hole
from src.fairing import ordered_boundary_loop
from src.grid_patcher import (_complete_triangles, _pentagon_corner, _slide_boundary,
                              _nearest_corrected, _right_angle_error, _refine_boundary,
                              _quad_rectangle_error, _t_shape_error, _turn, prepare_grid_steps,
                              _expand_triangle, _merge_collapsed_ring_edges, _tracked_loop)


def clipped_grid():
    """Eight grid cells around a hole that cuts off four source-grid corners."""
    c = 1.25
    polygons = [
        [(-2, -2), (-1, -2), (-1, -c), (-c, -1), (-2, -1)],
        [(-1, -2), (1, -2), (1, -c), (0, -c), (-1, -c)],
        [(1, -2), (2, -2), (2, -1), (c, -1), (1, -c)],
        [(c, -1), (2, -1), (2, 1), (c, 1)],
        [(c, 1), (2, 1), (2, 2), (1, 2), (1, c)],
        [(-1, c), (1, c), (1, 2), (-1, 2)],
        [(-2, 1), (-c, 1), (-1, c), (-1, 2), (-2, 2)],
        [(-2, -1), (-c, -1), (-c, 1), (-2, 1)],
    ]
    bm = bmesh.new()
    vertices = {}
    for polygon in polygons:
        for point in polygon:
            if point not in vertices:
                vertices[point] = bm.verts.new((*point, 0.))
        bm.faces.new([vertices[point] for point in polygon])
    bm.normal_update()
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    rim = [edge for edge in bm.edges if len(edge.link_faces) == 1
           and all(max(abs(vert.co.x), abs(vert.co.y)) < 1.5 for vert in edge.verts)]
    loop, edges = ordered_boundary_loop(edges=rim)
    return bm, loop, edges


def assert_clean(bm, expected_boundary):
    assert len(bm.faces) == 8  # Neither a cap nor a lost source cell.
    assert all(face.calc_area() > 1.0e-8 for face in bm.faces)
    assert all(len(edge.link_faces) in {1, 2} for edge in bm.edges)
    assert len(expected_boundary) >= 3
    assert not any(name.startswith(".holey_grid_boundary") for name in bm.verts.layers.int.keys())


bm, loop, edges = clipped_grid()
outer = {vert: vert.co.copy() for vert in bm.verts
         if max(abs(vert.co.x), abs(vert.co.y)) == 2.}
step1 = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'DISSOLVE')
assert step1['dissolved'] == 1 and step1['repaired'] == 0
assert len(step1['boundary_edges']) == 8
assert sum(len(face.verts) == 5 for face in bm.faces) == 4
assert all(len(vert.link_edges) != 2 for vert in step1['boundary_verts'])
assert all(vert.is_valid and (vert.co - co).length < 1.0e-7 for vert, co in outer.items())
assert_clean(bm, step1['boundary_edges'])
# Step 2 may be applied sequentially, or directly to the original input in redo.
step2 = prepare_grid_steps(bm, step1['boundary_verts'], step1['boundary_edges'],
                           Matrix.Identity(4), 'PENTAGONS')
assert step2['dissolved'] == 0 and step2['repaired'] == 4 and step2['skipped'] == 0
assert len(step2['boundary_edges']) == 4
assert all(len(face.verts) == 4 for face in bm.faces)
assert {tuple(vert.co) for vert in step2['boundary_verts']} == {
    (-1., -1., 0.), (-1., 1., 0.), (1., -1., 0.), (1., 1., 0.)}
assert all(vert.is_valid and (vert.co - co).length < 1.0e-7 for vert, co in outer.items())
assert_clean(bm, step2['boundary_edges'])
repeat = prepare_grid_steps(bm, step2['boundary_verts'], step2['boundary_edges'],
                            Matrix.Identity(4), 'PENTAGONS')
assert repeat['repaired'] == repeat['dissolved'] == 0
bm.free()

# The ring-face 0.001 merge must precede pentagon detection: this clipped
# corner becomes a quad by welding rather than entering the pentagon repair.
bm, loop, edges = clipped_grid()
near_corner = [vert for vert in loop if vert.co.x < 0. and vert.co.y < 0.]
assert len(near_corner) == 2
for vert in near_corner:
    if vert.co.x == -1.:
        vert.co.y = -1.0005
    else:
        vert.co.x = -1.0005
outer_positions = {vert: vert.co.copy() for vert in bm.verts if vert not in loop}
result = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'PENTAGONS')
assert result['merged'] == 1 and result['repaired'] == 3
assert len(result['boundary_edges']) == 4
assert all(len(face.verts) == 4 for face in bm.faces)
assert all(vert.is_valid and (vert.co - point).length < 1.0e-7 for vert, point in outer_positions.items())
assert_clean(bm, result['boundary_edges'])
bm.free()

# All vertices of ring faces are merged, including their outer support verts.
# Nearby vertices outside those faces must not participate in the weld.
bm = bmesh.new()
inner = [bm.verts.new(co) for co in [(-1., -1., 0.), (0., -1., 0.), (.0005, -1., 0.),
                                    (1., -1., 0.), (1., 1., 0.), (-1., 1., 0.)]]
outer = [bm.verts.new(co) for co in [(-2., -2., 0.), (0., -2., 0.), (.0005, -2., 0.),
                                    (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.)]]
for i in range(6):
    bm.faces.new([outer[i], outer[(i + 1) % 6], inner[(i + 1) % 6], inner[i]])
unrelated = bm.verts.new((0., -1., .00025))
unrelated_face_verts = [bm.verts.new(co) for co in [(0., -2., .00025),
                                                  (.0005, -2., .00025), (0., -3., .1)]]
bm.faces.new(unrelated_face_verts)
fixed = {vert: vert.co.copy() for vert in
         [outer[i] for i in (0, 3, 4, 5)] + [unrelated] + unrelated_face_verts}
outer_pair = outer[1:3]
bm.edges.index_update()
loop, edges = ordered_boundary_loop(edges=[bm.edges.get((inner[i], inner[(i + 1) % 6])) for i in range(6)])
result = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'PENTAGONS')
assert result['merged'] == 2 and len(result['boundary_edges']) == 5
assert sum(vert.is_valid for vert in outer_pair) == 1
assert all(vert.is_valid and (vert.co - point).length < 1.0e-7 for vert, point in fixed.items())
assert all(face.calc_area() > 1.0e-8 for face in bm.faces)
bm.free()

# Collect merge inputs only after Step 1 has removed its redundant rim vertex.
bm, loop, edges = clipped_grid()
redundant = next(vert for vert in loop if len(vert.link_edges) == 2)
original_bmesh = grid_patcher.bmesh
merge_calls = []


def check_merge_inputs(mesh, *, verts, dist):
    assert not redundant.is_valid and redundant not in verts
    assert set(verts) == set(mesh.verts)
    assert dist == .001
    merge_calls.append(1)
    return bmesh.ops.find_doubles(mesh, verts=verts, dist=dist)


grid_patcher.bmesh = SimpleNamespace(ops=SimpleNamespace(
    dissolve_verts=bmesh.ops.dissolve_verts, find_doubles=check_merge_inputs,
    weld_verts=bmesh.ops.weld_verts,
))
try:
    result = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'PENTAGONS')
finally:
    grid_patcher.bmesh = original_bmesh
assert merge_calls == [1] and result['dissolved'] == 1
assert result['merged'] == 0 and result['repaired'] == 4
bm.free()

# Track the surviving support vertex when a boundary vertex welds into it.
bm = bmesh.new()
inner = [bm.verts.new(co) for co in [(-1., -1., 0.), (1., -1., 0.),
                                    (1., 1., 0.), (-1., 1., 0.)]]
outer = [bm.verts.new(co) for co in [(-1.0005, -1., 0.), (2., -2., 0.),
                                    (2., 2., 0.), (-2., 2., 0.)]]
for i in range(4):
    bm.faces.new([outer[i], outer[(i + 1) % 4], inner[(i + 1) % 4], inner[i]])
bm.edges.index_update()
loop, edges = ordered_boundary_loop(edges=[bm.edges.get((inner[i], inner[(i + 1) % 4])) for i in range(4)])


def keep_support(mesh, *, verts, dist):
    return bmesh.ops.find_doubles(mesh, verts=verts, dist=dist, keep_verts=[outer[0]])


grid_patcher.bmesh = SimpleNamespace(ops=SimpleNamespace(
    find_doubles=keep_support, weld_verts=bmesh.ops.weld_verts,
))
try:
    result = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'PENTAGONS')
finally:
    grid_patcher.bmesh = original_bmesh
assert result['merged'] == 1 and len(result['boundary_edges']) == 4
assert not inner[0].is_valid and outer[0] in result['boundary_verts']
assert all(face.calc_area() > 1.0e-8 for face in bm.faces)
bm.free()

# World transforms don't move the intersection away from its source-grid corner.
bm, loop, edges = clipped_grid()
transform = (Matrix.Translation((5., -7., 3.))
             @ Matrix.Rotation(.7, 4, 'X')
             @ Matrix.Diagonal((2., .5, 1., 1.)))
result = prepare_grid_steps(bm, loop, edges, transform, 'PENTAGONS')
assert result['dissolved'] == 1 and result['repaired'] == 4
assert all(abs(abs(vert.co.x) - 1.) < 1.0e-6
           and abs(abs(vert.co.y) - 1.) < 1.0e-6 for vert in result['boundary_verts'])
assert_clean(bm, result['boundary_edges'])
bm.free()

# The two skipped clipped corners from fairing test.blend's Sphere have tiny
# non-coplanarity beyond the old 1e-5 relative gap gate. Preserve the remaining
# orientation/convexity checks while accepting their closest-approach midpoint.
sphere_corners = [
    [(-.5404460430145264, 3.982922077178955, .41216278076171875),
     (-.5022290945053101, 3.954061985015869, .4229167699813843),
     (-.49071717262268066, 3.9679253101348877, .4229167699813843),
     (-.5088152885437012, 3.984423875808716, .41741740703582764),
     (-.5374245643615723, 3.986560821533203, .4121628403663635)],
    [(-.9689568281173706, 4.5192484855651855, -.17518502473831177),
     (-.9555561542510986, 4.504549026489258, -.13031339645385742),
     (-.9348742961883545, 4.521722793579102, -.13031336665153503),
     (-.9025911092758179, 4.572819709777832, -.1725125014781952),
     (-.9033050537109375, 4.573765277862549, -.17518502473831177)],
]
for corner_index, coordinates in enumerate(sphere_corners):
    for recenter in (False, True):
        bm = bmesh.new()
        vertices = [bm.verts.new(co) for co in coordinates]
        face = bm.faces.new(vertices)
        pair = (3, 4) if corner_index == 0 else (2, 3)
        edge = bm.edges.get(tuple(vertices[i] for i in pair))
        transform = Matrix.Translation(-Vector(coordinates[0])) if recenter else Matrix.Identity(4)
        before = {vert: vert.co.copy() for vert in vertices}
        corner = _pentagon_corner(face, edge, transform)
        assert corner is not None, (corner_index, recenter)
        first, second, point = corner
        assert all(vert.co == co for vert, co in before.items())
        first.co = second.co = point
        bmesh.ops.weld_verts(bm, targetmap={second: first})
        assert len(bm.faces) == 1 and len(next(iter(bm.faces)).verts) == 4
        assert next(iter(bm.faces)).calc_area() > 1.0e-7
        bm.free()

# Float32 world-transform uncertainty can exceed the local geometric floor.
# Moving the object far away must still never accept appreciably skew lines:
# its precision allowance is capped at 0.1% of the local face extent.
bm, loop, edges = clipped_grid()
edge = next(edge for edge in edges if len(edge.link_faces[0].verts) == 5
            and all(vert.co.x < 0 and vert.co.y < 0 for vert in edge.verts))
face = edge.link_faces[0]
edge.verts[0].co.z = .0001
assert _pentagon_corner(face, edge, Matrix.Identity(4)) is None
assert _pentagon_corner(face, edge, Matrix.Translation((500., 0., 0.))) is not None
edge.verts[0].co.z = .1
assert _pentagon_corner(face, edge, Matrix.Translation((1000000., 0., 0.))) is None
bm.free()

# Unsafe hypotheses are skipped, not merged at a made-up midpoint.
bm, loop, edges = clipped_grid()
edge = next(edge for edge in edges if len(edge.link_faces[0].verts) == 5
            and all(vert.co.x < 0 and vert.co.y < 0 for vert in edge.verts))
face = edge.link_faces[0]
first = edge.verts[0]
first.co.z = .1
assert _pentagon_corner(face, edge, Matrix.Identity(4)) is None
bm.free()

# A valid intersection must still be rejected if welding its endpoint across
# another incident triangle would invert that triangle.
bm, loop, edges = clipped_grid()
edge = next(edge for edge in edges if len(edge.link_faces[0].verts) == 5
            and all(vert.co.x < 0 and vert.co.y < 0 for vert in edge.verts))
face = edge.link_faces[0]
first = min(edge.verts, key=lambda vert: vert.co.y)
bm.faces.new([first, bm.verts.new((-.5, -1.125, 0.)), bm.verts.new((.5, -1.125, 0.))])
assert _pentagon_corner(face, edge, Matrix.Identity(4)) is None
bm.free()

# Parallel source sides do not define a recoverable rectangular corner.
bm = bmesh.new()
parallel = [bm.verts.new(co) for co in [(-1., 0., 0.), (1., 0., 0.),
                                      (1., 1., 0.), (0., 2., 0.), (-1., 1., 0.)]]
face = bm.faces.new(parallel)
edge = bm.edges.get((parallel[0], parallel[1]))
assert _pentagon_corner(face, edge, Matrix.Identity(4)) is None
bm.free()

# A spoke-connected notch slides along its spoke to a straight border; the
# neighboring source faces retain positive area and fixed outer vertices.
bm = bmesh.new()
inner = [bm.verts.new(co) for co in [(-1., -1., 0.), (0., -1.3, 0.),
                                    (1., -1., 0.), (1., 1., 0.), (-1., 1., 0.)]]
outer = [bm.verts.new(co) for co in [(-2., -2., 0.), (0., -2., 0.),
                                    (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.)]]
for i in range(5):
    bm.faces.new([outer[i], outer[(i + 1) % 5], inner[(i + 1) % 5], inner[i]])
# Additional non-boundary edges make only the notch eligible in this fixture.
for i in (0, 2, 3, 4):
    bm.edges.new((inner[i], bm.verts.new(inner[i].co + (outer[i].co - inner[i].co) * .5)))
bm.normal_update()
bm.edges.index_update()
loop, edges = ordered_boundary_loop(edges=[bm.edges.get((inner[i], inner[(i + 1) % 5])) for i in range(5)])
outer_positions = [vert.co.copy() for vert in outer]
slid, skipped = _slide_boundary(bm, loop, edges, Matrix.Identity(4))
assert slid == 1 and skipped == 0
assert abs(inner[1].co.x) < 1.0e-6 and abs(inner[1].co.y + 1.) < 1.0e-5
assert _turn(inner[0].co, inner[1].co, inner[2].co) < 1.0e-5
assert all((vert.co - original).length < 1.0e-7 for vert, original in zip(outer, outer_positions))
assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
bm.free()


def unfinished_side(kind='TRIANGLE', chain=False):
    """A corrected left corner, quad slides, and an unfinished right region."""
    points = [(-1., -1.), (0., -1.3)]
    if chain:
        points.append((.5, -1.4))
    points += [(1., -1.45), (1., 1.), (-1., 1.)]
    outer_points = [(-2., -2.), (0., -2.)]
    if chain:
        outer_points.append((.5, -2.))
    outer_points += [(1., -2.), (2., 2.), (-2., 2.)]
    mesh = bmesh.new()
    inner = [mesh.verts.new((*point, 0.)) for point in points]
    outer = [mesh.verts.new((*point, 0.)) for point in outer_points]
    bad = len(points) - 3
    for i in range(len(points)):
        vertices = [outer[i], outer[(i + 1) % len(points)],
                    inner[(i + 1) % len(points)], inner[i]]
        if i == bad and kind == 'TRIANGLE':
            mesh.faces.new([vertices[0], vertices[1], vertices[3]])
            mesh.faces.new([vertices[1], vertices[2], vertices[3]])
        else:
            if i == bad and kind == 'NGON':
                vertices.insert(1, mesh.verts.new(outer[i].co.lerp(outer[i + 1].co, .5)))
            mesh.faces.new(vertices)
    # Only the vertices between the corrected corner and bad region can slide.
    for i in (0, bad, bad + 1, bad + 2):
        mesh.edges.new((inner[i], mesh.verts.new(inner[i].co.lerp(outer[i].co, .5))))
    mesh.normal_update()
    mesh.edges.index_update()
    loop, edges = ordered_boundary_loop(edges=[mesh.edges.get((inner[i], inner[(i + 1) % len(points)]))
                                              for i in range(len(points))])
    return mesh, loop, edges, inner, outer


# The closer right neighbor is triangular or an unrepaired ngon. Its direction
# must not pull the quad slide toward a straight line through both neighbors.
for kind in ('TRIANGLE', 'NGON'):
    bm, loop, edges, inner, outer = unfinished_side(kind)
    positions = {vert: vert.co.copy() for vert in bm.verts if vert is not inner[1]}
    slid, skipped = _slide_boundary(bm, loop, edges, Matrix.Identity(4))
    assert slid == 1 and skipped == 0, (kind, slid, skipped)
    assert abs(inner[1].co.y + 1.) < 1.0e-5, (kind, tuple(inner[1].co))
    assert _right_angle_error(outer[1].co, inner[1].co, inner[0].co) < 1.0e-5
    assert _turn(inner[0].co, inner[1].co, inner[2].co) > .1
    assert all((vert.co - point).length < 1.0e-7 for vert, point in positions.items())
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    bm.free()

# Reach the nearest corrected corner through unfinished quad vertices, then
# propagate successful right-angle corrections instead of using them as noise.
bm, loop, edges, inner, outer = unfinished_side(chain=True)
reference = _nearest_corrected(bm, loop, loop.index(inner[2]), {inner[0]}, Matrix.Identity(4))
assert reference is not None and reference[1] is inner[0]
slid, skipped = _slide_boundary(bm, loop, edges, Matrix.Identity(4))
assert slid == 2 and skipped == 0, (slid, skipped)
assert all(abs(vert.co.y + 1.) < 1.0e-5 for vert in inner[1:3])
assert _right_angle_error(outer[2].co, inner[2].co, inner[1].co) < 1.0e-5
assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
bm.free()

# Without a corrected corner or fixed quad junction, don't guess using the
# other unfinished side of a valence-three vertex.
bm, loop, edges, inner, outer = unfinished_side('QUAD')
for vert in list(bm.verts):
    if not vert.link_faces:
        bmesh.ops.delete(bm, geom=[vert], context='VERTS')
positions = {vert: vert.co.copy() for vert in bm.verts}
slid, skipped = _slide_boundary(bm, loop, edges, Matrix.Identity(4))
assert slid == 0 and skipped > 0
assert all((vert.co - point).length < 1.0e-7 for vert, point in positions.items())
bm.free()


def triangle_band(bowtie=False, crease=0.):
    if bowtie:
        polygons = [
            [(-1, -1), (0, 0), (-1, 1)],
            [(1, -1), (1, 1), (0, 0)],
            [(-1, -1), (0, -2), (1, -1), (0, 0)],
            [(-1, -1), (-1, 1), (-2, 3), (-3, 4)],
            [(1, -1), (3, 4), (2, 3), (1, 1)],
            [(3, 4), (-3, 4), (-2, 3), (2, 3)],
        ]
        hole = [(-1, 1), (0, 0), (1, 1), (2, 3), (-2, 3)]
    else:
        hole = [(-1, -1), (1, -1), (2, -2), (3, 3), (-3, 3), (-2, -2)]
        outer = [(0, -2), (0, -2), (1, -3), (5, 5), (-5, 5), (-1, -3)]
        polygons = [[outer[i], outer[(i + 1) % 6], hole[(i + 1) % 6], hole[i]]
                    for i in range(6)]
        polygons[0] = [(0, -2), (1, -1), (-1, -1)]
    mesh = bmesh.new()
    uv = mesh.loops.layers.uv.new("UVMap")
    tag = mesh.faces.layers.int.new("source_tag")
    verts = {}
    for polygon in polygons:
        for x, y in polygon:
            if (x, y) not in verts:
                verts[(x, y)] = mesh.verts.new((x, y, -abs(x) * crease))
        face = mesh.faces.new([verts[point] for point in polygon])
        face.material_index = 2
        face.smooth = True
        face[tag] = 37
        for loop in face.loops:
            loop[uv].uv = loop.vert.co.xy
            loop[uv].pin_uv = True
    mesh.normal_update()
    mesh.edges.index_update()
    loop, edges = ordered_boundary_loop(edges=[mesh.edges.get((verts[point], verts[hole[(i + 1) % len(hole)]]))
                                               for i, point in enumerate(hole)])
    return mesh, loop, edges, verts


# A legacy near-zero bowtie spoke disguises both source triangles as quads.
# Reclassify by a connected-edge weld, not saved IDs or a broad spatial merge.
for lock_new in (False, True):
    bm, loop, edges, verts = triangle_band(bowtie=True)
    original = verts[(0, 0)]
    source_positions = {v: v.co.copy() for v in bm.verts}
    remote = bm.verts.new((0., 2e-6, 0.))  # Nearby, but not connected to the ring.
    new = bm.verts.new((0., 1e-6, 0.))
    for edge in [e for e in edges if original in e.verts]:
        _expand_triangle(bm, edge.link_faces[0], edge, new)
    boundary = (set(loop) - {original}) | {new}
    loop, edges = _tracked_loop(bm, boundary)
    assert all(len(f.verts) == 4 for f in bm.faces)
    repaired = {new} if lock_new else {original}
    fixed = new if lock_new else original
    fixed_position = fixed.co.copy()
    loop, edges, merged = _merge_collapsed_ring_edges(bm, loop, edges, repaired)
    assert merged == 1 and fixed.is_valid and fixed.co == fixed_position
    assert repaired == {fixed} and fixed in loop
    assert sum(len(f.verts) == 3 for f in bm.faces) == 2
    assert remote.is_valid and remote.co == Vector((0., 2e-6, 0.))
    assert all(v.co == p for v, p in source_positions.items() if v.is_valid)
    assert all(len(e.link_faces) in {1, 2} for e in bm.edges)
    assert all(f.calc_area() > 1e-7 for f in bm.faces)
    uv = bm.loops.layers.uv.get('UVMap')
    assert all((corner[uv].uv - corner.vert.co.xy).length < 2e-6
               for face in bm.faces for corner in face.loops)
    again_loop, again_edges, again_count = _merge_collapsed_ring_edges(bm, loop, edges, repaired)
    assert again_count == 0 and set(again_loop) == set(loop)
    loop, edges, bowties, mirrored, skipped = _complete_triangles(bm, loop, edges, Matrix.Identity(4))
    assert (bowties, mirrored, skipped) == (1, 0, 0)
    assert all(len(f.verts) == 4 for f in bm.faces)
    bm.free()

# A nearly straight boundary is not intrinsically disqualified as a bowtie:
# retain genuine crease completions when their new common edge is substantial.
bm, loop, edges, verts = triangle_band(bowtie=True, crease=.25)
verts[(0, 0)].co.y = .75
bm.normal_update()
loop, edges, bowties, mirrored, skipped = _complete_triangles(bm, loop, edges, Matrix.Identity(4))
assert (bowties, mirrored, skipped) == (1, 0, 0)
assert all(e.calc_length() > .001 for e in bm.edges)
bm.free()


# A triangle-adjacent T point targets its own and both neighboring quad angles,
# not a straight border or the triangle-side branch. Even an already-right
# moved corner must improve its neighboring angles. Reverse the walk and rotate
# the object without changing the rectangular target or moving fixed corners.
for reverse, already_right in ((False, False), (True, False), (False, True), (True, True)):
    bm, loop, edges, verts = triangle_band()
    target = verts[(-1, -1)]
    support = verts[(0, -2)]
    reference = verts[(-2, -2)]
    verts[(1, -1)].co.y = -1.2 if reverse else -.8
    if already_right:
        center = (support.co + reference.co) * .5
        radial = Vector((.3, .9, 0.)).normalized()
        target.co = center + radial * (reference.co - support.co).length * .5
        assert _right_angle_error(support.co, target.co, reference.co) < 1.0e-5
    else:
        target.co += Vector((.25, -.2, 0.))
    transform = (Matrix.Translation((3., -2., 5.)) @ Matrix.Rotation(.7, 4, 'X')
                 if reverse else Matrix.Identity(4))
    if reverse:
        loop.reverse()
    before = {vert: vert.co.copy() for vert in bm.verts if vert is not target}
    position = target.co.copy()
    axis = position - support.co
    expected = Vector((-1., -1., 0.))
    quad = next(face for face in target.link_faces if len(face.verts) == 4)
    initial_error = _quad_rectangle_error(quad, target, transform @ position, transform)
    assert initial_error > .01
    locked = set(loop) - {target}
    counts = _refine_boundary(bm, loop, edges, transform, locked, {})
    assert len(counts) == 3 and counts[0] == 1
    assert (target.co - expected).length < 1.0e-5, tuple(target.co)
    assert _right_angle_error(transform @ support.co, transform @ target.co,
                              transform @ reference.co) < 1.0e-5
    assert _quad_rectangle_error(quad, target, transform @ target.co, transform) < 1.0e-10
    # Explicitly check both neighboring corners, not just the moved angle.
    ring = list(quad.verts)
    corner = ring.index(target)
    for j in ((corner - 1) % 4, (corner + 1) % 4):
        assert _right_angle_error(transform @ ring[j - 1].co, transform @ ring[j].co,
                                  transform @ ring[(j + 1) % 4].co) < 1.0e-5
    assert (target.co - position).cross(axis).length > .01  # Both planar axes may move.
    assert abs(target.co.z) < 1.0e-6
    i = loop.index(target)
    assert _t_shape_error(loop[i - 1].co, target.co, loop[(i + 1) % len(loop)].co,
                          support.co) > .1  # The ignored T terms are still nonzero.
    assert _refine_boundary(bm, loop, edges, transform, locked, {}) == [0, 0, 0]
    assert all((vert.co - point).length < 1.0e-7 for vert, point in before.items())
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    bm.free()

# Two triangle sides have no unambiguous completed-side angle. Don't silently
# fall back to full-T straightening or pick a side based on edge ordering.
bm, loop, edges, verts = triangle_band()
target = verts[(-1, -1)]
quad = next(face for face in target.link_faces if len(face.verts) == 4)
ring = list(quad.verts)
i = ring.index(target)
bm.faces.new([ring[i - 1], target, ring[(i + 1) % 4]])
bm.faces.new([ring[i - 1], ring[(i + 1) % 4], ring[(i + 2) % 4]])
bm.faces.remove(quad)
assert len(target.link_edges) == 3 and all(len(face.verts) == 3 for face in target.link_faces)
before = {vert: vert.co.copy() for vert in bm.verts}
assert _refine_boundary(bm, loop, edges, Matrix.Identity(4), set(loop) - {target}, {}) == [0, 0, 0]
assert all((vert.co - point).length == 0. for vert, point in before.items())
bm.free()


# A boundary walk can encounter triangle vertices before quad-only vertices.
# All quad-only slide attempts must nevertheless finish before the first
# triangle-touching attempt, including vertices shared by a quad and triangle.
bm, loop, edges, _verts = triangle_band(bowtie=True)
triangle_index = next(i for i, vert in enumerate(loop)
                      if any(len(face.verts) == 3 for face in vert.link_faces))
loop = loop[triangle_index:] + loop[:triangle_index]
edges = edges[triangle_index:] + edges[:triangle_index]
attempts = []
original_safe_move = grid_patcher._safe_vertex_move


def record_slide(vert, point, matrix_world):
    attempts.append(any(len(face.verts) == 3 for face in vert.link_faces))
    return original_safe_move(vert, point, matrix_world)


grid_patcher._safe_vertex_move = record_slide
try:
    _slide_boundary(bm, loop, edges, Matrix.Identity(4), {_verts[(2, 3)]})
finally:
    grid_patcher._safe_vertex_move = original_safe_move
assert False in attempts and True in attempts
assert attempts == sorted(attempts), attempts
assert sum(len(face.verts) == 3 for face in bm.faces) == 2
bm.free()


# Bowties complete both triangles using a single new shared vertex. On a
# crease, the new connecting edge belongs to both original triangle planes.
# The new middle corner minimizes the turn toward its two rim neighbors;
# the old parallelogram target would unnecessarily kink this boundary.
for crease in (0., .25):
    bm, loop, edges, verts = triangle_band(bowtie=True, crease=crease)
    triangles = [face for face in bm.faces if len(face.verts) == 3]
    planes = [(face.verts[0].co.copy(), face.normal.copy()) for face in triangles]
    old_positions = {vert: vert.co.copy() for vert in bm.verts}
    bowtie_vertices = set()
    loop, edges, bowties, mirrored, skipped = _complete_triangles(
        bm, loop, edges, Matrix.Identity(4), bowtie_vertices=bowtie_vertices,
    )
    assert (bowties, mirrored, skipped) == (1, 0, 0), (bowties, mirrored, skipped)
    assert len(bm.faces) == 6 and all(len(face.verts) == 4 for face in bm.faces)
    new = next(vert for vert in bm.verts if vert not in old_positions)
    assert (new.co - Vector((0., 1., 0.))).length < 1.0e-6
    assert bowtie_vertices == {new}
    a, b = verts[(-1, 1)], verts[(1, 1)]
    error = _turn(a.co, new.co, b.co)
    assert error < _turn(a.co, Vector((0., 2., 0.)), b.co) - .1
    if not crease:
        assert error < 1.0e-7
    assert all(abs((new.co - point).dot(normal)) < 1.0e-6 for point, normal in planes)
    assert verts[(0, 0)] not in loop and new in loop
    assert len(bm.edges.get((verts[(0, 0)], new)).link_faces) == 2
    assert all((vert.co - point).length < 1.0e-7 for vert, point in old_positions.items())
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    assert all(len(edge.link_faces) in {1, 2} for edge in bm.edges)
    assert all(face.material_index == 2 and face.smooth for face in bm.faces)
    uv = bm.loops.layers.uv.get("UVMap")
    tag = bm.faces.layers.int.get("source_tag")
    assert all(face[tag] == 37 for face in bm.faces)
    assert all((loop[uv].uv - loop.vert.co.xy).length < 1.0e-6 for face in bm.faces for loop in face.loops)
    # Keep the same bowtie-specific target during post-mirror refinement.
    counts = _refine_boundary(bm, loop, edges, Matrix.Identity(4), set(loop) - {new}, {},
                              bowtie_vertices=bowtie_vertices)
    assert counts == [0, 0, 0] and _turn(a.co, new.co, b.co) <= error + 1.0e-7
    bm.free()

# A skew/asymmetric bowtie's minimum need not be the projected chord midpoint.
# Search along the actual plane intersection, invariant to rim walk/object pose.
for reverse in (False, True):
    bm, loop, edges, verts = triangle_band(bowtie=True, crease=.25)
    a, b = verts[(-1, 1)], verts[(1, 1)]
    b.co = Vector((.6, 1.5, -.15))
    bm.normal_update()
    planes = [(face.verts[0].co.copy(), face.normal.copy())
              for face in bm.faces if len(face.verts) == 3]
    transform = Matrix.Translation((3., -2., 5.)) @ Matrix.Rotation(.7, 4, 'X')
    if reverse:
        loop.reverse()
    before = {vert: vert.co.copy() for vert in bm.verts}
    bowtie_vertices = set()
    loop, edges, bowties, mirrored, skipped = _complete_triangles(
        bm, loop, edges, transform, bowtie_vertices=bowtie_vertices,
    )
    assert (bowties, mirrored, skipped) == (1, 0, 0)
    new = next(vert for vert in bm.verts if vert not in before)
    error = _turn(a.co, new.co, b.co)
    assert error < _turn(a.co, Vector((0., 1.25, 0.)), b.co) - 1.0e-4
    sampled_minimum = min(_turn(a.co, Vector((0., j / 1000., 0.)), b.co)
                          for j in range(100, 3001))
    assert error <= sampled_minimum + 1.0e-5, (tuple(new.co), error, sampled_minimum)
    assert all(abs((new.co - point).dot(normal)) < 1.0e-6 for point, normal in planes)
    assert all((vert.co - point).length < 1.0e-7 for vert, point in before.items())
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    assert all(len(edge.link_faces) in {1, 2} for edge in bm.edges)
    new.co.y += .1
    disturbed = _turn(a.co, new.co, b.co)
    counts = _refine_boundary(bm, loop, edges, transform, set(loop) - {new}, {},
                              bowtie_vertices=bowtie_vertices)
    assert len(counts) == 3 and counts[0] == 1
    assert _turn(a.co, new.co, b.co) < disturbed - 1.0e-4
    assert _turn(a.co, new.co, b.co) <= sampled_minimum + 1.0e-5
    assert all(abs((new.co - point).dot(normal)) < 1.0e-6 for point, normal in planes)
    assert all((vert.co - point).length < 1.0e-7 for vert, point in before.items())
    bm.free()

# An isolated boundary triangle completes by a half-turn of its opposite
# vertex about the face normal through the exposed edge midpoint.
bm, loop, edges, verts = triangle_band()
old_positions = {vert: vert.co.copy() for vert in bm.verts}
loop, edges, bowties, mirrored, skipped = _complete_triangles(bm, loop, edges, Matrix.Identity(4))
assert (bowties, mirrored, skipped) == (0, 1, 0), (bowties, mirrored, skipped)
assert len(bm.faces) == 6 and all(len(face.verts) == 4 for face in bm.faces)
new = next(vert for vert in bm.verts if vert not in old_positions)
assert (new.co - Vector((0., 0., 0.))).length < 1.0e-6
assert bm.edges.get((verts[(-1, -1)], verts[(1, -1)])) is None
assert all((vert.co - point).length < 1.0e-7 for vert, point in old_positions.items())
assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
assert all(len(edge.link_faces) in {1, 2} for edge in bm.edges)
bm.free()

# Asymmetric triangles distinguish a half-turn from an edge reflection. The
# new corner is A + B - C in world space, also on a rotated/scaled object.
for transform in (Matrix.Identity(4), Matrix.Translation((2., -3., 4.))
                  @ Matrix.Rotation(.7, 4, 'Y') @ Matrix.Diagonal((2., 1., .5, 1.))):
    bm, loop, edges, verts = triangle_band()
    verts[(0, -2)].co.x = .25
    bm.normal_update()
    before = {vert: vert.co.copy() for vert in bm.verts}
    a, b, c = (transform @ verts[key].co for key in ((-1, -1), (1, -1), (0, -2)))
    normal = (b - a).cross(c - a).normalized()
    loop, edges, bowties, mirrored, skipped = _complete_triangles(bm, loop, edges, transform)
    assert (bowties, mirrored, skipped) == (0, 1, 0)
    new = next(vert for vert in bm.verts if vert not in before)
    assert (transform @ new.co - (a + b - c)).length < 1.0e-6
    assert abs((transform @ new.co - a).dot(normal)) < 1.0e-6
    assert abs(new.co.x + .25) < 1.0e-6
    assert all((vert.co - co).length == 0. for vert, co in before.items())
    assert len(bm.faces) == 6
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    assert all(len(edge.link_faces) in {1, 2} for edge in bm.edges)
    assert all(len(edge.link_faces) == 1 for edge in edges)
    bm.free()

# Edge-adjacent triangles are both completed, regardless of traversal order.
# Their common edge and all old vertices remain intact; each exposed edge is
# replaced by two sides of the new quad, without overlapping the other repair.
for reverse in (False, True):
    bm = bmesh.new()
    hole = [(-1, 0), (0, -1), (1, 0), (2, 0), (2, 3), (-2, 3), (-2, 0)]
    outer = [(0, -2), (0, -2), (0, -2), (3, -1), (3, 4), (-3, 4), (-3, -1)]
    verts = {point: bm.verts.new((*point, 0.)) for point in set(hole + outer)}
    for i in range(len(hole)):
        ring = [outer[i], outer[(i + 1) % len(hole)], hole[(i + 1) % len(hole)], hole[i]]
        if ring[0] == ring[1]:
            ring.pop(0)
        bm.faces.new([verts[point] for point in ring])
    bm.normal_update()
    border = [bm.edges.get((verts[p], verts[hole[(i + 1) % len(hole)]])) for i, p in enumerate(hole)]
    loop, edges = ordered_boundary_loop(edges=border)
    if reverse:
        loop.reverse()
        edges.reverse()
    common = bm.edges.get((verts[(0, -2)], verts[(0, -1)]))
    assert all(len(face.verts) == 3 for face in common.link_faces)
    before = {vert: vert.co.copy() for vert in bm.verts}
    loop, edges, bowties, mirrored, skipped = _complete_triangles(bm, loop, edges, Matrix.Identity(4))
    assert (bowties, mirrored, skipped) == (0, 2, 0), (bowties, mirrored, skipped)
    assert common.is_valid and len(common.link_faces) == 2
    assert all(len(face.verts) == 4 for face in bm.faces)
    assert {tuple(v.co) for v in bm.verts if v not in before} == {(-1., 1., 0.), (1., 1., 0.)}
    assert all((vert.co - co).length == 0. for vert, co in before.items())
    assert all(not edge.is_valid for edge in border[:2])
    assert len(bm.faces) == 7
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    assert all(len(edge.link_faces) in {1, 2} for edge in bm.edges)
    assert all(len(edge.link_faces) == 1 for edge in edges)
    bm.free()

# Step 3 stops after three T-only passes: triangle topology stays available
# for inspection and later completion in Step 4.
for bowtie in (False, True):
    bm, loop, edges, _verts = triangle_band(bowtie=bowtie)
    triangle_count = sum(len(face.verts) == 3 for face in bm.faces)
    result = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'SLIDE')
    assert result['grid_step'] == 'SLIDE' and result['boundary_edges']
    assert not result['faces']
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    assert all(len(edge.link_faces) in {1, 2} for edge in bm.edges)
    assert result['bowties'] == result['mirrored'] == 0
    assert sum(len(face.verts) == 3 for face in bm.faces) == triangle_count
    assert len(result['t_refinement_counts']) == 3
    assert result['refinement_counts'] == []
    bm.free()

# Additional passes optimize the full T, including both border branches, and
# can slide back along the same spoke after Step 3's one-sided alignment.
bm, loop, edges, inner, outer = unfinished_side('NGON')
_slide_boundary(bm, loop, edges, Matrix.Identity(4))
point = inner[1].co.copy()
direction = point - outer[1].co
locked = set(loop) - {inner[1]}
fixed = {vert: vert.co.copy() for vert in bm.verts if vert is not inner[1]}
errors = [_t_shape_error(inner[0].co, inner[1].co, inner[2].co, outer[1].co)]
counts = []
for _pass in range(3):
    counts.extend(_refine_boundary(bm, loop, edges, Matrix.Identity(4), locked, {}, passes=1))
    errors.append(_t_shape_error(inner[0].co, inner[1].co, inner[2].co, outer[1].co))
assert len(counts) == 3 and counts[0] == 1
assert errors[-1] < errors[0] - 1.0e-4
assert all(after <= before + 1.0e-9 for before, after in zip(errors, errors[1:]))
assert inner[1].co.y < point.y  # Bidirectional sliding, not another outward-only pass.
assert (inner[1].co - point).cross(direction).length < 1.0e-7
assert all((vert.co - value).length < 1.0e-7 for vert, value in fixed.items())
assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
bm.free()

# A T point diagonally opposite an actual repaired pentagon corner has the
# same planar freedom. The repair remains fixed; arbitrary locks or nearby
# unrelated vertices must not accidentally enable this case.
for freed in (False, True):
    bm = bmesh.new()
    inner = [bm.verts.new(co) for co in [(-1., -1., 0.), (.3, -1.3, 0.),
                                        (1., -1., 0.), (1., 1., 0.), (-1., 1., 0.)]]
    outer = [bm.verts.new(co) for co in [(-1., -2., 0.), (0., -2., 0.),
                                        (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.)]]
    faces = [bm.faces.new([outer[i], outer[(i + 1) % 5], inner[(i + 1) % 5], inner[i]])
             for i in range(5)]
    bm.edges.index_update()
    loop, edges = ordered_boundary_loop(edges=[bm.edges.get((inner[i], inner[(i + 1) % 5]))
                                              for i in range(5)])
    target, repaired = inner[1], outer[0]
    initial = target.co.copy()
    spoke = initial - outer[1].co
    fixed = {vert: vert.co.copy() for vert in bm.verts if vert is not target}
    unrelated = bm.verts.new(repaired.co)
    repairs = {repaired} if freed else {unrelated}
    transform = Matrix.Translation((3., -2., 5.)) @ Matrix.Rotation(.7, 4, 'X')
    counts = _refine_boundary(bm, loop, edges, transform, set(loop) - {target}, {},
                              repaired_vertices=repairs)
    assert len(counts) == 3 and sum(counts) > 0
    assert all((vert.co - point).length < 1.0e-7 for vert, point in fixed.items())
    if freed:
        assert (target.co - Vector((0., -1., 0.))).length < 1.0e-5, tuple(target.co)
        assert (target.co - initial).cross(spoke).length > .1
        assert _quad_rectangle_error(faces[0], target, transform @ target.co, transform) < 1.0e-8
    else:
        assert (target.co - initial).cross(spoke).length < 1.0e-6
        assert _quad_rectangle_error(faces[0], target, transform @ target.co, transform) > .001
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    bm.free()

# A mirrored triangle's new corner is free in two planar directions, not
# restricted to a fictitious single spoke. Both components improve toward a
# rectangle, with unchanged source vertices and no added/removed topology.
for transform in (Matrix.Identity(4), Matrix.Translation((3., -2., 5.)) @ Matrix.Rotation(.7, 4, 'X')):
    bm, loop, edges, verts = triangle_band()
    corners = {}
    loop, edges, _, mirrored, _ = _complete_triangles(bm, loop, edges, transform, corners)
    assert mirrored == 1 and len(corners) == 1
    new, face = next(iter(corners.items()))
    new.co += Vector((.3, .4, 0.))
    fixed = {vert: vert.co.copy() for vert in bm.verts if vert is not new}
    old = transform @ new.co
    initial = _quad_rectangle_error(face, new, old, transform)
    topology = len(bm.verts), len(bm.edges), len(bm.faces)
    counts = _refine_boundary(bm, loop, edges, transform, set(loop) - {new}, corners)
    assert len(counts) == 3 and counts[0] == 1
    assert _quad_rectangle_error(face, new, transform @ new.co, transform) < initial - .01
    assert new.co.length < 1.0e-5, tuple(new.co)
    assert all((vert.co - value).length < 1.0e-7 for vert, value in fixed.items())
    assert topology == (len(bm.verts), len(bm.edges), len(bm.faces))
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    # Explicitly protected corners stay fixed even when their geometry could
    # improve; the lock takes priority over either optimization branch.
    new.co += Vector((.2, .3, 0.))
    protected = new.co.copy()
    assert _refine_boundary(bm, loop, edges, transform, set(loop), corners) == [0, 0, 0]
    assert (new.co - protected).length == 0.
    bm.free()

# Cumulative Step 4 always keeps the repaired pentagon corners at their fitted
# positions. Redo can inspect one or all three post-mirror passes, while the
# three T-only passes always precede triangle completion.
for passes in (0, 1, 2, 3):
    bm, loop, edges = clipped_grid()
    result = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'REFINE', passes)
    assert result['repaired'] == 4 and result['refinement_counts'] == [0] * passes
    assert result['t_refinement_counts'] == [0, 0, 0]
    assert {tuple(vert.co) for vert in result['boundary_verts']} == {
        (-1., -1., 0.), (-1., 1., 0.), (1., -1., 0.), (1., 1., 0.)}
    assert not result['faces']
    assert_clean(bm, result['boundary_edges'])
    bm.free()

# Explicitly check the actual pipeline order on a mesh containing triangles,
# rather than only relying on counters from an already rectangular fixture.
bm, loop, edges, _verts = triangle_band()
events = []
native_refine = grid_patcher._refine_boundary
native_complete = grid_patcher._complete_triangles


def record_refinement(mesh, verts, rim, transform, locked, corners, passes=3,
                      repaired_vertices=(), bowtie_vertices=()):
    phase = 'POST' if events else 'PRE'
    if phase == 'PRE':
        assert not corners and any(len(face.verts) == 3 for face in mesh.faces)
        assert any(len(vert.link_edges) == 3 and any(len(face.verts) == 3 for face in vert.link_faces)
                   for vert in verts)
    else:
        assert events == ['PRE', 'MIRROR']
    before = {vert: vert.co.copy() for vert in locked if vert.is_valid}
    counts = native_refine(mesh, verts, rim, transform, locked, corners, passes,
                           repaired_vertices=repaired_vertices, bowtie_vertices=bowtie_vertices)
    assert len(counts) == 3
    assert all((vert.co - position).length == 0. for vert, position in before.items())
    events.append(phase)
    return counts


def record_completion(*args):
    assert events == ['PRE']
    result = native_complete(*args)
    if not result[3]:
        # Correcting neighboring T points can leave too little opening for the
        # full half-turn completion. Preserve the triangle rather than force an overlap.
        assert result[4] == 1
        mesh, verts, rim, transform, _corners, _bowties = args
        triangle = next(face for face in mesh.faces if len(face.verts) == 3)
        border = next(edge for edge in triangle.edges if edge in rim)
        a, b = border.verts
        opposite = next(vert for vert in triangle.verts if vert not in border.verts)
        first, second, third = [transform @ vert.co for vert in (a, b, opposite)]
        assert not grid_patcher._inside_repair(first + second - third, a, b, verts, transform)
    events.append('MIRROR')
    return result


grid_patcher._refine_boundary = record_refinement
grid_patcher._complete_triangles = record_completion
try:
    result = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'REFINE', refinement_passes=3)
finally:
    grid_patcher._refine_boundary = native_refine
    grid_patcher._complete_triangles = native_complete
assert events == ['PRE', 'MIRROR', 'POST'], events
assert len(result['t_refinement_counts']) == len(result['refinement_counts']) == 3
assert result['mirrored'] + result['triangle_skipped'] == 1
bm.free()

for bowtie in (False, True):
    bm, loop, edges, _verts = triangle_band(bowtie=bowtie)
    result = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'REFINE')
    assert result['refinement_counts'] == [] and result['boundary_edges']
    assert result['bowties'] + result['mirrored'] + result['triangle_skipped'] >= 1
    assert not result['faces']
    assert all(face.calc_area() > 1.0e-7 for face in bm.faces)
    assert all(len(edge.link_faces) in {1, 2} for edge in bm.edges)
    bm.free()

# An unrelated opening is never visited, even if it has redundant rim vertices.
bm, loop, edges = clipped_grid()
unrelated = [bm.verts.new(co) for co in [(10., 0., 0.), (11., 0., 0.),
                                       (12., 0., 0.), (12., 2., 0.), (10., 2., 0.)]]
remote_face = bm.faces.new(unrelated)
result = prepare_grid_steps(bm, loop, edges, Matrix.Identity(4), 'PENTAGONS')
assert remote_face.is_valid and len(remote_face.verts) == 5
assert all(vert.is_valid for vert in unrelated)
assert result['repaired'] == 4
bm.free()

# Real operator registration/redo properties and cumulative execution.
fair_hole_module.DEVELOPMENT_CONTROLS = True
src.register()
assert bpy.ops.mesh.holey_tools_fair_hole.get_rna_type().properties['fill_style'].default == 'AUTO'
assert bpy.ops.mesh.holey_tools_fair_hole.get_rna_type().properties['grid_step'].default == 'ROWS'
assert 'REFINE' in bpy.ops.mesh.holey_tools_fair_hole.get_rna_type().properties['grid_step'].enum_items
assert 'grid_refinement_passes' not in bpy.ops.mesh.holey_tools_fair_hole.get_rna_type().properties
assert 'ROWS' in bpy.ops.mesh.holey_tools_fair_hole.get_rna_type().properties['grid_step'].enum_items
for fill, stage, expected_edges in [('GRID', 'DISSOLVE', 8),
                                   ('GRID', 'PENTAGONS', 4),
                                   ('GRID', 'SLIDE', 4),
                                   ('GRID', 'REFINE', 4),
                                   ('AUTO', 'PENTAGONS', 4)]:
    bm, loop, edges = clipped_grid()
    mesh = bpy.data.meshes.new("staged_grid_" + stage)
    bm.to_mesh(mesh)
    indices = ','.join(str(edge.index) for edge in edges)
    bm.free()
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    operator = SimpleNamespace(edge_indices=indices, fill_style=fill, grid_step=stage,
                               fairing_method='ORGANIC',
                               final_simplify=True, report=lambda *_args: None)
    assert MESH_OT_holey_tools_fair_hole.execute(operator, bpy.context) == {'FINISHED'}
    assert operator.result_method == 'GRID_PREP'
    assert operator.result_grid_dissolved == 1
    assert operator.result_grid_repaired == (0 if stage == 'DISSOLVE' else 4)
    if stage in {'SLIDE', 'REFINE'}:
        assert operator.result_grid_t_refinement_counts == '0 / 0 / 0'
    assert operator.result_grid_refinement_counts == ('0' if stage == 'REFINE' else '')
    live = bmesh.from_edit_mesh(obj.data)
    assert sum(edge.select for edge in live.edges) == expected_edges
    assert_clean(live, [edge for edge in live.edges if edge.select])
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)

# A failed stage restores the mesh and edit mode, rather than leaving a partial
# boundary edit or a replacement mesh with stale topology.
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
live = bmesh.from_edit_mesh(obj.data)
before = (len(live.verts), len(live.edges), len(live.faces))
bad = SimpleNamespace(edge_indices=','.join(str(edge.index) for edge in live.edges if edge.select),
                      fill_style='GRID', grid_step='UNKNOWN', report=lambda *_args: None)
assert MESH_OT_holey_tools_fair_hole.execute(bad, bpy.context) == {'CANCELLED'}
live = bmesh.from_edit_mesh(obj.data)
assert (len(live.verts), len(live.edges), len(live.faces)) == before
assert obj.mode == 'EDIT'

# A merge threshold that collapses a tiny opening must restore the input, not
# leave an invalid boundary or partially welded source surface.
for vert in live.verts:
    vert.co *= .0001
live.edges.index_update()
before_coordinates = sorted(tuple(vert.co) for vert in live.verts)
tiny = SimpleNamespace(edge_indices=','.join(str(edge.index) for edge in live.edges if edge.select),
                       fill_style='GRID', grid_step='PENTAGONS', report=lambda *_args: None)
assert MESH_OT_holey_tools_fair_hole.execute(tiny, bpy.context) == {'CANCELLED'}
live = bmesh.from_edit_mesh(obj.data)
assert (len(live.verts), len(live.edges), len(live.faces)) == before
assert sorted(tuple(vert.co) for vert in live.verts) == before_coordinates
assert obj.mode == 'EDIT'
bpy.ops.object.mode_set(mode='OBJECT')
src.unregister()
fair_hole_module.DEVELOPMENT_CONTROLS = False
print("HOLEY_TOOLS_GRID_PATCHER_TESTS_OK")
