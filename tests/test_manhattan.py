"""Run with Blender --background --factory-startup --python tests/test_manhattan.py."""
import sys
from pathlib import Path

import bmesh
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fairing import FairingError, _signed_area_2d, ordered_boundary_loop
from src.manhattan import analyze_boundary, grid_layout
from src.row_patcher import fill_refined_rows


# Minimal fixture from Suzanne's saved Refined Edge. Source indices are labels
# for assertions only; the planner receives geometry and total edge valence.
LABELS = [1730, 1738, 1739, 1735, 1736, 1731, 1726, 1724, 1720, 1721,
          1723, 1725, 1727, 1728, 1734, 1729, 1732, 1722, 1737, 1740]
VALENCES = [2, 3, 4, 2, 4, 4, 3, 4, 3, 4, 3, 3, 3, 4, 2, 4, 3, 3, 3, 4]
POINTS = [(.119243972, .077919334), (.155656859, .128404707), (.172306463, .174803868),
          (.124040455, .172723204), (.127886668, .222160026), (.037373267, .217916727),
          (-.158121482, .211082727), (-.342895359, .202630863), (-.351457626, -.018524839),
          (-.353682578, -.204691783), (-.190817133, -.195427537), (-.050497539, -.186382130),
          (.035104435, -.199152797), (.080251560, -.226392627), (.066896230, -.142168716),
          (.105050005, -.187765449), (.078902088, -.097713523), (.076238945, -.051552162),
          (.105827749, .021045076), (.162691504, .081085227)]
WORLD = [(-2.867172241, 4.210918903, -1.162132025), (-2.914009571, 4.215342522, -1.242253065),
         (-2.958370209, 4.222244263, -1.288295269), (-2.958370209, 4.255420208, -1.249940157),
         (-3.007364750, 4.255607605, -1.260635614), (-3.007364750, 4.312788486, -1.180448651),
         (-3.008733273, 4.450443745, -1.030562997), (-3.007364750, 4.590487003, -.905038238),
         (-2.785794497, 4.613558292, -.912891865), (-2.597859621, 4.647344589, -.952564299),
         (-2.600741625, 4.525909901, -1.066595793), (-2.604062796, 4.424588680, -1.170312166),
         (-2.588645697, 4.350753307, -1.212789774), (-2.561739683, 4.286788940, -1.192989588),
         (-2.648795128, 4.260496616, -1.129496336), (-2.603887796, 4.200998783, -1.102480292),
         (-2.696319580, 4.198153019, -1.053192139), (-2.742651701, 4.197980881, -1.050621271),
         (-2.813310146, 4.185626030, -1.091550350), (-2.869789362, 4.161465168, -1.164676547)]


def check(layout, points):
    coords = layout['coords']
    faces = [face for row in layout['row_faces'] for face in row]
    assert all(len(face) == 4 for face in faces)
    assert all(_signed_area_2d([coords[i] for i in face]) > 1e-10 for face in faces)
    assert abs(sum(_signed_area_2d([coords[i] for i in face]) for face in faces)
               - _signed_area_2d([Vector(p) for p in points])) < 1e-5
    assert all(coords[i] == Vector(p) for i, p in enumerate(points))
    for row, counts in zip(layout['row_faces'], layout['row_edge_counts']):
        assert abs(counts[0] - counts[1]) in {0, 2}
        # Mutation vertices belong to this row alone, never a two-row patch.
        others = {i for other in layout['row_faces'] if other is not row for face in other for i in face}
        unique = {i for face in row for i in face} - others - set(range(len(points)))
        assert len(unique) >= (2 if abs(counts[0] - counts[1]) == 2 else 0)
    return layout


for seed in range(len(POINTS)):
    points = POINTS[seed:] + POINTS[:seed]
    valences = VALENCES[seed:] + VALENCES[:seed]
    world = WORLD[seed:] + WORLD[:seed]
    labels = LABELS[seed:] + LABELS[:seed]
    analysis = analyze_boundary(points, valences, world)
    assert analysis['side_counts'] == (8, 5, 2, 5)
    assert {labels[i] for i, kind in analysis['corners'] if kind == 'EXTERIOR'} == {1735, 1730, 1734}
    assert {labels[i] for i, kind in analysis['corners'] if kind == 'INTERIOR'} == {1724, 1721, 1736, 1739, 1740, 1729, 1728}
    assert {labels[i] for i in analysis['side_edges'][2]} == {1724, 1720}
    side_labels = [{labels[i] for i in side} for side in analysis['side_edges']]
    assert side_labels == [{1730, 1738, 1735, 1728, 1729, 1732, 1722, 1737},
                           {1739, 1736, 1731, 1726, 1740},
                           {1724, 1720}, {1721, 1723, 1725, 1727, 1734}]
    # Four groups are interlaced runs, not four contiguous chains.
    assert len(analysis['side_edges'][0]) == 8
    layout = check(grid_layout(points, valences, world), points)
    assert layout['grid_size'] == (8, 5)
    assert layout['mutations'] == 3 and layout['triangles'] == 0 and not layout['splits']
    assert sum(map(len, layout['row_faces'])) == 26 and layout['trimmed_cells'] == 5


def edge_energy(layout):
    edges = {tuple(sorted((a, b))) for row in layout['row_faces'] for face in row
             for a, b in zip(face, face[1:] + face[:1])}
    return sum((layout['coords'][a] - layout['coords'][b]).length_squared for a, b in edges)


raw = grid_layout(POINTS, VALENCES, WORLD, relaxation_iterations=0)
relaxed = grid_layout(POINTS, VALENCES, WORLD, relaxation_iterations=20)
assert raw['row_faces'] == relaxed['row_faces'] and raw['splits'] == relaxed['splits']
assert relaxed['relaxation_passes'] > 0 and raw['relaxation_passes'] == 0
print('RELAXATION_ENERGY', edge_energy(raw), edge_energy(relaxed), flush=True)
assert edge_energy(relaxed) < edge_energy(raw)
assert max((a - b).length for a, b in zip(raw['coords'], relaxed['coords'])) > 1e-3
disabled = grid_layout(POINTS, VALENCES, WORLD, relaxation_strength=0.)
assert disabled['coords'] == raw['coords']

def rectangle(bottom=4, top=4, left=4, right=4):
    return ([(4. * i / bottom, 0.) for i in range(bottom)]
            + [(4., 4. * i / right) for i in range(right)]
            + [(4. - 4. * i / top, 4.) for i in range(top)]
            + [(0., 4. - 4. * i / left) for i in range(left)])


for counts in ((4, 4, 4, 4), (8, 2, 5, 5), (5, 4, 4, 4), (8, 2, 3, 5)):
    points = rectangle(*counts)
    layout = check(grid_layout(points, [3] * len(points)), points)
    assert len(layout['splits']) == (1 if counts == (5, 4, 4, 4) else 2 if counts == (8, 2, 3, 5) else 0)
    for edge, factor, identity in layout['splits']:
        assert 0 < factor < 1 and identity >= len(points)
        assert (layout['coords'][identity] - Vector(points[edge]).lerp(Vector(points[(edge + 1) % len(points)]), factor)).length < 1e-7


def ring(points, curved=False):
    bm = bmesh.new()
    center = Vector((2., 2.))
    def coordinate(p):
        return (p[0], p[1], .01 * (p[0] ** 2 + p[1] ** 2) if curved else 0.)
    inner = [bm.verts.new(coordinate(p)) for p in points]
    outer = [bm.verts.new(coordinate(center + (Vector(p) - center) * 1.2)) for p in points]
    for i in range(len(points)):
        j = (i + 1) % len(points)
        face = bm.faces.new([inner[i], outer[i], outer[j], inner[j]])
        face.material_index = 2
        face.smooth = True
    bm.normal_update()
    edges = [bm.edges.get((inner[i], inner[(i + 1) % len(points)])) for i in range(len(points))]
    loop, edges = ordered_boundary_loop(edges=edges)
    return bm, loop, edges


for curved in (False, True):
    for counts in ((4, 4, 4, 4), (5, 4, 4, 4), (8, 2, 3, 5), (6, 2, 1, 6)):
        for partial in (0, 1, 2):
            bm, loop, edges = ring(rectangle(*counts), curved)
            original = {v: v.co.copy() for v in bm.verts}
            source_faces = list(bm.faces)
            uv = bm.loops.layers.uv.new('SourceUV')
            for face in source_faces:
                for corner in face.loops:
                    corner[uv].uv = corner.vert.co.xy
            result = fill_refined_rows(bm, loop, edges, Matrix.Identity(4),
                                       fill_rows=partial)
            assert all(v.is_valid and v.co == co for v, co in original.items())
            assert all(face.is_valid for face in source_faces)
            assert all((corner[uv].uv - corner.vert.co.xy).length < 1e-6
                       for face in source_faces for corner in face.loops)
            assert all(face.material_index == 2 and face.smooth for face in result['faces'])
            assert all(len(face.verts) == 4 and face.calc_area() > 1e-8 for face in result['faces'])
            assert all(len(e.link_faces) in {1, 2} for e in bm.edges)
            # Required cuts remain exactly on the original 3D boundary edges;
            # only brand-new interior points receive the surface projection.
            cuts = set(result['boundary_verts']) - set(loop)
            assert len(cuts) == result['grid_cut_count']
            for v in cuts:
                assert any((v.co - a.co).cross(b.co - a.co).length < 1e-6
                           and 0 < (v.co - a.co).dot(b.co - a.co) < (b.co - a.co).length_squared
                           for a, b in zip(loop, loop[1:] + loop[:1]))
            if partial:
                assert result['boundary_edges']
                assert all(len(e.link_faces) == 1 for e in result['boundary_edges'])
            else:
                assert not result['boundary_edges']
                assert all(len(e.link_faces) == 2 for v in result['boundary_verts'] for e in v.link_edges)
            bm.free()

# Project the relaxed points, never their pre-relaxation positions, onto the
# same frozen C2 field for every row inspection limit.
from src.fairing import _organic_grid_frame
from src.ideal_surface import capture_curvature_surface
for partial in (0, 1, 2):
    bm, loop, edges = ring(rectangle(), curved=True)
    center, axis, x, y, _points, _values = _organic_grid_frame(loop, edges, Matrix.Identity(4))
    surface = capture_curvature_surface(loop, Matrix.Identity(4), center, x, y, axis)
    assert surface is not None
    source = {v: v.co.copy() for v in bm.verts}
    result = fill_refined_rows(bm, loop, edges, Matrix.Identity(4), fill_rows=partial)
    assert result['surface_continuity'] == 'C2_ESTIMATED'
    for v in result['verts']:
        offset = v.co - center
        assert abs(offset.dot(axis) - surface([(offset.dot(x), offset.dot(y))])[0]) < 1e-6
    assert all(v.co == p for v, p in source.items())
    bm.free()

import src
src.register()
from src.fair_hole import MESH_OT_holey_tools_fair_hole
from types import SimpleNamespace
import bpy
properties = bpy.ops.mesh.holey_tools_fair_hole.get_rna_type().properties
assert 'grid_backend' not in properties
assert properties['grid_relaxation_iterations'].default == 20
assert properties['grid_match_curvature'].default
# There is only one grid planner, shared with Resize Hole.
bm = bmesh.new()
verts = {(i, j): bm.verts.new((i, j, .01 * (i * i + j * j)))
         for j in range(-1, 6) for i in range(-1, 6)}
for j in range(-1, 5):
    for i in range(-1, 5):
        if 0 <= i < 4 and 0 <= j < 4:
            continue
        bm.faces.new([verts[i, j], verts[i + 1, j], verts[i + 1, j + 1], verts[i, j + 1]])
for v in list(bm.verts):
    if not v.link_edges:
        bm.verts.remove(v)
edges = [e for e in bm.edges if len(e.link_faces) == 1
         and all(0 <= v.co.x <= 4 and 0 <= v.co.y <= 4 for v in e.verts)]
bm.edges.index_update()
indices = ','.join(str(e.index) for e in edges)
mesh = bpy.data.meshes.new('relaxed_operator'); bm.to_mesh(mesh); bm.free()
obj = bpy.data.objects.new(mesh.name, mesh); bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj; obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
op = SimpleNamespace(edge_indices=indices, fill_style='GRID', grid_step='ROWS',
                     report=lambda _level, message: print(message, flush=True))
assert MESH_OT_holey_tools_fair_hole.execute(op, bpy.context) == {'FINISHED'}
assert op.result_grid_row_method == 'MANHATTAN_GRID'
assert op.result_grid_surface_continuity == 'C2_ESTIMATED'
bpy.ops.object.mode_set(mode='OBJECT')
src.unregister()


# Reject inconsistent labels, folds, and disconnected strip templates without
# mutating any input data or falling back to another backend.
for points, valences in (([(0, 0), (1, 0), (1, 1), (0, 1)], [2] * 4),
                          ([(0, 0), (0, 0), (1, 1), (0, 1)], [3] * 4)):
    try:
        grid_layout(points, valences)
    except FairingError:
        pass
    else:
        raise AssertionError('Invalid Manhattan boundary accepted')

u_shape = [(0, 0), (.5, 0), (1, 0), (1.5, 0), (2, 0), (2.5, 0), (3, 0), (3.5, 0),
           (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
           (3, 4), (3, 3), (3, 2), (2, 2), (1, 2), (1, 3), (1, 4),
           (0, 4), (0, 3), (0, 2), (0, 1)]
valences = []
for i, p in enumerate(map(Vector, u_shape)):
    incoming, outgoing = p - Vector(u_shape[i - 1]), Vector(u_shape[(i + 1) % len(u_shape)]) - p
    valences.append(2 if incoming.x * outgoing.y - incoming.y * outgoing.x < 0 else 3)
try:
    grid_layout(u_shape, valences)
except FairingError as error:
    assert 'connected interval' in str(error)
else:
    raise AssertionError('Disconnected Manhattan rows accepted')

print('HOLEY_TOOLS_MANHATTAN_OK', flush=True)
