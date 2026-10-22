"""Close one-cell rim notches without subdividing the source grid."""

import json
from pathlib import Path
import sys

import bmesh
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fairing_pipeline import fair_hole_patch
from src.grid_patcher import _cap_refined_notches, prepare_grid_steps
from tests.blender_support import edit_mesh


def notch():
    bm = bmesh.new()
    verts = {(x, y): bm.verts.new((x, y, 0.))
             for x in range(-2, 4) for y in range(-2, 4)}
    uv = bm.loops.layers.uv.new('UVMap')
    for x in range(-2, 3):
        for y in range(-2, 3):
            if (-1 <= x <= 1 and 0 <= y <= 1) or (x, y) == (0, -1):
                continue
            face = bm.faces.new([verts[x, y], verts[x+1, y], verts[x+1, y+1], verts[x, y+1]])
            face.material_index, face.smooth = 2, True
            for corner in face.loops:
                corner[uv].uv = corner.vert.co.xy
    points = [(0, 0), (0, -1), (1, -1), (1, 0), (2, 0), (2, 1),
              (2, 2), (1, 2), (0, 2), (-1, 2), (-1, 1), (-1, 0)]
    loop = [verts[p] for p in points]
    edges = [bm.edges.get((v, loop[(i+1) % len(loop)])) for i, v in enumerate(loop)]
    bm.normal_update()
    return bm, loop, edges, verts, uv


for reverse in (False, True):
    for seed in (0, 5, 11):
        bm, loop, edges, verts, uv = notch()
        if reverse:
            loop.reverse()
        loop = loop[seed:] + loop[:seed]
        positions = {v: v.co.copy() for v in bm.verts}
        untouched = {e for e in bm.edges if e not in edges}
        original_faces = set(bm.faces)
        old_edges = [bm.edges.get((verts[a], verts[b])) for a, b in
                     [((0, 0), (0, -1)), ((0, -1), (1, -1)), ((1, -1), (1, 0))]]
        transform = Matrix.Translation((4., -2., 3.)) @ Matrix.Rotation(.7, 4, 'X')
        loop, edges, faces = _cap_refined_notches(bm, loop, edges, transform)
        assert len(faces) == 1 and set(bm.faces) == original_faces | set(faces)
        assert set(faces[0].verts) == {verts[p] for p in ((0, 0), (0, -1), (1, -1), (1, 0))}
        assert len(loop) == 10 and all(len(e.link_faces) == 2 for e in old_edges)
        assert bm.edges.get((verts[0, 0], verts[1, 0])) in edges
        assert all(v.is_valid and v.co == co for v, co in positions.items())
        assert len(bm.verts) == len(positions)  # No inserted vertices or edge cuts.
        assert all(e.is_valid for e in untouched)
        assert faces[0].normal.z > .99 and faces[0].material_index == 2 and faces[0].smooth
        assert all((c[uv].uv - c.vert.co.xy).length < 1e-7 for c in faces[0].loops)
        # Do not recursively fill another row, or fill the remaining rectangle.
        _, _, more = _cap_refined_notches(bm, loop, edges, transform)
        assert not more
        bm.free()

# Hidden/masked support must not be modified by this pass.
bm, loop, edges, verts, _ = notch()
positions = {v: v.co.copy() for v in bm.verts}
_, _, faces = _cap_refined_notches(bm, loop, edges, Matrix.Identity(4), {verts[0, -1]})
assert not faces and all(v.co == co for v, co in positions.items())
bm.free()

# A folded candidate must remain open, even if its two source junctions have
# the expected valence. Connectivity alone doesn't make it a valid cell.
bm, loop, edges, verts, _ = notch()
verts[0, -1].co = (1.2, -.2, 0.)
positions = {v: v.co.copy() for v in bm.verts}
_, _, faces = _cap_refined_notches(bm, loop, edges, Matrix.Identity(4))
assert not faces and all(v.co == co for v, co in positions.items())
bm.free()

# Current saved Suzanne: short spokes disguise triangles, adjacent repairs
# expose another degree-two rim point, and independent corners aren't bowties.
data = json.loads(Path(__file__).with_name('fixtures').joinpath('suzanne_triangle_grid_hole.json').read_text())


def fixture(seed=0, reverse=False):
    obj, bm = edit_mesh('Suzanne triangle corners', data['vertices'], data['faces'])
    obj.matrix_world = Matrix(data['matrix'])
    tag = bm.faces.layers.int.new('source_face')
    for face, source in zip(bm.faces, data['source_faces']):
        face[tag] = source
    loop = [bm.verts[i] for i in data['boundary']]
    loop = loop[seed:] + loop[:seed]
    if reverse:
        loop.reverse()
    edges = [bm.edges.get((v, loop[(i+1) % len(loop)])) for i, v in enumerate(loop)]
    bm.normal_update()
    return obj, bm, loop, edges, tag


obj, bm, loop, edges, tag = fixture()
result = prepare_grid_steps(bm, loop, edges, obj.matrix_world, 'PENTAGONS')
by_source = {f[tag]: f for f in bm.faces}
assert len(by_source[1937].verts) == len(by_source[1945].verts) == 3
assert len(by_source[1944].verts) == 4
assert result['collapsed_merged'] > 0

for seed, reverse, topology in ((0, False, 'GRID'), (13, True, 'GRID'), (29, False, 'AUTO')):
    obj, bm, loop, edges, tag = fixture(seed, reverse)
    outside = {e for e in bm.edges if e.is_boundary and e not in edges}
    positions = {v: v.co.copy() for e in outside for v in e.verts}
    bm, result = fair_hole_patch(obj, bm, loop, edges, dict(method='ORGANIC', residual=0.), topology)
    assert result['fill_style'] == 'GRID', result.get('grid_fallback_reason')
    assert result['notch_caps'] == 1 and result['mirrored'] == 6 and result['bowties'] == 1
    assert result['grid_cut_count'] == result['transition_triangles'] == 0
    assert result['mutation_patches'] == 3
    assert all(len(f.verts) == 4 and f.calc_area() > 1e-14 for f in result['faces'])
    assert {e for e in bm.edges if e.is_boundary} == outside
    assert all(v.is_valid and v.co == co for v, co in positions.items())
    by_source = {f[tag]: f for f in bm.faces}
    assert len(by_source[1937].verts) == len(by_source[1945].verts) == 4

print('HOLEY_TOOLS_GRID_NOTCHES_OK')
