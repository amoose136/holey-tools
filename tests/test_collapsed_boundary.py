"""Run with Blender --background --factory-startup --python tests/test_collapsed_boundary.py."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fairing import (COLLAPSED_BOUNDARY_DISTANCE, FairingError, choose_hypothesis,
                         ordered_boundary_loop, weld_collapsed_boundary_edges)
from src.fairing_pipeline import fair_hole_patch


def annulus(triangulated=False, gap=0.):
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new('SourceUV')
    tag = bm.faces.layers.int.new('source_tag')
    outer = [bm.verts.new(point) for point in ((2., 2., 0.), (-2., 2., 0.),
                                             (-2., -2., 0.), (2., -2., 0.))]
    inner = []
    stations = {}
    for i in range(32):
        stations[i] = len(inner)
        vert = bm.verts.new((math.cos(math.tau * i / 32), math.sin(math.tau * i / 32), 0.))
        inner.append(vert)
        if i in (3, 19):
            inner.append(bm.verts.new(vert.co + Vector((0., 0., gap))))
    source = []
    for i in range(4):
        indices = [(j % len(inner)) for j in range(stations[8 * i],
                   (stations[8 * (i + 1)] if i < 3 else len(inner)) + 1)]
        face = bm.faces.new([outer[i], outer[(i + 1) % 4]] + [inner[j] for j in reversed(indices)])
        face.material_index = 2
        face.smooth = True
        face[tag] = i + 10
        for corner in face.loops:
            corner[uv].uv = corner.vert.co.xy
            corner[uv].pin_uv = True
        source.append(face)
    if triangulated:
        bmesh.ops.triangulate(bm, faces=source)
    # A separate source layer sits exactly on a duplicate corner. It is not
    # connected to the selected rim and must not participate in the weld.
    near = bm.verts.new(inner[3].co)
    remote = bm.faces.new([near, bm.verts.new(near.co + Vector((0., 0., 2.))),
                          bm.verts.new(near.co + Vector((0., 2., 0.)))])
    bm.normal_update(); bm.edges.index_update()
    edges = [bm.edges.get((v, inner[(i + 1) % len(inner)])) for i, v in enumerate(inner)]
    loop, edges = ordered_boundary_loop(edges=edges)
    return bm, loop, edges, remote, uv, tag


for triangulated in (False, True):
    for gap in (0., COLLAPSED_BOUNDARY_DISTANCE * .5):
        if triangulated and gap:
            continue  # A nonzero ear is valid geometry, not removable debris.
        for fill in ('AUTO', 'CIRCLE'):
            bm, loop, edges, remote, uv, tag = annulus(triangulated, gap)
            before = {v: v.co.copy() for v in bm.verts}
            protected = tuple(remote.verts)
            original_face_count = len(bm.faces)
            degenerate_count = sum(f.calc_area() <= 1.e-14 for f in bm.faces)
            loop, edges = weld_collapsed_boundary_edges(bm, loop, edges)
            assert len(loop) == 32 and len(bm.verts) == len(before) - 2
            assert len(bm.faces) == original_face_count - degenerate_count
            assert all(v.co == co for v, co in before.items() if v.is_valid)
            assert remote.is_valid and tuple(remote.verts) == protected
            assert all(f.calc_area() > 1.e-14 for f in bm.faces)
            for face in bm.faces:
                if face is remote:
                    continue
                assert face.material_index == 2 and face.smooth
                assert 10 <= face[tag] < 14
                assert all(c[uv].pin_uv and (c[uv].uv - c.vert.co.xy).length < 1.e-6 for c in face.loops)
            again, _edges = weld_collapsed_boundary_edges(bm, loop, edges)
            assert again == loop  # Idempotent.
            hypothesis = choose_hypothesis(loop, edges, Matrix.Identity(4),
                                           lambda _faces: {}, lambda face: face.normal)
            assert hypothesis['method'] == 'PLANES'
            bm, result = fair_hole_patch(SimpleNamespace(matrix_world=Matrix.Identity(4)),
                                         bm, loop, edges, hypothesis, fill, final_simplify=False)
            assert result['fill_style'] == ('PLANE' if fill == 'AUTO' else 'CIRCLE')
            assert all(len(e.link_faces) == 2 for e in result['boundary_edges'])
            assert all(f.calc_area() > 1.e-14 for f in bm.faces)
            assert remote.is_valid and tuple(remote.verts) == protected
            bm.free()

            # Exercise the shared entry point on dirty input as well, without
            # calling the weld directly: both operators use this preparation.
            bm, loop, edges, remote, _uv, _tag = annulus(triangulated, gap)
            hypothesis = choose_hypothesis(loop, edges, Matrix.Identity(4),
                                           lambda _faces: {}, lambda face: face.normal)
            bm, result = fair_hole_patch(SimpleNamespace(matrix_world=Matrix.Identity(4)),
                                         bm, loop, edges, hypothesis, fill, final_simplify=False)
            assert len(result['boundary_verts']) == 32
            assert all(len(e.link_faces) == 2 for e in result['boundary_edges'])
            assert all(f.calc_area() > 1.e-14 for f in bm.faces)
            assert remote.is_valid
            bm.free()

# Near-zero does not mean zero-area: do not remove a valid narrow triangle
# from a larger ring just to make the topology easier to fill.
bm, loop, edges, remote, _uv, _tag = annulus(True, COLLAPSED_BOUNDARY_DISTANCE * .5)
original = {v: v.co.copy() for v in bm.verts}
face_count = len(bm.faces)
try:
    weld_collapsed_boundary_edges(bm, loop, edges)
except FairingError:
    assert len(bm.faces) == face_count and len(bm.verts) == len(original)
    assert all(v.is_valid and v.co == co for v, co in original.items())
else:
    raise AssertionError('A valid near-zero triangle was discarded')
bm.free()

# A geometrically narrow but nondegenerate triangle cannot be erased merely
# because its edge happens to lie below the weld threshold.
bm = bmesh.new()
a, b, c = [bm.verts.new(point) for point in ((0., 0., 0.), (5.e-9, 0., 0.), (0., 1., 0.))]
face = bm.faces.new([a, b, c])
bm.edges.index_update()
loop, edges = ordered_boundary_loop(edges=list(bm.edges))
try:
    weld_collapsed_boundary_edges(bm, loop, edges)
except FairingError:
    assert face.is_valid and len(bm.verts) == 3 and len(bm.faces) == 1
else:
    raise AssertionError('A valid triangular opening was collapsed')
bm.free()
print('HOLEY_TOOLS_COLLAPSED_BOUNDARY_OK', flush=True)
