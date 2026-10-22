"""Grid topology and surface hypothesis must be independent choices."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.fair_hole as fair_operator
from src.fairing import ordered_boundary_loop
from src.fairing_pipeline import fair_hole_patch
from tests.blender_support import bind_operator_methods, edit_mesh


def fixture(surface='SPHERE', transformed=False):
    coordinates = []
    indices = {}
    for i in range(-4, 5):
        for j in range(-4, 5):
            if -2 < i < 2 and -2 < j < 2:
                continue
            x, y = i * .15, j * .15
            z = (.2 * x + .1 * y if surface == 'PLANES' else
                 math.sqrt(1. - x * x) if surface == 'CONIC' else
                 math.sqrt(1. - x * x - y * y) + .2 * x * x * x + .1 * x * y)
            indices[i, j] = len(coordinates)
            coordinates.append((x, y, z))
    faces = []
    for i in range(-4, 4):
        for j in range(-4, 4):
            if -2 <= i < 2 and -2 <= j < 2:
                continue
            faces.append([indices[i, j], indices[i + 1, j],
                          indices[i + 1, j + 1], indices[i, j + 1]])
    obj, bm = edit_mesh('Grid fit regression', coordinates, faces)
    if transformed:
        obj.matrix_world = (Matrix.Translation((2., -3., 4.)) @
                            Matrix.Rotation(.6, 4, 'X') @ Matrix.Scale(1.7, 4))
    bm.normal_update()
    edges = [e for e in bm.edges if e.is_boundary and
             all(abs(v.co.x) <= .301 and abs(v.co.y) <= .301 for v in e.verts)]
    loop, edges = ordered_boundary_loop(edges=edges)
    return obj, bm, loop, edges


for transformed in (False, True):
    outputs = {}
    for method in ('ORGANIC', 'SPHERE'):
        for curvature in (False, True):
            obj, bm, loop, edges = fixture(transformed=transformed)
            source = {v: v.co.copy() for v in bm.verts if v not in loop}
            scale = 1.7 if transformed else 1.
            center = obj.matrix_world @ Vector()
            hypothesis = dict(method=method, center=center, radius=scale, residual=.012)
            bm, result = fair_hole_patch(obj, bm, loop, edges, hypothesis, 'GRID',
                                        match_curvature=curvature)
            assert result['method'] == 'GRID_ROWS' and result['surface_method'] == method
            assert result['residual'] == .012
            assert len(result['verts']) == 9 and len(result['faces']) == 16
            assert all(v.co == co for v, co in source.items())
            assert all(len(f.verts) == 4 and f.calc_area() > 1e-10 for f in result['faces'])
            if method == 'SPHERE':
                assert result['surface_continuity'] == 'ANALYTIC'
                assert all(abs((obj.matrix_world @ v.co - center).length - scale) < 1e-6
                           for v in result['verts'])
                assert all(abs((obj.matrix_world @ v.co - center).length - scale) < 1e-6
                           for v in result['relaxed_rim_verts'])
            else:
                assert result['surface_continuity'] == ('C2_ESTIMATED' if curvature else 'THIN_PLATE')
            outputs[method, curvature] = [v.co.copy() for v in result['verts']]
    assert max((a - b).length for a, b in zip(outputs['SPHERE', True], outputs['ORGANIC', True])) > 1e-4
    assert all((a - b).length < 1e-7 for a, b in zip(outputs['SPHERE', True], outputs['SPHERE', False]))

# Exercise the actual operator's fit selection, not just a supplied hypothesis.
# Its Grid branch previously passed None, making the Fit dropdown a no-op.
for requested, surface in (('SPHERE', 'SPHERE'), ('ORGANIC', 'SPHERE'),
                           ('CYLINDER', 'CONIC'), ('PLANES', 'PLANES'), ('AUTO', 'PLANES')):
    obj, bm, loop, edges = fixture(surface)
    bm.edges.index_update()
    op = bind_operator_methods(SimpleNamespace(), fair_operator.MESH_OT_holey_tools_fair_hole)
    op.edge_indices = ','.join(str(edge.index) for edge in edges)
    op.fill_style, op.fairing_method = 'GRID', requested
    op.report = lambda *_args: None
    captured = []
    def capture(*args, **kwargs):
        output = fair_hole_patch(*args, **kwargs)
        captured.append((args[4], output[1]))
        return output
    with patch.object(fair_operator, 'fair_hole_patch', side_effect=capture):
        assert op.execute(bpy.context) == {'FINISHED'}, requested
    hypothesis, result = captured[0]
    assert hypothesis is not None
    expected = {'CYLINDER': 'CONIC', 'AUTO': 'PLANES'}.get(requested, requested)
    assert hypothesis['method'] == result['surface_method'] == op.result_method == expected
    assert op.result_fill_style == 'GRID'

print('HOLEY_TOOLS_GRID_SURFACE_FIT_OK')
