"""Sphere fairing should batch work without lowering relaxation quality."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fairing import choose_hypothesis
from src.fairing_pipeline import fair_hole_patch
from src.grid_patcher import _tracked_loop
import src.grid_quality as quality
from src.surface_projection import project_hypothesis_along_axis, sphere_axis_projector
from tests.blender_support import bind_operator_methods, edit_mesh


# The batch projector retains the nearest ray root, rejects genuinely missing
# intersections, and stays frozen when the caller changes its fit dictionary.
for scale in (.01, 1., 100.):
    for offset in (0., .7, -.7):
        rotation = Matrix.Rotation(.4, 4, 'X') @ Matrix.Rotation(-.3, 4, 'Z')
        x, y, axis = [rotation.to_3x3() @ Vector(v) for v in ((1.,0.,0.), (0.,1.,0.), (0.,0.,1.))]
        center = Vector((2., -3., 4.)) * scale
        origin = center + axis * (offset*scale)
        hypothesis = dict(method='SPHERE', center=center.copy(), radius=scale)
        project = sphere_axis_projector(hypothesis, origin, x, y, axis)
        queries = np.asarray(((0.,0.), (.3,.4), (.95,0.), (1.2,0.))) * scale
        actual = project(queries)
        for i, q in enumerate(queries):
            p = origin + x*float(q[0]) + y*float(q[1])
            expected = project_hypothesis_along_axis(hypothesis, p, p, axis)
            if expected is None:
                assert np.isnan(actual[i]).all()
            elif offset == 0.:
                # A chart through the sphere center has equal-distance roots.
                # Tiny float32 scalar rounding can break that exact tie, so
                # check sphere/ray membership rather than an arbitrary side.
                assert abs(np.linalg.norm(actual[i]-np.asarray(center))-scale) < 1e-7*scale
                assert np.linalg.norm(np.cross(actual[i]-np.asarray(p), np.asarray(axis))) < 1e-6*scale
            else:
                assert np.linalg.norm(actual[i]-np.asarray(expected)) < 2e-6*scale
        frozen = actual.copy()
        hypothesis['radius'] *= 2.
        hypothesis['center'].z += scale
        assert np.array_equal(project(queries), frozen, equal_nan=True)

# A commuting wavefront must match the original sequential update trajectory
# for general Organic fits; multicolor scheduling is opt-in for Sphere only.
boundary = [(x,y) for x in range(7) for y in range(7) if x in (0,6) or y in (0,6)]
inside = [(x,y) for x in range(1,6) for y in range(1,6)]
indices = {p:i for i,p in enumerate(boundary+inside)}
layout = dict(coords=[Vector(p) for p in boundary+inside], splits=[], row_faces=[
    [[indices[x,y], indices[x+1,y], indices[x+1,y+1], indices[x,y+1]] for x in range(6)]
    for y in range(6)])
def project(queries):
    q = np.asarray(queries)
    return np.column_stack((q, .5*np.exp(-((q[:,0]-3.)**2+(q[:,1]-3.)**2)/3.)))
def sequential(free, neighbors, **_options):
    return [[np.asarray([i]) for i in order] for order in (free, list(reversed(free)))]
reference, batched = deepcopy(layout), deepcopy(layout)
with patch.object(quality, '_independent_sweeps', side_effect=sequential):
    quality.improve_projected_shapes(reference, len(boundary), project, iterations=5)
quality.improve_projected_shapes(batched, len(boundary), project, iterations=5)
assert np.array_equal(np.asarray(reference['coords']), np.asarray(batched['coords']))

# A singular member in the stacked solve must fall back individually rather
# than discard every independent update in that batch.
linear_solve = np.linalg.solve
fallback = deepcopy(layout)
def unstack(matrix, target):
    if matrix.ndim == 3 and len(matrix) > 1:
        raise np.linalg.LinAlgError('injected singular batch')
    return linear_solve(matrix, target)
with patch.object(quality.np.linalg, 'solve', side_effect=unstack) as solve_calls:
    quality.improve_projected_shapes(fallback, len(boundary), project, iterations=5)
assert solve_calls.call_count > 0
assert np.array_equal(np.asarray(fallback['coords']), np.asarray(batched['coords']))

neighbors = {i:set() for i in indices.values()}
for row in layout['row_faces']:
    for face in row:
        for i in face:
            neighbors[i].update(j for j in face if i != j)
free = list(range(len(boundary), len(indices)))
for colored in (False, True):
    for sweep in quality._independent_sweeps(free, neighbors, colored=colored):
        assert sorted(i for batch in sweep for i in batch) == free
        for batch in sweep:
            assert all(not (set(batch)-{i}) & neighbors[i] for i in batch)
wide = list(range(1025))
for colored in (False, True):
    for sweep in quality._independent_sweeps(wide, {i:set() for i in wide}, colored=colored):
        assert max(map(len, sweep)) <= 128
        assert sorted(i for batch in sweep for i in batch) == wide


data = json.loads(Path(__file__).with_name('fixtures').joinpath('sphere_fairing_hole.json').read_text())
def distortion(faces, matrix):
    result = 0.
    for face in faces:
        p = np.asarray([matrix @ v.co for v in face.verts], dtype=float)
        u, v = p-np.roll(p,1,axis=0), np.roll(p,-1,axis=0)-p
        normals = np.cross(u,v)
        area = np.linalg.norm(normals,axis=1)
        assert np.all(area > 1e-12)
        normals /= area[:,None]
        result += np.sum(np.sum(u*u+v*v,axis=1)/(2.*area)
                         + .5*(1.-np.sum(normals*np.roll(normals,-1,axis=0),axis=1)))
    return result

solve = quality.improve_projected_shapes
for iterations in (20, 100):
    results, timings, losses = [], [], []
    for colored in (False, True):
        obj, bm = edit_mesh('Saved sphere opening', data['vertices'], data['faces'])
        obj.matrix_world = Matrix(data['matrix'])
        bm.normal_update()
        loop = [bm.verts[i] for i in data['boundary']]
        edges = [bm.edges.get((v,loop[(i+1)%len(loop)])) for i,v in enumerate(loop)]
        class LocalEdges:
            def index_update(self): bm.edges.index_update()
            def __iter__(self): raise AssertionError('Boundary tracking scanned the whole mesh')
        _verts, tracked = _tracked_loop(SimpleNamespace(edges=LocalEdges()), set(loop))
        assert set(tracked) == set(edges)
        operator = bind_operator_methods(SimpleNamespace(_object=obj))
        hypothesis = choose_hypothesis(loop, edges, obj.matrix_world, operator._fit_cylinder, operator._world_normal)
        assert hypothesis['method'] == 'SPHERE'
        def run(*args, **kwargs):
            assert kwargs['color_sweeps']  # Both operators dispatch Sphere to the fast path.
            kwargs['color_sweeps'] = colored
            return solve(*args, **kwargs)
        start = time.perf_counter()
        with patch.object(quality, 'improve_projected_shapes', side_effect=run) as optimizer:
            bm, result = fair_hole_patch(obj, bm, loop, edges, hypothesis, 'AUTO',
                                         relaxation_iterations=iterations)
        timings.append(time.perf_counter()-start)
        assert optimizer.call_count == 1
        assert result['fill_style'] == 'GRID' and result['surface_method'] == 'SPHERE'
        assert len(result['verts']) == 312 and len(result['faces']) == 354
        assert all(len(f.verts) == 4 for f in result['faces'])
        losses.append(distortion(result['faces'], obj.matrix_world))
        results.append(np.asarray([tuple(v.co) for v in bm.verts]))
        for vert in result['verts'] + result['relaxed_rim_verts']:
            radius = (obj.matrix_world @ vert.co-hypothesis['center']).length
            assert abs(radius-hypothesis['radius']) < 1e-6
    # Same pass count and safety checks, near-identical converged patch shape.
    assert losses[1] <= 1.001*losses[0], losses
    assert np.max(np.linalg.norm(results[0]-results[1],axis=1)) < .002
    print('Sphere ordered/colored:', iterations, timings, 'distortion:', losses)

print('HOLEY_TOOLS_SPHERE_FAIRING_OK')
