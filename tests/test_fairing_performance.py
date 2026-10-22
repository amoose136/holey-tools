"""Avoid duplicate fairing work without reducing topology/relaxation quality."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.fairing_pipeline as pipeline
import src.grid_quality as quality
import src.ideal_surface as surfaces
import src.manhattan as manhattan
from src.grid_patcher import prepare_grid_steps
from src.row_patcher import fill_refined_rows, _projection_key
from tests.blender_support import edit_mesh

fixtures = Path(__file__).with_name('fixtures')


def fixture(name):
    data = json.loads((fixtures / (name+'.json')).read_text())
    obj, bm = edit_mesh(name, data['vertices'], data['faces'])
    obj.matrix_world = Matrix(data['matrix'])
    bm.normal_update()
    loop = [bm.verts[i] for i in data['boundary']]
    edges = [bm.edges.get((v, loop[(i+1) % len(loop)])) for i, v in enumerate(loop)]
    return obj, bm, loop, edges


prepare, attempt = pipeline.prepare_grid_steps, manhattan._grid_layout_attempt


def uncached_prepare(*args, **kwargs):
    kwargs.pop('_row_plan_cache', None)
    return prepare(*args, **kwargs)


def uncached_attempt(*args, **kwargs):
    kwargs.pop('solutions', None)
    return attempt(*args, **kwargs)


for name in ('suzanne_triangle_grid_hole', 'suzanne_notched_grid_hole'):
    outputs, timings, counts = [], [], []
    for cached in (False, True):
        obj, bm, loop, edges = fixture(name)
        original_faces = set(bm.faces)
        start = time.perf_counter()
        with ExitStack() as stack:
            if not cached:
                stack.enter_context(patch.object(pipeline, 'prepare_grid_steps', side_effect=uncached_prepare))
                stack.enter_context(patch.object(manhattan, '_grid_layout_attempt', side_effect=uncached_attempt))
            capture = stack.enter_context(patch.object(surfaces, 'capture_organic_surface', wraps=surfaces.capture_organic_surface))
            project = stack.enter_context(patch.object(quality, 'improve_projected_shapes', wraps=quality.improve_projected_shapes))
            shapes = stack.enter_context(patch.object(manhattan, 'improve_quad_shapes', wraps=manhattan.improve_quad_shapes))
            bm, result = pipeline.fair_hole_patch(obj, bm, loop, edges,
                                                 dict(method='ORGANIC', residual=0.), 'AUTO')
            counts.append((capture.call_count, project.call_count, shapes.call_count))
        timings.append(time.perf_counter()-start)
        # Neither plan reuse nor private probe adoption replaces original faces.
        assert any(f.is_valid for f in original_faces)
        assert result['fill_style'] == 'GRID'
        bm.verts.index_update()
        outputs.append(([(tuple(v.co), v.hide) for v in bm.verts],
                        [tuple(v.index for v in f.verts) for f in bm.faces],
                        result['side_counts'], result['surface_continuity']))
    assert outputs[0] == outputs[1], name
    assert counts[0][:2] == (2, 2) and counts[1][:2] == (1, 1), counts
    assert counts[1][2] <= counts[0][2] // 2
    if name.endswith('notched_grid_hole'):
        assert counts[1][2] < counts[0][2] // 2  # Also deduplicate candidates within each sweep.
    print('Fairing timings uncached/cached:', name, timings, 'solve counts:', counts)

# Reuse is local, numeric, and guarded by source geometry/settings. Replay on
# a copied Refined Edge must work after the probe BMesh has been freed.
obj, bm, loop, edges = fixture('suzanne_triangle_grid_hole')
refined = prepare_grid_steps(bm, loop, edges, obj.matrix_world, 'REFINE', refinement_passes=1)
loop, edges = refined['boundary_verts'], refined['boundary_edges']
bm.verts.index_update()
bm.edges.index_update()
vertex_ids, edge_ids = [v.index for v in loop], [e.index for e in edges]
hypothesis, cache = dict(method='ORGANIC', residual=0.), {}
for variant in ('capture', 'replay', 'partial', 'source', 'matrix', 'settings', 'protected'):
    trial = bm.copy()
    trial.verts.ensure_lookup_table()
    trial.edges.ensure_lookup_table()
    rim, border = [trial.verts[i] for i in vertex_ids], [trial.edges[i] for i in edge_ids]
    matrix = obj.matrix_world.copy()
    options = dict(relaxation_iterations=3)
    if variant == 'source':
        neighbor = next(e.other_vert(rim[0]) for e in rim[0].link_edges if e.other_vert(rim[0]) not in rim)
        neighbor.co.z += .001
    elif variant == 'matrix':
        matrix.translation.x += .01
    elif variant == 'settings':
        options['relaxation_iterations'] = 4
    elif variant == 'protected':
        options['protected_vertices'] = {rim[0]}
    elif variant == 'partial':
        options['fill_rows'] = 2
    # Each invalidation is compared to the original cached source, not the
    # preceding altered variant. There are no live mesh handles in the plan.
    if variant == 'replay':
        baseline = dict(cache)
    elif variant != 'capture':
        cache = dict(baseline)
    with patch.object(manhattan, 'grid_layout', wraps=manhattan.grid_layout) as planner:
        result = fill_refined_rows(trial, rim, border, matrix, hypothesis=hypothesis,
                                   _plan_cache=cache, **options)
    assert planner.call_count == (0 if variant in {'replay', 'partial'} else 1), variant
    if variant == 'capture':
        expected = np.asarray([v.co for v in result['verts']])
    elif variant == 'replay':
        assert np.array_equal(expected, np.asarray([v.co for v in result['verts']]))
    elif variant == 'partial':
        assert all(np.min(np.linalg.norm(expected-np.asarray(v.co), axis=1)) < 1e-6 for v in result['verts'])
    trial.free()

# Numeric snapshots must notice edits to an existing hypothesis dictionary,
# not merely a changed object address. Non-geometric fit diagnostics need not
# invalidate a projection plan.
for hypothesis, mutate in (
        (dict(method='SPHERE', center=Vector(), radius=1., residual=0.),
         lambda h: h.update(radius=2.)),
        (dict(method='CONIC', fit=dict(center=Vector(), axis=Vector((0., 0., 1.)), radius=1., slope=.1)),
         lambda h: h['fit'].update(slope=.2)),
        (dict(method='PLANES', planes=[dict(point=Vector(), normal=Vector((0., 0., 1.)))]),
         lambda h: h['planes'][0].update(point=Vector((0., 0., .1)))),
):
    key = _projection_key(hypothesis)
    hypothesis['residual'] = .9
    assert key == _projection_key(hypothesis)
    mutate(hypothesis)
    assert key != _projection_key(hypothesis)

print('HOLEY_TOOLS_FAIRING_PERFORMANCE_OK')
