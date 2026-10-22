"""A notched Suzanne rim needs a perpendicular Manhattan row sweep."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.manhattan as manhattan
from src.fairing import FairingError, _organic_grid_frame
from src.fairing_pipeline import fair_hole_patch
from src.grid_patcher import prepare_grid_steps
from src.grid_quality import shape_loss
from src.visible_geometry import visible_valence
from tests.blender_support import edit_mesh


data = json.loads(Path(__file__).with_name('fixtures').joinpath('suzanne_notched_grid_hole.json').read_text())


def fixture(seed=0, reverse=False):
    obj, bm = edit_mesh('Notched Suzanne opening', data['vertices'], data['faces'])
    obj.matrix_world = Matrix(data['matrix'])
    bm.normal_update()
    loop = [bm.verts[index] for index in data['boundary']]
    loop = loop[seed:] + loop[:seed]
    if reverse:
        loop.reverse()
    edges = [bm.edges.get((vert, loop[(i + 1) % len(loop)])) for i, vert in enumerate(loop)]
    return obj, bm, loop, edges


obj, bm, loop, edges = fixture()
prepared = prepare_grid_steps(bm, loop, edges, obj.matrix_world, 'REFINE', refinement_passes=1)
loop, edges = prepared['boundary_verts'], prepared['boundary_edges']
_center, _axis, _x, _y, points, _values = _organic_grid_frame(loop, edges, obj.matrix_world)
world = [obj.matrix_world @ v.co for v in loop]
valences = [visible_valence(v) for v in loop]
analysis = manhattan.analyze_boundary(points, valences, world)
assert analysis['side_counts'] == (10, 6, 4, 7)
assert prepared['notch_caps'] == 1
try:
    manhattan._grid_layout(points, analysis, 20, .5)
except FairingError as error:
    assert 'one connected interval per row' in str(error)
else:
    raise AssertionError('The preferred sweep must encounter the two-interval notch')

# The old first-valid layout was topologically sound but had severe slivers.
# Quality-aware planning must beat it without using boundary subdivisions.
perpendicular = dict(analysis)
for key in ('side_edges', 'side_counts'):
    perpendicular[key] = analysis[key][1:] + analysis[key][:1]
perpendicular['directions'] = [(d - 1) % 4 for d in analysis['directions']]
with patch.object(manhattan, 'improve_quad_shapes', return_value=0):
    first_valid = manhattan._grid_layout_attempt(points, perpendicular, 20, .5, hold_mask=0)
selected = manhattan.grid_layout(points, valences, world)
assert shape_loss(selected) < .6 * shape_loss(first_valid)
assert len(selected['splits']) == len(points) % 2 and selected['mutations'] > 0

def maximum_edge_ratio(layout):
    coords = layout['coords']
    return max(max(lengths) / min(lengths)
               for row in layout['row_faces'] for face in row
               for lengths in [[(coords[b] - coords[a]).length
                                for a, b in zip(face, face[1:] + face[:1])]])

# The notch cap already removes the old worst sliver from both layouts;
# shape selection must still at least halve the remaining worst aspect ratio.
assert maximum_edge_ratio(selected) < .5 * maximum_edge_ratio(first_valid)

# Retry is seed-independent and does not modify caller-owned chart/analysis.
for seed in range(len(points)):
    p = points[seed:] + points[:seed]
    v = valences[seed:] + valences[:seed]
    w = world[seed:] + world[:seed]
    source = [point.copy() for point in p]
    result = manhattan.grid_layout(p, v, w)
    assert p == source
    assert result['sweep_rotation'] == 1
    assert result['side_counts'] == (6, 4, 7, 10)
    assert result['grid_size'] == (7, 10)
    assert len(result['splits']) == len(points) % 2
    assert all(len(face) == 4 for row in result['row_faces'] for face in row)
assert analysis == manhattan.analyze_boundary(points, valences, world)

# Both failures preserve the preferred diagnostic, with exactly two bounded
# attempts. Invalid boundary analysis must fail before any sweep is attempted.
with patch.object(manhattan, '_grid_layout', side_effect=[FairingError('preferred'), FairingError('alternate')]) as planner:
    try:
        manhattan.grid_layout(points, valences, world)
    except FairingError as error:
        assert str(error) == 'preferred'
    else:
        raise AssertionError('Two invalid sweeps must not produce a patch')
    assert planner.call_count == 2
with patch.object(manhattan, '_grid_layout') as planner:
    try:
        manhattan.grid_layout([], [])
    except FairingError:
        pass
    else:
        raise AssertionError('Invalid boundary must remain invalid')
    planner.assert_not_called()

# Exercise Auto's probe and replay as well as explicit Grid, including the
# original boundary preparation and actual frozen-surface mesh projection.
for seed in (0, 13, 29):
    for reverse in (False, True):
        for topology in ('AUTO', 'GRID'):
            obj, bm, loop, edges = fixture(seed, reverse)
            outside = {e for e in bm.edges if e.is_boundary and e not in edges}
            outside_points = {v: v.co.copy() for e in outside for v in e.verts}
            bm, result = fair_hole_patch(obj, bm, loop, edges,
                                        dict(method='ORGANIC', residual=0.), topology)
            assert result['method'] == 'GRID_ROWS'
            assert result['fill_style'] == 'GRID' and result['surface_method'] == 'ORGANIC'
            assert result['grid_size'] == (7, 10)
            assert result['rows_filled'] == result['rows_total'] == 10
            assert result['notch_caps'] == 1
            assert result['grid_cut_count'] == 1  # Odd rim after redundant-corner cleanup.
            assert result['corner_transitions'] > 0
            assert result['transition_triangles'] == 0
            assert all(len(f.verts) == 4 and f.calc_area() > 1e-14 for f in result['faces'])
            assert all(v.is_valid and v.co == co for v, co in outside_points.items())
            assert {e for e in bm.edges if e.is_boundary} == outside
            assert all(len(e.link_faces) in {1, 2} for e in bm.edges)

print('HOLEY_TOOLS_GRID_SWEEP_OK')
