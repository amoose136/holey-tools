"""A small bore cut across spherical grid cells must retain a usable rim."""

import json
from pathlib import Path
import sys

from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fairing_pipeline import fair_hole_patch
from src.grid_patcher import _slide_boundary, prepare_grid_steps
from tests.blender_support import edit_mesh


data = json.loads(Path(__file__).with_name('fixtures').joinpath('small_sphere_grid_bore.json').read_text())


def fixture(seed=0, reverse=False):
    obj, bm = edit_mesh('Small spherical grid hole', data['vertices'], data['faces'])
    obj.matrix_world = Matrix(data['matrix'])
    bm.normal_update()
    loop = [bm.verts[index] for index in data['boundary']]
    loop = loop[seed:] + loop[:seed]
    if reverse:
        loop.reverse()
    edges = [bm.edges.get((vert, loop[(i + 1) % len(loop)])) for i, vert in enumerate(loop)]
    return obj, bm, loop, edges


# Six-sided clipped cells cannot supply the finished-quad reference used by a
# triangle slide. The old turn-only objective nearly collapsed both endpoints
# of their exposed edge, then post-mirror refinement produced a 39-degree fold.
obj, bm, loop, edges = fixture()
prepared = prepare_grid_steps(bm, loop, edges, obj.matrix_world, 'PENTAGONS')
loop, edges = prepared['boundary_verts'], prepared['boundary_edges']
ambiguous = {vert: vert.co.copy() for vert in loop
             if any(len(face.verts) == 3 for face in vert.link_faces)
             and any(len(face.verts) > 4 for face in vert.link_faces)}
assert len(ambiguous) == 4
_slide_boundary(bm, loop, edges, obj.matrix_world)
assert all(vert.co == point for vert, point in ambiguous.items())

sphere = dict(method='SPHERE', center=Vector((-1.0472322702, 10.498541832, -2.880855799)),
              radius=1.0675839503, residual=.00303712965)
for seed in (0, 13, 29):
    for reverse in (False, True):
        for style in ('AUTO', 'GRID'):
            obj, bm, loop, edges = fixture(seed, reverse)
            outside = {edge for edge in bm.edges if edge.is_boundary and edge not in edges}
            outside_points = {vert: vert.co.copy() for edge in outside for vert in edge.verts}
            bm, result = fair_hole_patch(obj, bm, loop, edges, sphere, style)
            assert result['method'] == 'GRID_ROWS', result.get('grid_fallback_reason')
            assert result['fill_style'] == 'GRID'
            assert result['grid_size'] == (2, 1) and result['side_counts'] == (2, 1, 2, 1)
            assert len(result['faces']) == 2 and all(len(face.verts) == 4 for face in result['faces'])
            assert result['rows_filled'] == result['rows_total'] == 1
            assert result['grid_cut_count'] == result['transition_triangles'] == 0
            assert all(face.calc_area() > 1e-14 for face in bm.faces)
            assert all(edge.is_valid and edge.is_boundary for edge in outside)
            assert all(vert.is_valid and vert.co == point for vert, point in outside_points.items())
            assert {edge for edge in bm.edges if edge.is_boundary} == outside

print('HOLEY_TOOLS_SMALL_GRID_HOLE_OK')
