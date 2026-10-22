"""Quad shape optimization must improve slivers without changing the rim."""

from copy import deepcopy
import math
from pathlib import Path
import sys

import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.grid_quality import improve_projected_shapes, improve_quad_shapes, shape_loss


def fixture(scale=1.):
    # Seven original vertices, one explicitly fixed parity cut, one nearly
    # collapsed interior point. Four quads already cover the complete disk.
    coordinates = [(0, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1),
                   (1, 0), (1, .015)]
    return dict(coords=[Vector(p) * scale for p in coordinates],
                row_faces=[[[0, 7, 8, 6], [7, 1, 2, 8]], [[8, 2, 3, 4], [6, 8, 4, 5]]],
                splits=[(0, .5, 7)])


for scale in (.001, 1., 1000.):
    layout = fixture(scale)
    original = deepcopy(layout)
    before = shape_loss(layout)
    passes = improve_quad_shapes(layout, 7)
    assert 0 < passes <= 20
    assert shape_loss(layout) < before * .1
    assert (layout['coords'][8] / scale - Vector((1., 1.))).length < .1
    assert layout['coords'][:8] == original['coords'][:8]
    assert layout['row_faces'] == original['row_faces']
    assert layout['splits'] == original['splits']
    assert all((layout['coords'][face[j]] - layout['coords'][face[j-1]]).cross(
                   layout['coords'][face[(j+1)%4]] - layout['coords'][face[j]]) > 0.
               for row in layout['row_faces'] for face in row for j in range(4))
    # Starting almost on the barrier needs more than twenty damped steps to
    # converge fully, but another pass must approach the exact square layout.
    improve_quad_shapes(layout, 7)
    assert (layout['coords'][8] / scale - Vector((1., 1.))).length < 1e-4

for options in (dict(iterations=0), dict(strength=0.)):
    layout = fixture()
    original = deepcopy(layout)
    assert improve_quad_shapes(layout, 7, **options) == 0
    assert layout == original

square = dict(coords=[Vector(p) for p in ((0,0),(1,0),(1,1),(0,1))],
              row_faces=[[[0,1,2,3]]], splits=[])
assert abs(shape_loss(square) - 1.) < 1e-7
assert improve_quad_shapes(square, 4) == 0
folded = deepcopy(square)
folded['coords'][2] = Vector((0., -1.))
assert math.isinf(shape_loss(folded))


def projected_loss(layout, project):
    world = project(np.asarray(layout['coords']))
    total = 0.
    for row in layout['row_faces']:
        for face in row:
            points = world[face]
            u, v = points - np.roll(points, 1, axis=0), np.roll(points, -1, axis=0) - points
            normals = np.cross(u, v)
            area = np.linalg.norm(normals, axis=1)
            total += np.sum(np.sum(u*u + v*v, axis=1) / (2.*area))
            normals /= area[:, None]
            total += .5 * np.sum(1. - np.sum(normals * np.roll(normals, -1, axis=0), axis=1))
    return total


# Perfect squares in the flat chart become distorted cells on a steep fit.
# More iterations of 2D relaxation cannot fix their projected shape.
for scale in (.001, 1., 1000.):
    layout = fixture(scale)
    layout['coords'][8] = Vector((1., 1.)) * scale
    original = deepcopy(layout)
    def project(queries):
        q = np.asarray(queries) / scale
        z = .7 * np.exp(-((q[:, 0]-.9)**2 + (q[:, 1]-.8)**2)/.1)
        return np.column_stack((q, z)) * scale
    before = projected_loss(layout, project)
    passes = improve_projected_shapes(layout, 7, project, iterations=100, strength=1.)
    assert 0 < passes <= 100
    assert projected_loss(layout, project) < .99 * before, (scale, before, projected_loss(layout, project))
    assert (layout['coords'][8] - original['coords'][8]).length > .05 * scale
    assert layout['coords'][:8] == original['coords'][:8]
    assert layout['row_faces'] == original['row_faces']
    assert layout['splits'] == original['splits']
    assert math.isfinite(shape_loss(layout))
    for options in (dict(iterations=0), dict(strength=0.)):
        disabled = deepcopy(original)
        assert improve_projected_shapes(disabled, 7, project, **options) == 0
        assert disabled == original

print('HOLEY_TOOLS_GRID_QUALITY_OK')
