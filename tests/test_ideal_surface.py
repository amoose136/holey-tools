"""Run with Blender --background --factory-startup --python tests/test_ideal_surface.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ideal_surface import _basis, fit_curvature_surface


def height(p):
    x, y = np.asarray(p).T
    return .1 + .05 * x + .03 * y + .1 * x * x + .02 * x * y + .2 * y * y


points = np.asarray([(x, y) for y in np.linspace(-2., 2., 9) for x in np.linspace(-2., 2., 9)
                     if abs(x) >= 1. or abs(y) >= 1.])
boundary = [i for i, p in enumerate(points) if max(abs(p)) == 1.]
neighborhoods = [[j for j, q in enumerate(points) if np.linalg.norm(q - points[i]) <= 1.5]
                 for i in boundary]
queries = np.asarray([[-.7, -.4], [.1, .3], [.6, -.2]])
for scale in (.01, 1., 100.):
    surface = fit_curvature_surface(points * scale, height(points) * scale, boundary, neighborhoods)
    assert surface is not None
    assert np.max(np.abs(surface(points[boundary] * scale) - height(points[boundary]) * scale)) < 1e-7 * scale
    assert np.max(np.abs(surface(queries * scale) - height(queries) * scale)) < 1e-6 * scale
    assert np.max(np.abs(surface(queries * scale, (0,)) - (.05 + .2 * queries[:, 0] + .02 * queries[:, 1]))) < 1e-6
    assert np.max(np.abs(surface(queries * scale, (1,)) - (.03 + .02 * queries[:, 0] + .4 * queries[:, 1]))) < 1e-6
    for derivative, expected in (((0, 0), .2), ((0, 1), .02), ((1, 1), .4)):
        assert np.max(np.abs(surface(queries * scale, derivative) - expected / scale)) < 1e-6 / scale

# Check analytic radial/polynomial derivatives, including coincident centers.
centers = np.asarray([[0., 0.], [.4, -.7]])
query = np.asarray([[.2, .3], [0., 0.]])
epsilon = 1e-5
for axis in (0, 1):
    offset = np.eye(2)[axis] * epsilon
    first = (_basis(query + offset, centers) - _basis(query - offset, centers)) / (2 * epsilon)
    assert np.max(np.abs(first - _basis(query, centers, (axis,)))) < 1e-7
    for second_axis in (0, 1):
        second = (_basis(query + offset, centers, (second_axis,))
                  - _basis(query - offset, centers, (second_axis,))) / (2 * epsilon)
        assert np.max(np.abs(second - _basis(query, centers, (axis, second_axis)))) < 1e-7
        assert np.isfinite(_basis(centers, centers, (axis, second_axis))).all()

# Noisy samples still have exact rim heights and continuous second derivatives.
noisy = height(points) + np.random.default_rng(42).normal(0., .002, len(points))
surface = fit_curvature_surface(points, noisy, boundary, neighborhoods)
assert surface is not None
assert np.max(np.abs(surface(points[boundary]) - noisy[boundary])) < 1e-7
for p in points[boundary]:
    for derivative in ((0,), (1,), (0, 0), (0, 1), (1, 1)):
        assert np.max(np.abs(surface(p + (epsilon, 0), derivative)
                             - surface(p - (epsilon, 0), derivative))) < .002
assert fit_curvature_surface([(i, 0.) for i in range(8)], np.zeros(8), [0, 1, 2], [list(range(8))] * 3) is None
print('HOLEY_TOOLS_IDEAL_SURFACE_OK', flush=True)
