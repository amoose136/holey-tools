"""Run with Blender --background --factory-startup --python tests/test_ideal_surface.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ideal_surface import _basis, _curvature_jet, _interior_samples, fit_curvature_surface


def height(p):
    x, y = np.asarray(p).T
    return .1 + .05 * x + .03 * y + .1 * x * x + .02 * x * y + .2 * y * y


points = np.asarray([(x, y) for y in np.linspace(-2., 2., 9) for x in np.linspace(-2., 2., 9)
                     if abs(x) >= 1. or abs(y) >= 1.])
boundary = [i for i, p in enumerate(points) if max(abs(p)) == 1.]
neighborhoods = [[j for j, q in enumerate(points) if np.linalg.norm(q - points[i]) <= 1.5]
                 for i in boundary]
queries = np.asarray([[-.7, -.4], [.1, .3], [.6, -.2]])

# Interior guidance must not place anchors across a completed corner outside
# a concave opening, and cannot depend on the boundary's walking direction.
notch = np.asarray([(0., 0.), (2., 0.), (2., 1.), (1., 1.), (1., 2.), (0., 2.)])
inside = _interior_samples(notch, resolution=12)
assert len(inside) == 108 and np.all((inside[:, 0] < 1.) | (inside[:, 1] < 1.))
assert np.array_equal(inside, _interior_samples(notch[::-1], resolution=12))
assert np.max(np.abs(_interior_samples(notch * .01, resolution=12) - inside * .01)) < 1e-12

# Local conditioning must depend on stencil shape, not its size relative to
# the complete hole. Exact quadratic curvature survives dense neighborhoods.
stencil = np.asarray([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
identity = next(i for i, p in enumerate(stencil) if not p.any())
for density in (.001, .1, 1.):
    local = stencil * density
    jet = _curvature_jet(local, height(local), identity, list(range(len(local))))
    assert jet is not None
    assert np.max(np.abs(jet - (.05, .03, .2, .02, .4))) < 1e-8

# Near-collinear stencils have enough samples and algebraic rank, but cannot
# estimate transverse curvature reliably. Do not pass their noise to the RBF.
narrow = np.column_stack((np.linspace(-1., 1., 9),
                          np.asarray([.002, -.003, .001, .004, 0., -.002, .003, -.004, .001])))
for angle in (0., .7):
    rotation = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    local = narrow @ rotation.T
    noisy_local = height(local) + np.random.default_rng(23).normal(0., .002, len(local))
    assert _curvature_jet(local, noisy_local, 4, list(range(len(local)))) is None

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
# Resolve the rounded thin-plate core when checking its Hessian at a center.
epsilon = 1e-7
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
epsilon = 1e-5
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
