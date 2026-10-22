"""Refined border faces supply frozen tangent and curvature constraints."""

import sys
from pathlib import Path

import bmesh
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.face_curvature import capture_face_samples, estimate_normal_jet
from src.ideal_surface import capture_curvature_surface, fit_curvature_surface


# Recover all five derivatives from face-normal variation, including the
# symmetric mixed derivative, regardless of scale or stencil density.
stencil = np.asarray([(-1., -1.), (0., -1.), (1., -1.), (-1., 0.), (1., 0.),
                      (-1., 1.), (0., 1.), (1., 1.)])
hessian = np.asarray(((.2, .04), (.04, .3)))
for scale in (.01, 1., 100.):
    for density in (.001, 1.):
        points = stencil * density * scale
        gradients = (.05, -.03) + points @ (hessian / scale)
        result = estimate_normal_jet(points, gradients, np.ones(8), np.zeros(2), list(range(8)))
        assert result is not None
        jet, radius = result
        expected = np.asarray((.05, -.03, .2 / scale, .04 / scale, .3 / scale))
        assert np.max(np.abs(jet - expected)) < 1e-8 / scale
        assert abs(radius - np.sqrt(2.) * density * scale) < 1e-12 * scale
narrow = np.column_stack((np.linspace(-1., 1., 8), np.zeros(8)))
assert estimate_normal_jet(narrow, np.ones((8, 2)), np.ones(8), np.zeros(2), list(range(8))) is None
assert estimate_normal_jet(stencil, np.ones((8, 2)), np.zeros(8), np.zeros(2), list(range(8))) is None


def ring(height):
    bm = bmesh.new()
    verts = {(x, y): bm.verts.new((x, y, height(x, y)))
             for x in range(-3, 4) for y in range(-3, 4)}
    for x in range(-3, 3):
        for y in range(-3, 3):
            if -1 <= x < 1 and -1 <= y < 1:
                continue
            bm.faces.new([verts[x, y], verts[x+1, y], verts[x+1, y+1], verts[x, y+1]])
    boundary = [verts[p] for p in ((-1, -1), (0, -1), (1, -1), (1, 0),
                                  (1, 1), (0, 1), (-1, 1), (-1, 0))]
    bm.normal_update()
    return bm, boundary, verts


origin, x_axis, y_axis, axis = Vector(), Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
identity = Matrix.Identity(4)

# World-space planes, areas, and normals must survive rotation, non-uniform
# scale, shear, and reflection (not the object rotation alone).
for reflected in (False, True):
    bm, boundary, _ = ring(lambda x, y: .12 * x - .23 * y)
    shear = Matrix.Identity(4)
    shear[0][1] = .17
    transform = (Matrix.Translation((3., -4., 2.)) @ Matrix.Rotation(.3, 4, 'X') @ shear
                 @ Matrix.Diagonal(((-2. if reflected else 2.), .7, 1.4, 1.)))
    samples = capture_face_samples(boundary, transform, transform @ origin, x_axis, y_axis, axis)
    assert samples is not None and np.count_nonzero(samples['primary']) == 8
    normal = transform.to_3x3().inverted().transposed() @ Vector((-.12, .23, 1.))
    expected = np.asarray((-normal.x / normal.z, -normal.y / normal.z))
    assert np.max(np.abs(samples['gradients'] - expected)) < 1e-6
    assert np.all(samples['areas'] > 0.)
    reversed_samples = capture_face_samples(list(reversed(boundary)), transform, transform @ origin,
                                             x_axis, y_axis, axis)
    assert np.allclose(samples['points'], reversed_samples['points'])
    assert np.array_equal(samples['primary'], reversed_samples['primary'])
    bm.free()

# The normals of planar quad samples reproduce a quadratic surface's slope
# at each face centroid and its curvature across adjacent faces.
bm, boundary, verts = ring(lambda x, y: .03*x*x + .07*y*y + .04*x - .02*y)
samples = capture_face_samples(boundary, identity, origin, x_axis, y_axis, axis)
assert samples is not None and samples['primary'].sum() == 8
p = samples['points']
expected = np.column_stack((.06*p[:, 0] + .04, .14*p[:, 1] - .02))
assert np.max(np.abs(samples['gradients'] - expected)) < 1e-6
corner = np.flatnonzero(np.linalg.norm(p - (1.5, 1.5), axis=1) < 1e-6)
assert len(corner) == 1 and not samples['primary'][corner[0]]
for vertex, support in zip(boundary, samples['neighborhoods']):
    result = estimate_normal_jet(p, samples['gradients'], samples['areas'],
                                 np.asarray(vertex.co.xy), support)
    assert result is not None
    assert np.max(np.abs(result[0][2:] - (.06, 0., .14))) < 1e-6

# A dissolvable vertex on the border must not alter a face's area centroid,
# normal, primary classification, or weight.
edge = next(e for e in boundary[0].link_edges if e.other_vert(boundary[0]) is boundary[1])
_, new = bmesh.utils.edge_split(edge, boundary[0], .33)
boundary.insert(1, new)
split = capture_face_samples(boundary, identity, origin, x_axis, y_axis, axis)
for key in ('points', 'gradients', 'areas', 'confidence', 'primary'):
    assert np.allclose(split[key], samples[key], atol=1e-6), key

# A hidden incident face is excluded before collecting or evaluating its
# geometry, including its deliberately pathological remote vertex.
queries = np.asarray([[-.3, -.2], [.1, .4], [.5, -.1]])
unmasked_surface = capture_curvature_surface(boundary, identity, origin, x_axis, y_axis, axis)
assert unmasked_surface is not None
remote = bm.verts.new((20., 30., 100.))
hidden = bm.faces.new([boundary[0], boundary[-1], remote])
hidden.hide = True
masked = capture_face_samples(boundary, identity, origin, x_axis, y_axis, axis)
for key in ('points', 'gradients', 'areas', 'confidence', 'primary'):
    assert np.array_equal(masked[key], split[key]), key

surface = capture_curvature_surface(boundary, identity, origin, x_axis, y_axis, axis)
assert surface is not None
assert np.allclose(surface(queries), unmasked_surface(queries), atol=1e-10)
saved = surface(queries).copy()
remote.co.z = -1000.
verts[-3, -3].co.z += 20.
assert np.array_equal(surface(queries), saved)  # No live BMesh dependency.
bm.free()

# Tangents are independent fitting evidence, not ignored metadata. With the
# same height samples, changing the supplied face normals must affect the fit
# and reduce disagreement with those normals, while keeping the rim fixed.
bm, boundary, _ = ring(lambda *_: 0.)
samples = capture_face_samples(boundary, identity, origin, x_axis, y_axis, axis)
points = np.asarray([(x, y) for x in np.arange(-2., 2.1, .5) for y in np.arange(-2., 2.1, .5)
                     if max(abs(x), abs(y)) >= 1.])
rim = [i for i, p in enumerate(points) if max(abs(p)) == 1.]
neighborhoods = [[j for j, q in enumerate(points) if np.linalg.norm(q - points[i]) <= 1.5]
                 for i in rim]
# Pure-array callers can supply the adjacent normal stencils directly.
samples['neighborhoods'] = [list(range(len(samples['points']))) for _ in rim]
samples['gradients'] = -.1 * samples['points']
flat = fit_curvature_surface(points, np.zeros(len(points)), rim, neighborhoods)
guided = fit_curvature_surface(points, np.zeros(len(points)), rim, neighborhoods, face_samples=samples)
assert flat is not None and guided is not None
primary = samples['primary']
q, target = samples['points'][primary], samples['gradients'][primary]
gradients = np.column_stack((guided(q, (0,)), guided(q, (1,))))
assert np.mean((gradients-target)**2) < np.mean(target**2)
assert np.max(np.abs(guided(points[rim]))) < 1e-7
assert np.max(np.abs(guided(queries) - flat(queries))) > 1e-5
bm.free()

print('HOLEY_TOOLS_FACE_CURVATURE_OK')
