# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smooth height-field continuation with estimated boundary curvature jets."""

import numpy as np


# The chart is normalized to unit extent. Round the thin-plate singularity in
# a small, scale-independent core so its Hessian is finite at source points.
_CORE_SQUARED = 1e-6


class HeightSurface:
    """Frozen chart evaluator shared by Circle and Grid."""

    def __init__(self, center, x, y, axis, field, continuity):
        self.center, self.x, self.y, self.axis = [np.array(v, dtype=float) for v in (center, x, y, axis)]
        self.field, self.continuity = field, continuity

    def evaluate_chart(self, queries):
        queries = np.atleast_2d(queries)
        return (self.center + queries @ np.asarray((self.x, self.y))
                + self.field(queries)[:, None]*self.axis)


def capture_organic_surface(boundary, matrix_world, center, x, y, axis, *,
                            match_curvature=True, planned_splits=()):
    """Capture once, before output topology exists; never fit new patch faces.

    A grazing or folded support plane makes scalar height interpolation
    unreliable. Use actual 3D source connectivity for that case rather than
    merging incompatible projected samples or clipping the fitted heights.
    """
    from .intrinsic_surface import capture_bending_surface, folded_chart
    if folded_chart(boundary, matrix_world, axis):
        return capture_bending_surface(boundary, matrix_world, center, x, y)
    field = (capture_curvature_surface(boundary, matrix_world, center, x, y, axis,
                                       planned_splits=planned_splits) if match_curvature else None)
    continuity = 'C2_ESTIMATED'
    if field is None:
        from .fairing import _organic_depth_samples
        field = fit_thin_plate_surface(*_organic_depth_samples([], boundary, matrix_world, center, x, y, axis))
        continuity = 'THIN_PLATE'
    return HeightSurface(center, x, y, axis, field, continuity) if field is not None else None


def _curvature_jet(points, heights, identity, support):
    """Estimate derivatives only when the local chart constrains them.

    Normalize each neighborhood independently: a dense, small neighborhood
    must not appear deficient just because the complete hole is much larger.
    Narrow/near-collinear stencils can still interpolate heights while giving
    arbitrary second derivatives. Reject those instead of injecting them as
    curvature constraints into the global field.
    """
    support = [i for i in support if i != identity]
    if len(support) < 5:
        return None
    delta = points[support] - points[identity]
    radius = float(np.max(np.linalg.norm(delta, axis=1)))
    if radius <= 1e-12:
        return None
    local = delta / radius
    x, y = local.T
    design = np.column_stack((x, y, .5 * x * x, x * y, .5 * y * y))
    weights = 1. / np.maximum(np.linalg.norm(local, axis=1), .05)
    try:
        values, _residual, rank, singular = np.linalg.lstsq(
            design * weights[:, None], (heights[support] - heights[identity]) * weights,
            rcond=1e-7)
    except np.linalg.LinAlgError:
        return None
    # Bound amplification of sample/float32 noise, not curvature magnitude.
    # Reliable high-curvature quadratic surfaces remain exactly reproducible.
    if rank < 5 or singular[-1] < singular[0] * .01 or not np.isfinite(values).all():
        return None
    values[:2] /= radius
    values[2:] /= radius * radius
    return values


def _polynomial(query, derivative=None):
    query = np.atleast_2d(query)
    x, y = query.T
    zero, one = np.zeros(len(query)), np.ones(len(query))
    if derivative is None:
        poly = (one, x, y, .5 * x * x, x * y, .5 * y * y)
    elif len(derivative) == 1:
        i = derivative[0]
        poly = (zero, one, zero, x, y, zero) if i == 0 else (zero, zero, one, zero, x, y)
    else:
        i, j = derivative
        poly = (zero, zero, zero, one if i == j == 0 else zero,
                one if i != j else zero, one if i == j == 1 else zero)
    return np.column_stack(poly)


def _basis(query, centers, derivative=None):
    """Regularized thin-plate basis, including its exact first/second jets.

    The former r^5 continuation coupled distant rim constraints too strongly:
    even well-conditioned boundary jets could produce a broad interior dome.
    Thin-plate growth keeps the minimum-bending guide representable without
    those high-order excursions. Rounding r^2 log(r) at the origin makes the
    field C2 there as well, rather than merely smooth between source samples.
    """
    query = np.atleast_2d(query)
    d = query[:, None, :] - centers[None, :, :]
    squared = np.sum(d * d, axis=2) + _CORE_SQUARED
    logarithm = np.log(squared)
    if derivative is None:
        radial = .5 * squared * logarithm
    elif len(derivative) == 1:
        radial = (logarithm + 1.) * d[:, :, derivative[0]]
    else:
        i, j = derivative
        radial = 2. * d[:, :, i] * d[:, :, j] / squared
        if i == j:
            radial += logarithm + 1.
    return np.column_stack((radial, _polynomial(query, derivative)))


def _convex_hull(points):
    """Default domain for unordered boundary samples supplied to the fitter."""
    points = sorted(set(map(tuple, points)))
    def half(walk):
        hull = []
        for p in walk:
            while len(hull) >= 2:
                a, b = hull[-2:]
                if (b[0]-a[0])*(p[1]-b[1]) - (b[1]-a[1])*(p[0]-b[0]) > 0.:
                    break
                hull.pop()
            hull.append(p)
        return hull
    return np.asarray(half(points)[:-1] + half(reversed(points))[:-1])


def _interior_samples(polygon, resolution=15):
    """Deterministic cell-center quadrature, independent of infill topology."""
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3:
        return np.empty((0, 2))
    lower, upper = polygon.min(axis=0), polygon.max(axis=0)
    axes = [np.linspace(a, b, resolution, endpoint=False) + (b-a)/(2.*resolution)
            for a, b in zip(lower, upper)]
    x, y = np.meshgrid(*axes)
    queries = np.column_stack((x.ravel(), y.ravel()))
    inside = np.zeros(len(queries), dtype=bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if a[1] == b[1]:
            continue
        crosses = (a[1] > queries[:, 1]) != (b[1] > queries[:, 1])
        crossing_x = a[0] + (queries[:, 1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
        inside ^= crosses & (queries[:, 0] < crossing_x)
    return queries[inside]


def _bending_guide(points, heights, queries):
    """Minimum-bending-style shape prior, with exact quadratic reproduction.

    A quadratic trend is removed before the ordinary thin-plate solve. Its
    interior predictions guide the C2 field, not the final vertices directly:
    the resulting field still uses the C2 basis and exact rim constraints. Reference
    data comes only from the frozen source, never from newly filled faces.
    """
    polynomial = _polynomial(points)
    n = len(points)
    def kernel(distances):
        return distances * distances * np.log(np.maximum(distances, 1e-30))
    radial = kernel(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2))
    affine = polynomial[:, :3]
    system = np.block([[radial + np.eye(n)*1e-8, affine], [affine.T, np.zeros((3, 3))]])
    try:
        trend = np.linalg.lstsq(polynomial, heights, rcond=1e-10)[0]
        target = np.concatenate((heights - polynomial @ trend, np.zeros(3)))
        coefficients, _, rank, _ = np.linalg.lstsq(system, target, rcond=1e-10)
    except np.linalg.LinAlgError:
        return None
    if rank < n + 3 or not np.isfinite(coefficients).all():
        return None
    query_poly = _polynomial(queries)
    query_radial = kernel(np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=2))
    return query_radial @ coefficients[:n] + query_poly[:, :3] @ coefficients[n:] + query_poly @ trend


def fit_thin_plate_surface(points, heights):
    """Freeze the ordinary organic height fit for batched surface projection."""
    if points is None or len(points) < 3:
        return None
    points, heights = np.asarray(points, dtype=float), np.asarray(heights, dtype=float)
    if not np.isfinite(points).all() or not np.isfinite(heights).all():
        return None
    center = points.mean(axis=0)
    scale = float(np.max(np.linalg.norm(points - center, axis=1)))
    if scale <= 1e-12:
        return None
    normalized = (points - center) / scale
    polynomial = np.column_stack((np.ones(len(points)), normalized))
    if np.linalg.matrix_rank(polynomial) < 3:
        return None
    def kernel(distances):
        return distances**2 * np.log(np.maximum(distances, 1e-30))
    radial = kernel(np.linalg.norm(normalized[:, None] - normalized[None], axis=2))
    system = np.block([[radial + np.eye(len(points))*1e-8, polynomial],
                       [polynomial.T, np.zeros((3, 3))]])
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(
            system, np.concatenate((heights, np.zeros(3))), rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < len(points) + 3 or not np.isfinite(coefficients).all():
        return None
    def evaluate(queries):
        queries = (np.atleast_2d(queries) - center) / scale
        distances = np.linalg.norm(queries[:, None] - normalized[None], axis=2)
        return (kernel(distances) @ coefficients[:-3]
                + np.column_stack((np.ones(len(queries)), queries)) @ coefficients[-3:])
    return evaluate


def fit_curvature_surface(points, heights, boundary_indices, neighborhoods, *, boundary_polygon=None,
                          face_samples=None):
    """Fit exact rim heights and soft exterior slope/curvature constraints.

    Refined border-face normals and their local variation supply the primary
    tangent/curvature evidence; vertex-height quadratics are a fallback. A
    global C2 field balances these estimates with exterior
    heights and a frozen thin-plate interior shape prior; arbitrary faceted or
    noisy inputs need not admit an exact C2 match. Domain samples are not mesh
    subdivisions and are independent of the eventual patch topology.
    None requests the existing thin-plate fallback on a deficient chart.
    """
    points, heights = np.asarray(points, dtype=float), np.asarray(heights, dtype=float)
    if len(points) < 6 or not np.isfinite(points).all() or not np.isfinite(heights).all():
        return None
    center = points.mean(axis=0)
    scale = float(np.max(np.linalg.norm(points - center, axis=1)))
    if scale <= 1e-12:
        return None
    normalized = (points - center) / scale
    n = len(points)
    value_basis = _basis(normalized, normalized)
    # Polynomial moments avoid an arbitrary affine/quadratic radial drift.
    moments = np.column_stack((value_basis[:, n:].T, np.zeros((6, 6))))
    exact = np.vstack((value_basis[boundary_indices], moments))
    exact_rhs = np.concatenate((heights[boundary_indices], np.zeros(6)))
    jets, jet_points, jet_radii = [], [], []
    if face_samples is not None:
        from .face_curvature import estimate_normal_jet
        face_points = (face_samples['points'] - center) / scale
        face_gradients = face_samples['gradients'] * scale
        face_weights = face_samples['areas'] * face_samples['confidence']
    for slot, (identity, support) in enumerate(zip(boundary_indices, neighborhoods)):
        face_jet = (estimate_normal_jet(face_points, face_gradients, face_weights, normalized[identity],
                                       face_samples['neighborhoods'][slot])
                    if face_samples is not None else None)
        if face_jet is not None:
            values, radius = face_jet
        else:
            values = _curvature_jet(normalized, heights, identity, support)
            if values is None:
                continue
            radius = np.max(np.linalg.norm(normalized[support] - normalized[identity], axis=1))
        jets.append(values)
        jet_points.append(normalized[identity])
        jet_radii.append(radius)
    if len(jets) < 3:
        return None
    jets = np.asarray(jets)
    polygon = (_convex_hull(normalized[boundary_indices]) if boundary_polygon is None
               else (np.asarray(boundary_polygon) - center) / scale)
    guide_points = _interior_samples(polygon)
    if not len(guide_points):
        return None
    guide_heights = _bending_guide(normalized, heights, guide_points)
    if guide_heights is None or not np.isfinite(guide_heights).all():
        return None
    # Convert derivative errors to height errors over their actual stencil,
    # not the whole opening: tiny neighborhoods must not dominate the patch.
    rows, rhs = [value_basis], [heights]
    for column, derivative in enumerate(((0,), (1,), (0, 0), (0, 1), (1, 1))):
        weight = np.asarray(jet_radii) ** len(derivative) * (.2 if column < 2 else .05)
        rows.append(_basis(jet_points, normalized, derivative) * weight[:, None])
        rhs.append(jets[:, column] * weight)
    if face_samples is not None:
        primary = face_samples['primary']
        # Direct face-plane tangents are sampled at area centroids, not moved
        # rim vertices. These observed planes take priority over extrapolated
        # jets; area weighting avoids bias from support tessellation.
        weight = 4. * np.sqrt(face_weights[primary]) / scale
        for column in (0, 1):
            rows.append(_basis(face_points[primary], normalized, (column,)) * weight[:, None])
            rhs.append(face_gradients[primary, column] * weight)
    # Boundary jets alone leave broad interior bulges unconstrained. Anchor
    # the missing domain to a fair shape while fitting curvature near the rim.
    weight = 6. * np.sqrt(n / len(guide_points))
    rows.append(_basis(guide_points, normalized) * weight)
    rhs.append(guide_heights * weight)
    rows.append(np.column_stack((np.eye(n) * 1e-5, np.zeros((n, 6)))))
    rhs.append(np.zeros(n))
    system, target = np.vstack(rows), np.concatenate(rhs)
    try:
        u, singular, vh = np.linalg.svd(exact, full_matrices=True)
        rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
        particular = vh[:rank].T @ ((u[:, :rank].T @ exact_rhs) / singular[:rank])
        if np.max(np.abs(exact @ particular - exact_rhs)) > max(np.ptp(heights) * 1e-6, 1e-8):
            return None
        nullspace = vh[rank:].T
        correction = np.linalg.lstsq(system @ nullspace, target - system @ particular, rcond=1e-10)[0]
        coefficients = particular + nullspace @ correction
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(coefficients).all():
        return None

    def evaluate(query, derivative=None):
        values = _basis((np.atleast_2d(query) - center) / scale, normalized, derivative) @ coefficients
        return values / scale ** (len(derivative) if derivative else 0)

    return evaluate


def capture_curvature_surface(boundary, matrix_world, center, x, y, axis, planned_splits=()):
    """Freeze refined faces and exterior heights before cuts or row infill."""
    neighborhoods = []
    source = set(boundary)
    for vert in boundary:
        region = {vert}
        for _ring in range(2):
            region |= {edge.other_vert(v) for v in list(region) for edge in v.link_edges
                       if not edge.hide and not edge.other_vert(v).hide
                       and any(not face.hide for face in edge.link_faces)}
        neighborhoods.append(region)
        source.update(region)
    # Stable source ordering makes the continuation independent of set hashes.
    source = list(boundary) + sorted(source - set(boundary), key=lambda v: tuple(v.co))
    indices = {v: i for i, v in enumerate(source)}
    offsets = [matrix_world @ v.co - center for v in source]
    points = [(p.dot(x), p.dot(y)) for p in offsets]
    heights = [p.dot(axis) for p in offsets]
    rim_indices = list(range(len(boundary)))
    support_indices = [sorted(indices[v] for v in region) for region in neighborhoods]
    from .face_curvature import capture_face_samples
    face_samples = capture_face_samples(boundary, matrix_world, center, x, y, axis)
    for edge, factor, _identity in planned_splits:
        following = (edge + 1) % len(boundary)
        offset = offsets[edge].lerp(offsets[following], factor)
        rim_indices.append(len(points))
        points.append((offset.dot(x), offset.dot(y)))
        heights.append(offset.dot(axis))
        support_indices.append(sorted(indices[v] for v in neighborhoods[edge] | neighborhoods[following]))
        if face_samples is not None:
            face_samples['neighborhoods'].append(sorted(
                set(face_samples['neighborhoods'][edge]) | set(face_samples['neighborhoods'][following])))
    return fit_curvature_surface(points, heights, rim_indices, support_indices,
                                 boundary_polygon=points[:len(boundary)], face_samples=face_samples)
