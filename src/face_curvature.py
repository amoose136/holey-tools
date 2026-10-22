# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen face-plane tangents and normal-variation curvature at a refined rim."""

import numpy as np


def _face_plane(face, matrix_world, coordinates):
    for vert in face.verts:
        if vert not in coordinates:
            coordinates[vert] = np.asarray(matrix_world @ vert.co, dtype=float)
    points = np.asarray([coordinates[v] for v in face.verts])
    origin = points[0]
    offsets = points - origin
    wedges = np.cross(offsets, np.roll(offsets, -1, axis=0))
    normal = wedges.sum(axis=0)
    magnitude = np.linalg.norm(normal)
    extent = np.max(np.linalg.norm(offsets, axis=1))
    if magnitude <= max(extent * extent * 1e-10, 1e-24):
        return None
    normal /= magnitude
    # Signed fan areas also handle concave n-gons. Unlike a vertex average,
    # this centroid is unchanged by inserting a redundant point on an edge.
    areas = wedges @ normal
    centroids = (offsets + np.roll(offsets, -1, axis=0)) / 3.
    center = origin + np.sum(centroids * areas[:, None], axis=0) / areas.sum()
    return center, normal, magnitude * .5


def capture_face_samples(boundary, matrix_world, center, x, y, axis):
    """Capture the visible border faces and one edge-connected exterior ring.

    All returned data is numeric, with no live BMesh references. The caller
    captures this after refinement but before constructing any infill faces.
    """
    border = []
    for first, second in zip(boundary, boundary[1:] + boundary[:1]):
        edge = next((e for e in first.link_edges if e.other_vert(first) is second and not e.hide), None)
        border.append({f for f in edge.link_faces if not f.hide} if edge is not None else set())
    primary = set().union(*border)
    neighborhoods = []
    for i in range(len(boundary)):
        seed = border[i-1] | border[i]
        neighborhoods.append(seed | {other for face in seed for edge in face.edges if not edge.hide
                                      for other in edge.link_faces if not other.hide})
    faces = set().union(*neighborhoods)
    if not faces:
        return None
    coordinates, geometry = {}, {}
    for face in faces:
        if any(v.hide for v in face.verts):
            continue
        plane = _face_plane(face, matrix_world, coordinates)
        if plane is not None:
            geometry[face] = plane
    axis, x, y, center = map(np.asarray, (axis, x, y, center))
    facing = sum(area * normal.dot(axis) for face, (_point, normal, area) in geometry.items()
                 if face in primary)
    sign = 1. if facing >= 0. else -1.
    samples = []
    for face, (point, normal, area) in geometry.items():
        along = normal.dot(axis)
        # Reject other sheets and nearly vertical planes in this height chart;
        # do not turn a tiny denominator into an enormous slope constraint.
        if sign * along < .2:
            continue
        offset = point - center
        samples.append((face, (offset.dot(x), offset.dot(y)),
                        (-normal.dot(x) / along, -normal.dot(y) / along), area, along * along))
    samples.sort(key=lambda s: (s[1], s[2], s[3]))
    if not samples:
        return None
    indices = {sample[0]: i for i, sample in enumerate(samples)}
    return {
        'points': np.asarray([s[1] for s in samples]),
        'gradients': np.asarray([s[2] for s in samples]),
        'areas': np.asarray([s[3] for s in samples]),
        'confidence': np.asarray([s[4] for s in samples]),
        'primary': np.asarray([s[0] in primary for s in samples]),
        'neighborhoods': [sorted(indices[f] for f in region if f in indices) for region in neighborhoods],
    }


def estimate_normal_jet(points, gradients, weights, anchor, support):
    """Fit a symmetric Hessian to changes of neighboring face-plane slopes."""
    if len(support) < 3:
        return None
    delta = points[support] - anchor
    radius = float(np.max(np.linalg.norm(delta, axis=1)))
    if radius <= 1e-12:
        return None
    local = delta / radius
    count = len(support)
    design = np.zeros((2 * count, 5))
    design[::2, 0], design[1::2, 1] = 1., 1.
    design[::2, 2:4] = local
    design[1::2, 3:5] = local
    maximum = np.max(weights[support])
    if maximum <= 0.:
        return None
    influence = np.sqrt(weights[support] / maximum)
    influence /= np.maximum(np.linalg.norm(local, axis=1), .2)
    influence = np.repeat(influence, 2)
    try:
        values, _, rank, singular = np.linalg.lstsq(
            design * influence[:, None], gradients[support].ravel() * influence, rcond=1e-7)
    except np.linalg.LinAlgError:
        return None
    if rank < 5 or singular[-1] < singular[0] * .01 or not np.isfinite(values).all():
        return None
    values[2:] /= radius
    return values, radius
