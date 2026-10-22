# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later
"""Topology-independent 3D bending continuation for folded height charts."""

import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import tessellate_polygon

from .face_curvature import _face_plane


def _source_faces(boundary):
    border = {f for v in boundary for f in v.link_faces
              if not f.hide and not any(v.hide for v in f.verts)}
    return border | {other for f in border for e in f.edges if not e.hide
                     for other in e.link_faces if not other.hide and not any(v.hide for v in other.verts)}


def folded_chart(boundary, matrix, axis):
    """A grazing/reversed border plane cannot safely constrain a height fit."""
    faces = set()
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        edge = next((e for e in a.link_edges if e.other_vert(a) is b), None)
        if edge is not None:
            faces.update(f for f in edge.link_faces if not f.hide and not any(v.hide for v in f.verts))
    cache = {}
    planes = [p for f in faces if (p := _face_plane(f, matrix, cache)) is not None]
    direction = np.asarray(axis)
    facing = sum(area * np.dot(normal, direction) for _, normal, area in planes)
    sign = 1. if facing >= 0. else -1.
    return any(sign * np.dot(normal, direction) < .2 for _, normal, _ in planes)


def _smooth_triangles(points, parameters, triangles, patch, subdivisions=3):
    """Tessellate cubic point/normal triangles without changing output topology.

    Shared edge controls use the same endpoint normals in both triangles.
    The reference, not the eventual Circle/Grid mesh, determines this fixed
    tessellation. Original source normals contribute at the rim.
    """
    normals = np.zeros_like(points)
    for a, b, c in triangles:
        normal = np.cross(points[b] - points[a], points[c] - points[a])
        for i in (a, b, c):
            normals[i] += normal
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    vertices, uv, faces = [], [], []
    for ids in patch:
        p, n = points[list(ids)], normals[list(ids)]
        controls = {}
        for i, j in ((0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)):
            controls[i, j] = (2.*p[i] + p[j] - np.dot(p[j]-p[i], n[i])*n[i]) / 3.
        edge_mean = np.mean(list(controls.values()), axis=0)
        middle = edge_mean + .5*(edge_mean - p.mean(axis=0))
        local = {}
        for i in range(subdivisions + 1):
            for j in range(subdivisions + 1 - i):
                u, v = i/subdivisions, j/subdivisions
                w = 1. - u - v
                q = u**3*p[0] + v**3*p[1] + w**3*p[2] + 6.*u*v*w*middle
                q += 3.*(u*u*v*controls[0, 1] + v*v*u*controls[1, 0]
                          + v*v*w*controls[1, 2] + w*w*v*controls[2, 1]
                          + w*w*u*controls[2, 0] + u*u*w*controls[0, 2])
                local[i, j] = len(vertices)
                vertices.append(Vector(q))
                uv.append(u*parameters[ids[0]] + v*parameters[ids[1]] + w*parameters[ids[2]])
        for i in range(subdivisions):
            for j in range(subdivisions - i):
                faces.append((local[i, j], local[i+1, j], local[i, j+1]))
                if i+j+1 < subdivisions:
                    faces.append((local[i+1, j], local[i+1, j+1], local[i, j+1]))
    return vertices, np.asarray(uv), faces


def _ears(parameters):
    """Triangulate the actual concave rim, retaining collinear rim stations."""
    live, triangles = list(range(len(parameters))), []
    p = np.asarray(parameters)
    area = np.sum(p[:, 0]*np.roll(p[:, 1], -1) - p[:, 1]*np.roll(p[:, 0], -1))
    if area < 0.:
        live.reverse()
    tolerance = max(np.ptp(p, axis=0).max()**2 * 1e-12, 1e-24)
    while len(live) > 3:
        for j, b in enumerate(live):
            a, c = live[j-1], live[(j+1) % len(live)]
            u, v = p[b]-p[a], p[c]-p[a]
            cross = u[0]*v[1]-u[1]*v[0]
            if cross <= tolerance:
                continue
            others = [i for i in live if i not in (a, b, c)]
            d = p[others]-p[a]
            s = (d[:, 0]*v[1]-d[:, 1]*v[0])/cross
            t = (u[0]*d[:, 1]-u[1]*d[:, 0])/cross
            if np.any((s >= -1e-10) & (t >= -1e-10) & (s+t <= 1.+1e-10)):
                continue
            triangles.append((a, b, c))
            live.pop(j)
            break
        else:
            raise ValueError('The folded rim cannot be parameterized without crossing')
    triangles.append(tuple(live))
    return triangles


def _delaunay(triangles, points):
    """Remove the arbitrary long diagonals of the initial ear triangulation."""
    triangles = list(triangles)
    for _iteration in range(8):
        owners = {}
        for i, t in enumerate(triangles):
            for a, b in zip(t, t[1:] + t[:1]):
                owners.setdefault(tuple(sorted((a, b))), []).append(i)
        changed, used = False, set()
        for (a, b), pair in sorted(owners.items()):
            if len(pair) != 2 or any(i in used for i in pair):
                continue
            i, j = pair
            c = next(v for v in triangles[i] if v not in (a, b))
            d = next(v for v in triangles[j] if v not in (a, b))
            def cross(u, v):
                return u[0]*v[1] - u[1]*v[0]
            u, v = points[a]-points[c], points[b]-points[c]
            s, t = points[a]-points[d], points[b]-points[d]
            if cross(u, v)*cross(s, t) >= 0.:
                continue
            if cross(points[d]-points[c], points[a]-points[c])*cross(points[d]-points[c], points[b]-points[c]) >= 0.:
                continue
            angles = np.arctan2(abs(cross(u, v)), np.dot(u, v)) + np.arctan2(abs(cross(s, t)), np.dot(s, t))
            if angles <= np.pi + 1e-8:
                continue
            candidate = [(c, d, a), (d, c, b)]
            sign = np.sign(cross(points[triangles[i][1]]-points[triangles[i][0]],
                                 points[triangles[i][2]]-points[triangles[i][0]]))
            candidate = [tri if cross(points[tri[1]]-points[tri[0]], points[tri[2]]-points[tri[0]])*sign > 0.
                         else tuple(reversed(tri)) for tri in candidate]
            triangles[i], triangles[j] = candidate
            used.update(pair)
            changed = True
        if not changed:
            break
    return triangles


class BendingSurface:
    """Frozen reference surface; retains no BMesh, object or fitted output."""

    continuity = 'INTRINSIC_BENDING'

    def __init__(self, origin, scale, vertices, parameters, faces):
        self.origin, self.scale = origin, scale
        self.tree = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
        self.chart = BVHTree.FromPolygons([Vector((p[0], p[1], 0.)) for p in parameters], faces,
                                          all_triangles=True)
        self.vertices = np.asarray(vertices)[np.asarray(faces)]
        self.parameters = parameters[np.asarray(faces)]
        edges = self.parameters[:, 1:] - self.parameters[:, :1]
        self.inverse = np.linalg.inv(np.swapaxes(edges, 1, 2))

    def evaluate_chart(self, queries):
        queries = np.atleast_2d(queries) / self.scale
        hits = [self.chart.find_nearest(Vector((q[0], q[1], 0.))) for q in queries]
        if any(hit[0] is None for hit in hits):
            raise ValueError('The 3D bending chart could not be evaluated')
        identities = np.asarray([hit[2] for hit in hits])
        offsets = np.asarray([tuple(hit[0])[:2] for hit in hits]) - self.parameters[identities, 0]
        weights = np.einsum('ijk,ik->ij', self.inverse[identities], offsets)
        weights = np.column_stack((1.-weights.sum(axis=1), weights))
        return np.einsum('ij,ijk->ik', weights, self.vertices[identities])*self.scale + self.origin

    def project(self, points):
        queries = (np.atleast_2d(points) - self.origin) / self.scale
        targets = [self.tree.find_nearest(Vector(p))[0] for p in queries]
        if any(p is None for p in targets):
            raise ValueError('The 3D bending reference could not be projected')
        return np.asarray(targets)*self.scale + self.origin


def capture_bending_surface(boundary, matrix, chart_center, x, y):
    """Minimize discrete bending in XYZ with the source rim/two-ring fixed.

    Cotangents use actual 3D triangle geometry. Only a private reference
    patch is solved; source positions never become ambiguous chart heights.
    A separate triangulated parameter domain is independent of output rows
    or Circle Loops. Exterior coordinates never enter that flattened domain.
    """
    source_faces = _source_faces(boundary)
    if len(boundary) > 768:
        raise ValueError('The 3D bending reference exceeds the rim safety limit')
    rim = set(boundary)
    source = list(boundary) + sorted({v for f in source_faces for v in f.verts} - rim,
                                    key=lambda v: tuple(v.co))
    identities = {v: i for i, v in enumerate(source)}
    source_faces = sorted(source_faces, key=lambda f: tuple(sorted(identities[v] for v in f.verts)))
    points = [np.asarray(matrix @ v.co, dtype=float) for v in source]
    origin = np.mean(points[:len(boundary)], axis=0)
    scale = float(np.max(np.linalg.norm(np.asarray(points) - origin, axis=1)))
    if scale <= 1e-12:
        raise ValueError('The 3D bending reference has no extent')
    points = [(p-origin)/scale for p in points]
    triangles = []
    for face in source_faces:
        ids = [identities[v] for v in face.verts]
        polygon = np.asarray([points[i] for i in ids])
        normal = np.cross(polygon-polygon[0], np.roll(polygon, -1, axis=0)-polygon[0]).sum(axis=0)
        turns = np.cross(np.roll(polygon, -1, axis=0)-polygon,
                         np.roll(polygon, -2, axis=0)-np.roll(polygon, -1, axis=0)) @ normal
        if np.any(turns < -1e-12*np.dot(normal, normal)):
            # A centroid fan can leave a concave source n-gon and constrain
            # the solve with triangles that do not belong to the source.
            vectors = [Vector(p) for p in polygon]
            triangles.extend(tuple(ids[i] for i in tri) for tri in tessellate_polygon([vectors]))
            continue
        center = len(points)
        points.append(np.mean([points[i] for i in ids], axis=0))
        triangles.extend((a, b, center) for a, b in zip(ids, ids[1:] + ids[:1]))
    parameters = (np.asarray(points)*scale + origin - np.asarray(chart_center)) @ np.asarray((x, y)).T / scale
    patch = _delaunay(_ears(parameters[:len(boundary)]), parameters)
    first, second = boundary[:2]
    edge = next(e for e in first.link_edges if e.other_vert(first) is second)
    support = next(f for f in edge.link_faces if f in source_faces)
    # Ear clipping has positive chart winding. Match the source's opposite
    # directed rim edge, independent of the caller's boundary walk direction.
    first_triangle = next(t for t in patch if identities[first] in t and identities[second] in t)
    a, b = identities[first], identities[second]
    along = any(first_triangle[j] == a and first_triangle[(j+1) % 3] == b for j in range(3))
    source_along = any(loop.vert is first and loop.link_loop_next.vert is second for loop in support.loops)
    if along == source_along:
        patch = [(c, b, a) for a, b, c in patch]
    levels = 2 if len(boundary) <= 128 else 1
    if (len(triangles)+len(patch))*4**levels > 50000:
        raise ValueError('The 3D bending reference exceeds the tessellation safety limit')
    fixed_flags = [True]*len(points)
    for _level in range(levels):
        source_edges = {tuple(sorted((a, b))) for t in triangles for a, b in zip(t, t[1:] + t[:1])}
        midpoints = {}
        def midpoint(a, b):
            edge = tuple(sorted((a, b)))
            if edge not in midpoints:
                midpoints[edge] = len(points)
                points.append((points[a]+points[b])*.5)
                fixed_flags.append(edge in source_edges)
            return midpoints[edge]
        def subdivide(faces):
            result = []
            for a, b, c in faces:
                ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
                result.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
            return result
        triangles, patch = subdivide(triangles), subdivide(patch)
        uv = (np.asarray(points)*scale + origin - np.asarray(chart_center)) @ np.asarray((x, y)).T / scale
        patch = _delaunay(patch, uv)
    triangles.extend(patch)
    p = np.asarray(points)
    parameters = (p*scale + origin - np.asarray(chart_center)) @ np.asarray((x, y)).T / scale
    fixed_indices = np.flatnonzero(fixed_flags)
    free_indices = np.flatnonzero(~np.asarray(fixed_flags))
    order = np.concatenate((fixed_indices, free_indices))
    remap = np.empty(len(order), dtype=int)
    remap[order] = np.arange(len(order))
    p, parameters = p[order], parameters[order]
    triangles = [tuple(remap[list(t)]) for t in triangles]
    patch = [tuple(remap[list(t)]) for t in patch]
    fixed, unknown = len(fixed_indices), len(free_indices)
    if len(p) > 12000 or unknown > 1536:
        raise ValueError('The 3D bending reference exceeds the solve safety limit')
    # A harmonic XYZ seed depends only on the actual boundary, not on the
    # ear diagonals' arbitrary interpolation through a non-planar opening.
    seed_matrix, seed_target = np.zeros((unknown, unknown)), np.zeros((unknown, 3))
    for ids in patch:
        for j in range(3):
            i, k, l = ids[j], ids[(j+1) % 3], ids[(j+2) % 3]
            u, v = parameters[k]-parameters[i], parameters[l]-parameters[i]
            area = abs(u[0]*v[1]-u[1]*v[0])
            weight = .5*np.dot(u, v)/max(area, 1e-24)
            for row, column, value in ((k, k, weight), (l, l, weight), (k, l, -weight), (l, k, -weight)):
                if row < fixed:
                    continue
                if column < fixed:
                    seed_target[row-fixed] -= value*p[column]
                else:
                    seed_matrix[row-fixed, column-fixed] += value
    try:
        p[fixed:] = np.linalg.solve(seed_matrix, seed_target)
    except np.linalg.LinAlgError as error:
        raise ValueError('The 3D bending parameterization was singular') from error
    columns = np.zeros((len(p), unknown))
    fixed_laplacian = np.zeros((len(p), 3))
    mass = np.zeros(len(p))

    def entry(row, column, weight):
        if column >= fixed:
            columns[row, column-fixed] += weight
        else:
            fixed_laplacian[row] += weight * p[column]

    for ids in triangles:
        a, b, c = ids
        area = np.linalg.norm(np.cross(p[b]-p[a], p[c]-p[a]))
        if area <= 1e-16:
            continue
        for j in range(3):
            i, k, l = ids[j], ids[(j+1) % 3], ids[(j+2) % 3]
            weight = .5 * np.dot(p[k]-p[i], p[l]-p[i]) / area
            entry(k, k, weight)
            entry(l, l, weight)
            entry(k, l, -weight)
            entry(l, k, -weight)
            mass[i] += area/6.
    mass = np.maximum(mass, max(float(np.median(mass))*.01, 1e-16))
    system = columns.T @ (columns / mass[:, None])
    target = -columns.T @ (fixed_laplacian / mass[:, None])
    try:
        solution = np.linalg.solve(system, target)
    except np.linalg.LinAlgError as error:
        raise ValueError('The 3D bending reference was singular') from error
    if not np.isfinite(solution).all():
        raise ValueError('The 3D bending reference was not finite')
    p[fixed:] = solution
    vertices, parameters, faces = _smooth_triangles(p, parameters, triangles, patch)
    return BendingSurface(origin, scale, vertices, parameters, faces)
