# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Boundary-locked quad shape quality, independent of mesh construction."""

import math

import numpy as np
from mathutils import Vector


def shape_loss(layout):
    """Mean inverse corner quality: one is square, slivers tend to infinity."""
    coords = layout['coords']
    losses = []
    for row in layout['row_faces']:
        for face in row:
            for j, b in enumerate(face):
                a, c = face[j - 1], face[(j + 1) % len(face)]
                first, second = coords[b] - coords[a], coords[c] - coords[b]
                cross = first.x * second.y - first.y * second.x
                if cross <= 0.:
                    return math.inf
                losses.append((first.length_squared + second.length_squared) / (2. * cross))
    return sum(losses) / len(losses) if losses else math.inf


def _independent_sweeps(free, neighbors, *, colored=False):
    """Batch noninteracting updates, with bounded temporary-array sizes.

    By default every earlier vertex sharing a face is a dependency, including
    diagonal corners. Those DAG levels preserve ordered Gauss-Seidel updates.
    Sphere opts into multicolor Gauss-Seidel: a different sweep order with the
    same objective, pass count, and per-update fold/line-search safeguards.
    """
    def bounded(batches):
        return [np.asarray(batch[start:start+128], dtype=int)
                for batch in batches for start in range(0, len(batch), 128)]
    if colored:
        assigned, batches = {}, []
        for i in free:
            used = {assigned[j] for j in neighbors[i] if j in assigned}
            color = next(j for j in range(len(used)+1) if j not in used)
            assigned[i] = color
            if color == len(batches):
                batches.append([])
            batches[color].append(i)
        batches = bounded(batches)
        return batches, list(reversed(batches))
    sweeps = []
    for order in (free, list(reversed(free))):
        levels, batches = {}, []
        for i in order:
            level = 1 + max((levels[j] for j in neighbors[i] if j in levels), default=-1)
            levels[i] = level
            if level == len(batches):
                batches.append([])
            batches[level].append(i)
        sweeps.append(bounded(batches))
    return sweeps


def improve_quad_shapes(layout, boundary_count, *, iterations=20, strength=.5):
    """Damped local Newton steps minimize angle/aspect distortion.

    Each corner contributes (|u|²+|v|²)/(2 cross(u,v)). With one vertex
    moving, the cross product is affine and the numerator quadratic, giving
    an analytic 2D gradient/Hessian. Backtracking preserves positive corners
    and decreases the complete incident-face objective, not just edge length.
    Original/cut boundary vertices and all topology remain fixed.
    """
    if not iterations or strength == 0.:
        return 0
    coords = layout['coords']
    fixed = set(range(boundary_count)) | {i for _edge, _factor, i in layout['splits']}
    incident, neighbors = {}, {}
    for row in layout['row_faces']:
        for face in row:
            for i in face:
                incident.setdefault(i, []).append(face)
            for a, b in zip(face, face[1:] + face[:1]):
                neighbors.setdefault(a, set()).add(b)
                neighbors.setdefault(b, set()).add(a)
    free = sorted(incident.keys() - fixed)
    if not free:
        return 0
    tolerance = max(max((p - coords[0]).length_squared for p in coords) * 1e-10, 1e-16)

    def evaluate(identity, derivatives=True):
        energy = gx = gy = hxx = hxy = hyy = 0.
        for face in incident[identity]:
            for j, b in enumerate(face):
                a, c = face[j - 1], face[(j + 1) % len(face)]
                u, v = coords[b] - coords[a], coords[c] - coords[b]
                cross = u.x * v.y - u.y * v.x
                if cross <= tolerance:
                    return math.inf, 0., 0., 0., 0., 0.
                square = u.length_squared + v.length_squared
                energy += square / (2. * cross)
                if not derivatives or identity not in (a, b, c):
                    continue
                if identity == a:
                    ds, dc, k = -2. * u, Vector((-v.y, v.x)), 1.
                elif identity == b:
                    ds, dc, k = 2. * (u - v), Vector((u.y + v.y, -u.x - v.x)), 2.
                else:
                    ds, dc, k = 2. * v, Vector((-u.y, u.x)), 1.
                gx += ds.x / (2. * cross) - square * dc.x / (2. * cross ** 2)
                gy += ds.y / (2. * cross) - square * dc.y / (2. * cross ** 2)
                hxx += k / cross - ds.x * dc.x / cross ** 2 + square * dc.x ** 2 / cross ** 3
                hxy += -(ds.x * dc.y + dc.x * ds.y) / (2. * cross ** 2) + square * dc.x * dc.y / cross ** 3
                hyy += k / cross - ds.y * dc.y / cross ** 2 + square * dc.y ** 2 / cross ** 3
        return energy, gx, gy, hxx, hxy, hyy

    completed = 0
    for iteration in range(iterations):
        maximum_move = 0.
        for i in (free if iteration % 2 == 0 else reversed(free)):
            energy, gx, gy, hxx, hxy, hyy = evaluate(i)
            if not math.isfinite(energy):
                continue
            trace = hxx + hyy
            minimum_eigenvalue = (trace - math.hypot(hxx - hyy, 2. * hxy)) * .5
            regularizer = max(0., trace * 1e-8 - minimum_eigenvalue)
            hxx += regularizer
            hyy += regularizer
            determinant = hxx * hyy - hxy * hxy
            if determinant <= max(trace * trace * 1e-14, 1e-300):
                continue
            step = Vector(((-hyy * gx + hxy * gy) / determinant,
                           (hxy * gx - hxx * gy) / determinant))
            connected = sorted(neighbors[i])
            extent = sum((coords[i] - coords[j]).length for j in connected) / len(connected)
            if step.length > extent:
                step *= extent / step.length
            original = coords[i].copy()
            for backtrack in range(12):
                coords[i] = original + step * (strength * .5 ** backtrack)
                if evaluate(i, False)[0] < energy:
                    maximum_move = max(maximum_move, (coords[i] - original).length_squared)
                    break
                coords[i] = original
        completed += 1
        if maximum_move <= tolerance * 1e-4:
            break
    return completed


def improve_projected_shapes(layout, boundary_count, project, *, fixed_positions=None,
                             iterations=20, strength=.5, movable_boundary=False,
                             support=None, pinned_boundary=(), color_sweeps=False):
    """Optimize the actual 3D cells while remaining on a frozen surface.

    A good flat chart can become a spike or twisted quad after projection.
    Minimize 3D corner distortion and corner-normal disagreement instead of
    merely running more flat relaxation. Optional refined-rim movement also
    includes exterior support faces, whose other vertices remain fixed.
    Only free chart coordinates move;
    every trial is reprojected and must retain positive chart corners. Finite
    differences include the surface's metric without assuming it is planar.
    The complete layout is processed even for a partial-row inspection.
    General fits retain ordered wavefronts; Sphere may opt into larger
    multicolor batches. No two vertices in a batch share an incident face.
    """
    if not iterations or strength == 0.:
        return 0
    coords = np.asarray(layout['coords'], dtype=float)
    faces = [face for row in layout['row_faces'] for face in row]
    patch_count, patch_faces = len(coords), len(faces)
    fixed = (set() if movable_boundary else set(range(boundary_count)))
    fixed |= {i for _edge, _factor, i in layout['splits']} | set(pinned_boundary)
    world = np.asarray(project(coords), dtype=float)
    for i, position in (fixed_positions or {}).items():
        world[i] = position
    if support is not None:
        coords = np.vstack((coords, support['coords']))
        world = np.vstack((world, support['positions']))
        faces += support['faces']
        fixed.update(range(patch_count, len(coords)))
    original_world = world.copy()
    incident, neighbors = {}, {}
    for face in faces:
        for i in face:
            incident.setdefault(i, []).append(face)
            neighbors.setdefault(i, set()).update(j for j in face if j != i)
    free = sorted(incident.keys() - fixed)
    if not free:
        return 0
    extent = float(np.max(np.linalg.norm(coords[:patch_count] - coords[0], axis=1)))
    tolerance = max(extent * extent * 1e-10, 1e-30)
    groups = {}
    support_ids = {id(face) for face in faces[patch_faces:]}
    for i in free:
        groups[i] = []
        for size, exterior in sorted({(len(face), id(face) in support_ids) for face in incident[i]}):
            indices = np.asarray([face for face in incident[i]
                                  if len(face) == size and (id(face) in support_ids) == exterior])
            p = world[indices]
            cross = np.cross(p - np.roll(p, 1, axis=1), np.roll(p, -1, axis=1) - p)
            normal = cross.sum(axis=1)
            normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], tolerance)
            signed = np.sum(cross * normal[:, None], axis=2)
            previous = np.roll(np.arange(size), 1)
            following = np.roll(np.arange(size), -1)
            slots = np.nonzero(indices == i)
            active = np.abs(signed) > tolerance if exterior else np.ones(indices.shape, dtype=bool)
            groups[i].append((indices, exterior, normal, signed, previous, following, slots, active))
    rim_limits = {i: .5 * np.mean(np.linalg.norm(world[sorted(neighbors[i])] - world[i], axis=1))
                  for i in free if i < boundary_count}

    sweeps = _independent_sweeps(free, neighbors, colored=color_sweeps)
    prepared, neighbor_groups = {}, {}
    for sweep in sweeps:
        for batch in sweep:
            entries = {}
            for slot, i in enumerate(batch):
                for group in groups[i]:
                    indices, exterior = group[:2]
                    entries.setdefault((indices.shape[1], exterior, len(indices)), []).append((slot, group))
            merged = []
            for (_size, exterior, _count), entries in sorted(entries.items()):
                slots = np.asarray([slot for slot, _group in entries])
                indices = np.asarray([group[0] for _slot, group in entries])
                normals = np.asarray([group[2] for _slot, group in entries])
                corners = np.asarray([group[3] for _slot, group in entries])
                active = np.asarray([group[7] for _slot, group in entries])
                changing = np.nonzero(indices == batch[slots, None, None])
                previous, following = entries[0][1][4:6]
                merged.append((slots, indices, exterior, normals, corners, previous, following, changing, active))
            prepared[tuple(batch)] = merged
            by_count = {}
            for slot, i in enumerate(batch):
                by_count.setdefault(len(neighbors[i]), []).append(slot)
            neighbor_groups[tuple(batch)] = [
                (np.asarray(slots), np.asarray([sorted(neighbors[batch[slot]]) for slot in slots]))
                for slots in by_count.values()]

    def losses(batch, queries, positions):
        total = np.zeros(queries.shape[:2])
        for (slots, indices, exterior, reference_normal, reference_corners,
             previous, following, (vertex_slot, face_slot, corner_slot), active) in prepared[tuple(batch)]:
            shape = (len(slots), queries.shape[1], *indices.shape[1:])
            uv, points = np.empty((*shape, 2)), np.empty((*shape, 3))
            uv[:] = coords[indices][:, None]
            points[:] = world[indices][:, None]
            uv[vertex_slot, :, face_slot, corner_slot] = queries[slots[vertex_slot]]
            points[vertex_slot, :, face_slot, corner_slot] = positions[slots[vertex_slot]]
            first = uv - uv[:, :, :, previous]
            second = uv[:, :, :, following] - uv
            cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
            valid = (np.ones((len(slots), queries.shape[1]), dtype=bool) if exterior else
                     np.all(cross > tolerance, axis=(2, 3)))
            u = points - points[:, :, :, previous]
            v = points[:, :, :, following] - points
            normals = np.empty_like(u)
            normals[..., 0] = u[..., 1]*v[..., 2] - u[..., 2]*v[..., 1]
            normals[..., 1] = u[..., 2]*v[..., 0] - u[..., 0]*v[..., 2]
            normals[..., 2] = u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]
            area = np.linalg.norm(normals, axis=4)
            valid &= np.all((area > tolerance) | ~active[:, None], axis=(2, 3))
            if exterior:
                signed = np.sum(normals * reference_normal[:, None, :, None], axis=4)
                # Keep existing convex corners and face orientation; do not
                # demand convexity from an originally concave support n-gon.
                valid &= np.all((signed > np.maximum(reference_corners[:, None] * .1, tolerance))
                                | (reference_corners <= tolerance)[:, None], axis=(2, 3))
                valid &= np.all(np.sum(signed, axis=3) > .1 * reference_corners.sum(axis=2)[:, None], axis=2)
            area = np.maximum(area, tolerance)
            quality = np.sum(u*u + v*v, axis=4) / (2. * area)
            normals /= area[..., None]
            # All corner normals agree for a planar convex cell. This term
            # discourages the extreme twisting that corner aspect alone can
            # accept after a highly nonlinear surface projection.
            twist = 1. - np.clip(np.sum(normals * normals[:, :, :, following], axis=4), -1., 1.)
            value = np.sum((quality + .5 * twist) * active[:, None], axis=(2, 3))
            value[~valid] = np.inf
            total[slots] += value
        limits = np.asarray([rim_limits.get(i, np.inf) for i in batch])
        total[np.linalg.norm(positions-original_world[batch, None], axis=2) > limits[:, None]] = np.inf
        return total

    def project_batch(queries):
        return np.asarray(project(queries.reshape((-1, 2)))).reshape((*queries.shape[:2], 3))

    offsets = np.asarray(((1, 0), (-1, 0), (0, 1), (0, -1),
                          (1, 1), (-1, -1), (1, -1), (-1, 1)))
    completed = 0
    for iteration in range(iterations):
        maximum = 0.
        for batch in sweeps[iteration % 2]:
            original = coords[batch].copy()
            energy = losses(batch, original[:, None], world[batch, None])[:, 0]
            radius = np.empty(len(batch))
            for slots, connected in neighbor_groups[tuple(batch)]:
                radius[slots] = np.mean(np.linalg.norm(coords[connected]-original[slots, None], axis=2), axis=1)
            epsilon = radius * .001
            eligible = (epsilon > extent * 1e-12) & np.isfinite(energy)
            if not eligible.any():
                continue
            safe_epsilon = np.where(eligible, epsilon, 0.)
            queries = original[:, None] + offsets[None] * safe_epsilon[:, None, None]
            costs = losses(batch, queries, project_batch(queries))
            eligible &= np.isfinite(costs).all(axis=1)
            step = np.zeros_like(original)
            slots = np.flatnonzero(eligible)
            if len(slots):
                a, b, c, d, e, f, g, h = costs[slots].T
                gradient = np.column_stack((a-b, c-d)) / (2.*epsilon[slots, None])
                hessian = np.empty((len(slots), 2, 2))
                hessian[:, 0, 0], hessian[:, 1, 1] = a+b-2.*energy[slots], c+d-2.*energy[slots]
                hessian[:, 0, 1] = hessian[:, 1, 0] = (e+f-g-h)*.25
                hessian /= epsilon[slots, None, None]**2
                eigenvalues = np.linalg.eigvalsh(hessian)
                regularizer = np.maximum(0., np.max(np.abs(eigenvalues), axis=1)*1e-5 - eigenvalues[:, 0])
                hessian += np.eye(2)[None] * regularizer[:, None, None]
                try:
                    step[slots] = -np.linalg.solve(hessian, gradient[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    # A singular member must not suppress independent valid
                    # updates in the same wavefront.
                    for local, slot in enumerate(slots):
                        try:
                            step[slot] = -np.linalg.solve(hessian[local], gradient[local])
                        except np.linalg.LinAlgError:
                            eligible[slot] = False
            eligible &= np.isfinite(step).all(axis=1)
            step[~eligible] = 0.
            length = np.linalg.norm(step, axis=1)
            step *= np.minimum(1., radius / np.maximum(length, 1e-300))[:, None]
            queries = original[:, None] + step[:, None] * (strength * .5**np.arange(12))[None, :, None]
            positions = project_batch(queries)
            accepted = losses(batch, queries, positions) < energy[:, None]
            accepted &= eligible[:, None]
            moving = np.flatnonzero(accepted.any(axis=1))
            if len(moving):
                slots = accepted[moving].argmax(axis=1)
                coords[batch[moving]], world[batch[moving]] = queries[moving, slots], positions[moving, slots]
                maximum = max(maximum, np.max(np.linalg.norm(coords[batch[moving]] - original[moving], axis=1)))
        completed += 1
        if maximum <= extent * 1e-7:
            break
    for i in free:
        layout['coords'][i] = Vector(coords[i])
    return completed
