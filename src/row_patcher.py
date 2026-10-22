# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Manhattan row construction, validation and frozen surface projection."""

import math
from copy import deepcopy
from .visible_geometry import update_visible_normals, visible_valence

import bmesh
import numpy as np
from mathutils import Vector

from .fairing import (FairingError, _fair_organic_depth, _organic_depth_samples,
                     _organic_grid_frame, _signed_area_2d, ordered_boundary_loop)
from .grid_patcher import _inside_boundary
from .surface_projection import project_hypothesis_along_axis


MAX_ROWS = 256
MAX_COLUMNS = 256
MAX_PATCH_FACES = 65536


def _projection_key(hypothesis):
    """Snapshot exactly the analytic parameters consumed by axis projection."""
    method = (hypothesis or {}).get('method', 'ORGANIC')
    if method == 'SPHERE':
        return method, tuple(hypothesis['center']), float(hypothesis['radius'])
    if method == 'CONIC':
        fit = hypothesis['fit']
        return method, tuple(fit['center']), tuple(fit['axis']), float(fit['radius']), float(fit.get('slope', 0.))
    if method == 'PLANES':
        return method, tuple((tuple(p['point']), tuple(p['normal'])) for p in hypothesis['planes'])
    return (method,)


def _row_plan_key(loop, matrix, frame, protected, options):
    """Numeric local input identity for one Auto probe/replay, never a global cache.

    Include both vertex-height stencils and the face-connected fitting ring.
    Element indices/addresses differ between the private probe and live mesh.
    Ambiguous coincident source vertices simply disable reuse.
    """
    from .intrinsic_surface import _source_faces
    faces = _source_faces(loop)
    vertices = set(loop)
    for _ring in range(2):
        vertices |= {e.other_vert(v) for v in list(vertices) for e in v.link_edges
                     if not e.hide and not e.other_vert(v).hide
                     and (not e.link_faces or any(not f.hide for f in e.link_faces))}
    vertices.update(v for f in faces for v in f.verts)
    ordered = sorted(vertices, key=lambda v: tuple(v.co))
    positions = tuple(tuple(v.co) for v in ordered)
    if len(set(positions)) != len(positions):
        return None
    identities = {v: i for i, v in enumerate(ordered)}
    polygons = []
    for face in faces:
        ids = [identities[v] for v in face.verts]
        start = ids.index(min(ids))
        polygons.append(tuple(ids[start:] + ids[:start]))
    edges = {tuple(sorted((identities[e.verts[0]], identities[e.verts[1]])))
             for v in vertices for e in v.link_edges if not e.hide
             and all(w in identities for w in e.verts)
             and (not e.link_faces or any(not f.hide for f in e.link_faces))}
    pinned = tuple(v in protected or v.hide or any(f.hide for f in v.link_faces) for v in loop)
    return (tuple(map(tuple, matrix)), tuple(map(tuple, frame)), options,
            tuple(identities[v] for v in loop), tuple(visible_valence(v) for v in loop), pinned,
            positions, tuple(sorted(edges)), tuple(sorted(polygons)))


def _rim_relaxation_support(loop, layout, matrix_world, center, x, y):
    """Freeze the attached visible faces, pinning everything beyond the rim."""
    identities = {v: i for i, v in enumerate(loop)}
    rim = set(loop)
    pinned = {i for i, v in enumerate(loop) if v.hide or any(f.hide for f in v.link_faces)}
    faces = {f for v in loop for f in v.link_faces if not f.hide}
    for face in faces:
        if any(v.hide for v in face.verts):
            pinned.update(identities[v] for v in face.verts if v in rim)
    faces = [f for f in faces if not any(v.hide for v in f.verts)]
    # A parity cut is an exact interpolation on an existing source edge.
    # Retain that edge and its endpoints rather than making a new kink in
    # its support face during an unrelated rim relaxation.
    for edge, _factor, _identity in layout['splits']:
        pinned.update((edge, (edge + 1) % len(loop)))
    outside = sorted({v for f in faces for v in f.verts} - rim, key=lambda v: tuple(v.co))
    for i, v in enumerate(outside, len(layout['coords'])):
        identities[v] = i
    positions = [matrix_world @ v.co for v in outside]
    offsets = [p - center for p in positions]
    faces.sort(key=lambda f: tuple(sorted(identities[v] for v in f.verts)))
    return {'coords': np.asarray([(p.dot(x), p.dot(y)) for p in offsets]).reshape((-1, 2)),
            'positions': np.asarray(positions).reshape((-1, 3)),
            'faces': [[identities[v] for v in f.verts] for f in faces]}, pinned


def _strip(lower, upper, coords, tolerance, *, allow_unrelaxed=False, quads_only=False):
    """Bridge one row: quads, one local 3-to-1 mutation, or one odd triangle."""
    invert = len(lower) < len(upper)
    if invert:
        lower, upper = upper, lower
    n, m = len(lower) - 1, len(upper) - 1
    difference = n - m
    if not 0 <= difference <= 2:
        raise FairingError("Row edge counts changed too quickly")

    def candidate(position):
        faces = [[lower[i], lower[i + 1], upper[i + 1], upper[i]] for i in range(position)]
        extra = []
        if difference == 1:
            faces.append([lower[position], lower[position + 1], upper[position]])
            for i in range(position, m):
                faces.append([lower[i + 1], lower[i + 2], upper[i + 1], upper[i]])
        elif difference == 2:
            b0, b1, b2, b3 = lower[position:position + 4]
            t0, t1 = upper[position:position + 2]
            # Two local interior vertices create four convex quads. A single
            # center vertex would leave three collinear corners in one quad.
            extra = [coords[b1].lerp(coords[t0].lerp(coords[t1], 1. / 3.), .5),
                     coords[b2].lerp(coords[t0].lerp(coords[t1], 2. / 3.), .5)]
            a, b = len(coords), len(coords) + 1
            faces += [[b0, b1, a, t0], [b1, b2, b, a],
                      [b2, b3, t1, b], [a, b, t1, t0]]
            for i in range(position + 1, m):
                faces.append([lower[i + 2], lower[i + 3], upper[i + 1], upper[i]])
        else:
            faces = [[lower[i], lower[i + 1], upper[i + 1], upper[i]] for i in range(n)]
        if invert:
            faces = [list(reversed(face)) for face in faces]
        # A shorter side may hold its existing endpoint for one row. Collapse
        # repeated identities, not source edges: this creates a local corner
        # triangle (or consumes an already-present corner), never an edge split.
        faces = [list(dict.fromkeys(face)) for face in faces]
        faces = [face for face in faces if len(face) >= 3]
        if quads_only and any(len(face) != 4 for face in faces):
            return None
        all_coords = coords + extra
        score = 100.0 * sum(len(face) == 3 for face in faces)
        for face in faces:
            polygon = [all_coords[i] for i in face]
            if _signed_area_2d(polygon) <= tolerance and not allow_unrelaxed:
                return None
            edges = [polygon[(i + 1) % len(face)] - p for i, p in enumerate(polygon)]
            for i, edge in enumerate(edges):
                previous = edges[i - 1]
                cross = previous.x * edge.y - previous.y * edge.x
                if cross <= tolerance and not allow_unrelaxed:
                    return None
                denominator = previous.length_squared * edge.length_squared
                if denominator <= tolerance * tolerance:
                    return None
                score += previous.dot(edge) ** 2 / denominator
                score += 10000. * max(0.0, tolerance - cross) / math.sqrt(denominator)
            sizes = [edge.length_squared for edge in edges]
            score += .05 * max(sizes) / min(sizes)
        return score, extra, faces

    positions = range(m + 1) if difference == 1 else range(m) if difference == 2 else (0,)
    choices = [choice for k in positions if (choice := candidate(k)) is not None]
    if not choices:
        raise FairingError("A row would fold or cross; inspect the Refined Edge state")
    _score, extra, faces = min(choices, key=lambda choice: choice[0])
    coords.extend(extra)
    return faces, int(difference == 2), sum(len(face) == 3 for face in faces)


def _place_interior(coords, faces, fixed, tolerance):
    """Untangle approximate rows, locking the entire refined boundary.

    Project violated convex-corner constraints using their analytic 2D
    gradients. This changes only new points, not the refined edge or row
    connectivity. All faces must pass strict validation before returning.
    """
    def constraints():
        for face in faces:
            for j, b in enumerate(face):
                a, c = face[j - 1], face[(j + 1) % len(face)]
                pa, pb, pc = coords[a], coords[b], coords[c]
                first, second = pb - pa, pc - pb
                cross = first.x * second.y - first.y * second.x
                margin = max(tolerance * 16., first.length * second.length * .02)
                if cross < margin:
                    gradients = ((a, Vector((pb.y - pc.y, pc.x - pb.x))),
                                 (b, Vector((pc.y - pa.y, pa.x - pc.x))),
                                 (c, Vector((pa.y - pb.y, pb.x - pa.x))))
                    yield margin - cross, [(i, g) for i, g in gradients if i not in fixed]

    for _iteration in range(256):
        changed = False
        for deficit, gradients in constraints():
            denominator = sum(g.length_squared for _i, g in gradients)
            if denominator <= tolerance * tolerance:
                continue
            for i, gradient in gradients:
                coords[i] += gradient * (deficit / denominator)
            changed = True
        if not changed:
            break
    for face in faces:
        polygon = [coords[i] for i in face]
        for j, p in enumerate(polygon):
            first, second = p - polygon[j - 1], polygon[(j + 1) % len(face)] - p
            if first.x * second.y - first.y * second.x <= tolerance:
                raise FairingError("A row would fold or cross; inspect the Refined Edge state")


def _validate_layout(layout, points, *, allow_splits=False):
    """Validate a footprint with source positions and explicit cuts locked."""
    coords, splits = layout['coords'], layout['splits']
    if splits and not allow_splits:
        raise FairingError("Row fill must not split refined boundary edges")
    faces = [face for row in layout['row_faces'] for face in row]
    if len(faces) > MAX_PATCH_FACES:
        raise FairingError("Row fill exceeds the face safety limit")
    extent = max((p - coords[0]).length for p in coords)
    tolerance = max(extent * extent * 1.0e-10, 1.0e-16)
    identities = set()
    for edge, factor, identity in splits:
        if (not isinstance(edge, int) or not 0 <= edge < len(points)
                or not isinstance(identity, int) or not len(points) <= identity < len(coords)
                or identity in identities or not 0. < factor < 1.):
            raise FairingError("Invalid planned boundary cut")
        expected = coords[edge].lerp(coords[(edge + 1) % len(points)], factor)
        if (coords[identity] - expected).length_squared > tolerance * 1e-4:
            raise FairingError("A planned cut leaves its source edge")
        identities.add(identity)
    fixed = set(range(len(points))) | {identity for _edge, _factor, identity in splits}
    _place_interior(coords, faces, fixed, tolerance)
    area = sum(_signed_area_2d([coords[i] for i in face]) for face in faces)
    if abs(area - _signed_area_2d(coords[:len(points)])) > tolerance * len(faces) * 10.:
        raise FairingError("The refined edge cannot be unfolded into non-crossing rows")
    for identity, p in enumerate(coords[len(points):], len(points)):
        if identity not in fixed and not _inside_boundary(p, coords[:len(points)]):
            raise FairingError("A grid row leaves the refined boundary")
    boundary = []
    for i in range(len(points)):
        boundary.append(i)
        boundary.extend(identity for edge, factor, identity in sorted(splits) if edge == i)
    boundary_edges = {frozenset((a, b)) for a, b in zip(boundary, boundary[1:] + boundary[:1])}
    owners = {}
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            key = frozenset((a, b))
            owners[key] = owners.get(key, 0) + 1
    if (any(count != (1 if edge in boundary_edges else 2) for edge, count in owners.items())
            or not boundary_edges <= owners.keys()):
        raise FairingError("Row fill produced a non-conforming patch")
    return layout


def fill_refined_rows(bm, loop_verts, loop_edges, matrix_world, *, fill_rows=0,
                      relaxation_iterations=20, relaxation_strength=.5, match_curvature=True,
                      hypothesis=None, protected_vertices=(), _plan_cache=None):
    """Fill rows, relaxing the infill and refined rim on a frozen surface."""
    if not isinstance(fill_rows, int) or not 0 <= fill_rows <= MAX_ROWS:
        raise FairingError("Rows to Fill must be between zero and 256")
    center, axis, x, y, points, _values = _organic_grid_frame(loop_verts, loop_edges, matrix_world)
    key = (_row_plan_key(loop_verts, matrix_world, (center, axis, x, y), set(protected_vertices),
                        (relaxation_iterations, relaxation_strength, match_curvature, _projection_key(hypothesis)))
           if _plan_cache is not None else None)
    cached = _plan_cache.get('plan') if key is not None and _plan_cache.get('key') == key else None
    from .manhattan import grid_layout
    layout = (deepcopy(cached[0]) if cached is not None else
              grid_layout(points, [visible_valence(v) for v in loop_verts],
                          [matrix_world @ v.co for v in loop_verts],
                          relaxation_iterations=relaxation_iterations,
                          relaxation_strength=relaxation_strength))
    # Freeze the fit at the Refined Edge state. The number of inspection rows
    # must not change the surface used for the first rows.
    surface_method = (hypothesis or {}).get('method', 'ORGANIC')
    analytic = surface_method in {'PLANES', 'CONIC', 'SPHERE'}
    depth_samples = None
    surface = None
    if cached is not None:
        _layout, surface, depth_samples, surface_relaxation_passes = cached
    elif not analytic:
        from .ideal_surface import capture_organic_surface
        surface = capture_organic_surface(loop_verts, matrix_world, center, x, y, axis,
                                          match_curvature=match_curvature, planned_splits=layout['splits'])
        if surface is None:
            depth_samples = _organic_depth_samples([], loop_verts, matrix_world, center, x, y, axis)
    if cached is None:
        surface_relaxation_passes = 0
    project = None
    if surface is not None or analytic:
        from .grid_quality import improve_projected_shapes
        from .surface_projection import sphere_axis_projector
        sphere_project = (sphere_axis_projector(hypothesis, center, x, y, axis)
                          if surface_method == 'SPHERE' else None)
        def project(queries):
            if surface is not None:
                return surface.evaluate_chart(queries)
            if sphere_project is not None:
                return sphere_project(queries)
            points = np.asarray(center) + np.asarray(queries) @ np.asarray((x, y))
            targets = [project_hypothesis_along_axis(hypothesis, Vector(p), Vector(p), axis) for p in points]
            # Trial steps can leave a sphere/cone's axis-projection domain.
            # Reject those trials in the optimizer, not the whole grid fit.
            return np.asarray([p if p is not None else (np.nan, np.nan, np.nan) for p in targets])
        if cached is None:
            fixed_positions = {i: matrix_world @ v.co for i, v in enumerate(loop_verts)}
            for edge, factor, identity in layout['splits']:
                fixed_positions[identity] = fixed_positions[edge].lerp(
                    fixed_positions[(edge + 1) % len(loop_verts)], factor)
            support, pinned = _rim_relaxation_support(loop_verts, layout, matrix_world, center, x, y)
            protected = set(protected_vertices)
            pinned.update(i for i, v in enumerate(loop_verts) if v in protected)
            surface_relaxation_passes = improve_projected_shapes(
                layout, len(loop_verts), project, fixed_positions=fixed_positions,
                iterations=relaxation_iterations, strength=relaxation_strength,
                movable_boundary=True, support=support, pinned_boundary=pinned,
                color_sweeps=surface_method == 'SPHERE')
    if key is not None and cached is None:
        _plan_cache.update(key=key, plan=(deepcopy(layout), surface, depth_samples, surface_relaxation_passes))
    inverse = matrix_world.inverted_safe()
    moved_rim = [i for i in range(len(loop_verts)) if Vector(layout['coords'][i]) != Vector(points[i])]
    if moved_rim and project is not None:
        targets = project([layout['coords'][i] for i in moved_rim])
        changed = []
        for i, target in zip(moved_rim, targets):
            original = loop_verts[i].co.copy()
            loop_verts[i].co = inverse @ Vector(target)
            if loop_verts[i].co != original:
                changed.append(i)
        moved_rim = changed
    filled = min(fill_rows or layout['rows'], layout['rows'])
    planned_faces = [face for row in layout['row_faces'][:filled] for face in row]
    used = {i for face in planned_faces for i in face}
    output = {i: vert for i, vert in enumerate(loop_verts)}
    # Capture winding and appearance before an original rim edge is cut.
    a, b = loop_verts[:2]
    support = bm.edges.get((a, b)).link_faces[0]
    reverse = any(loop.vert is a and loop.link_loop_next.vert is b for loop in support.loops)
    boundary = []
    for i, original in enumerate(loop_verts):
        boundary.append(original)
        previous, factor_before = original, 0.
        endpoint = loop_verts[(i + 1) % len(loop_verts)]
        for _edge, factor, identity in sorted(s for s in layout['splits'] if s[0] == i):
            edge = bm.edges.get((previous, endpoint))
            _new_edge, vert = bmesh.utils.edge_split(
                edge, previous, (factor - factor_before) / (1. - factor_before))
            output[identity] = vert
            boundary.append(vert)
            previous, factor_before = vert, factor
    new = []
    for identity in sorted(used - output.keys()):
        p = layout['coords'][identity]
        vert = bm.verts.new(inverse @ (center + x * p.x + y * p.y))
        output[identity] = vert
        new.append(vert)
    faces = []
    for indices in planned_faces:
        vertices = [output[i] for i in (reversed(indices) if reverse else indices)]
        face = bm.faces.new(vertices)
        face.material_index = support.material_index
        face.smooth = support.smooth
        faces.append(face)
    projected = False
    if new and analytic:
        targets = []
        for vert in new:
            point = matrix_world @ vert.co
            target = project_hypothesis_along_axis(hypothesis, point, point, axis)
            if target is None or not all(math.isfinite(value) for value in target):
                raise FairingError("The grid cannot be projected onto the selected surface fit")
            targets.append(target)
        for vert, target in zip(new, targets):
            vert.co = inverse @ target
        projected = True
    elif new and surface is not None:
        queries = [Vector(((matrix_world @ v.co - center).dot(x),
                           (matrix_world @ v.co - center).dot(y))) for v in new]
        targets = surface.evaluate_chart(queries)
        if np.isfinite(targets).all():
            for vert, target in zip(new, targets):
                vert.co = inverse @ Vector(target)
            projected = True
    if new and not projected:
        if not _fair_organic_depth(new, boundary, matrix_world, center, axis, x, y,
                                   depth_samples=depth_samples):
            raise FairingError("The refined-edge surface continuation could not be fitted")
    update_visible_normals(bm)
    if any(face.calc_area() <= 1.0e-14 for face in faces):
        raise FairingError("Surface projection produced a degenerate row face")
    remaining_edges = []
    remaining_vertices = []
    if filled < layout['rows']:
        scope = set(boundary) | set(new)
        pending = {edge for edge in bm.edges if len(edge.link_faces) == 1
                   and not edge.hide and not edge.link_faces[0].hide
                   and all(vert in scope for vert in edge.verts)}
        while pending:
            component = {next(iter(pending))}
            stack = list(component)
            while stack:
                edge = stack.pop()
                for vert in edge.verts:
                    for other in vert.link_edges:
                        if other in pending and other not in component:
                            component.add(other)
                            stack.append(other)
            vertices, edges = ordered_boundary_loop(edges=component)
            remaining_vertices.extend(vertices)
            remaining_edges.extend(edges)
            pending -= component
    return {"method": 'GRID_ROWS', "faces": faces, "verts": new,
            "boundary_verts": boundary, "boundary_edges": remaining_edges,
            "remaining_verts": remaining_vertices, "grid_layout": 'ROW_PATCH',
            "row_layout_method": layout['layout_method'],
            "grid_estimated_edge_length": layout['estimated_edge_length'],
            "rows_filled": filled, "rows_total": layout['rows'],
            "mutation_patches": sum(abs(a - b) == 2 for a, b in
                                    layout.get('row_edge_counts', [(len(a) - 1, len(b) - 1)
                                        for a, b in zip(layout['chains'], layout['chains'][1:])])[:filled]),
            "corner_transitions": sum(abs(a - b) == 1 for a, b in layout['row_edge_counts'][:filled]),
            "transition_triangles": sum(len(face) == 3 for face in planned_faces),
            "side_counts": layout.get('side_counts', ()),
            "grid_size": layout.get('grid_size', ()),
            "grid_cut_count": len(layout['splits']),
            "grid_trimmed_cells": layout.get('trimmed_cells', 0),
            "relaxation_passes": layout.get('relaxation_passes', 0),
            "surface_relaxation_passes": surface_relaxation_passes,
            "relaxed_rim_vertices": len(moved_rim),
            "relaxed_rim_verts": [loop_verts[i] for i in moved_rim],
            "surface_method": surface_method,
            "residual": (hypothesis or {}).get('residual', 0.0),
            "surface_continuity": 'ANALYTIC' if analytic else
                                  surface.continuity if surface is not None else 'THIN_PLATE'}
