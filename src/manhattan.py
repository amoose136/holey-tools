# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later
"""Topology-led four-wall analysis and a preplanned Manhattan quad template."""

import math

from mathutils import Vector

from .fairing import FairingError, _signed_area_2d
from .grid_quality import improve_quad_shapes, shape_loss


def analyze_boundary(points, valences, world_points=None):
    """Count interlaced direction groups, ignoring walk seed choice.

    Valence is total mesh edge valence, not the always-two boundary valence.
    Angles are measured on the source surface; projection can badly distort
    corner angles on curved meshes. Interior/exterior corners turn the grid
    heading by +1/-1; a direction may appear in several perimeter runs.
    """
    points = [Vector(p) for p in points]
    spatial = points if world_points is None else [Vector(p) for p in world_points]
    n = len(points)
    if (n < 4 or len(valences) != n or len(spatial) != n
            or any(not math.isfinite(v) for p in points + spatial for v in p)
            or _signed_area_2d(points) <= 0.):
        raise FairingError("Manhattan Grid needs one positively oriented refined boundary")
    corners = []
    for i, p in enumerate(spatial):
        first, second = spatial[i - 1] - p, spatial[(i + 1) % n] - p
        if min(first.length_squared, second.length_squared) <= 1.0e-20:
            raise FairingError("Manhattan Grid has a collapsed boundary edge")
        angle = math.degrees(first.angle(second))
        if abs(angle - 90.) <= 40.:
            incoming, outgoing = points[i] - points[i - 1], points[(i + 1) % n] - points[i]
            turn = incoming.x * outgoing.y - incoming.y * outgoing.x
            if valences[i] == 3 and turn < 0.:
                # A two-quad T junction can bend on a curved source without
                # becoming a concave grid corner. It continues the same wall;
                # true concave corners have no outward spoke (valence two).
                continue
            kind = 'EXTERIOR' if valences[i] == 2 else 'INTERIOR'
            if valences[i] < 2 or (turn <= 0. if kind == 'INTERIOR' else turn >= 0.):
                raise FairingError("Manhattan corner topology disagrees with the refined boundary turn")
            corners.append((i, kind))
        elif angle < 130.:
            raise FairingError("Manhattan Grid found a non-right-angle boundary fold")
    turns = {i: 1 if kind == 'INTERIOR' else -1 for i, kind in corners}
    if sum(turns.values()) != 4:
        raise FairingError("Manhattan Grid needs a four-sided quarter-turn boundary")
    heading = 0
    directions = [0]
    for i in range(1, n):
        heading = (heading + turns.get(i, 0)) % 4
        directions.append(heading)
    if (heading + turns.get(0, 0)) % 4:
        raise FairingError("Manhattan Grid headings do not close")
    groups = [[i for i, direction in enumerate(directions) if direction == j] for j in range(4)]
    if any(not group for group in groups):
        raise FairingError("Manhattan Grid needs all four side directions")
    # Longest direction group is U1, with a seed-independent tie-break.
    start = min(range(4), key=lambda j: (-len(groups[j]), min(tuple(spatial[i]) for i in groups[j])))
    groups = groups[start:] + groups[:start]
    directions = [(direction - start) % 4 for direction in directions]
    counts = tuple(map(len, groups))
    return {'side_edges': groups, 'side_counts': counts, 'corners': corners, 'directions': directions,
            'grid_size': (max(counts[0], counts[2]), max(counts[1], counts[3]))}


def _relax_template(layout, boundary_count, *, iterations=20, strength=.5):
    """Boundary-locked Laplacian relaxation of the complete patch chart.

    Minimize edge Dirichlet energy, not a quad-angle objective that can stop
    smoothing early. Backtrack each local move to keep every incident cell
    convex; one constrained corner must not prevent other vertices relaxing.
    The caller subsequently projects the relaxed chart onto its frozen fit.
    """
    coords = layout['coords']
    faces = [face for row in layout['row_faces'] for face in row]
    fixed = set(range(boundary_count)) | {i for _edge, _factor, i in layout['splits']}
    neighbors = {}
    incident = {}
    for face in faces:
        for identity in face:
            incident.setdefault(identity, []).append(face)
        for a, b in zip(face, face[1:] + face[:1]):
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
    free = sorted(neighbors.keys() - fixed)
    if not free or not iterations or strength == 0.:
        return 0
    tolerance = max(max((p - coords[0]).length_squared for p in coords) * 1e-10, 1e-16)

    def safe(identity):
        for face in incident[identity]:
            for j, b in enumerate(face):
                a, c = face[j - 1], face[(j + 1) % len(face)]
                first, second = coords[b] - coords[a], coords[c] - coords[b]
                if first.x * second.y - first.y * second.x <= tolerance * 16.:
                    return False
        return True

    completed = 0
    for iteration in range(iterations):
        maximum_move = 0.
        # Alternating walks avoid a persistent sweep-direction bias.
        for i in (free if iteration % 2 == 0 else reversed(free)):
            connected = sorted(neighbors[i])
            target = sum((coords[j] for j in connected), Vector((0., 0.))) / len(connected)
            original = coords[i]
            energy = sum((original - coords[j]).length_squared for j in connected)
            for backtrack in range(10):
                coords[i] = original.lerp(target, strength * .5 ** backtrack)
                improved = sum((coords[i] - coords[j]).length_squared for j in connected) < energy
                if improved and safe(i):
                    maximum_move = max(maximum_move, (coords[i] - original).length_squared)
                    break
                coords[i] = original
        completed += 1
        if maximum_move <= tolerance * 1e-4:
            break
    return completed


def grid_layout(points, valences, world_points=None, *, relaxation_iterations=20, relaxation_strength=.5):
    """Prefer the canonical sweep; retry perpendicular before rejecting Grid.

    Analyze the source once, then plan each orientation independently without
    touching BMesh. A notch can split rows in one direction but not the other.
    Existing successful layouts retain their ordering and topology.
    """
    if (not isinstance(relaxation_iterations, int) or not 0 <= relaxation_iterations <= 100
            or not math.isfinite(relaxation_strength) or not 0. <= relaxation_strength <= 1.):
        raise FairingError("Interior relaxation needs 0–100 iterations and strength between 0 and 1")
    points = [Vector(p) for p in points]
    analysis = analyze_boundary(points, valences, world_points)
    try:
        layout = _grid_layout(points, analysis, relaxation_iterations, relaxation_strength)
    except FairingError as preferred_error:
        perpendicular = dict(analysis)
        for key in ('side_edges', 'side_counts'):
            perpendicular[key] = analysis[key][1:] + analysis[key][:1]
        perpendicular['directions'] = [(direction - 1) % 4 for direction in analysis['directions']]
        try:
            layout = _grid_layout(points, perpendicular, relaxation_iterations, relaxation_strength)
        except FairingError:
            # Keep the original diagnostic when neither orientation is valid.
            raise preferred_error from None
        layout['sweep_rotation'] = 1
    else:
        layout['sweep_rotation'] = 0
    return layout


def _grid_layout(points, analysis, relaxation_iterations, relaxation_strength):
    """Choose the best valid fixed-boundary transition layout, not the first."""
    preferred_error = None
    best = None
    solutions = {}
    # Odd rims require one parity cut. Try a few bounded locations rather
    # than introducing more cuts when the first location blocks transitions.
    parity_slots = (0, -1, 1, -2, 2, -3) if len(points) % 2 else (0,)
    # Uniform and alternating endpoint holds cover skew/notched station
    # alignments without inserting vertices into any source edge.
    for hold_mask in (0, -1, 0x55555555, 0xAAAAAAAA, 0x33333333, 0xCCCCCCCC):
        for parity_slot in parity_slots:
            for geometric_schedule in (False, True):
                try:
                    candidate = _grid_layout_attempt(
                        points, analysis, relaxation_iterations, relaxation_strength,
                        hold_mask=hold_mask, parity_slot=parity_slot,
                        geometric_schedule=geometric_schedule,
                        solutions=solutions,
                    )
                    if not candidate['station_holds']:
                        return candidate
                    score = shape_loss(candidate)
                    if best is None or score < best[0]:
                        best = score, candidate
                except FairingError as error:
                    if preferred_error is None:
                        preferred_error = error
    if best is not None:
        return best[1]
    raise preferred_error from None


def _grid_layout_attempt(points, analysis, relaxation_iterations, relaxation_strength, *, hold_mask,
                         parity_slot=0, geometric_schedule=False, solutions=None):
    """Trim and validate one sweep's rectangle, including quad mutations."""
    from .row_patcher import MAX_COLUMNS, MAX_ROWS, _strip, _validate_layout
    from .grid_patcher import _inside_boundary

    counts = list(analysis['side_counts'])
    target = counts.copy()
    if (counts[0] - counts[2]) % 2:
        target[0 if counts[0] < counts[2] else 2] += 1
    width = max(target[0], target[2])
    height = max(target[1], target[3], abs(target[0] - target[2]) // 2)
    target[1] = target[3] = height
    if width > MAX_COLUMNS or height > MAX_ROWS:
        raise FairingError("Manhattan Grid exceeds the 256-row/column safety limit")

    coords, splits = [p.copy() for p in points], []
    divisions = [1] * len(points)
    for side, wanted in zip(analysis['side_edges'], target):
        for _cut in range(wanted - len(side)):
            i = max(side, key=lambda j: (points[(j + 1) % len(points)] - points[j]).length / divisions[j])
            divisions[i] += 1
    boundary, chart = [], {}
    cursor = Vector((0., 0.))
    steps = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))
    for i, pieces in enumerate(divisions):
        direction = analysis['directions'][i]
        step = Vector(steps[direction]) * (width / target[direction] if direction % 2 == 0 else 1.)
        for j in range(pieces):
            identity = i if j == 0 else len(coords)
            if j:
                factor = j / pieces
                coords.append(points[i].lerp(points[(i + 1) % len(points)], factor))
                splits.append((i, factor, identity))
            boundary.append(identity)
            chart[identity] = cursor.copy()
            cursor += step
    if cursor.length > 1.0e-4:
        raise FairingError("Manhattan template does not close")
    low = Vector((min(p.x for p in chart.values()), min(p.y for p in chart.values())))
    chart = {i: p - low for i, p in chart.items()}
    polygon = [chart[i] for i in boundary]
    keys = [(round(p.x, 5), round(p.y, 5)) for p in polygon]
    if len(set(keys)) != len(keys) or _signed_area_2d(polygon) <= 0.:
        raise FairingError("Manhattan template touches or crosses itself")
    rows = round(max(p.y for p in polygon))
    # Each open row is a strip. Disconnected intervals need a later multi-strip
    # extension, not a guessed connection across already-completed geometry.
    spans = []
    for y in range(rows):
        crossings = sorted(a.x + (b.x - a.x) * ((y + .5 - a.y) / (b.y - a.y))
                           for a, b in zip(polygon, polygon[1:] + polygon[:1])
                           if min(a.y, b.y) < y + .5 < max(a.y, b.y))
        if len(crossings) != 2 or crossings[1] - crossings[0] <= 1.0e-6:
            raise FairingError("Manhattan Grid currently requires one connected interval per row")
        spans.append(tuple(crossings))
    if not spans:
        raise FairingError("Manhattan template has no missing cells")

    # A profile includes fixed horizontal rim segments and at most one free
    # interior segment shared by its neighboring rows. This accounts for cells
    # already occupied beside each exterior corner.
    profiles = []
    for y in range(rows + 1):
        neighbors = spans[max(0, y - 1):min(rows, y + 1)]
        extent = min(a for a, b in neighbors), max(b for a, b in neighbors)
        anchors = sorted((i for i in boundary if abs(chart[i].y - y) < 1.0e-5
                          and extent[0] - 1.0e-5 <= chart[i].x <= extent[1] + 1.0e-5),
                         key=lambda i: chart[i].x)
        free = None
        if 0 < y < rows:
            overlap = max(spans[y - 1][0], spans[y][0]), min(spans[y - 1][1], spans[y][1])
            if overlap[1] - overlap[0] <= 1.0e-5:
                raise FairingError("Manhattan row front pinches at an exterior corner")
            endpoints = [next((i for i in anchors if abs(chart[i].x - x) < 1.0e-5), None)
                         for x in overlap]
            if None in endpoints:
                raise FairingError("Manhattan template needs an unresolved boundary cut")
            free = tuple(endpoints)
            if any(overlap[0] + 1.0e-5 < chart[i].x < overlap[1] - 1.0e-5 for i in anchors):
                raise FairingError("Manhattan row intersects an existing boundary intrusion")
        profiles.append({'anchors': anchors, 'free': free})

    # Stations are planning coordinates, not permission to cut the source.
    # A shorter wall holds an existing endpoint for a row while the opposite
    # front advances. Only an odd perimeter needs one real cut for all-quads.
    retained_splits = [splits[parity_slot % len(splits)]] if len(points) % 2 else []
    retained = {identity for _edge, _factor, identity in retained_splits}
    mapped = {identity: edge if factor < .5 or (factor == .5 and not (hold_mask >> (slot % 32) & 1))
              else (edge + 1) % len(points)
              for slot, (edge, factor, identity) in enumerate(splits) if identity not in retained}

    def anchors_for(profile, span):
        return list(dict.fromkeys(mapped.get(i, i) for i in profile['anchors']
                                  if span[0] - 1.0e-5 <= chart[i].x <= span[1] + 1.0e-5))

    def chain_for(profile, count, target_coords):
        chain = anchors_for(profile, (-math.inf, math.inf))
        if profile['free'] is not None:
            a, b = (mapped.get(i, i) for i in profile['free'])
            if a == b:
                raise FairingError("A held row front collapses at a boundary corner")
            insertion = chain.index(a) + 1
            interior = []
            for j in range(1, count):
                interior.append(len(target_coords))
                target_coords.append(coords[a].lerp(coords[b], j / count))
            chain[insertion:insertion] = interior
        return chain

    def base_count(profile, span):
        anchors = anchors_for(profile, span)
        return len(anchors) - 1 - int(profile['free'] is not None)

    # Solve integer profile counts before creating any vertices. Ordinary
    # strips change by two using 3-to-1 quad mutations. A held corner changes
    # by one: the nominal triangle is a repeated identity, not a real face.
    # Prefer no growth, then the minimum changes, centered in the template.
    states, parents = {0: 0.}, []
    for y, span in enumerate(spans):
        following = {}
        lower_base, upper_base = base_count(profiles[y], span), base_count(profiles[y + 1], span)
        lower_anchors, upper_anchors = (anchors_for(profile, span) for profile in profiles[y:y + 2])
        shared_ends = (int(lower_anchors[0] == upper_anchors[0])
                       + int(lower_anchors[-1] == upper_anchors[-1]))
        lower_width = sum((coords[b] - coords[a]).length for a, b in zip(lower_anchors, lower_anchors[1:]))
        upper_width = sum((coords[b] - coords[a]).length for a, b in zip(upper_anchors, upper_anchors[1:]))
        row_height = .5 * ((coords[lower_anchors[0]] - coords[upper_anchors[0]]).length
                          + (coords[lower_anchors[-1]] - coords[upper_anchors[-1]]).length)
        for previous, cost in states.items():
            lower = lower_base + previous
            choices = ([lower + delta - upper_base for delta in (-2, -1, 0, 1, 2)]
                       if profiles[y + 1]['free'] is not None else [0])
            for count in choices:
                upper = upper_base + count
                difference = upper - lower
                if (min(lower, upper) < 1 or abs(difference) > 2
                        or (shared_ends == 0 and difference % 2)
                        or (shared_ends == 1 and abs(difference) != 1)
                        or shared_ends == 2
                        or (profiles[y + 1]['free'] is not None and not 1 <= count <= width)):
                    continue
                change_cost = 10000. * max(difference, 0) + 1000. * abs(difference)
                center_cost = (y - (rows - 1) * .5) ** 2 if difference else 0.
                if geometric_schedule and mapped:
                    # Source station spacing, not an arbitrary grid density,
                    # supplies the desired row aspect. Permit local growth/
                    # reduction mutations when that avoids long, thin cells.
                    aspect = (lower_width + upper_width) / max(row_height * (lower + upper), 1e-12)
                    change_cost = .5 * (max(difference, 0) + abs(difference))
                    change_cost += 2. * (lower + upper) * (aspect + 1. / max(aspect, 1e-9) - 2.)
                    center_cost *= .01
                value = cost + change_cost + center_cost
                if count not in following or value < following[count][0]:
                    following[count] = (value, previous)
        if not following:
            raise FairingError("The fixed boundary needs a different quad transition pattern")
        parents.append(following)
        states = {i: value[0] for i, value in following.items()}
    schedule = [0] * (rows + 1)
    for y in range(rows - 1, -1, -1):
        schedule[y] = parents[y][schedule[y + 1]][1]

    chains = [chain_for(profile, count, coords) for profile, count in zip(profiles, schedule)]
    tolerance = max(max((p - points[0]).length_squared for p in points) * 1.0e-10, 1.0e-16)
    row_faces, row_counts, mutations = [], [], 0
    for y, span in enumerate(spans):
        pair = []
        for k in (0, 1):
            allowed = set(anchors_for(profiles[y + k], span))
            pair.append([i for i in chains[y + k] if i not in chart or i in allowed])
        faces, mutation, triangles = _strip(*pair, coords, tolerance,
                                            allow_unrelaxed=True, quads_only=True)
        if triangles:
            raise FairingError("Manhattan mutation would leave a triangle")
        row_faces.append(faces)
        row_counts.append(tuple(len(chain) - 1 for chain in pair))
        mutations += mutation
    # Virtual stations never become BMesh vertices. Compact them out while
    # retaining source identities and the optional single parity cut.
    used = set(range(len(points))) | retained | {i for row in row_faces for face in row for i in face}
    renumber = {old: new for new, old in enumerate(sorted(used))}
    coords = [coords[old] for old in sorted(used)]
    row_faces = [[[renumber[i] for i in face] for face in row] for row in row_faces]
    splits = [(edge, factor, renumber[identity]) for edge, factor, identity in retained_splits]
    layout = {'coords': coords, 'row_faces': row_faces, 'splits': splits, 'chains': [],
              'station_holds': bool(mapped),
              'rows': rows, 'mutations': mutations, 'triangles': 0,
              'layout_method': 'MANHATTAN_GRID', 'estimated_edge_length': 0.,
              'row_edge_counts': row_counts, 'cut_count': len(splits),
              'grid_cells': [[(x, y) for x in range(width)
                              if _inside_boundary(Vector((x + .5, y + .5)), polygon)]
                             for y in range(rows)]}
    layout.update(analysis)
    layout['grid_size'] = (width, height)
    layout['trimmed_cells'] = width * height - sum(map(len, layout['grid_cells']))
    # Different hold masks/schedules often produce exactly the same compact
    # candidate. Solve each distinct numeric footprint once within this sweep;
    # preserve every candidate and the same fully relaxed quality comparison.
    # The search is bounded, but retaining dozens of very large footprints
    # would still consume unnecessary memory. Large candidates use the same
    # uncached path; ordinary hole patches fit comfortably within this limit.
    if len(coords) > 2048:
        solutions = None
    key = ((tuple(map(tuple, coords)), tuple(tuple(map(tuple, row)) for row in row_faces), tuple(splits))
           if solutions is not None else None)
    cached = solutions.get(key) if solutions is not None else None
    if cached is not None:
        if isinstance(cached, str):
            raise FairingError(cached)
        layout['coords'] = [Vector(p) for p in cached[0]]
        layout['relaxation_passes'], layout['quality_passes'] = cached[1:]
        return layout
    try:
        _validate_layout(layout, points, allow_splits=True)
        layout['relaxation_passes'] = _relax_template(
            layout, len(points), iterations=relaxation_iterations, strength=relaxation_strength)
        layout['quality_passes'] = improve_quad_shapes(
            layout, len(points), iterations=relaxation_iterations, strength=relaxation_strength)
        _validate_layout(layout, points, allow_splits=True)
    except FairingError as error:
        if solutions is not None:
            solutions[key] = str(error)
        raise
    if solutions is not None:
        solutions[key] = (tuple(map(tuple, layout['coords'])), layout['relaxation_passes'], layout['quality_passes'])
    return layout
