# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Inspectable boundary preparation and row fill for the in-house grid patcher."""

import math

import bmesh
from mathutils import Vector

from .fairing import FairingError, ordered_boundary_loop
from .visible_geometry import update_visible_normals, hidden_boundary_vertices, visible_valence


BOUNDARY_MERGE_DISTANCE = 0.001


def _polygon_normal(points):
    origin = points[0]
    return sum(((point - origin).cross(points[(i + 1) % len(points)] - origin)
                for i, point in enumerate(points)), Vector())


def _tracked_loop(bm, boundary):
    local_edges = {edge for vert in boundary if vert.is_valid for edge in vert.link_edges}
    edges = [edge for edge in local_edges if edge.is_valid and not edge.hide
             and len(edge.link_faces) == 1
             and not edge.link_faces[0].hide
             and all(vert in boundary for vert in edge.verts)]
    if not edges:
        raise FairingError("Grid preparation lost the selected boundary")
    bm.edges.index_update()  # Stable walk order for newly constructed rim edges.
    return ordered_boundary_loop(edges=edges)


def _merge_collapsed_ring_edges(bm, loop_verts, loop_edges, corrected_vertices, protected_vertices=(),
                               relative_tolerance=0.):
    """Collapse short connected edges in ring faces, never spatial neighbors.

    Repair can move a corner onto an existing support vertex after the initial
    merge. A short spoke then disguises triangles as quads. Keep repaired
    corners fixed, otherwise prefer an existing support point over the rim.
    Reject pinched links, lost faces, and inverted incident geometry.
    """
    boundary = set(loop_verts)
    merged = 0
    while True:
        ring_faces = {face for edge in loop_edges for face in edge.link_faces}
        def collapse_limit(edge):
            limit = BOUNDARY_MERGE_DISTANCE
            # A clipped source corner may leave a short spoke larger than the
            # absolute weld threshold. Normalize only rim-to-support spokes
            # in quads; never join two nearby, disconnected boundary points.
            if relative_tolerance and sum(v in boundary for v in edge.verts) == 1:
                sizes = [sorted(e.calc_length() for e in face.edges)[-2]
                         for face in edge.link_faces if len(face.verts) == 4]
                limit = max([limit] + [relative_tolerance * size for size in sizes])
            return limit
        candidates = {edge for face in ring_faces for edge in face.edges
                      if not edge.hide and not any(v.hide for v in edge.verts)
                      and any(v in boundary for v in edge.verts)
                      and edge.calc_length() <= collapse_limit(edge)}
        welded = False
        for edge in sorted(candidates, key=lambda e: (e.calc_length(),
                            sorted(tuple(v.co) for v in e.verts))):
            if any(vert in protected_vertices for vert in edge.verts):
                continue
            if not any(len(f.verts) == 4 for f in edge.link_faces):
                continue
            first, second = edge.verts
            first_neighbors = {e.other_vert(first) for e in first.link_edges} - {second}
            second_neighbors = {e.other_vert(second) for e in second.link_edges} - {first}
            if first_neighbors & second_neighbors:
                continue  # A triangular link or duplicate connectivity would pinch.
            target = max(edge.verts, key=lambda v: (v in corrected_vertices, v not in boundary,
                                                    len(v.link_edges), tuple(v.co)))
            source = edge.other_vert(target)
            if source in corrected_vertices and target in corrected_vertices:
                continue
            safe = True
            for face in set(source.link_faces) | set(target.link_faces):
                old = list(face.verts)
                changed = list(dict.fromkeys(target if v is source else v for v in old))
                if len(changed) < 3:
                    safe = False
                    break
                old_normal = _polygon_normal([v.co for v in old])
                new_normal = _polygon_normal([v.co for v in changed])
                extent = max((v.co - target.co).length_squared for v in old)
                if (new_normal.length <= max(extent * 1e-8, 1e-16)
                        or old_normal.dot(new_normal) <= 0.):
                    safe = False
                    break
                for i, vert in enumerate(changed):
                    j = old.index(vert if vert in old else source)
                    before = (old[j].co - old[j - 1].co).cross(old[(j + 1) % len(old)].co - old[j].co).dot(old_normal)
                    after = (vert.co - changed[i - 1].co).cross(changed[(i + 1) % len(changed)].co - vert.co).dot(old_normal)
                    if before > old_normal.length_squared * 1e-6 and after <= 0.:
                        safe = False
                        break
                if not safe:
                    break
            if not safe:
                continue
            if source in boundary:
                boundary.remove(source)
                boundary.add(target)
            if source in corrected_vertices:
                corrected_vertices.remove(source)
                corrected_vertices.add(target)
            bmesh.ops.weld_verts(bm, targetmap={source: target})
            merged += 1
            loop_verts, loop_edges = _tracked_loop(bm, boundary)
            welded = True
            break  # Refresh live faces/edges after every weld.
        if not welded:
            break
    update_visible_normals(bm)
    return loop_verts, loop_edges, merged


def _pentagon_corner(face, edge, matrix_world):
    """Return the intersection beyond both ends of a clipped quad corner.

    The exposed edge is removed, not used as the sliding direction. Each end
    slides along its other edge in the pentagon. Skew/parallel directions,
    distant intersections and folded faces are left for manual inspection.
    """
    ring = list(face.verts)
    index = next(i for i, vert in enumerate(ring)
                 if {vert, ring[(i + 1) % 5]} == set(edge.verts))
    first, second = ring[index], ring[(index + 1) % 5]
    previous, following = ring[index - 1], ring[(index + 2) % 5]
    a, b, p, q = [matrix_world @ vert.co
                  for vert in (first, second, previous, following)]
    first_direction, second_direction = a - p, b - q
    scale = max((matrix_world @ vert.co - a).length for vert in ring)
    tolerance = max(scale * 1.0e-5, 1.0e-9)
    if min(first_direction.length, second_direction.length) <= tolerance:
        return None
    first_direction.normalize()
    second_direction.normalize()
    dot = first_direction.dot(second_direction)
    # Restoring a roughly rectangular corner requires two distinct source
    # directions; a shallow line crossing is not a safe corner hypothesis.
    if abs(dot) > math.cos(math.radians(45.0)):
        return None
    delta = b - a
    denominator = 1.0 - dot * dot
    first_distance = (delta.dot(first_direction)
                      - dot * delta.dot(second_direction)) / denominator
    second_distance = (dot * delta.dot(first_direction)
                       - delta.dot(second_direction)) / denominator
    if (min(first_distance, second_distance) < -tolerance
            or max(first_distance, second_distance) > scale * 2.0):
        return None
    first_hit = a + first_direction * first_distance
    second_hit = b + second_direction * second_distance
    # Keep the orientation/length gates below at their original tolerance.
    # Only the closest-approach test allows near intersections: four times
    # the local geometric tolerance plus a float32 transform-roundoff floor.
    # Absolute transform terms include local-coordinate/translation cancellation;
    # world position alone would underestimate that mesh precision loss.
    transform_magnitude = max(
        sum(abs(matrix_world[row][column] * vert.co[column]) for column in range(3))
        + abs(matrix_world[row][3])
        for vert in ring for row in range(3)
    )
    roundoff = 8.0 * (2.0 ** -23) * transform_magnitude
    intersection_tolerance = max(tolerance * 4.0, min(roundoff, scale * 1.0e-3))
    if (first_hit - second_hit).length > intersection_tolerance:
        return None
    point = (first_hit + second_hit) * 0.5
    quad = [point] + [matrix_world @ ring[(index + offset) % 5].co
                      for offset in (2, 3, 4)]
    normal = (quad[1] - quad[0]).cross(quad[2] - quad[1])
    if normal.length <= tolerance * scale:
        return None
    normal.normalize()
    if _polygon_normal([matrix_world @ vert.co for vert in ring]).dot(normal) <= 0.0:
        return None
    for i, vertex in enumerate(quad):
        incoming = vertex - quad[i - 1]
        outgoing = quad[(i + 1) % 4] - vertex
        if (min(incoming.length, outgoing.length) <= tolerance
                or incoming.cross(outgoing).dot(normal) <= tolerance * scale):
            return None
    # Moving these corners must not invert or collapse another incident face.
    for neighbor in set(first.link_faces) | set(second.link_faces):
        if neighbor is face:
            continue
        if first in neighbor.verts and second in neighbor.verts:
            return None  # A second shared face would be pinched by the weld.
        old = [matrix_world @ vert.co for vert in neighbor.verts]
        new = [point if vert in {first, second} else matrix_world @ vert.co
               for vert in neighbor.verts]
        old_normal = _polygon_normal(old)
        new_normal = _polygon_normal(new)
        if (new_normal.length <= tolerance * scale
                or old_normal.dot(new_normal) <= 0.0):
            return None
    return first, second, matrix_world.inverted_safe() @ point


def _turn(previous, point, following):
    incoming, outgoing = point - previous, following - point
    if min(incoming.length_squared, outgoing.length_squared) <= 1.0e-20:
        return math.pi
    return math.atan2(incoming.cross(outgoing).length, incoming.dot(outgoing))


def _safe_vertex_move(vert, point, matrix_world):
    for face in vert.link_faces:
        old = [matrix_world @ vertex.co for vertex in face.verts]
        new = [point if vertex is vert else matrix_world @ vertex.co for vertex in face.verts]
        normal = _polygon_normal(old)
        changed = _polygon_normal(new)
        if (normal.length_squared <= 1.0e-20
                or changed.dot(normal) < normal.length_squared * 0.05):
            return False
        # Preserve the orientation of each formerly convex corner, not just
        # the face's aggregate area: a quad can fold while its area stays positive.
        for i in range(len(old)):
            original = (old[i] - old[i - 1]).cross(old[(i + 1) % len(old)] - old[i]).dot(normal)
            updated = (new[i] - new[i - 1]).cross(new[(i + 1) % len(new)] - new[i]).dot(normal)
            if original > normal.length_squared * 1.0e-6 and updated < original * 0.02:
                return False
    return True


def _boundary_chart(loop_verts, matrix_world):
    points = [matrix_world @ vert.co for vert in loop_verts]
    normal = _polygon_normal(points)
    if normal.length_squared <= 1.0e-20:
        return None
    normal.normalize()
    origin = points[0]
    axis = max((point - origin - normal * (point - origin).dot(normal)
                for point in points), key=lambda direction: direction.length_squared)
    if axis.length_squared <= 1.0e-20:
        return None
    axis.normalize()
    other = normal.cross(axis)
    project = lambda point: Vector(((point - origin).dot(axis), (point - origin).dot(other)))
    return project, [project(point) for point in points]


def _inside_boundary(point, polygon):
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        delta = b - a
        factor = max(0.0, min(1.0, (point - a).dot(delta) / max(delta.length_squared, 1.0e-20)))
        if (point - (a + factor * delta)).length_squared < 1.0e-16:
            return True
        if (a.y > point.y) != (b.y > point.y):
            crossing = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y)
            if point.x < crossing:
                inside = not inside
    return inside


def _inside_repair(point, first, second, loop_verts, matrix_world):
    """Keep the completed corner and its two new sides inside this opening."""
    chart = _boundary_chart(loop_verts, matrix_world)
    if chart is None:
        return False
    project, polygon = chart
    candidate = project(point)
    scale = max((coordinate - polygon[0]).length for coordinate in polygon)
    if any((candidate - project(matrix_world @ vert.co)).length < scale * 1.0e-6
           for vert in loop_verts):
        return False
    # A new point on another rim segment would create a geometric T-junction
    # even if the new segments have no proper crossing.
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        delta = b - a
        factor = max(0.0, min(1.0, (candidate - a).dot(delta) / max(delta.length_squared, 1.0e-20)))
        if (candidate - (a + delta * factor)).length < scale * 1.0e-7:
            return False
    segments = [(project(matrix_world @ vert.co), candidate) for vert in (first, second)]
    if not _inside_boundary(candidate, polygon):
        return False
    for start, end in segments:
        if not all(_inside_boundary(start.lerp(end, fraction), polygon)
                   for fraction in (0.25, 0.5, 0.75)):
            return False
        # Samples alone can miss a narrow notch. Reject proper crossings of
        # other rim segments; endpoint contact with the two anchors is allowed.
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            cross = lambda p, q: p.x * q.y - p.y * q.x
            denominator = cross(end - start, b - a)
            if abs(denominator) <= max(scale * scale * 1.0e-10, 1.0e-16):
                continue
            t = cross(a - start, b - a) / denominator
            u = cross(a - start, end - start) / denominator
            if 1.0e-6 < t < 1.0 - 1.0e-6 and 1.0e-6 < u < 1.0 - 1.0e-6:
                return False
    return True


def _right_angle_error(support, point, corrected):
    """Angular error between the inward spoke and the corrected-side edge."""
    inward, along = support - point, corrected - point
    if min(inward.length_squared, along.length_squared) <= 1.0e-20:
        return math.pi * 0.5
    return abs(math.atan2(inward.cross(along).length, inward.dot(along)) - math.pi * 0.5)


def _nearest_corrected(bm, loop_verts, index, corrected, matrix_world):
    """Find the nearest corrected anchor along a quad-supported boundary run.

    Uncorrected quad vertices may be passed, but triangle/ngon edges must not
    supply a direction or lead us into an unrelated, unfinished support run.
    """
    choices = []
    count = len(loop_verts)
    for direction in (-1, 1):
        previous = loop_verts[index]
        distance = 0.0
        for offset in range(1, count):
            other = loop_verts[(index + direction * offset) % count]
            edge = bm.edges.get((previous, other))
            if edge is None or any(len(face.verts) != 4 for face in edge.link_faces):
                break
            distance += (matrix_world @ other.co - matrix_world @ previous.co).length
            if other in corrected:
                choices.append((distance, other))
                break
            previous = other
    return min(choices, key=lambda choice: choice[0]) if choices else None


def _slide_boundary(bm, loop_verts, loop_edges, matrix_world, corrected_vertices=()):
    slid = skipped = 0
    boundary = set(loop_edges)
    inverse = matrix_world.inverted_safe()
    # Repaired pentagon corners and fixed quad junctions anchor the quad pass.
    # Successful slides then propagate that corrected region along the rim.
    corrected = set(corrected_vertices)
    corrected.update(vert for vert in loop_verts
                     if vert.link_faces and all(len(face.verts) == 4 for face in vert.link_faces)
                     and sum(edge not in boundary for edge in vert.link_edges) > 1)
    quad_walk = []
    triangle_walk = []
    for item in enumerate(loop_verts):
        vert = item[1]
        if vert in corrected:
            continue
        if any(len(face.verts) == 3 for face in vert.link_faces):
            if any(len(face.verts) > 4 for face in vert.link_faces):
                # An unrepaired n-gon is not a completed grid direction. The
                # triangle turn objective can otherwise slide its two rim
                # endpoints onto each other before bowtie/mirror completion.
                skipped += 1
            else:
                triangle_walk.append(item)
        elif vert.link_faces and all(len(face.verts) == 4 for face in vert.link_faces):
            quad_walk.append(item)
        elif sum(edge not in boundary for edge in vert.link_edges) == 1:
            skipped += 1  # Unrepaired ngon support is not a quad slide target.
    # Correct the nearest eligible quad vertices first, rather than letting
    # arbitrary boundary start/direction choose an unfinished reference side.
    quad_walk = [item for item in quad_walk
                 if sum(edge not in boundary for edge in item[1].link_edges) == 1]
    # Resolve each anchor immediately before its slide, after earlier edits.
    while quad_walk or triangle_walk:
        anchor = None
        if quad_walk:
            choices = [(item, _nearest_corrected(bm, loop_verts, item[0], corrected, matrix_world))
                       for item in quad_walk]
            choices = [(item, reference) for item, reference in choices if reference is not None]
            if not choices:
                skipped += len(quad_walk)
                quad_walk.clear()
                continue
            (i, vert), (_, anchor) = min(choices, key=lambda choice: (choice[1][0], choice[0][0]))
            quad_walk.remove((i, vert))
        else:
            i, vert = triangle_walk.pop(0)
        spokes = [edge for edge in vert.link_edges if edge not in boundary]
        if len(spokes) != 1:
            continue
        previous = matrix_world @ loop_verts[i - 1].co
        point = matrix_world @ vert.co
        following = matrix_world @ loop_verts[(i + 1) % len(loop_verts)].co
        support = matrix_world @ spokes[0].other_vert(vert).co
        direction = point - support
        if anchor is not None:
            reference = matrix_world @ anchor.co
            error = lambda candidate: _right_angle_error(support, candidate, reference)
        else:
            error = lambda candidate: _turn(previous, candidate, following)
        initial = error(point)
        if initial <= 1.0e-5 or direction.length_squared <= 1.0e-20:
            if anchor is not None and initial <= 1.0e-5:
                corrected.add(vert)
            continue
        limit = max(direction.length, (point - previous).length, (following - point).length) * 2.0
        direction.normalize()
        score = lambda distance: error(point + direction * distance)
        samples = [limit * sample / 32.0 for sample in range(33)]
        best = min(range(len(samples)), key=lambda sample: score(samples[sample]))
        lower, upper = samples[max(0, best - 1)], samples[min(32, best + 1)]
        # Quad slides target 90 degrees to their corrected-side reference.
        # Triangle slides retain their separate turning-angle continuation.
        for _ in range(28):
            a = lower + (upper - lower) / 3.0
            b = upper - (upper - lower) / 3.0
            if score(a) < score(b):
                upper = b
            else:
                lower = a
        samples.extend(((lower + upper) * 0.5, lower, upper))
        accepted = None
        for distance in sorted(samples, key=score):
            candidate = point + direction * distance
            if (score(distance) >= initial - 1.0e-5
                    or distance < limit * 1.0e-6):
                continue
            if (_safe_vertex_move(vert, candidate, matrix_world)
                    and _inside_repair(candidate, loop_verts[i - 1],
                                       loop_verts[(i + 1) % len(loop_verts)], loop_verts, matrix_world)):
                accepted = candidate
                break
        if accepted is None:
            skipped += 1
        else:
            vert.co = inverse @ accepted
            slid += 1
            if anchor is not None:
                corrected.add(vert)
    update_visible_normals(bm)
    return slid, skipped


def _quad_with_insert(face, edge, point, matrix_world):
    points = []
    ring = list(face.verts)
    for i, vert in enumerate(ring):
        points.append(matrix_world @ vert.co)
        if {vert, ring[(i + 1) % 3]} == set(edge.verts):
            points.append(point)
    normal = _polygon_normal([matrix_world @ vert.co for vert in ring])
    scale = max((vertex - points[0]).length for vertex in points)
    tolerance = max(scale * scale * 1.0e-6, 1.0e-16)
    if len(points) != 4 or normal.length <= tolerance:
        return False
    normal.normalize()
    return all((points[i] - points[i - 1]).cross(points[(i + 1) % 4] - points[i]).dot(normal) > tolerance
               for i in range(4))


def _expand_triangle(bm, face, exposed, new_vert):
    old_loops = {loop.vert: loop for loop in face.loops}
    ring = list(face.verts)
    vertices = []
    for i, vert in enumerate(ring):
        vertices.append(vert)
        if {vert, ring[(i + 1) % 3]} == set(exposed.verts):
            vertices.append(new_vert)
    quad = bm.faces.new(vertices)
    quad.material_index = face.material_index
    quad.smooth = face.smooth
    for kind in ('bool', 'int', 'float', 'string', 'float_vector', 'color', 'float_color', 'byte_color'):
        layers = getattr(bm.faces.layers, kind, None)
        if layers is not None:
            for layer in layers.values():
                quad[layer] = face[layer]
    # Extend triangle UVs affinely into the completed corner. Existing loop UVs
    # and pin/selection flags are copied unchanged; no custom-data copy call
    # is allowed to invalidate the live BMesh wrappers during this edit.
    first, second = ring[1].co - ring[0].co, ring[2].co - ring[0].co
    offset = new_vert.co - ring[0].co
    d00, d01, d11 = first.dot(first), first.dot(second), second.dot(second)
    denominator = d00 * d11 - d01 * d01
    w1 = (d11 * offset.dot(first) - d01 * offset.dot(second)) / denominator
    w2 = (d00 * offset.dot(second) - d01 * offset.dot(first)) / denominator
    weights = (1.0 - w1 - w2, w1, w2)
    for layer in bm.loops.layers.uv.values():
        values = [old_loops[vert][layer].uv.copy() for vert in ring]
        for loop in quad.loops:
            if loop.vert in old_loops:
                source = old_loops[loop.vert][layer]
                loop[layer].uv = source.uv
                loop[layer].pin_uv = source.pin_uv
            else:
                loop[layer].uv = sum((uv * weight for uv, weight in zip(values, weights)), Vector((0., 0.)))
    for loop in quad.loops:
        if loop.vert in old_loops:
            source = old_loops[loop.vert]
            loop.uv_select_vert_set(source.uv_select_vert)
            loop.uv_select_edge_set(source.uv_select_edge)
    for vert in exposed.verts:
        new_edge = bm.edges.get((vert, new_vert))
        new_edge.seam = new_edge.seam or exposed.seam
        new_edge.smooth = new_edge.smooth and exposed.smooth
    bm.faces.remove(face)
    if not exposed.link_faces:
        bm.edges.remove(exposed)
    return quad


def _bowtie_corner(shared, a, b, faces, border, loop_verts, matrix_world):
    """Minimize the new boundary corner's turn within both source planes."""
    origin, first, second = [matrix_world @ vert.co for vert in (shared, a, b)]
    normals = [_polygon_normal([matrix_world @ vert.co for vert in face.verts])
               for face in faces]
    if min(normal.length_squared for normal in normals) <= 1.0e-20:
        return None
    for normal in normals:
        normal.normalize()
    line = normals[0].cross(normals[1])
    crease = line.length_squared > 1.0e-12
    if crease:
        line.normalize()
        project = lambda point: origin + line * (point - origin).dot(line)
        axes = (line,)
    else:
        project = lambda point: point - normals[0] * (point - origin).dot(normals[0])
        axis = second - first
        axis -= normals[0] * axis.dot(normals[0])
        if axis.length_squared <= 1.0e-20:
            return None
        axis.normalize()
        axes = (axis, normals[0].cross(axis))
    scale = max((matrix_world @ vert.co - origin).length
                for face in faces for vert in face.verts)
    lengths = sorted((matrix_world @ edge.verts[0].co - matrix_world @ edge.verts[1].co).length
                     for face in faces for edge in face.edges)
    minimum_spoke = lengths[len(lengths) // 2] * .1
    inverse = matrix_world.inverted_safe()
    limit = scale * 2.0
    midpoint = (first + second) * .5
    # The chord midpoint gives zero turn on a flat bowtie. On a crease its
    # projection is a useful seed, but the minimum may lie elsewhere along
    # the plane-intersection line when the two rim vertices are asymmetric.
    target = project(midpoint)
    fallback = project(first + second - origin)
    candidates = [target] + [origin.lerp(fallback, factor)
                            for factor in (1., .875, .75, .625, .5, .375, .25, .125)]
    if crease:
        candidates.extend(origin + line * (limit * j / 32.) for j in range(-32, 33))
    valid = lambda point: (
        (point - origin).length <= limit
        and (inverse @ point - shared.co).length > BOUNDARY_MERGE_DISTANCE
        and (point - origin).length > minimum_spoke
        and _turn(first, point, second) < _turn(first, origin, second) - 1e-6
        and max(abs((point - origin).dot(normal)) for normal in normals) <= scale * 1.0e-5
        and all(_quad_with_insert(face, edge, point, matrix_world)
                for face, edge in zip(faces, border))
        and _inside_repair(point, a, b, loop_verts, matrix_world)
    )
    score = lambda point: _turn(first, point, second)
    best = min((point for point in candidates if valid(point)),
               key=lambda point: (score(point), (point - target).length_squared), default=None)
    if best is None:
        return None
    spacing = limit / 32. if crease else limit * .25
    for _iteration in range(16):
        if score(best) <= 1.0e-7:
            break
        candidates = ([best + line * (offset * spacing) for offset in (-1, 1)] if crease
                      else [best + axes[0] * (x * spacing) + axes[1] * (y * spacing)
                            for x in (-1, 0, 1) for y in (-1, 0, 1)])
        for point in sorted(candidates, key=score):
            if score(point) < score(best) - 1.0e-8 and valid(point):
                best = point
                break
        spacing *= .5
    return best


def _complete_triangles(bm, loop_verts, loop_edges, matrix_world, mirrored_corners=None,
                        bowtie_vertices=None, protected_vertices=()):
    boundary_vertices = set(loop_verts)
    bowties = mirrored = skipped = 0
    inverse = matrix_world.inverted_safe()
    # Snapshot the walk: newly completed corners don't recursively propagate
    # into a full fill during this inspection stage.
    for shared in list(loop_verts):
        if not shared.is_valid or shared not in boundary_vertices or shared in protected_vertices:
            continue
        boundary_edges = set(loop_edges)
        border = [edge for edge in shared.link_edges if edge in boundary_edges]
        if len(border) != 2:
            continue
        first, second = [edge.link_faces[0] for edge in border]
        if any(vert in protected_vertices for face in (first, second) for vert in face.verts):
            continue
        if (first is second or len(first.verts) != 3 or len(second.verts) != 3
                or set(first.verts) & set(second.verts) != {shared}):
            continue
        a, b = [edge.other_vert(shared) for edge in border]
        targets = []
        for face, edge in zip((first, second), border):
            opposite = next(v for v in face.verts if v not in edge.verts)
            target = sum((matrix_world @ v.co for v in edge.verts), Vector()) - matrix_world @ opposite.co
            targets.append(target)
        scale = max((matrix_world @ v.co - matrix_world @ shared.co).length
                    for face in (first, second) for v in face.verts)
        directions = [target - matrix_world @ shared.co for target in targets]
        if ((targets[0] - targets[1]).length > scale * .2
                and min(direction.length for direction in directions) > scale * 1e-6
                and directions[0].normalized().dot(directions[1].normalized()) < math.cos(math.radians(45.))
                and all(_quad_with_insert(face, edge, point, matrix_world)
                        and _inside_repair(point, *edge.verts, loop_verts, matrix_world)
                        for face, edge, point in zip((first, second), border, targets))):
            # A shared vertex alone isn't a bowtie. Distinct completion
            # directions belong to separate source cells. A true bowtie's
            # corners lie along the same spoke, even for asymmetric triangles.
            continue
        point = _bowtie_corner(shared, a, b, (first, second), border, loop_verts, matrix_world)
        if point is None:
            # A creased pair may have only a collapsed feasible common spoke.
            # It is not a failed repair until separate half-turn completion
            # below also fails. Do not disguise it as two degenerate quads.
            continue
        new_vert = bm.verts.new(inverse @ point)
        _expand_triangle(bm, first, border[0], new_vert)
        _expand_triangle(bm, second, border[1], new_vert)
        boundary_vertices.remove(shared)
        boundary_vertices.add(new_vert)
        if bowtie_vertices is not None:
            bowtie_vertices.add(new_vert)
        bowties += 1
        loop_verts, loop_edges = _tracked_loop(bm, boundary_vertices)

    for edge in list(loop_edges):
        if not edge.is_valid or len(edge.link_faces) != 1:
            continue
        face = edge.link_faces[0]
        if any(vert in protected_vertices for vert in face.verts):
            continue
        if len(face.verts) != 3 or sum(other in loop_edges for other in face.edges) != 1:
            continue
        a, b = edge.verts
        opposite = next(vert for vert in face.verts if vert not in edge.verts)
        first, second, third = [matrix_world @ vert.co for vert in (a, b, opposite)]
        direction = second - first
        if direction.length_squared <= 1.0e-20:
            skipped += 1
            continue
        # A half-turn about the face normal through the edge midpoint keeps
        # the fourth corner in the source plane and completes a parallelogram.
        # Unlike reflection across the edge, it also reverses the tangential
        # component. Neighboring faces need not already be quads.
        point = first + second - third
        if (not _quad_with_insert(face, edge, point, matrix_world)
                or not _inside_repair(point, a, b, loop_verts, matrix_world)):
            skipped += 1
            continue
        new_vert = bm.verts.new(inverse @ point)
        quad = _expand_triangle(bm, face, edge, new_vert)
        if mirrored_corners is not None:
            mirrored_corners[new_vert] = quad
        boundary_vertices.add(new_vert)
        mirrored += 1
        loop_verts, loop_edges = _tracked_loop(bm, boundary_vertices)
    update_visible_normals(bm)
    return loop_verts, loop_edges, bowties, mirrored, skipped


def _t_shape_error(previous, point, following, support):
    directions = [other - point for other in (previous, following, support)]
    if min(direction.length_squared for direction in directions) <= 1.0e-20:
        return math.inf
    for direction in directions:
        direction.normalize()
    first, second, inward = directions
    return ((first.dot(second) + 1.0) ** 2
            + first.dot(inward) ** 2 + second.dot(inward) ** 2)


def _quad_rectangle_error(face, vert, point, matrix_world):
    points = [point if other is vert else matrix_world @ other.co for other in face.verts]
    corner_index = list(face.verts).index(vert)
    score = 0.0
    # Only these three angles change when this corner moves. The opposite
    # corner is fixed; don't let its constant error bias quad selection.
    for i in (corner_index - 1, corner_index, (corner_index + 1) % 4):
        corner = points[i]
        first, second = points[i - 1] - corner, points[(i + 1) % 4] - corner
        denominator = first.length_squared * second.length_squared
        if denominator <= 1.0e-20:
            return math.inf
        score += first.dot(second) ** 2 / denominator
    others = [matrix_world @ other.co for other in face.verts if other is not vert]
    normal = (others[1] - others[0]).cross(others[2] - others[0])
    if normal.length_squared <= 1.0e-20:
        return math.inf
    scale = max((other - others[0]).length_squared for other in others)
    return score + (point - others[0]).dot(normal) ** 2 / (normal.length_squared * scale)


def _safe_boundary_slide(point, index, loop_verts, matrix_world):
    """Allow sliding either way, but reject crossed edges and T-junctions."""
    chart = _boundary_chart(loop_verts, matrix_world)
    if chart is None:
        return False
    project, polygon = chart
    candidate = project(point)
    count = len(polygon)
    scale = max((other - polygon[0]).length for other in polygon)
    cross = lambda first, second: first.x * second.y - first.y * second.x
    for j, a in enumerate(polygon):
        if j == index:
            continue
        if (candidate - a).length < max(scale * 1.0e-7, 1.0e-10):
            return False
        if j == (index - 1) % count:
            continue
        b = polygon[(j + 1) % count]
        delta = b - a
        factor = max(0.0, min(1.0, (candidate - a).dot(delta) / max(delta.length_squared, 1.0e-20)))
        if (candidate - (a + delta * factor)).length < max(scale * 1.0e-7, 1.0e-10):
            return False
        for start in (polygon[index - 1], polygon[(index + 1) % count]):
            denominator = cross(candidate - start, delta)
            if abs(denominator) <= max(scale * scale * 1.0e-10, 1.0e-16):
                continue
            t = cross(a - start, delta) / denominator
            u = cross(a - start, candidate - start) / denominator
            if 1.0e-6 < t < 1.0 - 1.0e-6 and 1.0e-6 < u < 1.0 - 1.0e-6:
                return False
    return True


def _refine_planar_quad(vert, face, point, matrix_world, valid):
    """Optimize the moved corner and both adjacent angles in the quad plane."""
    if not face.is_valid or len(face.verts) != 4:
        return point, math.inf, math.inf
    score = lambda candidate: _quad_rectangle_error(face, vert, candidate, matrix_world)
    initial = score(point)
    others = [matrix_world @ other.co for other in face.verts if other is not vert]
    axis = others[1] - others[0]
    normal = axis.cross(others[2] - others[0])
    if min(axis.length_squared, normal.length_squared) <= 1.0e-20:
        return point, initial, initial
    axis.normalize()
    normal.normalize()
    other_axis = normal.cross(axis)
    best, best_score = point, initial
    planar = point - normal * (point - others[0]).dot(normal)
    ring = list(face.verts)
    corner = ring.index(vert)
    rectangle = (matrix_world @ ring[corner - 1].co
                 + matrix_world @ ring[(corner + 1) % 4].co
                 - matrix_world @ ring[(corner + 2) % 4].co)
    limit = max((other - point).length for other in others)
    for candidate in (planar, rectangle):
        error = score(candidate)
        if (candidate - point).length <= limit and error < best_score - 1.0e-10 and valid(candidate):
            best, best_score = candidate, error
    # Shared by mirrored corners, triangle-adjacent T points, and T points
    # opposite repaired pentagons. Constraints may block the rectangle seed.
    spacing = limit * .25
    for _iteration in range(12):
        center = best - normal * (best - others[0]).dot(normal)
        candidates = [center + axis * (a * spacing) + other_axis * (b * spacing)
                      for a in (-1, 0, 1) for b in (-1, 0, 1)]
        for candidate in sorted(candidates, key=score):
            error = score(candidate)
            if ((candidate - point).length <= limit
                    and error < best_score - 1.0e-10 and valid(candidate)):
                best, best_score = candidate, error
                break
        spacing *= .5
    return best, initial, best_score


def _refine_triangle_t(vert, edge, support, point, matrix_world, valid):
    """Fit all three variable quad angles, ignoring the unfinished triangle."""
    return _refine_planar_quad(vert, edge.link_faces[0], point, matrix_world, valid)


def _refine_boundary(bm, loop_verts, loop_edges, matrix_world, locked_vertices,
                     mirrored_corners, passes=3, repaired_vertices=(), bowtie_vertices=()):
    """Refine T points, freeing planar axes beside triangles or repaired diagonals."""
    boundary = set(loop_edges)
    repaired = set(repaired_vertices)
    bowties = set(bowtie_vertices)
    inverse = matrix_world.inverted_safe()
    counts = []
    for _pass in range(passes):
        moves = 0
        for i, vert in enumerate(loop_verts):
            if vert in locked_vertices:
                continue
            point = matrix_world @ vert.co
            previous = matrix_world @ loop_verts[i - 1].co
            following = matrix_world @ loop_verts[(i + 1) % len(loop_verts)].co
            valid = lambda candidate: (
                _safe_vertex_move(vert, candidate, matrix_world)
                and _safe_boundary_slide(candidate, i, loop_verts, matrix_world)
            )
            planar_face = mirrored_corners.get(vert)
            if (planar_face is None and vert not in bowties and len(vert.link_edges) == 3
                    and not any(len(face.verts) == 3 for face in vert.link_faces)):
                # Match real Step-2 repairs by live topology, not every locked
                # vertex or a nearby position. The repaired corner stays fixed;
                # its opposite T corner may optimize both planar coordinates.
                opposite_faces = []
                for face in vert.link_faces:
                    if len(face.verts) != 4:
                        continue
                    ring = list(face.verts)
                    if ring[(ring.index(vert) + 2) % 4] in repaired:
                        opposite_faces.append(face)
                if opposite_faces:
                    planar_face = max(opposite_faces, key=lambda face:
                                      _quad_rectangle_error(face, vert, point, matrix_world))
            if planar_face is not None:
                best, initial, best_score = _refine_planar_quad(
                    vert, planar_face, point, matrix_world, valid,
                )
            else:
                spokes = [edge for edge in vert.link_edges if edge not in boundary]
                if len(vert.link_edges) != 3 or len(spokes) != 1:
                    continue
                support = matrix_world @ spokes[0].other_vert(vert).co
                if any(len(face.verts) == 3 for face in vert.link_faces):
                    # The triangle-side border is unfinished. Only the
                    # completed quad's movable corner and its two neighboring
                    # angles should all approach 90 degrees together.
                    completed = [edge for edge in vert.link_edges if edge in boundary
                                 and edge.link_faces
                                 and all(len(face.verts) != 3 for face in edge.link_faces)]
                    if len(completed) != 1:
                        continue  # No unambiguous completed-side angle.
                    best, initial, best_score = _refine_triangle_t(
                        vert, completed[0], support, point, matrix_world, valid,
                    )
                else:
                    direction = point - support
                    if direction.length_squared <= 1.0e-20:
                        continue
                    spoke_length = direction.length
                    direction.normalize()
                    limit = max(spoke_length, (previous - point).length, (following - point).length) * .5
                    lower, upper = max(-limit, -spoke_length * .9), limit
                    # Bowtie completion targets a straight rim, not a rectangle
                    # or a right angle to its newly inserted inward edge.
                    if vert in bowties:
                        score = lambda candidate: _turn(previous, candidate, following)
                    else:
                        score = lambda candidate: _t_shape_error(previous, candidate, following, support)
                    initial = score(point)
                    best, best_score = point, initial
                    samples = [lower + (upper - lower) * j / 32.0 for j in range(33)]
                    winner = min(range(33), key=lambda j: score(point + direction * samples[j]))
                    a, b = samples[max(0, winner - 1)], samples[min(32, winner + 1)]
                    for _iteration in range(28):
                        first, second = a + (b - a) / 3., b - (b - a) / 3.
                        if score(point + direction * first) < score(point + direction * second):
                            b = second
                        else:
                            a = first
                    samples.append((a + b) * .5)
                    for distance in sorted(samples, key=lambda distance: score(point + direction * distance)):
                        candidate = point + direction * distance
                        error = score(candidate)
                        if error < best_score - 1.0e-10 and valid(candidate):
                            best, best_score = candidate, error
                            break
            if best_score < initial - 1.0e-10 and (best - point).length > 1.0e-8:
                vert.co = inverse @ best
                moves += 1
        counts.append(moves)
    update_visible_normals(bm)
    return counts


def _dissolve_two_edge_rim(bm, loop_verts, loop_edges, locked=()):
    """Refresh redundant rim vertices after neighbor repairs, not only once."""
    boundary = set(loop_verts)
    dissolved = 0
    while len(loop_verts) > 3:
        candidate = next((v for v in loop_verts if len(v.link_edges) == 2 and v not in locked
                          and all(len(f.verts) > 3 for f in v.link_faces)), None)
        if candidate is None:
            break
        boundary.remove(candidate)
        bmesh.ops.dissolve_verts(bm, verts=[candidate], use_face_split=False, use_boundary_tear=False)
        dissolved += 1
        loop_verts, loop_edges = _tracked_loop(bm, boundary)
    return loop_verts, loop_edges, dissolved


def _cap_refined_notches(bm, loop_verts, loop_edges, matrix_world, protected_vertices=()):
    """Fill single-cell three-edge U runs bounded by two inside corners."""
    original = list(loop_verts)
    boundary = set(loop_verts)
    faces = []
    for i in range(len(original)):
        if len(boundary) <= 4:
            break
        vertices = [original[(i + j) % len(original)] for j in range(4)]
        if len(set(vertices)) != 4 or any(v not in boundary or v in protected_vertices for v in vertices):
            continue
        a, b, c, d = vertices
        if min(visible_valence(b), visible_valence(c)) < 4 or bm.edges.get((a, d)) is not None:
            continue
        edges = [bm.edges.get(pair) for pair in zip(vertices, vertices[1:])]
        if any(e is None or e not in loop_edges or len(e.link_faces) != 1 for e in edges):
            continue
        orientations = [any(l.vert is first and l.link_loop_next.vert is second
                            for l in edge.link_faces[0].loops)
                        for edge, first, second in zip(edges, vertices, vertices[1:])]
        if len(set(orientations)) != 1:
            continue  # Do not bridge inconsistently oriented support faces.
        points = [matrix_world @ v.co for v in vertices]
        normal = _polygon_normal(points)
        scale = max((p - points[0]).length for p in points)
        if normal.length <= max(scale * scale * 1e-8, 1e-16):
            continue
        normal.normalize()
        if any((points[j] - points[j-1]).cross(points[(j+1)%4] - points[j]).dot(normal)
               <= max(scale * scale * 1e-6, 1e-16) for j in range(4)):
            continue
        if any(abs(math.degrees((points[j-1] - points[j]).angle(points[j+1] - points[j])) - 90.) > 40.
               for j in (1, 2)):
            continue
        # The closing chord and quad interior must remain within this opening;
        # no distance-based face search or boundary subdivision is involved.
        midpoint = (points[0] + points[3]) * .5
        if not _inside_repair(midpoint, a, d, loop_verts, matrix_world):
            continue
        chart = _boundary_chart(loop_verts, matrix_world)
        if chart is None:
            continue
        project, polygon = chart
        if not _inside_boundary(project(sum(points, Vector()) * .25), polygon):
            continue
        support = edges[0].link_faces[0]
        face = bm.faces.new(list(reversed(vertices)) if orientations[0] else vertices)
        face.material_index, face.smooth = support.material_index, support.smooth
        for layer in bm.loops.layers.uv.values():
            for corner in face.loops:
                source = next((l for l in corner.vert.link_loops if l.face is not face), None)
                if source is not None:
                    corner[layer].uv = source[layer].uv
        faces.append(face)
        boundary.difference_update((b, c))
        loop_verts, loop_edges = _tracked_loop(bm, boundary)
    update_visible_normals(bm)
    return loop_verts, loop_edges, faces


def prepare_grid_steps(bm, loop_verts, loop_edges, matrix_world, step='DISSOLVE', refinement_passes=0,
                       fill_rows=0,
                       relaxation_iterations=20, relaxation_strength=.5, match_curvature=True,
                       protected_vertices=None, hypothesis=None, _row_plan_cache=None):
    """Run cumulative preparation, optionally filling the refined edge in rows."""
    if step not in {'DISSOLVE', 'PENTAGONS', 'SLIDE', 'REFINE', 'ROWS'}:
        raise FairingError("Unknown grid preparation step")
    if refinement_passes not in {0, 1, 2, 3}:
        raise FairingError("Grid refinement needs zero to three additional passes")
    # Track the selected loop through topology edits using live vertices,
    # not stored indices or distance to old positions.
    boundary_vertices = set(loop_verts)
    protected_vertices = (hidden_boundary_vertices(bm) if protected_vertices is None
                          else set(protected_vertices))
    corrected_vertices = set()
    dissolved = merged = repaired = skipped = collapsed_merged = 0
    loop_verts, loop_edges, dissolved = _dissolve_two_edge_rim(
        bm, loop_verts, loop_edges, protected_vertices)
    boundary_vertices = set(loop_verts)
    if step in {'PENTAGONS', 'SLIDE', 'REFINE', 'ROWS'}:
        # Gather the ring's incident faces after dissolving two-edge vertices.
        # Merge their complete vertex set, including outer support vertices,
        # without including nearby geometry outside these faces.
        ring_faces = {face for edge in loop_edges for face in edge.link_faces}
        merge_vertices = {vert for face in ring_faces for vert in face.verts}
        # Use the same distance weld as remove_doubles, but retain its mapping:
        # a rim vertex may weld into a support vertex, which must then remain
        # part of the tracked boundary rather than losing the selected loop.
        targetmap = bmesh.ops.find_doubles(
            bm, verts=list(merge_vertices - protected_vertices), dist=BOUNDARY_MERGE_DISTANCE,
        )['targetmap']
        boundary_vertices = {targetmap.get(vert, vert) for vert in boundary_vertices}
        before_merge = len(merge_vertices)
        bmesh.ops.weld_verts(bm, targetmap=targetmap)
        merged = before_merge - sum(vert.is_valid for vert in merge_vertices)
        loop_verts, loop_edges = _tracked_loop(bm, boundary_vertices)
        update_visible_normals(bm)
        visited = set()
        while len(loop_verts) > 3:
            boundary_edges = set(loop_edges)
            candidate = next((edge for edge in loop_edges
                              if len(edge.link_faces[0].verts) == 5
                              and edge.link_faces[0] not in visited), None)
            if candidate is None:
                break
            face = candidate.link_faces[0]
            visited.add(face)
            exposed = [edge for edge in face.edges if edge in boundary_edges]
            corner = (_pentagon_corner(face, candidate, matrix_world)
                      if len(exposed) == 1 else None)
            if corner is None:
                skipped += 1
                continue
            first, second, point = corner
            if first in protected_vertices or second in protected_vertices:
                skipped += 1
                continue
            boundary_vertices.remove(second)
            first.co = point
            second.co = point
            bmesh.ops.weld_verts(bm, targetmap={second: first})
            corrected_vertices.add(first)
            repaired += 1
            loop_verts, loop_edges = _tracked_loop(bm, boundary_vertices)
        loop_verts, loop_edges, collapsed = _merge_collapsed_ring_edges(
            bm, loop_verts, loop_edges, corrected_vertices, protected_vertices,
            relative_tolerance=.16)
        collapsed_merged += collapsed
        loop_verts, loop_edges, extra = _dissolve_two_edge_rim(
            bm, loop_verts, loop_edges, corrected_vertices | protected_vertices)
        dissolved += extra
    slid = bowties = mirrored = slide_skipped = triangle_skipped = 0
    mirrored_corners = {}
    bowtie_vertices = set()
    t_refinement_counts = []
    refinement_counts = []
    if step in {'SLIDE', 'REFINE', 'ROWS'}:
        slid, slide_skipped = _slide_boundary(
            bm, loop_verts, loop_edges, matrix_world, corrected_vertices | protected_vertices,
        )
        # Finish three T-only walks on the original perimeter before adding
        # mirrored corners. Triangle-touching valence-three points participate
        # too; repaired pentagon corners remain explicitly locked.
        t_refinement_counts = _refine_boundary(
            bm, loop_verts, loop_edges, matrix_world, corrected_vertices | protected_vertices, {}, passes=3,
            repaired_vertices=corrected_vertices,
        )
    if step in {'REFINE', 'ROWS'}:
        loop_verts, loop_edges, collapsed = _merge_collapsed_ring_edges(
            bm, loop_verts, loop_edges, corrected_vertices, protected_vertices,
            relative_tolerance=.16)
        collapsed_merged += collapsed
        loop_verts, loop_edges, bowties, mirrored, triangle_skipped = _complete_triangles(
            bm, loop_verts, loop_edges, matrix_world, mirrored_corners, bowtie_vertices,
            protected_vertices=protected_vertices,
        )
        if refinement_passes:
            refinement_counts = _refine_boundary(
                bm, loop_verts, loop_edges, matrix_world, corrected_vertices | protected_vertices,
                mirrored_corners, refinement_passes,
                repaired_vertices=corrected_vertices, bowtie_vertices=bowtie_vertices,
            )
    notch_faces = []
    if step in {'REFINE', 'ROWS'}:
        loop_verts, loop_edges, notch_faces = _cap_refined_notches(
            bm, loop_verts, loop_edges, matrix_world, protected_vertices)
    update_visible_normals(bm)
    result = {
        "method": "GRID_PREP", "fill_style": "GRID", "residual": 0.0,
        "faces": notch_faces, "verts": loop_verts, "boundary_verts": loop_verts,
        "notch_caps": len(notch_faces),
        "boundary_edges": loop_edges, "grid_step": step,
        "dissolved": dissolved, "merged": merged, "repaired": repaired, "skipped": skipped,
        "collapsed_merged": collapsed_merged,
        "slid": slid, "bowties": bowties, "mirrored": mirrored,
        "slide_skipped": slide_skipped, "triangle_skipped": triangle_skipped,
        "refinement_counts": refinement_counts,
        "t_refinement_counts": t_refinement_counts,
    }
    if step == 'REFINE':
        from .fairing import _organic_grid_frame
        from .manhattan import analyze_boundary
        _center, _axis, _x, _y, points, _values = _organic_grid_frame(
            loop_verts, loop_edges, matrix_world)
        try:
            result.update(analyze_boundary(points, [visible_valence(v) for v in loop_verts],
                                           [matrix_world @ v.co for v in loop_verts]))
        except FairingError as error:
            # The Refined Edge is independently inspectable, even when it is
            # not yet suitable for the Manhattan template required by Step 5.
            result['grid_analysis_error'] = str(error)
    if step == 'ROWS':
        from .row_patcher import fill_refined_rows
        result.update(fill_refined_rows(bm, loop_verts, loop_edges, matrix_world,
                                       fill_rows=fill_rows,
                                       relaxation_iterations=relaxation_iterations,
                                       relaxation_strength=relaxation_strength, match_curvature=match_curvature,
                                       hypothesis=hypothesis, protected_vertices=protected_vertices,
                                       _plan_cache=_row_plan_cache))
        result['faces'] = notch_faces + result['faces']
    return result
