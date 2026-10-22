# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hole-boundary validation and ordered analytic/organic fairing hypotheses."""

import math

import bmesh
import bpy
import numpy as np

from mathutils import Vector

from .constants import PARALLEL_PLANE_ANGLE_EPSILON
from .visible_geometry import touches_hidden, update_visible_normals


RELATIVE_RESIDUAL_TOLERANCE = 0.03
MAX_SUPPORT_PLANES = 3
PATCH_RINGS = 5
LIMITED_DISSOLVE_ANGLE = math.radians(0.1)
CYLINDER_MERGE_DISTANCE = 0.0001
COLLAPSED_BOUNDARY_DISTANCE = 1.0e-8


class FairingError(ValueError):
    """Raised when a selected edge component cannot produce a valid patch."""


def boundary_component(seed_edge):
    """Flood one connected component of one-face non-manifold edges."""
    if seed_edge is None or seed_edge.hide or len(seed_edge.link_faces) != 1 or seed_edge.link_faces[0].hide:
        return set()
    result = {seed_edge}
    stack = [seed_edge]
    while stack:
        edge = stack.pop()
        for vert in edge.verts:
            for neighbor in vert.link_edges:
                if (
                    neighbor not in result
                    and not neighbor.hide
                    and len(neighbor.link_faces) == 1
                    and not neighbor.link_faces[0].hide
                ):
                    result.add(neighbor)
                    stack.append(neighbor)
    return result


def ordered_boundary_loop(seed_edge=None, edges=None):
    """Return ordered vertices/edges for one simple closed boundary component."""
    component = set(edges or boundary_component(seed_edge))
    if len(component) < 3:
        raise FairingError("Choose a closed non-manifold edge loop")

    degree = {}
    for edge in component:
        if edge.hide or len(edge.link_faces) != 1:
            raise FairingError("The selected loop must contain one-face boundary edges")
        for vert in edge.verts:
            degree.setdefault(vert, []).append(edge)
    if any(len(linked) != 2 for linked in degree.values()):
        raise FairingError("The non-manifold edge component branches or is open")
    if len(degree) != len(component):
        raise FairingError("The boundary component is not one simple loop")

    first_edge = seed_edge if seed_edge in component else min(
        component, key=lambda edge: edge.index
    )
    start = first_edge.verts[0]
    current_vert = start
    current_edge = first_edge
    ordered_verts = []
    ordered_edges = []
    visited = set()
    while current_edge not in visited:
        visited.add(current_edge)
        ordered_verts.append(current_vert)
        ordered_edges.append(current_edge)
        next_vert = current_edge.other_vert(current_vert)
        next_edges = [
            edge for edge in degree[next_vert] if edge is not current_edge
        ]
        if len(next_edges) != 1:
            raise FairingError("The boundary component is not a simple loop")
        current_vert = next_vert
        current_edge = next_edges[0]

    if current_vert is not start or len(visited) != len(component):
        raise FairingError("The non-manifold edge component is not closed")
    return ordered_verts, ordered_edges


def weld_collapsed_boundary_edges(bm, loop_verts, loop_edges):
    """Weld only collapsed connected rim edges before angle-based cleanup.

    Coincident vertices are distinct topology: a fan still creates a face for
    their zero-length edge. Angle tests cannot classify that edge. Do not use
    a spatial merge, which could weld another nearby opening or source layer.
    Retain support coordinates, reject pinched links/face loss, and track the
    surviving loop with live elements rather than indices or nearest points.
    """
    loop_verts, loop_edges = list(loop_verts), list(loop_edges)
    while True:
        edge = next((edge for edge in loop_edges
                     if edge.calc_length() <= COLLAPSED_BOUNDARY_DISTANCE), None)
        if edge is None:
            break
        if len(loop_verts) <= 3 or any(vert.hide for vert in edge.verts):
            raise FairingError("A collapsed boundary edge cannot be welded safely")
        first, second = edge.verts
        first_neighbors = {e.other_vert(first) for e in first.link_edges} - {second}
        second_neighbors = {e.other_vert(second) for e in second.link_edges} - {first}
        collapsed_faces = {face for face in edge.link_faces
                           if len(face.verts) == 3 and face.calc_area() <= 1.e-14}
        # Triangulation can turn a repeated n-gon corner into a zero-area ear.
        # Its two redundant edges share a neighbor, but one edge belongs only
        # to that ear. This is removable debris, not a link between valid faces.
        for neighbor in first_neighbors & second_neighbors:
            if not any(not (set(bm.edges.get((vert, neighbor)).link_faces) - collapsed_faces)
                       for vert in (first, second)):
                raise FairingError("Welding a collapsed boundary edge would pinch the mesh")
        target = max(edge.verts, key=lambda vert: (len(vert.link_edges), tuple(vert.co)))
        source = edge.other_vert(target)
        for face in set(source.link_faces) | set(target.link_faces):
            old = list(face.verts)
            changed = list(dict.fromkeys(target if vert is source else vert for vert in old))
            if len(changed) < 3:
                if face in collapsed_faces:
                    continue  # Weld removes only this already-zero-area ear.
                raise FairingError("Welding a collapsed boundary edge would remove a face")
            def polygon_normal(vertices):
                origin = vertices[0].co
                return sum(((vert.co - origin).cross(vertices[(i + 1) % len(vertices)].co - origin)
                            for i, vert in enumerate(vertices)), Vector())
            old_normal, new_normal = polygon_normal(old), polygon_normal(changed)
            extent = max((vert.co - target.co).length_squared for vert in old)
            if (new_normal.length <= max(extent * 1.e-12, 1.e-30)
                    or old_normal.dot(new_normal) <= 0.):
                raise FairingError("Welding a collapsed boundary edge would fold a face")
        bmesh.ops.weld_verts(bm, targetmap={source: target})
        loop_verts.remove(source)
        loop_edges = [bm.edges.get((vert, loop_verts[(i + 1) % len(loop_verts)]))
                      for i, vert in enumerate(loop_verts)]
        if any(edge is None or len(edge.link_faces) != 1 for edge in loop_edges):
            raise FairingError("Welding a collapsed edge lost the selected boundary")
    update_visible_normals(bm)
    return loop_verts, loop_edges


def limited_dissolve_boundary(
    bm,
    loop_verts,
    loop_edges,
    angle_limit=LIMITED_DISSOLVE_ANGLE,
    use_all_boundaries=False,
    *,
    completed_faces=None,
):
    """Dissolve a ring; an empty returned loop means it was already closed."""
    loop_verts, loop_edges = weld_collapsed_boundary_edges(bm, loop_verts, loop_edges)
    original_points = [vert.co.copy() for vert in loop_verts]
    support_faces = {
        face for edge in loop_edges for face in edge.link_faces if face.is_valid
    }
    normal_limit = math.cos(angle_limit)
    collinear_indices = []
    for index, vert in enumerate(loop_verts):
        previous = loop_verts[index - 1]
        following = loop_verts[(index + 1) % len(loop_verts)]
        incoming = vert.co - previous.co
        outgoing = following.co - vert.co
        if (
            incoming.length_squared <= 1.0e-16
            or outgoing.length_squared <= 1.0e-16
            or incoming.angle(outgoing) > angle_limit
        ):
            continue
        linked_faces = {
            face
            for edge in loop_edges
            if vert in edge.verts
            for face in edge.link_faces
            if face.is_valid
        }
        if use_all_boundaries or (
            linked_faces
            and min(
                first.normal.dot(second.normal)
                for first in linked_faces
                for second in linked_faces
            ) >= normal_limit
        ):
            collinear_indices.append(index)
    # Dissolving every eligible vertex of a finely sampled smooth circle in one
    # BMesh operation can erase the component before accumulated turn is taken
    # into account. Choose a non-adjacent batch and always retain at least three
    # vertices; a dense circle therefore coarsens by at most every other sample.
    chosen_indices = []
    maximum = max(0, len(loop_verts) - 3)
    for index in collinear_indices:
        if len(chosen_indices) >= maximum:
            break
        previous = (index - 1) % len(loop_verts)
        following = (index + 1) % len(loop_verts)
        if previous in chosen_indices or following in chosen_indices:
            continue
        chosen_indices.append(index)
    if (
        len(loop_verts) > 2
        and 0 in chosen_indices
        and len(loop_verts) - 1 in chosen_indices
    ):
        chosen_indices.pop()
    collinear_verts = [loop_verts[index] for index in chosen_indices]
    if collinear_verts:
        operation = bmesh.ops.dissolve_verts(
            bm,
            verts=collinear_verts,
            use_face_split=False,
            use_boundary_tear=False,
        )
        support_faces.update(element for element in (operation or {}).get('region', ())
                             if isinstance(element, bmesh.types.BMFace))
    update_visible_normals(bm)

    return _rediscover_boundary(
        bm, original_points, support_faces,
        original_verts=loop_verts, original_edges=loop_edges,
        completed_faces=completed_faces,
    )


def _rediscover_boundary(
    bm, original_points, support_faces, *, original_verts, original_edges,
    completed_faces=None,
):
    """Recover only the original opening, never a nearby unrelated boundary."""

    support_faces = {face for face in support_faces if face.is_valid and not face.hide}
    if not support_faces:
        support_faces = {face for vert in original_verts if vert.is_valid
                         for face in vert.link_faces if face.is_valid and not face.hide}
    candidates = {
        edge
        for face in support_faces
        for edge in face.edges
        if edge.is_valid and not edge.hide and len(edge.link_faces) == 1
    }
    anchors = {vert for vert in original_verts if vert.is_valid}
    old_edges = {edge for edge in original_edges if edge.is_valid}
    candidates = {edge for edge in candidates
                  if edge in old_edges or any(vert in anchors for vert in edge.verts)}
    if not original_points:
        raise FairingError("Limited dissolve lost the original boundary scope")
    if not candidates:
        # Boundary-inclusive dissolve can close a planar opening by itself.
        # Keep the surviving local surface instead of trying another hole or
        # fitting/projecting a second patch over an already completed repair.
        if not support_faces or any(face.calc_area() <= 1.e-14 for face in support_faces):
            raise FairingError("Limited dissolve removed the boundary and its support surface")
        if completed_faces is not None:
            completed_faces.extend(support_faces)
        return [], []

    def distance_to_original(edge):
        midpoint = (edge.verts[0].co + edge.verts[1].co) * 0.5
        return min(
            (midpoint - point).length_squared for point in original_points
        )

    seed = min(candidates, key=distance_to_original)
    return ordered_boundary_loop(seed, boundary_component(seed))


def _limited_dissolve_analytic_boundary(
    obj, bm, loop_verts, loop_edges, *, completed_faces=None,
):
    """Run Blender's boundary-inclusive dissolve before an analytic repair."""
    loop_verts, loop_edges = weld_collapsed_boundary_edges(bm, loop_verts, loop_edges)
    original_points = [vert.co.copy() for vert in loop_verts]
    support_faces = {
        face for edge in loop_edges for face in edge.link_faces if face.is_valid
    }
    previous_select_mode = tuple(bpy.context.tool_settings.mesh_select_mode)
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)
    for vert in bm.verts:
        vert.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    for face in support_faces:
        if face.is_valid:
            face.select = True
    for vert in loop_verts:
        if vert.is_valid:
            vert.select = True
    for edge in loop_edges:
        if edge.is_valid:
            edge.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
    try:
        bpy.ops.mesh.dissolve_limited(
            angle_limit=LIMITED_DISSOLVE_ANGLE,
            use_dissolve_boundaries=True,
            delimit={'NORMAL'},
        )
    finally:
        bpy.context.tool_settings.mesh_select_mode = previous_select_mode
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    update_visible_normals(bm)
    # Native dissolve may replace every source support face. Selection tracks
    # its surviving output, while old element references alone cannot do so.
    support_faces.update(face for face in bm.faces if face.select and not face.hide)
    loop_verts, loop_edges = _rediscover_boundary(
        bm, original_points, support_faces,
        original_verts=loop_verts, original_edges=loop_edges,
        completed_faces=completed_faces,
    )
    return bm, loop_verts, loop_edges


def _limited_dissolve_cylinder_patch(obj, bm, patch_faces):
    """Grow a cylinder repair once and run Blender's final limited dissolve."""
    previous_select_mode = tuple(bpy.context.tool_settings.mesh_select_mode)
    bpy.ops.mesh.select_mode(
        use_extend=False, use_expand=False, type='FACE'
    )
    for vert in bm.verts:
        vert.select = False
    for edge in bm.edges:
        edge.select = False
    for face in bm.faces:
        face.select = False
    for face in patch_faces:
        if face.is_valid:
            face.select = True
    bm.select_flush_mode()
    bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
    try:
        bpy.ops.mesh.select_more(use_face_step=True)
        bpy.ops.mesh.select_mode(
            use_extend=False, use_expand=False, type='EDGE'
        )
        bpy.ops.mesh.dissolve_limited(
            angle_limit=LIMITED_DISSOLVE_ANGLE,
            use_dissolve_boundaries=False,
            delimit={'NORMAL'},
        )
    finally:
        bpy.context.tool_settings.mesh_select_mode = previous_select_mode
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    update_visible_normals(bm)
    faces = [face for face in bm.faces if face.select and face.is_valid]
    verts = list({vert for face in faces for vert in face.verts if vert.is_valid})
    return bm, {"verts": verts, "faces": faces}


def limited_dissolve_patch(
    bm,
    loop_verts,
    loop_edges,
    patch_verts,
    patch_faces,
    angle_limit=LIMITED_DISSOLVE_ANGLE,
    simplify=True,
):
    """Weld the patch and optionally run its final expanded dissolve."""
    affected_verts = {
        vert for vert in list(loop_verts) + list(patch_verts) if vert.is_valid
    }
    local_scale = _scale([vert.co for vert in affected_verts])
    merge_distance = max(local_scale * 1.0e-6, 1.0e-9)
    bmesh.ops.remove_doubles(
        bm, verts=list(affected_verts), dist=merge_distance
    )
    affected_verts = {
        vert for vert in affected_verts if vert.is_valid
    }

    seed_faces = {face for face in patch_faces if face.is_valid}
    if not seed_faces:
        seed_faces = {
            face
            for vert in affected_verts
            for face in vert.link_faces
            if face.is_valid
        }
    if not simplify:
        update_visible_normals(bm)
        faces = list(seed_faces)
        verts = list({
            vert for face in faces for vert in face.verts if vert.is_valid
        })
        return {"verts": verts, "faces": faces}
    return expanded_limited_dissolve(bm, seed_faces, angle_limit)


def expanded_limited_dissolve(
    bm,
    seed_faces,
    angle_limit=LIMITED_DISSOLVE_ANGLE,
):
    """Expand a face region by one ring and limited-dissolve that region."""
    seed_faces = {face for face in seed_faces if face.is_valid and not face.hide}
    expanded_faces = set(seed_faces)
    for face in tuple(seed_faces):
        for edge in face.edges:
            expanded_faces.update(
                neighbor for neighbor in edge.link_faces if neighbor.is_valid and not neighbor.hide
            )
    affected_verts = {
        vert for face in expanded_faces for vert in face.verts if vert.is_valid and not touches_hidden(vert)
    }
    affected_edges = {
        edge for face in expanded_faces for edge in face.edges
        if edge.is_valid and not edge.hide and not any(linked.hide for linked in edge.link_faces)
    }
    operation = bmesh.ops.dissolve_limit(
        bm,
        angle_limit=angle_limit,
        use_dissolve_boundaries=False,
        verts=list(affected_verts),
        edges=list(affected_edges),
        delimit={'NORMAL'},
    )
    update_visible_normals(bm)

    faces = {face for face in expanded_faces if face.is_valid}
    faces.update(
        element
        for element in operation.get("region", ())
        if isinstance(element, bmesh.types.BMFace) and element.is_valid
    )
    if not faces:
        faces = {
            face
            for vert in affected_verts
            if vert.is_valid
            for face in vert.link_faces
            if face.is_valid
        }
    verts = {
        vert for face in faces for vert in face.verts if vert.is_valid
    }
    return {"verts": list(verts), "faces": list(faces)}


def _world_points(verts, matrix_world):
    return [matrix_world @ vert.co for vert in verts]


def _scale(points):
    if not points:
        return 1.0
    center = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
    return max((point - center).length for point in points) or 1.0


def _fit_plane(points):
    """Fit a plane and return its point, normal, maximum error, and rank."""
    array = np.asarray(points, dtype=np.float64)
    center = array.mean(axis=0)
    centered = array - center
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > max(singular_values[0], 1.0) * 1e-10))
    normal = Vector(vh[-1]).normalized()
    errors = np.abs(centered @ np.asarray(normal))
    return {
        "point": Vector(center),
        "normal": normal,
        "max_error": float(errors.max(initial=0.0)),
        "rank": rank,
        "points": [Vector(point) for point in points],
    }


def _planes_compatible(points, tolerance):
    if len(points) < 3:
        return True
    plane = _fit_plane(points)
    return plane["rank"] >= 2 and plane["max_error"] <= tolerance


def _merge_coplanar_groups(groups, tolerance):
    """Merge any plane runs that describe the same geometric plane.

    A plane may occur in several non-adjacent rim regions because topology,
    missing faces, or the cyclic walk separates its supporting faces.  Prefer
    the compatible pair with the smallest combined residual, then refit after
    every merge so tolerance does not accumulate transitively.
    """
    groups = [list(group) for group in groups]
    while len(groups) > 1:
        best = None
        for first_index, first in enumerate(groups[:-1]):
            for second_index in range(first_index + 1, len(groups)):
                combined = first + groups[second_index]
                if not _planes_compatible(combined, tolerance):
                    continue
                residual = _fit_plane(combined)["max_error"]
                candidate = (
                    residual,
                    first_index,
                    second_index,
                    combined,
                )
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None:
            break
        _residual, first_index, second_index, combined = best
        groups[first_index] = combined
        groups.pop(second_index)
    return groups


def _support_planes(
    loop_edges, matrix_world, tolerance, support_faces=None
):
    """Progressively group consecutive rim supports into at most three planes."""
    edge_samples = []
    faces = (
        list(support_faces)
        if support_faces is not None
        else [edge.link_faces[0] for edge in loop_edges]
    )
    if len(faces) != len(loop_edges):
        raise ValueError("Each rim edge needs one adjacent support face")
    for face in faces:
        samples = []
        seen = set()
        for vert in face.verts:
            if vert.index not in seen:
                seen.add(vert.index)
                samples.append(matrix_world @ vert.co)
        edge_samples.append(samples)

    groups = []
    current = []
    for samples in edge_samples:
        candidate = current + samples
        if current and not _planes_compatible(candidate, tolerance):
            groups.append(current)
            current = list(samples)
        else:
            current = candidate
    if current:
        groups.append(current)

    # The cyclic seam is only one way a physical plane can be split. Merge all
    # compatible runs, including disconnected/non-adjacent regions, before the
    # three-distinct-plane limit is enforced.
    groups = _merge_coplanar_groups(groups, tolerance)
    if not groups or len(groups) > MAX_SUPPORT_PLANES:
        return None

    planes = [_fit_plane(points) for points in groups]
    if any(
        plane["rank"] < 2 or plane["max_error"] > tolerance
        for plane in planes
    ):
        return None
    return planes


def _plane_intersection(planes, reference, tolerance):
    """Return the point/line intersection nearest ``reference``."""
    if len(planes) == 1:
        plane = planes[0]
        return reference - plane["normal"] * plane["normal"].dot(
            reference - plane["point"]
        )

    parallel_threshold = math.cos(PARALLEL_PLANE_ANGLE_EPSILON)
    if any(
        abs(first["normal"].normalized().dot(second["normal"].normalized()))
        > parallel_threshold
        for index, first in enumerate(planes[:-1])
        for second in planes[index + 1:]
    ):
        return None

    system = np.asarray([plane["normal"] for plane in planes], dtype=np.float64)
    rhs = np.asarray(
        [plane["normal"].dot(plane["point"]) for plane in planes],
        dtype=np.float64,
    )
    point, _residuals, rank, _values = np.linalg.lstsq(system, rhs, rcond=None)
    if rank < 2:
        return None
    point = Vector(point)
    if rank == 2:
        _u, _s, vh = np.linalg.svd(system, full_matrices=True)
        direction = Vector(vh[-1]).normalized()
        point += direction * (reference - point).dot(direction)
    if max(
        abs(plane["normal"].dot(point - plane["point"]))
        for plane in planes
    ) > tolerance:
        return None
    return point


def fit_plane_hypothesis(
    loop_verts,
    loop_edges,
    matrix_world,
    support_faces=None,
    relative_tolerance=0.005,
):
    """Fit one to three consecutive support planes and their intersection."""
    points = _world_points(loop_verts, matrix_world)
    scale = _scale(points)
    tolerance = max(scale * relative_tolerance, 1.0e-7)
    planes = _support_planes(
        loop_edges, matrix_world, tolerance, support_faces
    )
    if planes is None:
        return None
    center = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
    intersection = _plane_intersection(planes, center, tolerance)
    if intersection is None or (intersection - center).length > scale * 4.0:
        return None
    return {
        "method": "PLANES",
        "center": intersection,
        "planes": planes,
        "residual": max(plane["max_error"] for plane in planes),
    }


def fit_conic_hypothesis(
    loop_edges,
    fit_cone,
    support_faces=None,
    residual_tolerance=RELATIVE_RESIDUAL_TOLERANCE,
    force_cylinder=False,
):
    """Accept the shared circular-cone fit when its radial residual is low."""
    faces = list(
        set(support_faces)
        if support_faces is not None
        else {face for edge in loop_edges for face in edge.link_faces}
    )
    faces = [face for face in faces if not face.hide]
    if len(faces) < 3:
        return None
    try:
        fit = fit_cone(faces)
    except (ValueError, np.linalg.LinAlgError):
        return None
    residual = float(fit.get("fit_relative_error", math.inf))
    if (
        fit.get("fit_rank", 0) < 2
        or (
            residual_tolerance is not None
            and residual > residual_tolerance
        )
    ):
        return None
    if force_cylinder:
        fit = dict(fit)
        fit.update({"slope": 0.0, "half_angle": 0.0, "is_cone": False})
    return {"method": "CONIC", "fit": fit, "residual": residual}


def fit_sphere_hypothesis(
    loop_edges,
    matrix_world,
    world_normal,
    support_faces=None,
    residual_tolerance=RELATIVE_RESIDUAL_TOLERANCE,
):
    """Fit a sphere by intersecting adjacent-face normal lines."""
    faces = list(
        set(support_faces)
        if support_faces is not None
        else {face for edge in loop_edges for face in edge.link_faces}
    )
    faces = [face for face in faces if not face.hide]
    if len(faces) < 3:
        return None
    projectors = []
    rhs = Vector((0.0, 0.0, 0.0))
    samples = []
    identity = np.identity(3)
    for face in faces:
        point = matrix_world @ face.calc_center_median()
        normal = world_normal(face).normalized()
        projector = identity - np.outer(normal, normal)
        projectors.append(projector)
        rhs += Vector(projector @ np.asarray(point))
        samples.append((point, normal))
    system = np.sum(projectors, axis=0)
    center_array, _residuals, rank, values = np.linalg.lstsq(
        system, np.asarray(rhs), rcond=None
    )
    if rank < 3 or values[-1] <= values[0] * 1.0e-8:
        return None
    center = Vector(center_array)
    radii = [(point - center).length for point, _normal in samples]
    radius = sum(radii) / len(radii)
    if radius <= 1.0e-9:
        return None
    radial_residual = math.sqrt(
        sum((value - radius) ** 2 for value in radii) / len(radii)
    ) / radius
    normal_residual = math.sqrt(
        sum(
            (1.0 - abs(normal.dot((point - center).normalized()))) ** 2
            for point, normal in samples
        ) / len(samples)
    )
    residual = max(radial_residual, normal_residual)
    if residual_tolerance is not None and residual > residual_tolerance:
        return None
    return {
        "method": "SPHERE",
        "center": center,
        "radius": radius,
        "residual": residual,
    }


def choose_hypothesis(
    loop_verts,
    loop_edges,
    matrix_world,
    fit_cone,
    world_normal,
    support_faces=None,
    analytic_faces=None,
    plane_relative_tolerance=0.005,
    forced_method='AUTO',
):
    """Try planes, compare conic/sphere residuals, then use organic fairing."""
    if forced_method == 'ORGANIC':
        return {"method": "ORGANIC", "residual": 0.0}

    try:
        plane = fit_plane_hypothesis(
            loop_verts,
            loop_edges,
            matrix_world,
            support_faces,
            plane_relative_tolerance,
        )
    except (ValueError, np.linalg.LinAlgError):
        plane = None
    if forced_method == 'PLANES':
        if plane is None:
            raise FairingError(
                "The selected rim does not define a stable N-plane fit"
            )
        return plane
    if plane is not None and forced_method == 'AUTO':
        return plane

    # These hypotheses explain many of the same curved samples. Evaluate both
    # before choosing so a merely acceptable cylinder cannot mask a distinctly
    # better spherical fit (or vice versa).
    analytic_faces = analytic_faces or support_faces
    conic = fit_conic_hypothesis(
        loop_edges,
        fit_cone,
        analytic_faces,
        None if forced_method == 'CYLINDER' else RELATIVE_RESIDUAL_TOLERANCE,
        force_cylinder=forced_method == 'CYLINDER',
    )
    try:
        sphere = fit_sphere_hypothesis(
            loop_edges,
            matrix_world,
            world_normal,
            analytic_faces,
            None if forced_method == 'SPHERE' else RELATIVE_RESIDUAL_TOLERANCE,
        )
    except (ValueError, np.linalg.LinAlgError):
        sphere = None
    if forced_method == 'CYLINDER':
        if conic is None:
            raise FairingError("The selected rim does not define a stable cylinder fit")
        return conic
    if forced_method == 'SPHERE':
        if sphere is None:
            raise FairingError("The selected rim does not define a stable sphere fit")
        return sphere
    analytic = [fit for fit in (conic, sphere) if fit is not None]
    if analytic:
        return min(analytic, key=lambda fit: fit["residual"])
    return {"method": "ORGANIC", "residual": 0.0}


def _project_conic(point, fit):
    offset = point - fit["center"]
    coordinate = offset.dot(fit["axis"])
    radial = offset - fit["axis"] * coordinate
    direction = (
        radial.normalized()
        if radial.length_squared > 1.0e-16
        else fit["basis_x"]
    )
    radius = max(1.0e-9, fit["radius"] + fit.get("slope", 0.0) * coordinate)
    return fit["center"] + fit["axis"] * coordinate + direction * radius


def _project_sphere(point, center, radius, fallback):
    direction = point - center
    if direction.length_squared <= 1.0e-16:
        direction = fallback
    return center + direction.normalized() * radius


def _orient_loop(loop_verts, loop_edges, center_local):
    """Orient the loop so new faces agree with the existing rim face winding."""
    first = loop_verts[0]
    following = loop_verts[1]
    normal = (following.co - first.co).cross(center_local - first.co)
    adjacent = loop_edges[0].link_faces[0].normal
    if normal.dot(adjacent) < 0.0:
        return list(reversed(loop_verts))
    return list(loop_verts)


def _new_face(bm, verts):
    try:
        return bm.faces.new(verts)
    except ValueError as error:
        raise FairingError("The selected opening is already filled") from error


def _cross_2d(first, second):
    return first.x * second.y - first.y * second.x


def _signed_area_2d(points):
    return 0.5 * sum(
        _cross_2d(point, points[(index + 1) % len(points)])
        for index, point in enumerate(points)
    )


def _organic_projection_frame(loop_verts, matrix_world):
    """Fit the hole plane and return an oriented 2D projection frame."""
    boundary = _world_points(loop_verts, matrix_world)
    plane = _fit_plane(boundary)
    if plane["rank"] < 2:
        raise FairingError("The organic boundary has no stable projection plane")
    center = plane["point"]
    axis = plane["normal"].normalized()
    projected = [
        offset - axis * offset.dot(axis)
        for point in boundary
        for offset in (point - center,)
    ]
    basis_x = max(projected, key=lambda vector: vector.length_squared)
    if basis_x.length_squared <= 1.0e-16:
        raise FairingError("The organic boundary has no stable projection axis")
    basis_x.normalize()
    basis_y = axis.cross(basis_x).normalized()

    def coordinates():
        return [
            Vector(((point - center).dot(basis_x), (point - center).dot(basis_y)))
            for point in boundary
        ]

    points_2d = coordinates()
    if _signed_area_2d(points_2d) < 0.0:
        axis.negate()
        basis_y.negate()
        points_2d = coordinates()
    return center, axis, basis_x, basis_y, points_2d


def _organic_direction_samples(loop_verts, loop_edges, matrix_world):
    """Snapshot the highlighted-style spokes incident to the boundary loop."""
    boundary_edges = set(loop_edges)
    samples = []
    seen = set()
    for vert in loop_verts:
        point = matrix_world @ vert.co
        for edge in vert.link_edges:
            if edge in boundary_edges or not edge.is_valid or edge.hide or (
                edge.link_faces and all(face.hide for face in edge.link_faces)
            ):
                continue
            key = (edge.index, vert.index)
            if key in seen:
                continue
            seen.add(key)
            samples.append((
                point.copy(),
                matrix_world @ edge.other_vert(vert).co - point,
            ))
    return samples


def _organic_grid_frame(
    loop_verts,
    loop_edges,
    matrix_world,
    direction_samples=None,
    grid_angle=0.0,
):
    """Align the fitted hole plane to SVD directions of exterior edges."""
    center, axis, initial_x, initial_y, _boundary = _organic_projection_frame(
        loop_verts, matrix_world
    )
    direction_samples = direction_samples or _organic_direction_samples(
        loop_verts, loop_edges, matrix_world
    )
    directions = []
    for _point, source_direction in direction_samples:
        delta = source_direction - axis * source_direction.dot(axis)
        if delta.length_squared <= 1.0e-16:
            continue
        delta.normalize()
        directions.append((delta.dot(initial_x), delta.dot(initial_y)))

    if directions:
        samples = np.asarray(directions, dtype=np.float64)
        angles = np.arctan2(samples[:, 1], samples[:, 0])
        fourth = np.asarray((
            np.cos(angles * 4.0).sum(),
            np.sin(angles * 4.0).sum(),
        ))
        if np.linalg.norm(fourth) > len(samples) * 1.0e-6:
            fold_angle = math.atan2(fourth[1], fourth[0]) * 0.25
            seed = Vector((math.cos(fold_angle), math.sin(fold_angle)))
            folded = []
            for sample in samples:
                direction = Vector(sample)
                candidates = (
                    direction,
                    Vector((-direction.y, direction.x)),
                    -direction,
                    Vector((direction.y, -direction.x)),
                )
                folded.append(max(candidates, key=lambda value: value.dot(seed)))
            samples = np.asarray(folded, dtype=np.float64)
    else:
        samples = np.asarray([
            (
                (matrix_world @ vert.co - center).dot(initial_x),
                (matrix_world @ vert.co - center).dot(initial_y),
            )
            for vert in loop_verts
        ], dtype=np.float64)
    _u, singular_values, vh = np.linalg.svd(samples, full_matrices=False)
    principal = Vector(vh[0]).normalized()
    basis_x = (
        initial_x * principal.x + initial_y * principal.y
    ).normalized()
    basis_y = axis.cross(basis_x).normalized()
    if abs(grid_angle) > 1.0e-12:
        cosine = math.cos(grid_angle)
        sine = math.sin(grid_angle)
        basis_x = (basis_x * cosine + basis_y * sine).normalized()
        basis_y = axis.cross(basis_x).normalized()
    boundary = [
        Vector((
            (matrix_world @ vert.co - center).dot(basis_x),
            (matrix_world @ vert.co - center).dot(basis_y),
        ))
        for vert in loop_verts
    ]
    if _signed_area_2d(boundary) < 0.0:
        axis.negate()
        basis_y.negate()
        boundary = [Vector((point.x, -point.y)) for point in boundary]
    return center, axis, basis_x, basis_y, boundary, singular_values


def _create_plane_patch(bm, loop_verts, loop_edges, center_local):
    ordered = _orient_loop(loop_verts, loop_edges, center_local)
    center_vert = bm.verts.new(center_local)
    faces = []
    for index, vert in enumerate(ordered):
        faces.append(
            _new_face(bm, (vert, ordered[(index + 1) % len(ordered)], center_vert))
        )
    return [center_vert], faces


def _radial_patch_positions(
    loop_verts, matrix_world, hypothesis, patch_rings=PATCH_RINGS
):
    boundary = _world_points(loop_verts, matrix_world)
    centroid = sum(boundary, Vector((0.0, 0.0, 0.0))) / len(boundary)
    method = hypothesis["method"]
    if method == "CONIC":
        fit = hypothesis["fit"]
        project = lambda point: _project_conic(point, fit)
        centroid_offset = centroid - fit["center"]
        centroid_axial = fit["axis"] * centroid_offset.dot(fit["axis"])
        centroid_radial = centroid_offset - centroid_axial
        mean_radius = sum(
            (
                point
                - fit["center"]
                - fit["axis"]
                * (point - fit["center"]).dot(fit["axis"])
            ).length
            for point in boundary
        ) / len(boundary)
        slope = float(fit.get("slope", 0.0))
        if (
            fit.get("is_cone", False)
            and abs(slope) > 1.0e-9
            and centroid_radial.length <= mean_radius * 0.25
        ):
            # A full circular conic rim closes at the fitted apex. Projecting
            # its centroid would choose an arbitrary radial direction instead.
            center = fit["center"] + fit["axis"] * (
                -float(fit["radius"]) / slope
            )
        else:
            center = project(centroid)
    elif method == "SPHERE":
        sphere_center = hypothesis["center"]
        fallback = boundary[0] - sphere_center
        project = lambda point: _project_sphere(
            point, sphere_center, hypothesis["radius"], fallback
        )
        center = project(centroid)
    elif method == "PLANES":
        project = lambda point: point
        center = hypothesis["center"]
    else:
        project = lambda point: point
        center = centroid

    rings = []
    patch_rings = max(0, int(patch_rings))
    for level in range(1, patch_rings + 1):
        factor = level / (patch_rings + 1)
        rings.append(
            [project(point.lerp(center, factor)) for point in boundary]
        )
    return center, rings


def _create_radial_patch(
    bm,
    loop_verts,
    loop_edges,
    matrix_world,
    hypothesis,
    patch_rings=PATCH_RINGS,
):
    inverse = matrix_world.inverted_safe()
    center_world, ring_positions = _radial_patch_positions(
        loop_verts, matrix_world, hypothesis, patch_rings
    )
    center_local = inverse @ center_world
    ordered = _orient_loop(loop_verts, loop_edges, center_local)
    if ordered[0] is not loop_verts[0]:
        ring_positions = [list(reversed(ring)) for ring in ring_positions]

    new_verts = []
    rings = []
    for positions in ring_positions:
        ring = [bm.verts.new(inverse @ point) for point in positions]
        rings.append(ring)
        new_verts.extend(ring)
    center_vert = bm.verts.new(center_local)
    new_verts.append(center_vert)

    faces = []
    previous = ordered
    for ring in rings:
        for index, vert in enumerate(previous):
            following = (index + 1) % len(previous)
            faces.append(
                _new_face(
                    bm,
                    (vert, previous[following], ring[following], ring[index]),
                )
            )
        previous = ring
    for index, vert in enumerate(previous):
        faces.append(
            _new_face(
                bm,
                (vert, previous[(index + 1) % len(previous)], center_vert),
            )
        )
    return new_verts, faces


def _create_cylinder_patch(
    bm, loop_verts, loop_edges, matrix_world, hypothesis
):
    """Close a cylindrical opening by collapsing its rim along the fit axis."""
    fit = hypothesis["fit"]
    axis = fit["axis"].normalized()
    boundary = _world_points(loop_verts, matrix_world)
    axial_center = sum(
        (point - fit["center"]).dot(axis) for point in boundary
    ) / len(boundary)
    inverse = matrix_world.inverted_safe()
    affected_faces = {
        face for edge in loop_edges for face in edge.link_faces if face.is_valid
    }
    affected_edges = {
        edge
        for face in affected_faces
        for edge in face.edges
        if edge.is_valid
    }
    for vert, point in zip(loop_verts, boundary):
        coordinate = (point - fit["center"]).dot(axis)
        collapsed = point + axis * (axial_center - coordinate)
        vert.co = inverse @ collapsed

    # This is the mesh equivalent of scaling the selected rim to zero along the
    # cylinder axis, followed by Blender's Merge by Distance at exactly 0.0001.
    bmesh.ops.remove_doubles(
        bm,
        verts=[vert for vert in loop_verts if vert.is_valid],
        dist=CYLINDER_MERGE_DISTANCE,
    )
    degenerate_edges = [
        edge for edge in affected_edges if edge.is_valid
        if edge.calc_length() <= CYLINDER_MERGE_DISTANCE
    ]
    if degenerate_edges:
        bmesh.ops.dissolve_degenerate(
            bm, edges=degenerate_edges, dist=CYLINDER_MERGE_DISTANCE
        )

    zero_area_faces = [
        face for face in affected_faces
        if face.is_valid and face.calc_area() <= 1.0e-14
    ]
    if zero_area_faces:
        bmesh.ops.delete(bm, geom=zero_area_faces, context='FACES_ONLY')

    faces = [face for face in affected_faces if face.is_valid]
    new_verts = [vert for vert in loop_verts if vert.is_valid]
    if not faces or not new_verts:
        raise FairingError("The cylindrical axis collapse removed the patch")
    return new_verts, faces


def _fair_organic_depth_laplacian(
    new_verts,
    boundary_verts,
    matrix_world,
    plane_center,
    axis,
):
    """Fallback graph solve for fitted-axis depth."""
    variables = {vert: index for index, vert in enumerate(new_verts)}
    rows = []
    right_hand_sides = []

    def height(vert):
        return (matrix_world @ vert.co - plane_center).dot(axis)

    for vert in list(new_verts) + list(boundary_verts):
        neighbors = [edge.other_vert(vert) for edge in vert.link_edges]
        if not neighbors:
            continue
        row = np.zeros(len(new_verts), dtype=np.float64)
        fixed = 0.0
        if vert in variables:
            row[variables[vert]] += 1.0
        else:
            fixed += height(vert)
        weight = 1.0 / len(neighbors)
        for neighbor in neighbors:
            if neighbor in variables:
                row[variables[neighbor]] -= weight
            else:
                fixed -= height(neighbor) * weight
        if np.any(row):
            rows.append(row)
            right_hand_sides.append(-fixed)

    # The new net starts on its best-fit plane. A tiny zero-height prior removes
    # null modes without allowing the solve to move points laterally.
    regularization = 1.0e-4
    for _vert, index in variables.items():
        row = np.zeros(len(new_verts), dtype=np.float64)
        row[index] = math.sqrt(regularization)
        rows.append(row)
        right_hand_sides.append(0.0)

    system = np.asarray(rows, dtype=np.float64)
    rhs = np.asarray(right_hand_sides, dtype=np.float64)
    try:
        solution, _residuals, rank, _values = np.linalg.lstsq(
            system, rhs, rcond=None
        )
    except (ValueError, np.linalg.LinAlgError):
        return False
    if rank < len(new_verts) or not np.isfinite(solution).all():
        return False
    inverse = matrix_world.inverted_safe()
    for vert, coordinate in zip(new_verts, solution):
        world = matrix_world @ vert.co
        world += axis * (float(coordinate) - (world - plane_center).dot(axis))
        vert.co = inverse @ world
    return True


def _thin_plate_kernel(distances):
    """Return the two-dimensional thin-plate radial basis function."""
    squared = distances * distances
    values = np.zeros_like(distances, dtype=np.float64)
    nonzero = distances > 1.0e-12
    values[nonzero] = squared[nonzero] * np.log(distances[nonzero])
    return values


def _organic_depth_samples(
    new_verts,
    boundary_verts,
    matrix_world,
    plane_center,
    basis_x,
    basis_y,
    axis,
):
    """Collect rim and exterior one-ring samples for surface continuation."""
    new_set = set(new_verts)
    fixed_verts = set(boundary_verts)
    for vert in boundary_verts:
        fixed_verts.update(
            edge.other_vert(vert)
            for edge in vert.link_edges
            if edge.other_vert(vert) not in new_set
            and not edge.hide and not edge.other_vert(vert).hide
            and (not edge.link_faces or any(not face.hide for face in edge.link_faces))
        )

    raw_samples = []
    for vert in fixed_verts:
        world = matrix_world @ vert.co
        offset = world - plane_center
        raw_samples.append(
            (
                np.asarray(
                    (offset.dot(basis_x), offset.dot(basis_y)),
                    dtype=np.float64,
                ),
                float(offset.dot(axis)),
            )
        )
    if not raw_samples:
        return None, None

    extent = max(
        (float(np.linalg.norm(point)) for point, _height in raw_samples),
        default=1.0,
    )
    merge_distance = max(extent * 1.0e-7, 1.0e-9)
    merged_points = []
    merged_heights = []
    merged_counts = []
    for point, height in raw_samples:
        match = next(
            (
                index
                for index, existing in enumerate(merged_points)
                if np.linalg.norm(point - existing) <= merge_distance
            ),
            None,
        )
        if match is None:
            merged_points.append(point.copy())
            merged_heights.append(height)
            merged_counts.append(1)
            continue
        count = merged_counts[match]
        merged_points[match] = (
            merged_points[match] * count + point
        ) / (count + 1)
        merged_heights[match] = (
            merged_heights[match] * count + height
        ) / (count + 1)
        merged_counts[match] += 1
    return (
        np.asarray(merged_points, dtype=np.float64),
        np.asarray(merged_heights, dtype=np.float64),
    )


def _fair_organic_depth(
    new_verts,
    boundary_verts,
    matrix_world,
    plane_center,
    axis,
    basis_x,
    basis_y,
    depth_samples=None,
):
    """Continue the surrounding surface with a smooth thin-plate height field."""
    points, heights = depth_samples if depth_samples is not None else _organic_depth_samples(
        new_verts,
        boundary_verts,
        matrix_world,
        plane_center,
        basis_x,
        basis_y,
        axis,
    )
    if points is None or len(points) < 3:
        return _fair_organic_depth_laplacian(
            new_verts, boundary_verts, matrix_world, plane_center, axis
        )

    point_center = np.mean(points, axis=0)
    point_scale = max(
        (float(np.linalg.norm(point - point_center)) for point in points),
        default=0.0,
    )
    if point_scale <= 1.0e-12:
        return _fair_organic_depth_laplacian(
            new_verts, boundary_verts, matrix_world, plane_center, axis
        )
    normalized = (points - point_center) / point_scale
    polynomial = np.column_stack(
        (np.ones(len(points), dtype=np.float64), normalized)
    )
    if np.linalg.matrix_rank(polynomial) < 3:
        return _fair_organic_depth_laplacian(
            new_verts, boundary_verts, matrix_world, plane_center, axis
        )

    separations = normalized[:, None, :] - normalized[None, :, :]
    distances = np.linalg.norm(separations, axis=2)
    kernel = _thin_plate_kernel(distances)
    # A very small diagonal term stabilizes nearly coincident support samples
    # without introducing a visible seam at the rim.
    kernel += np.eye(len(points), dtype=np.float64) * 1.0e-8
    system = np.block(
        [
            [kernel, polynomial],
            [polynomial.T, np.zeros((3, 3), dtype=np.float64)],
        ]
    )
    rhs = np.concatenate((heights, np.zeros(3, dtype=np.float64)))
    try:
        coefficients, _residuals, rank, _values = np.linalg.lstsq(
            system, rhs, rcond=None
        )
    except (ValueError, np.linalg.LinAlgError):
        coefficients = None
        rank = 0
    if (
        coefficients is None
        or rank < len(points) + 3
        or not np.isfinite(coefficients).all()
    ):
        return _fair_organic_depth_laplacian(
            new_verts, boundary_verts, matrix_world, plane_center, axis
        )

    radial_weights = coefficients[:len(points)]
    affine_weights = coefficients[len(points):]
    targets = []
    for vert in new_verts:
        world = matrix_world @ vert.co
        offset = world - plane_center
        query = np.asarray(
            (offset.dot(basis_x), offset.dot(basis_y)), dtype=np.float64
        )
        query = (query - point_center) / point_scale
        distances = np.linalg.norm(normalized - query, axis=1)
        target_height = float(
            _thin_plate_kernel(distances).dot(radial_weights)
            + np.asarray((1.0, query[0], query[1])).dot(affine_weights)
        )
        if not math.isfinite(target_height):
            return _fair_organic_depth_laplacian(
                new_verts, boundary_verts, matrix_world, plane_center, axis
            )
        targets.append((vert, world, offset, target_height))

    inverse = matrix_world.inverted_safe()
    for vert, world, offset, target_height in targets:
        world += axis * (target_height - offset.dot(axis))
        vert.co = inverse @ world
    return True


def resolved_fill_style(hypothesis, fill_style='AUTO'):
    """Resolve an explicit or automatic patch-topology choice."""
    if fill_style != 'AUTO':
        return fill_style
    if hypothesis["method"] == "PLANES":
        return 'PLANE'
    if (
        hypothesis["method"] == "CONIC"
        and not hypothesis["fit"].get("is_cone", False)
    ):
        return 'CYLINDER'
    if hypothesis["method"] in {"ORGANIC", "SPHERE"}:
        return 'GRID'
    return 'CIRCLE'


def prepare_fair_boundary(
    obj,
    bm,
    loop_verts,
    loop_edges,
    hypothesis,
    fill_style='AUTO',
):
    """Apply pre-fair simplification to a non-Grid analytic/circular fill."""
    topology = resolved_fill_style(hypothesis, fill_style)
    completed_faces = []
    if topology == 'CYLINDER' or hypothesis["method"] == "SPHERE":
        bm, loop_verts, loop_edges = _limited_dissolve_analytic_boundary(
            obj, bm, loop_verts, loop_edges, completed_faces=completed_faces,
        )
    else:
        loop_verts, loop_edges = limited_dissolve_boundary(
            bm, loop_verts, loop_edges, completed_faces=completed_faces,
        )
    result = None
    if not loop_edges:
        result = {
            'method': 'PLANES', 'fill_style': 'PLANE', 'residual': 0.0,
            'boundary_dissolved': True,
            'faces': completed_faces,
            'verts': list({vert for face in completed_faces for vert in face.verts}),
            'boundary_verts': [], 'boundary_edges': [],
        }
    forced_method = (
        'CYLINDER' if topology == 'CYLINDER'
        else 'SPHERE' if hypothesis["method"] == 'SPHERE'
        else None
    )
    return {
        "bm": bm,
        "loop_verts": loop_verts,
        "loop_edges": loop_edges,
        "forced_method": forced_method,
        "result": result,
    }


def finalize_fair_patch(
    obj,
    bm,
    result,
    loop_verts,
    loop_edges,
    final_simplify=True,
):
    """Apply the shared seam weld and optional post-fair simplification."""
    if result.get("method") in {'GRID_PREP', 'GRID_ROWS'}:
        # Boundary preparation already replaces the circular cut with source
        # grid corners. Dissolving the completed net would destroy those quads.
        return bm, result
    if result["fill_style"] == 'CYLINDER' and final_simplify:
        bm, dissolved = _limited_dissolve_cylinder_patch(
            obj, bm, result["faces"]
        )
    elif result["fill_style"] == 'CYLINDER':
        dissolved = {
            "verts": [vert for vert in result["verts"] if vert.is_valid],
            "faces": [face for face in result["faces"] if face.is_valid],
        }
    else:
        dissolved = limited_dissolve_patch(
            bm,
            loop_verts,
            loop_edges,
            result["verts"],
            result["faces"],
            simplify=final_simplify,
        )
    result["verts"] = dissolved["verts"]
    result["faces"] = dissolved["faces"]
    return bm, result


def fair_boundary_loop(
    bm,
    loop_verts,
    loop_edges,
    matrix_world,
    hypothesis,
    fill_style='AUTO',
    circle_loops=PATCH_RINGS,
    match_curvature=True,
):
    """Create the accepted patch and return its method/new topology."""
    topology = resolved_fill_style(hypothesis, fill_style)
    if topology == 'GRID':
        from .grid_patcher import prepare_grid_steps
        return prepare_grid_steps(bm, loop_verts, loop_edges, matrix_world,
                                  'ROWS', refinement_passes=1,
                                  match_curvature=match_curvature, hypothesis=hypothesis)

    layout = None
    organic_surface = None
    if hypothesis['method'] == 'ORGANIC':
        from .ideal_surface import capture_organic_surface
        center, axis, basis_x, basis_y, _points = _organic_projection_frame(loop_verts, matrix_world)
        layout = dict(center=center, axis=axis, basis_x=basis_x, basis_y=basis_y)
        organic_surface = capture_organic_surface(loop_verts, matrix_world, center, basis_x, basis_y, axis,
                                                   match_curvature=match_curvature)
    if topology == 'PLANE':
        inverse = matrix_world.inverted_safe()
        new_verts, new_faces = _create_plane_patch(
            bm,
            loop_verts,
            loop_edges,
            inverse @ hypothesis["center"],
        )
    elif topology == 'CYLINDER':
        new_verts, new_faces = _create_cylinder_patch(
            bm, loop_verts, loop_edges, matrix_world, hypothesis
        )
    else:
        new_verts, new_faces = _create_radial_patch(
            bm,
            loop_verts,
            loop_edges,
            matrix_world,
            hypothesis,
            circle_loops,
        )

    if hypothesis["method"] == 'ORGANIC':
        if organic_surface is not None:
            offsets = [matrix_world @ v.co - layout['center'] for v in new_verts]
            queries = [(p.dot(layout['basis_x']), p.dot(layout['basis_y'])) for p in offsets]
            targets = organic_surface.evaluate_chart(queries)
            if not np.isfinite(targets).all():
                raise FairingError('The organic surface projection was not finite')
            inverse = matrix_world.inverted_safe()
            for vert, target in zip(new_verts, targets):
                vert.co = inverse @ Vector(target)
        if layout is None:
            center, axis, basis_x, basis_y, _points = (
                _organic_projection_frame(loop_verts, matrix_world)
            )
            layout = {
                "center": center,
                "axis": axis,
                "basis_x": basis_x,
                "basis_y": basis_y,
            }
        if organic_surface is None and not _fair_organic_depth(
            new_verts,
            loop_verts,
            matrix_world,
            layout["center"],
            layout["axis"],
            layout["basis_x"],
            layout["basis_y"],
        ):
            bmesh.ops.delete(bm, geom=list(new_verts), context='VERTS')
            raise FairingError("Organic curvature solve was singular")

    update_visible_normals(bm)
    if any(face.calc_area() <= 1.0e-14 for face in new_faces):
        bmesh.ops.delete(bm, geom=list(new_verts), context='VERTS')
        raise FairingError("The fitted patch produced degenerate faces")
    return {
        "method": hypothesis["method"],
        "fill_style": topology,
        "grid_layout": layout.get("layout_method", "") if layout is not None else "",
        "grid_estimated_edge_length": layout.get("estimated_edge_length", 0.0) if layout else 0.0,
        "grid_target_edge_length": layout.get("target_edge_length", 0.0) if layout else 0.0,
        "residual": hypothesis.get("residual", 0.0),
        "verts": new_verts,
        "faces": new_faces,
        "boundary_verts": loop_verts,
        "boundary_edges": loop_edges,
        "surface_continuity": organic_surface.continuity if organic_surface is not None else
                              'THIN_PLATE' if hypothesis['method'] == 'ORGANIC' else 'ANALYTIC',
    }
