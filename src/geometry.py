# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Circular-cone fitting and mesh deformation for Resize Hole."""

import hashlib
import heapq
import math
import time

import bmesh
import bpy
import numpy as np

from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .constants import (
    PARALLEL_PLANE_ANGLE_EPSILON,
    PLANE_FIT_ABSOLUTE_EPSILON,
    PLANE_FIT_RELATIVE_EPSILON,
)
from .fairing import (
    FairingError,
    LIMITED_DISSOLVE_ANGLE,
    boundary_component,
    choose_hypothesis,
    expanded_limited_dissolve,
    fair_boundary_loop,
    finalize_fair_patch,
    ordered_boundary_loop,
    prepare_fair_boundary,
)


_CONE_ANGLE_EPSILON = math.radians(0.1)
_CONE_RELATIVE_FIT_EPSILON = 0.05
_MAX_GENERATOR_ALIGNMENT = 0.985
_BOOLEAN_PREVIEW_INTERVAL = 0.5


class GeometryMixin:
    """Provide selection, fitting, and deformation algorithms to the operator.

    Cone fits and boundary planes are expressed in world space. A cylinder is
    represented as the zero-slope special case. Original
    BMesh vertex coordinates are retained in object-local space so every update
    can be rebuilt from the same undeformed geometry instead of accumulating
    floating-point error.
    """

    @staticmethod
    def _canonical_topology_signature(vertex_count, edge_count, faces):
        """Return an order-independent digest of mesh face connectivity."""
        canonical_faces = sorted(
            tuple(sorted(int(index) for index in face)) for face in faces
        )
        digest = hashlib.sha256()
        digest.update(
            f"{int(vertex_count)}:{int(edge_count)}:{len(canonical_faces)}|".encode()
        )
        for face in canonical_faces:
            digest.update(",".join(str(index) for index in face).encode())
            digest.update(b";")
        return digest.hexdigest()

    @staticmethod
    def _bmesh_topology_signature(bm):
        """Return the canonical connectivity digest for a live BMesh."""
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.index_update()
        return GeometryMixin._canonical_topology_signature(
            len(bm.verts),
            len(bm.edges),
            ([vert.index for vert in face.verts] for face in bm.faces),
        )

    @staticmethod
    def _object_topology_signature(obj):
        """Return the canonical connectivity digest for an object's live mesh."""
        if obj.mode == 'EDIT':
            return GeometryMixin._bmesh_topology_signature(
                bmesh.from_edit_mesh(obj.data)
            )
        mesh = obj.data
        return GeometryMixin._canonical_topology_signature(
            len(mesh.vertices),
            len(mesh.edges),
            (polygon.vertices for polygon in mesh.polygons),
        )

    @staticmethod
    def _proportional_falloff(kind, value, vertex_index):
        """Return Blender-style proportional influence for a normalized value.

        ``value`` is clamped to the zero-to-one falloff interval. The vertex
        index is otherwise unused except to give RANDOM falloff a stable value
        while the operator is being adjusted.
        """
        value = min(1.0, max(0.0, value))
        if kind == 'SMOOTH':
            return value * value * (3.0 - 2.0 * value)
        if kind == 'SPHERE':
            return math.sqrt(max(0.0, 2.0 * value - value * value))
        if kind == 'ROOT':
            return math.sqrt(value)
        if kind == 'INVERSE_SQUARE':
            return value * (2.0 - value)
        if kind == 'SHARP':
            return value * value
        if kind == 'CONSTANT':
            return 1.0
        if kind == 'RANDOM':
            random_factor = (
                ((vertex_index * 1103515245 + 12345) & 0x7FFFFFFF)
                / 0x7FFFFFFF
            )
            return value * random_factor
        return value

    def _capture_deformation_state(self, context, bm, selected_verts):
        """Snapshot original coordinates and proportional-edit weights.

        Selected vertices receive full influence. With connected proportional
        editing, distances are shortest world-space paths over visible mesh
        edges; otherwise they are nearest-point distances, optionally flattened
        into view space for projected editing. The resulting weights and
        object-local coordinates are stored on the operator for repeatable live
        updates.
        """
        selected_verts = set(selected_verts)
        matrix_world = self._object.matrix_world
        weights = {vert.index: 1.0 for vert in selected_verts}
        tool_settings = context.scene.tool_settings

        if tool_settings.use_proportional_edit and selected_verts:
            radius = max(float(tool_settings.proportional_size), 1.0e-12)
            distances = {}

            if tool_settings.use_proportional_connected:
                queue = []
                for vert in selected_verts:
                    distances[vert.index] = 0.0
                    heapq.heappush(queue, (0.0, vert.index))
                bm.verts.ensure_lookup_table()
                while queue:
                    distance, vertex_index = heapq.heappop(queue)
                    if distance != distances.get(vertex_index) or distance >= radius:
                        continue
                    vert = bm.verts[vertex_index]
                    for edge in vert.link_edges:
                        neighbor = edge.other_vert(vert)
                        if neighbor.hide:
                            continue
                        edge_length = (
                            matrix_world @ neighbor.co - matrix_world @ vert.co
                        ).length
                        candidate = distance + edge_length
                        if candidate < radius and candidate < distances.get(
                            neighbor.index, math.inf
                        ):
                            distances[neighbor.index] = candidate
                            heapq.heappush(queue, (candidate, neighbor.index))
            else:
                selected_world = [matrix_world @ vert.co for vert in selected_verts]
                projected = tool_settings.use_proportional_projected
                region_data = getattr(self, "_region_data", None)
                if region_data is None:
                    region_data = getattr(context, "region_data", None)
                if region_data is None:
                    space_data = getattr(context, "space_data", None)
                    if getattr(space_data, "type", None) == 'VIEW_3D':
                        region_data = space_data.region_3d
                if projected and region_data is None:
                    raise RuntimeError(
                        "Projected proportional editing requires an active 3D view"
                    )
                view_matrix = region_data.view_matrix if projected else None
                if projected:
                    selected_world = [
                        Vector((view_point.x, view_point.y, 0.0))
                        for point in selected_world
                        for view_point in (view_matrix @ point,)
                    ]
                nearest_selected = KDTree(len(selected_world))
                for index, point in enumerate(selected_world):
                    nearest_selected.insert(point, index)
                nearest_selected.balance()
                for vert in bm.verts:
                    if vert.hide or vert in selected_verts:
                        continue
                    point = matrix_world @ vert.co
                    if projected:
                        view_point = view_matrix @ point
                        point = Vector((view_point.x, view_point.y, 0.0))
                    _nearest, _index, distance = nearest_selected.find(point)
                    if distance < radius:
                        distances[vert.index] = distance

            falloff = tool_settings.proportional_edit_falloff
            for vertex_index, distance in distances.items():
                if vertex_index in weights:
                    continue
                value = 1.0 - distance / radius
                weight = GeometryMixin._proportional_falloff(
                    falloff, value, vertex_index
                )
                if weight > 0.0:
                    weights[vertex_index] = weight

        bm.verts.ensure_lookup_table()
        self._proportional_weights = weights
        self._original_vertex_coords = {
            index: bm.verts[index].co.copy() for index in weights
        }

    def _world_normal(self, face):
        """Return a BMesh face normal transformed correctly into world space."""
        normal_matrix = self._object.matrix_world.to_3x3().inverted_safe().transposed()
        return (normal_matrix @ face.normal).normalized()

    def _flood_fill(self, seed):
        """Collect the smooth, edge-connected face region containing ``seed``.

        A visible neighboring face is admitted when its world-space normal is
        within ``self.flood_angle`` of the current face normal. Comparing each
        pair of adjacent faces allows the fill to follow a curved cylinder. The
        method returns faces without changing Blender's actual mesh selection.
        """
        threshold = math.cos(self.flood_angle)
        result = {seed}
        stack = [seed]
        normal_cache = {}

        def world_normal(face):
            normal = normal_cache.get(face)
            if normal is None:
                normal = self._world_normal(face)
                normal_cache[face] = normal
            return normal

        while stack:
            face = stack.pop()
            face_normal = world_normal(face)
            for edge in face.edges:
                for neighbor in edge.link_faces:
                    if neighbor in result or neighbor.hide:
                        continue
                    if face_normal.dot(world_normal(neighbor)) >= threshold:
                        result.add(neighbor)
                        stack.append(neighbor)
        return GeometryMixin._extend_flood_across_gaps(self, result)

    def _extend_flood_across_gaps(self, connected_faces):
        """Add disconnected wall remnants compatible with a provisional fit.

        A missing face breaks edge traversal even though the surfaces on either
        side still describe the same hole. Once the connected patch can define a
        cone, test other visible faces against its axial span, radial model, and
        expected wall normal. The deliberately tight tolerances bridge local
        defects without collecting end caps or remote coaxial features.
        """
        result = set(connected_faces)
        if len(result) < 3:
            return result
        try:
            fit = GeometryMixin._fit_cylinder(self, list(result))
        except (ValueError, np.linalg.LinAlgError):
            return result

        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        matrix_world = self._object.matrix_world
        axis = fit["axis"]
        center = fit["center"]
        slope = fit.get("slope", 0.0)
        axial_extent = max(fit["z_max"] - fit["z_min"], 1.0e-9)
        axial_tolerance = max(axial_extent * 0.08, fit["radius"] * 0.08)
        radial_tolerance = max(
            fit["radius"] * 0.08,
            fit.get("fit_relative_error", 0.0) * fit["radius"] * 3.0,
            1.0e-7,
        )
        normal_threshold = math.cos(math.radians(25.0))

        for face in bm.faces:
            if face in result or face.hide:
                continue
            point = matrix_world @ face.calc_center_median()
            offset = point - center
            coordinate = offset.dot(axis)
            if not (
                fit["z_min"] - axial_tolerance
                <= coordinate
                <= fit["z_max"] + axial_tolerance
            ):
                continue
            radial = offset - axis * coordinate
            if radial.length_squared <= 1.0e-16:
                continue
            expected_radius = fit["radius"] + slope * coordinate
            if expected_radius <= 1.0e-9:
                continue
            if abs(radial.length - expected_radius) > radial_tolerance:
                continue
            expected_normal = (radial.normalized() - axis * slope).normalized()
            if abs(self._world_normal(face).dot(expected_normal)) < normal_threshold:
                continue
            result.add(face)
        return result

    def _fit_cylinder(self, faces):
        """Fit a world-space circular cone, including its cylindrical limit.

        This mirrors the bundled ``Fit Circular Cone`` Geometry Nodes asset.
        PCA of centered wall normals finds the axis. Planes whose normals are
        ``axis × wall_normal`` all contain that axis, so their least-squares
        intersection recovers an axis line even when faces are missing. Radius
        is then regressed against distance along the line. A meaningful, low-
        residual slope identifies a cone; otherwise the fit remains cylindrical.

        The historical method name is retained because it is part of the
        operator's internal redo/test surface.
        """
        matrix_world = self._object.matrix_world
        verts = {vert for face in faces for vert in face.verts}
        if len(verts) < 6 or len(faces) < 3:
            raise ValueError("The flood fill needs at least three faces and six vertices")

        points = np.asarray([matrix_world @ vert.co for vert in verts], dtype=np.float64)
        face_points = []
        normals = []
        weights = []
        for face in faces:
            face_points.append(
                np.asarray(matrix_world @ face.calc_center_median(), dtype=np.float64)
            )
            normals.append(np.asarray(self._world_normal(face), dtype=np.float64))
            weights.append(max(face.calc_area(), 1.0e-12))
        face_points = np.asarray(face_points)
        normals = np.asarray(normals)
        weights = np.asarray(weights)

        # Circular-cone normals terminate in a plane. Centering them before SVD
        # makes its normal the cone axis, including for a partial angular patch.
        normal_mean = np.average(normals, axis=0, weights=weights)
        centered_normals = (normals - normal_mean) * np.sqrt(weights)[:, None]
        _u, _singular_values, vh = np.linalg.svd(
            centered_normals, full_matrices=False
        )
        axis = Vector(vh[-1]).normalized()
        if axis.length_squared <= 1.0e-16:
            raise ValueError("The selected wall does not define a stable axis")

        # The cross product is normal to each meridional plane through the cone
        # axis. Their least-squares intersection has rank two for a usable line.
        meridional_normals = np.cross(np.asarray(axis), normals)
        lengths = np.linalg.norm(meridional_normals, axis=1)
        usable = lengths > 1.0e-8
        if np.count_nonzero(usable) < 3:
            raise ValueError("The selected wall does not define a stable axis")
        meridional_normals = meridional_normals[usable] / lengths[usable, None]
        plane_weights = np.sqrt(weights[usable])
        system = meridional_normals * plane_weights[:, None]
        rhs = np.einsum(
            'ij,ij->i', meridional_normals, face_points[usable]
        ) * plane_weights
        axis_seed, _residuals, axis_rank, _values = np.linalg.lstsq(
            system, rhs, rcond=None
        )
        if axis_rank < 2:
            raise ValueError("The selected wall does not define a stable axis")

        mean = Vector(points.mean(axis=0))
        seed = Vector(axis_seed)
        center = seed + axis * (mean - seed).dot(axis)

        reference = Vector((0.0, 0.0, 1.0))
        if abs(axis.dot(reference)) > 0.9:
            reference = Vector((0.0, 1.0, 0.0))
        basis_x = axis.cross(reference).normalized()
        basis_y = axis.cross(basis_x).normalized()

        relative = [Vector(point) - center for point in points]
        x = np.asarray([point.dot(basis_x) for point in relative])
        y = np.asarray([point.dot(basis_y) for point in relative])
        z = np.asarray([point.dot(axis) for point in relative])
        radial_distances = np.hypot(x, y)

        regression = np.column_stack((z, np.ones_like(z)))
        solution, _residuals, rank, _values = np.linalg.lstsq(
            regression, radial_distances, rcond=None
        )
        if rank < 2:
            raise ValueError("The selected wall has no measurable axial extent")
        fitted_slope, fitted_radius = map(float, solution)
        if fitted_radius <= 1.0e-12:
            raise ValueError("The fitted wall radius is zero")
        predicted = fitted_radius + fitted_slope * z
        residual_rms = float(np.sqrt(np.mean((radial_distances - predicted) ** 2)))
        relative_error = residual_rms / max(float(radial_distances.mean()), 1.0e-12)
        half_angle = math.atan(abs(fitted_slope))
        is_cone = (
            half_angle >= _CONE_ANGLE_EPSILON
            and relative_error <= _CONE_RELATIVE_FIT_EPSILON
            and float(predicted.min()) > 1.0e-9
        )
        slope = fitted_slope if is_cone else 0.0

        if is_cone:
            ratios = radial_distances / predicted
            radius_estimates = {
                'MINIMUM': fitted_radius * float(ratios.min()),
                'AVERAGE': fitted_radius * float(ratios.mean()),
                'MAXIMUM': fitted_radius * float(ratios.max()),
            }
        else:
            radius_estimates = {
                'MINIMUM': float(radial_distances.min()),
                'AVERAGE': float(radial_distances.mean()),
                'MAXIMUM': float(radial_distances.max()),
            }
        estimator = getattr(self, "diameter_estimator", 'AVERAGE')
        if estimator == 'MINIMUM':
            radius = radius_estimates['MINIMUM']
        elif estimator == 'MAXIMUM':
            radius = radius_estimates['MAXIMUM']
        else:
            radius = radius_estimates['AVERAGE']
        if radius <= 1.0e-12:
            raise ValueError("The fitted wall radius is zero")

        # Mesh normals point radially away from a stud and toward the axis of a
        # hole. Keep this distinction for choosing which end is attached to the
        # surrounding model when depth changes.
        wall_orientation = 0.0
        for face in faces:
            face_center = matrix_world @ face.calc_center_median()
            face_offset = face_center - center
            face_axial = axis * face_offset.dot(axis)
            face_radial = face_offset - face_axial
            if face_radial.length_squared <= 1.0e-16:
                continue
            wall_orientation += (
                self._world_normal(face).dot(face_radial.normalized())
                * max(face.calc_area(), 1.0e-12)
            )
        return {
            "axis": axis,
            "basis_x": basis_x,
            "basis_y": basis_y,
            "center": center,
            "radius": radius,
            "radius_estimates": radius_estimates,
            "slope": slope,
            "half_angle": math.atan(abs(slope)),
            "is_cone": is_cone,
            "fit_relative_error": relative_error,
            "fit_rank": int(axis_rank),
            "z_min": float(z.min()),
            "z_max": float(z.max()),
            "is_stud": wall_orientation >= 0.0,
        }

    def _fit_boundary_planes(self, faces, cylinder_axis, fit=None):
        """Fit world-space end planes to connected wall-boundary components.

        Boundary edges have exactly one face in the supplied region. Edges that
        follow a fitted cone generator are missing-face defects, not end arcs,
        and are discarded. Remaining fragments are combined into two axial end
        groups before their planes are fitted, so neither closed loops nor
        uninterrupted topology are required.

        Each result records the component's vertex indices, a point on its
        plane, its normal, and scale-aware planarity diagnostics. The diagnostics
        let depth adjustment reject non-planar ends without losing the best-fit
        planes used to preserve skewed cuts during diameter changes.
        """
        selected = set(faces)
        boundary_edges = {
            edge
            for face in faces
            for edge in face.edges
            if sum(linked_face in selected for linked_face in edge.link_faces) == 1
        }
        matrix_world = self._object.matrix_world
        if fit is not None:
            axis_point = fit["center"]
            slope = fit.get("slope", 0.0)
            filtered_edges = set()
            for edge in boundary_edges:
                first = matrix_world @ edge.verts[0].co
                second = matrix_world @ edge.verts[1].co
                tangent = second - first
                midpoint = (first + second) * 0.5
                offset = midpoint - axis_point
                radial = offset - cylinder_axis * offset.dot(cylinder_axis)
                if (
                    tangent.length_squared <= 1.0e-16
                    or radial.length_squared <= 1.0e-16
                ):
                    continue
                generator = (
                    cylinder_axis + radial.normalized() * slope
                ).normalized()
                alignment = abs(tangent.normalized().dot(generator))
                if alignment < _MAX_GENERATOR_ALIGNMENT:
                    filtered_edges.add(edge)
            boundary_edges = filtered_edges

        remaining = set(boundary_edges)
        components = []

        while remaining:
            seed = remaining.pop()
            component_edges = {seed}
            component_verts = set(seed.verts)
            stack = [seed]
            while stack:
                edge = stack.pop()
                for vert in edge.verts:
                    for neighbor in vert.link_edges:
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component_edges.add(neighbor)
                            component_verts.update(neighbor.verts)
                            stack.append(neighbor)
            components.append((component_verts, component_edges))

        # Several omitted wall faces can split each circumferential end into
        # multiple arcs. Merge fragments into two axial bands with weighted
        # one-dimensional k-means before fitting the end planes.
        if len(components) > 2:
            component_data = []
            for component_verts, component_edges in components:
                midpoints = [
                    (
                        matrix_world @ edge.verts[0].co
                        + matrix_world @ edge.verts[1].co
                    ) * 0.5
                    for edge in component_edges
                ]
                lengths = [
                    (
                        matrix_world @ edge.verts[1].co
                        - matrix_world @ edge.verts[0].co
                    ).length
                    for edge in component_edges
                ]
                total = max(sum(lengths), 1.0e-12)
                coordinate = sum(
                    point.dot(cylinder_axis) * length
                    for point, length in zip(midpoints, lengths)
                ) / total
                component_data.append(
                    (coordinate, total, component_verts, component_edges)
                )
            centers = [
                min(item[0] for item in component_data),
                max(item[0] for item in component_data),
            ]
            if abs(centers[1] - centers[0]) > 1.0e-9:
                assignments = None
                for _iteration in range(12):
                    next_assignments = [
                        0
                        if abs(item[0] - centers[0]) <= abs(item[0] - centers[1])
                        else 1
                        for item in component_data
                    ]
                    if next_assignments == assignments:
                        break
                    assignments = next_assignments
                    for cluster in (0, 1):
                        members = [
                            item for item, assignment in zip(component_data, assignments)
                            if assignment == cluster
                        ]
                        if members:
                            centers[cluster] = sum(
                                item[0] * item[1] for item in members
                            ) / sum(item[1] for item in members)
                merged = []
                for cluster in (0, 1):
                    cluster_verts = set()
                    cluster_edges = set()
                    for item, assignment in zip(component_data, assignments or ()):
                        if assignment == cluster:
                            cluster_verts.update(item[2])
                            cluster_edges.update(item[3])
                    if cluster_edges:
                        merged.append((cluster_verts, cluster_edges))
                components = merged

        planes = []
        for verts, edges in components:
            if len(verts) < 3:
                continue
            points = np.asarray(
                [matrix_world @ vert.co for vert in verts], dtype=np.float64
            )
            center = points.mean(axis=0)
            _u, _singular_values, vh = np.linalg.svd(
                points - center, full_matrices=False
            )
            normal = Vector(vh[-1]).normalized()
            centered = points - center
            extent = float(np.linalg.norm(centered, axis=1).max(initial=0.0))
            max_error = float(
                np.abs(centered @ np.asarray(normal)).max(initial=0.0)
            )
            tolerance = max(
                getattr(
                    self,
                    "plane_fit_absolute_epsilon",
                    PLANE_FIT_ABSOLUTE_EPSILON,
                ),
                extent
                * getattr(
                    self,
                    "plane_fit_relative_epsilon",
                    PLANE_FIT_RELATIVE_EPSILON,
                ),
            )

            # A valid cylinder end surface must intersect lines parallel to the
            # fitted axis. Skip longitudinal/open-patch boundaries.
            if abs(normal.dot(cylinder_axis)) < 1.0e-5:
                continue
            adjacent_faces = {
                face
                for edge in edges
                for face in edge.link_faces
                if face not in selected and not face.hide
            }
            adjacent_normal = None
            if adjacent_faces:
                weighted_adjacent_normal = Vector((0.0, 0.0, 0.0))
                for face in adjacent_faces:
                    weighted_adjacent_normal += self._world_normal(face) * max(
                        face.calc_area(), 1.0e-12
                    )
                if weighted_adjacent_normal.length_squared > 1.0e-16:
                    adjacent_normal = weighted_adjacent_normal.normalized()
            planes.append(
                {
                    "verts": sorted(vert.index for vert in verts),
                    "point": Vector(center),
                    "normal": normal,
                    "max_error": max_error,
                    "tolerance": tolerance,
                    "adjacent_normal": adjacent_normal,
                }
            )
        return sorted(planes, key=lambda plane: plane["verts"][0])

    def _fit_adjustable_depth(self, planes, fit):
        """Return fixed-side axial depth metadata for two planar end loops.

        Depth is measured between each loop plane's intersection with the fitted
        axis, so non-parallel ends remain adjustable. The outer/model-side loop is
        selected from its adjacent face normal: for a hole that normal points away
        from the cylindrical region, while a stud reverses the test. Missing or
        ambiguous adjacent faces fall back to stable boundary-index ordering. Two
        outward-facing hole ends identify a through-hole, which has no adjustable
        depth.
        """
        if len(planes) != 2:
            return None
        if any(
            plane.get("max_error", math.inf) > plane.get("tolerance", 0.0)
            for plane in planes
        ):
            return None

        axis = fit["axis"]
        center = fit["center"]
        coordinates = []
        for plane in planes:
            denominator = plane["normal"].dot(axis)
            if abs(denominator) < 1.0e-5:
                return None
            coordinates.append(
                plane["normal"].dot(plane["point"] - center) / denominator
            )

        initial = abs(coordinates[1] - coordinates[0])
        if initial <= 1.0e-12:
            return None

        is_stud = bool(fit.get("is_stud", False))
        into_scores = []
        outer_scores = []
        for index, plane in enumerate(planes):
            adjacent_normal = plane.get("adjacent_normal")
            if adjacent_normal is None:
                into_scores.append(None)
                outer_scores.append(None)
                continue
            other_coordinate = coordinates[1 - index]
            into_region = axis * math.copysign(
                1.0, other_coordinate - coordinates[index]
            )
            into_score = adjacent_normal.dot(into_region)
            into_scores.append(into_score)
            outer_scores.append(into_score if is_stud else -into_score)

        # Both end faces of a manifold cylindrical through-hole point away from
        # the selected wall. A cone still has a meaningful top-to-bottom depth,
        # so retain its two recovered end arcs even when both rims are exposed.
        if not is_stud and not fit.get("is_cone", False) and all(
            score is not None and score <= 1.0e-5 for score in into_scores
        ):
            return None

        scored_indices = [
            index for index, score in enumerate(outer_scores) if score is not None
        ]
        if len(scored_indices) == 2 and not math.isclose(
            outer_scores[scored_indices[0]],
            outer_scores[scored_indices[1]],
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            outer_index = max(
                scored_indices, key=lambda index: outer_scores[index]
            )
        elif len(scored_indices) == 1:
            scored_index = scored_indices[0]
            outer_index = (
                scored_index
                if outer_scores[scored_index] > 0.0
                else 1 - scored_index
            )
        elif fit.get("is_cone", False):
            # Countersinks and tapered holes conventionally meet the model at
            # their wider circle. This also gives incomplete topology a stable
            # top-surface choice when adjacent-face evidence is unavailable.
            outer_index = max(
                range(2),
                key=lambda index: (
                    fit["radius"]
                    + fit.get("slope", 0.0) * coordinates[index]
                ),
            )
        else:
            outer_index = 0
        inner_index = 1 - outer_index
        parallel_threshold = math.cos(
            getattr(
                self,
                "parallel_plane_angle_epsilon",
                PARALLEL_PLANE_ANGLE_EPSILON,
            )
        )
        return {
            "initial": initial,
            "outer_coordinate": coordinates[outer_index],
            "inner_coordinate": coordinates[inner_index],
            "outer_plane_index": outer_index,
            "inner_plane_index": inner_index,
            "is_stud": is_stud,
            "parallel": abs(
                planes[0]["normal"].normalized().dot(
                    planes[1]["normal"].normalized()
                )
            ) > parallel_threshold,
        }

    def _target_depth_scale(self):
        """Return the axial scale requested by the optional depth property."""
        depth_fit = getattr(self, "_depth_fit", None)
        if not depth_fit or depth_fit["initial"] <= 1.0e-12:
            return 1.0
        return max(float(self.depth), 1.0e-9) / depth_fit["initial"]

    def _radius_coordinate(self):
        """Return the axial coordinate at which Diameter is measured."""
        depth_fit = getattr(self, "_depth_fit", None)
        if depth_fit:
            return depth_fit["outer_coordinate"]
        return float(self._fit.get("radius_coordinate", 0.0))

    def _fitted_radius_at(self, coordinate):
        """Evaluate the fitted cone radius at one axial coordinate."""
        fit = self._fit
        return max(
            1.0e-9,
            float(fit["radius"]) + float(fit.get("slope", 0.0)) * coordinate,
        )

    def _target_cone_slope(self):
        """Return the signed slope requested by the optional angle property."""
        fit_slope = float(self._fit.get("slope", 0.0))
        if not getattr(self, "angle_valid", False):
            return fit_slope
        sign = -1.0 if fit_slope < 0.0 else 1.0
        return sign * math.tan(
            min(math.radians(89.0), max(0.0, float(self.conic_angle)))
        )

    def _target_radius_at(self, coordinate):
        """Evaluate the requested cone radius, anchored at the top surface."""
        anchor = GeometryMixin._radius_coordinate(self)
        anchor_radius = (
            max(float(self.diameter), 1.0e-9) * 0.5
            if float(self.diameter) > 0.0
            else GeometryMixin._fitted_radius_at(self, anchor) * float(self.scale)
        )
        return max(
            1.0e-9,
            anchor_radius
            + GeometryMixin._target_cone_slope(self) * (coordinate - anchor),
        )

    def _target_scale(self):
        """Convert the requested diameter into a radial deformation scale.

        The legacy ``scale`` property remains a fallback for redo compatibility
        when no positive diameter has been assigned.
        """
        fitted_diameter = GeometryMixin._fitted_radius_at(
            self, GeometryMixin._radius_coordinate(self)
        ) * 2.0
        if fitted_diameter <= 1.0e-12:
            return 1.0
        if self.diameter <= 0.0:
            return self.scale
        return self.diameter / fitted_diameter

    @staticmethod
    def _surface_bvh(faces, matrix_world):
        """Build a world-space BVH and useful ray span from BMesh faces."""
        coordinates = []
        polygons = []
        for face in faces:
            polygon = []
            for vert in face.verts:
                polygon.append(len(coordinates))
                coordinates.append(matrix_world @ vert.co)
            if len(polygon) >= 3:
                polygons.append(polygon)
        if not polygons:
            return None, 0.0
        bvh = BVHTree.FromPolygons(coordinates, polygons, all_triangles=False)
        minimum = Vector((
            min(point.x for point in coordinates),
            min(point.y for point in coordinates),
            min(point.z for point in coordinates),
        ))
        maximum = Vector((
            max(point.x for point in coordinates),
            max(point.y for point in coordinates),
            max(point.z for point in coordinates),
        ))
        return bvh, max((maximum - minimum).length * 2.0, 1.0)

    @staticmethod
    def _ordered_supported_loop(component, support_by_edge):
        """Order one unbranched edge component and its outside support faces."""
        degree = {}
        for edge in component:
            for vert in edge.verts:
                degree.setdefault(vert, []).append(edge)
        endpoints = [
            vert for vert, linked in degree.items() if len(linked) == 1
        ]
        if (
            len(component) < 2
            or any(len(linked) not in {1, 2} for linked in degree.values())
            or len(endpoints) not in {0, 2}
        ):
            return None

        start = (
            min(endpoints, key=lambda vert: vert.index)
            if endpoints
            else min(component, key=lambda edge: edge.index).verts[0]
        )
        first_edge = min(degree[start], key=lambda edge: edge.index)
        current_vert = start
        current_edge = first_edge
        visited = set()
        verts = []
        edges = []
        supports = []
        while current_edge not in visited:
            visited.add(current_edge)
            verts.append(current_vert)
            edges.append(current_edge)
            supports.append(support_by_edge[current_edge])
            next_vert = current_edge.other_vert(current_vert)
            candidates = [
                edge for edge in degree[next_vert]
                if edge is not current_edge
            ]
            if not candidates:
                current_vert = next_vert
                break
            if len(candidates) != 1:
                return None
            current_vert = next_vert
            current_edge = candidates[0]
        closed = not endpoints
        if len(visited) != len(component) or (
            closed and current_vert is not start
        ):
            return None
        if not closed:
            verts.append(current_vert)
        return verts, edges, supports, closed

    def _support_patch_faces(
        self, support_faces, selected_faces, loop_edges, rings=2
    ):
        """Collect a small support patch without crossing the fitted wall rim."""
        selected = set(selected_faces)
        blocked = set(loop_edges)
        patch = set(support_faces)
        frontier = set(support_faces)
        for _ring in range(rings):
            following = set()
            for face in frontier:
                for edge in face.edges:
                    if edge in blocked:
                        continue
                    for neighbor in edge.link_faces:
                        if (
                            neighbor is not face
                            and neighbor not in selected
                            and neighbor not in patch
                            and not neighbor.hide
                            and self._world_normal(face).dot(
                                self._world_normal(neighbor)
                            ) >= 0.9
                        ):
                            following.add(neighbor)
            if not following:
                break
            patch.update(following)
            frontier = following
        return list(patch)

    def _support_extends_outward(self, loop_verts, support_faces):
        """Return whether a rim support continues away from the fitted wall."""
        loop_set = set(loop_verts)
        matrix_world = self._object.matrix_world
        fit = self._fit
        tolerance = max(float(fit["radius"]) * 1.0e-5, 1.0e-7)
        outward = 0
        tested = 0
        for face in support_faces:
            samples = [vert for vert in face.verts if vert not in loop_set]
            if samples:
                margins = []
                for vert in samples:
                    point = matrix_world @ vert.co
                    offset = point - fit["center"]
                    coordinate = offset.dot(fit["axis"])
                    radial = offset - fit["axis"] * coordinate
                    margins.append(
                        radial.length
                        - GeometryMixin._fitted_radius_at(self, coordinate)
                    )
                outward += max(margins) > tolerance
            else:
                # A cap n-gon may contain only rim vertices. Its centroid lies
                # inward, whereas an exterior support without extra vertices is
                # at worst neutral and should not create a false outward vote.
                point = matrix_world @ face.calc_center_median()
                offset = point - fit["center"]
                coordinate = offset.dot(fit["axis"])
                radial = offset - fit["axis"] * coordinate
                outward += (
                    radial.length
                    - GeometryMixin._fitted_radius_at(self, coordinate)
                ) > tolerance
            tested += 1
        # One genuine outward continuation is sufficient. Boolean-generated
        # cube faces can leave most rim edges supported by triangles containing
        # only boundary vertices, while just the corner-adjacent faces retain
        # vertices beyond the fitted radius. A blind cap has no such vertex.
        return tested > 0 and outward > 0

    def _build_resize_projection_state(self, bm, selected_faces):
        """Fit virtual endcap surfaces and snapshot the undeformed base mesh."""
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        selected = set(selected_faces)
        matrix_world = self._object.matrix_world
        saved_angular_segments = int(
            getattr(self, "fit_angular_segments", 0) or 0
        )
        self._resize_angular_segments = (
            saved_angular_segments
            if getattr(self, "fit_valid", False)
            and saved_angular_segments >= 3
            else GeometryMixin._estimate_angular_segments(self, selected)
        )
        base_faces = [
            face for face in bm.faces
            if face not in selected and not face.hide
        ]
        base_bvh, base_span = GeometryMixin._surface_bvh(
            base_faces, matrix_world
        )
        self._resize_base_bvh = base_bvh
        self._resize_base_span = base_span
        self._resize_surfaces_by_vertex = {}

        support_by_edge = {}
        for face in selected:
            for edge in face.edges:
                if sum(linked in selected for linked in edge.link_faces) != 1:
                    continue
                supports = [
                    linked for linked in edge.link_faces
                    if linked not in selected and not linked.hide
                ]
                if supports:
                    support_by_edge[edge] = max(
                        supports, key=lambda support: support.calc_area()
                    )

        remaining = set(support_by_edge)
        while remaining:
            seed = remaining.pop()
            component = {seed}
            stack = [seed]
            while stack:
                edge = stack.pop()
                for vert in edge.verts:
                    for neighbor in vert.link_edges:
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component.add(neighbor)
                            stack.append(neighbor)
            ordered = GeometryMixin._ordered_supported_loop(
                component, support_by_edge
            )
            if ordered is None:
                continue
            loop_verts, loop_edges, support_faces, closed = ordered
            support_patch_faces = GeometryMixin._support_patch_faces(
                self, support_faces, selected, loop_edges
            )
            try:
                hypothesis = choose_hypothesis(
                    loop_verts,
                    loop_edges,
                    matrix_world,
                    lambda faces: GeometryMixin._fit_cylinder(self, faces),
                    lambda face: GeometryMixin._world_normal(self, face),
                    support_faces,
                    analytic_faces=support_patch_faces,
                    # Direct deformation is safe only for an actually planar
                    # exterior. The looser default remains useful to Fair Hole
                    # when it must tolerate noisy hand-built boundaries.
                    plane_relative_tolerance=1.0e-5,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            surface = {
                "hypothesis": hypothesis,
                "verts": {vert.index for vert in loop_verts},
                "loop_verts": list(loop_verts),
                "loop_edges": list(loop_edges),
                "loop_world_points": [
                    matrix_world @ vert.co for vert in loop_verts
                ],
                "support_faces": list(set(support_faces)),
                "support_patch_faces": support_patch_faces,
                "closed": closed,
            }
            surface["exterior"] = GeometryMixin._support_extends_outward(
                self, loop_verts, surface["support_faces"]
            )
            if hypothesis.get("method") == "ORGANIC" and closed:
                organic_bvh, organic_span = (
                    GeometryMixin._build_virtual_organic_surface(
                        self,
                        bm,
                        selected,
                        loop_verts,
                        loop_edges,
                    )
                )
                surface["organic_bvh"] = organic_bvh
                surface["organic_span"] = organic_span
            for vert in loop_verts:
                self._resize_surfaces_by_vertex[vert.index] = surface
        unique_surfaces = {
            id(surface): surface
            for surface in self._resize_surfaces_by_vertex.values()
        }.values()
        self._resize_requires_boolean = (
            not bool(self._fit.get("is_stud", False))
            and any(
                surface["exterior"]
                and not (
                    surface["hypothesis"].get("method") == "PLANES"
                    and len(surface["hypothesis"].get("planes", ())) == 1
                )
                for surface in unique_surfaces
            )
        )
        self._resize_base_requires_boolean = self._resize_requires_boolean
        self._resize_dynamic_requires_boolean = False
        self._resize_projection_ready = True

    def _estimate_angular_segments(self, selected_faces):
        """Estimate the source mesh's full-circle angular resolution.

        Each distinct projected wall-face normal represents one angular facet.
        Missing wall faces create one or more large angular gaps. The median of
        the remaining cyclic gaps therefore preserves the source sampling while
        ignoring extra rim/interior vertices and axial face subdivisions.
        """
        fit = self._fit
        matrix_world = self._object.matrix_world
        angles = []
        axis = fit["axis"].normalized()
        for face in selected_faces:
            normal = GeometryMixin._world_normal(self, face)
            radial_normal = normal - axis * normal.dot(axis)
            if radial_normal.length <= 1.0e-9:
                continue
            # Holes and studs point their wall normals in opposite directions.
            # Orient every sample toward its face center so mixed winding does
            # not introduce artificial half-turn samples.
            center_offset = (
                matrix_world @ face.calc_center_median()
            ) - fit["center"]
            center_coordinate = center_offset.dot(axis)
            center_radial = center_offset - axis * center_coordinate
            if (
                center_radial.length > 1.0e-9
                and radial_normal.dot(center_radial) < 0.0
            ):
                radial_normal.negate()
            radial_normal.normalize()
            angles.append(
                math.atan2(
                    radial_normal.dot(fit["basis_y"]),
                    radial_normal.dot(fit["basis_x"]),
                ) % math.tau
            )
        angles.sort()
        unique = []
        tolerance = 1.0e-5
        for angle in angles:
            if not unique or angle - unique[-1] > tolerance:
                unique.append(angle)
        if (
            len(unique) > 1
            and unique[0] + math.tau - unique[-1] <= tolerance
        ):
            unique.pop()
        if len(unique) < 3:
            return max(3, min(512, len(getattr(self, "_locked_indices", ())) or 32))
        gaps = [
            (unique[(index + 1) % len(unique)] - angle) % math.tau
            for index, angle in enumerate(unique)
        ]
        ordinary_gap = float(np.median([gap for gap in gaps if gap > tolerance]))
        if ordinary_gap <= tolerance:
            return max(3, min(512, len(unique)))
        return max(3, min(512, round(math.tau / ordinary_gap)))

    def _fit_reconstruction_depth(self):
        """Measure blind depth between axis intersections with fitted surfaces."""
        if self._fit.get("is_stud", False):
            return None
        fit = self._fit
        axis = fit["axis"].normalized()
        center = fit["center"]
        unique_surfaces = list({
            id(surface): surface
            for surface in self._resize_surfaces_by_vertex.values()
        }.values())

        def intersection_coordinate(surface):
            points = surface.get("loop_world_points", ())
            if not points:
                return None
            reference = sum(
                points, Vector((0.0, 0.0, 0.0))
            ) / len(points)
            reference_coordinate = (reference - center).dot(axis)
            axis_point = center + axis * reference_coordinate
            if surface["hypothesis"].get("method") == "ORGANIC":
                intersection = GeometryMixin._project_bvh_along_axis(
                    surface.get("organic_bvh"),
                    surface.get("organic_span", 0.0),
                    axis_point,
                    reference,
                    axis,
                )
            else:
                intersection = GeometryMixin._project_hypothesis_along_axis(
                    surface["hypothesis"],
                    axis_point,
                    reference,
                    axis,
                )
            if intersection is None:
                return None
            return (intersection - center).dot(axis)

        exterior = [
            value
            for surface in unique_surfaces
            if surface["closed"] and surface["exterior"]
            for value in (intersection_coordinate(surface),)
            if value is not None
        ]
        interior = [
            value
            for surface in unique_surfaces
            if surface["closed"] and not surface["exterior"]
            for value in (intersection_coordinate(surface),)
            if value is not None
        ]
        if not exterior or not interior:
            return None
        initial, outer, inner = max(
            (abs(inner - outer), outer, inner)
            for outer in exterior
            for inner in interior
        )
        if initial <= max(float(fit["radius"]) * 1.0e-6, 1.0e-9):
            return None
        return {
            "initial": initial,
            "outer_coordinate": outer,
            "inner_coordinate": inner,
            "outer_plane_index": -1,
            "inner_plane_index": -1,
            "is_stud": False,
            "parallel": False,
            "surface_depth": True,
        }

    def _build_virtual_organic_surface(
        self, bm, selected_faces, loop_verts, loop_edges
    ):
        """Fair a temporary wall-free mesh and retain only its patch BVH."""
        copied = bm.copy()
        try:
            copied.verts.ensure_lookup_table()
            copied.edges.ensure_lookup_table()
            copied.faces.ensure_lookup_table()
            copied_loop_verts = [copied.verts[vert.index] for vert in loop_verts]
            copied_loop_edges = [copied.edges[edge.index] for edge in loop_edges]
            copied_selected = [
                copied.faces[face.index] for face in selected_faces
            ]
            bmesh.ops.delete(
                copied, geom=copied_selected, context='FACES'
            )
            if any(
                not edge.is_valid or len(edge.link_faces) != 1
                for edge in copied_loop_edges
            ):
                return None, 0.0
            result = fair_boundary_loop(
                copied,
                copied_loop_verts,
                copied_loop_edges,
                self._object.matrix_world,
                {"method": "ORGANIC", "residual": 0.0},
            )
            return GeometryMixin._surface_bvh(
                result["faces"], self._object.matrix_world
            )
        except (FairingError, IndexError, ValueError, np.linalg.LinAlgError):
            return None, 0.0
        finally:
            copied.free()

    @staticmethod
    def _project_bvh_along_axis(bvh, span, point, reference, axis):
        """Return the closest bidirectional axis-ray hit to ``reference``."""
        hit = GeometryMixin._project_bvh_hit_along_axis(
            bvh, span, point, reference, axis
        )
        return hit[0] if hit is not None else None

    @staticmethod
    def _project_bvh_hit_along_axis(bvh, span, point, reference, axis):
        """Return the closest axis-ray hit and normal to ``reference``."""
        if bvh is None:
            return None
        axis = axis.normalized()
        epsilon = max(span * 1.0e-7, 1.0e-8)
        candidates = []
        nearest = bvh.find_nearest(point)
        if nearest and nearest[0] is not None:
            location = nearest[0]
            delta = location - point
            perpendicular = delta - axis * delta.dot(axis)
            if perpendicular.length <= epsilon * 4.0:
                candidates.append((location, nearest[1]))
        for direction in (axis, -axis):
            hit = bvh.ray_cast(
                point + direction * epsilon,
                direction,
                span,
            )
            if hit[0] is not None:
                candidates.append((hit[0], hit[1]))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda candidate: (candidate[0] - reference).length,
        )

    def _enlargement_reaches_nonplanar_support(self):
        """Detect a target rim that grows beyond its original planar collar."""
        if GeometryMixin._target_scale(self) <= 1.0 + 1.0e-9:
            return False
        bvh = getattr(self, "_resize_base_bvh", None)
        span = getattr(self, "_resize_base_span", 0.0)
        if bvh is None:
            return False
        fit = self._fit
        axis = fit["axis"].normalized()
        # This is a mode switch, not a precision surface measurement. Allow
        # ordinary mesh noise and triangulation-normal variation to remain in
        # the direct one-plane path; the initial hypothesis has already
        # established that the collar itself is planar.
        tolerance = max(span * 5.0e-4, fit["radius"] * 1.0e-3, 1.0e-6)
        normal_alignment = math.cos(math.radians(2.0))
        unique_surfaces = {
            id(surface): surface
            for surface in getattr(
                self, "_resize_surfaces_by_vertex", {}
            ).values()
        }.values()
        for surface in unique_surfaces:
            hypothesis = surface.get("hypothesis", {})
            planes = hypothesis.get("planes", ())
            if (
                not surface.get("exterior")
                or not surface.get("closed")
                or hypothesis.get("method") != "PLANES"
                or len(planes) != 1
            ):
                continue
            plane = planes[0]
            plane_normal = plane["normal"].normalized()
            tested = 0
            nonplanar = 0
            for point in surface.get("loop_world_points", ()):
                offset = point - fit["center"]
                coordinate = offset.dot(axis)
                radial = offset - axis * coordinate
                if radial.length_squared <= 1.0e-16:
                    continue
                source_radius = GeometryMixin._fitted_radius_at(
                    self, coordinate
                )
                target_radius = GeometryMixin._target_radius_at(
                    self, coordinate
                )
                if target_radius <= source_radius + tolerance:
                    continue
                tested += 1
                probe = (
                    fit["center"]
                    + axis * coordinate
                    + radial.normalized() * target_radius
                )
                hit = GeometryMixin._project_bvh_hit_along_axis(
                    bvh, span, probe, point, axis
                )
                if hit is None:
                    nonplanar += 1
                    continue
                location, normal = hit
                plane_error = abs(
                    plane_normal.dot(location - plane["point"])
                )
                if plane_error > tolerance:
                    nonplanar += 1
                elif (
                    normal is not None
                    and normal.length_squared > 1.0e-16
                    and abs(normal.normalized().dot(plane_normal))
                    < normal_alignment
                ):
                    nonplanar += 1
            required = 1 if tested == 1 else max(2, math.ceil(tested * 0.1))
            if nonplanar >= required:
                return True
        return False

    @staticmethod
    def _project_hypothesis_along_axis(hypothesis, point, reference, axis):
        """Intersect an axis line with a virtual plane/sphere/conic surface."""
        method = hypothesis.get("method")
        axis = axis.normalized()
        candidates = []
        if method == "PLANES":
            for plane in hypothesis["planes"]:
                denominator = plane["normal"].dot(axis)
                if abs(denominator) <= 1.0e-9:
                    continue
                distance = plane["normal"].dot(
                    plane["point"] - point
                ) / denominator
                candidates.append(point + axis * distance)
        elif method == "SPHERE":
            relative = point - hypothesis["center"]
            linear = relative.dot(axis)
            constant = relative.length_squared - hypothesis["radius"] ** 2
            discriminant = linear * linear - constant
            tolerance = max(
                (linear * linear + abs(constant)) * 1.0e-10,
                1.0e-12,
            )
            if discriminant >= -tolerance:
                root = math.sqrt(max(0.0, discriminant))
                candidates.extend((
                    point + axis * (-linear - root),
                    point + axis * (-linear + root),
                ))
        elif method == "CONIC":
            fit = hypothesis["fit"]
            surface_axis = fit["axis"].normalized()
            relative = point - fit["center"]
            axial = relative.dot(surface_axis)
            ray_axial = axis.dot(surface_axis)
            radial = relative - surface_axis * axial
            ray_radial = axis - surface_axis * ray_axial
            slope = float(fit.get("slope", 0.0))
            radius = float(fit["radius"]) + slope * axial
            radius_rate = slope * ray_axial
            quadratic = ray_radial.length_squared - radius_rate * radius_rate
            linear = 2.0 * (
                radial.dot(ray_radial) - radius * radius_rate
            )
            constant = radial.length_squared - radius * radius
            if abs(quadratic) <= 1.0e-12:
                if abs(linear) > 1.0e-12:
                    candidates.append(point + axis * (-constant / linear))
            else:
                discriminant = linear * linear - 4.0 * quadratic * constant
                tolerance = max(
                    (
                        linear * linear
                        + abs(4.0 * quadratic * constant)
                    ) * 1.0e-10,
                    1.0e-12,
                )
                if discriminant >= -tolerance:
                    root = math.sqrt(max(0.0, discriminant))
                    candidates.extend((
                        point + axis * ((-linear - root) / (2.0 * quadratic)),
                        point + axis * ((-linear + root) / (2.0 * quadratic)),
                    ))
        if not candidates:
            return None
        return min(candidates, key=lambda candidate: (candidate - reference).length)

    def _boolean_cutter_geometry(
        self, obj, blind_coordinates=None, segment_override=None
    ):
        """Return local cutter vertices/faces for the requested hole dimensions."""
        fit = self._fit
        axis = fit["axis"].normalized()
        depth_fit = getattr(self, "_depth_fit", None)
        world_vertices = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
        if not world_vertices:
            raise FairingError("The reconstructed base mesh is empty")
        axial_bounds = [
            (point - fit["center"]).dot(axis) for point in world_vertices
        ]
        span = max(max(axial_bounds) - min(axial_bounds), fit["radius"], 1.0e-6)
        # Keep the cutter cap comfortably outside the reconstructed exterior.
        # Exact Boolean is much less reliable when the cap is coplanar with, or
        # only microscopically beyond, a large planar support face.
        target_outer_radius = max(
            abs(float(getattr(self, "diameter", fit["radius"] * 2.0))) * 0.5,
            float(fit["radius"]),
        )
        margin = max(
            span * 0.075 + target_outer_radius * 0.2,
            span * 1.0e-4,
            1.0e-6,
        )
        if depth_fit:
            outer = depth_fit["outer_coordinate"]
            inner = outer + (
                depth_fit["inner_coordinate"] - outer
            ) * GeometryMixin._target_depth_scale(self)
            direction = 1.0 if inner >= outer else -1.0
            coordinates = (outer - direction * margin, inner)
        elif blind_coordinates is not None:
            outer, inner = blind_coordinates
            direction = 1.0 if inner >= outer else -1.0
            coordinates = (outer - direction * margin, inner)
        else:
            coordinates = (
                min(axial_bounds) - margin,
                max(axial_bounds) + margin,
            )

        segments = max(3, min(512, int(
            segment_override
            or getattr(self, "boolean_segments", 0)
            or getattr(self, "_resize_angular_segments", 0)
            or len(self._locked_indices)
            or 32
        )))
        inverse = obj.matrix_world.inverted_safe()
        vertices = []
        for coordinate in coordinates:
            radius = GeometryMixin._target_radius_at(self, coordinate)
            for index in range(segments):
                angle = math.tau * index / segments
                radial = (
                    fit["basis_x"] * math.cos(angle)
                    + fit["basis_y"] * math.sin(angle)
                )
                vertices.append(
                    inverse @ (
                        fit["center"] + axis * coordinate + radial * radius
                    )
                )

        faces = []
        for index in range(segments):
            following = (index + 1) % segments
            faces.append((
                index,
                following,
                segments + following,
                segments + index,
            ))
        faces.append(tuple(reversed(range(segments))))
        faces.append(tuple(range(segments, segments * 2)))
        if coordinates[1] < coordinates[0]:
            faces = [tuple(reversed(face)) for face in faces]
        return vertices, faces

    def _cleanup_shrunk_boolean(self, obj, cutter_vertices, cutter_faces):
        """Dissolve Boolean splits around a newly inset, smaller hole."""
        cutter_bvh = BVHTree.FromPolygons(
            cutter_vertices, cutter_faces, all_triangles=False
        )
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        minimum = Vector((
            min(vert.x for vert in cutter_vertices),
            min(vert.y for vert in cutter_vertices),
            min(vert.z for vert in cutter_vertices),
        ))
        maximum = Vector((
            max(vert.x for vert in cutter_vertices),
            max(vert.y for vert in cutter_vertices),
            max(vert.z for vert in cutter_vertices),
        ))
        scale = max((maximum - minimum).length, 1.0)
        tolerance = max(scale * 1.0e-5, 1.0e-7)
        inset_faces = set()
        for face in bm.faces:
            nearest = cutter_bvh.find_nearest(face.calc_center_median())
            if nearest is not None and nearest[0] is not None:
                if nearest[3] <= tolerance:
                    inset_faces.add(face)
        if not inset_faces:
            return
        result = expanded_limited_dissolve(bm, inset_faces)
        for face in bm.faces:
            face.select = False
        for face in result["faces"]:
            if face.is_valid:
                face.select = True
        bm.normal_update()
        bmesh.update_edit_mesh(
            obj.data, loop_triangles=True, destructive=True
        )

    @staticmethod
    def _boolean_depth_unchanged(operator, blind_coordinates):
        """Return whether reconstruction retained the fitted blind depth."""
        depth_fit = getattr(operator, "_depth_fit", None)
        if depth_fit is None:
            return blind_coordinates is not None
        initial = float(depth_fit.get("initial", 0.0))
        current = float(getattr(operator, "depth", initial))
        return math.isclose(
            current,
            initial,
            rel_tol=1.0e-6,
            abs_tol=max(abs(initial) * 1.0e-6, 1.0e-9),
        )

    def _cleanup_unchanged_boolean_base(self, obj, cutter_vertices):
        """Dissolve coplanar Boolean splits in an unchanged blind-hole base."""
        segment_count = len(cutter_vertices) // 2
        if segment_count < 3 or len(cutter_vertices) != segment_count * 2:
            return

        fit = self._fit
        axis = fit["axis"].normalized()
        center = fit["center"]
        cap_points = [
            obj.matrix_world @ point
            for point in cutter_vertices[segment_count:]
        ]
        cap_coordinate = sum(
            (point - center).dot(axis) for point in cap_points
        ) / len(cap_points)
        cap_radius = max(
            (
                point
                - center
                - axis * (point - center).dot(axis)
            ).length
            for point in cap_points
        )
        tolerance = max(cap_radius * 1.0e-5, 1.0e-7)
        normal_limit = math.cos(PARALLEL_PLANE_ANGLE_EPSILON)
        normal_matrix = (
            obj.matrix_world.to_3x3().inverted_safe().transposed()
        )

        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()

        def is_base_face(face):
            if not face.is_valid:
                return False
            world_normal = normal_matrix @ face.normal
            if (
                world_normal.length_squared <= 1.0e-16
                or abs(world_normal.normalized().dot(axis)) < normal_limit
            ):
                return False
            for vert in face.verts:
                point = obj.matrix_world @ vert.co
                offset = point - center
                coordinate = offset.dot(axis)
                radial = offset - axis * coordinate
                if (
                    abs(coordinate - cap_coordinate) > tolerance
                    or radial.length > cap_radius + tolerance
                ):
                    return False
            return True

        base_faces = {face for face in bm.faces if is_base_face(face)}
        if len(base_faces) > 1:
            affected_verts = {
                vert for face in base_faces for vert in face.verts
            }
            affected_edges = {
                edge for face in base_faces for edge in face.edges
            }
            bmesh.ops.dissolve_limit(
                bm,
                angle_limit=LIMITED_DISSOLVE_ANGLE,
                use_dissolve_boundaries=False,
                verts=list(affected_verts),
                edges=list(affected_edges),
                delimit={'NORMAL'},
            )
            bm.normal_update()

        for face in bm.faces:
            face.select = False
        for face in bm.faces:
            if is_base_face(face):
                face.select = True
        bmesh.update_edit_mesh(
            obj.data, loop_triangles=True, destructive=True
        )

    def _replace_boolean_preview_with_source(self, consume=False):
        """Replace a live Boolean preview with a pristine source-mesh copy."""
        source = getattr(self, "_boolean_preview_source", None)
        obj = self._object
        if source is None or obj is None:
            return False
        if obj.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        preview_mesh = obj.data
        obj.data = source if consume else source.copy()
        if consume:
            self._boolean_preview_source = None
            self._boolean_preview_segments = 0
            self._boolean_preview_last_update = 0.0
            self._boolean_preview_pending = False
        if preview_mesh is not source and preview_mesh.users == 0:
            bpy.data.meshes.remove(preview_mesh)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        self._resize_projection_ready = False
        self._resize_surfaces_by_vertex = {}
        self._resize_base_bvh = None
        self._resize_base_span = 0.0
        self._original_vertex_coords = None
        self._proportional_weights = None
        return True

    def _discard_boolean_preview_source(self):
        """Release the pristine source retained while previewing a Boolean."""
        source = getattr(self, "_boolean_preview_source", None)
        self._boolean_preview_source = None
        self._boolean_preview_segments = 0
        self._boolean_preview_last_update = 0.0
        self._boolean_preview_pending = False
        if source is not None and source.users == 0:
            bpy.data.meshes.remove(source)

    def _update_boolean_preview(
        self, context, preview_segments=16, source_is_current=False
    ):
        """Show a reversible, low-resolution Exact Boolean mesh in Edit Mode."""
        obj = context.active_object
        if getattr(self, "_boolean_preview_source", None) is None:
            self._boolean_preview_source = obj.data.copy()
        self._boolean_preview_segments = preview_segments
        if not source_is_current:
            GeometryMixin._replace_boolean_preview_with_source(self)
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        selected_faces = [
            bm.faces[index] for index in sorted(self._locked_indices)
        ]
        GeometryMixin._build_resize_projection_state(
            self, bm, selected_faces
        )
        selected_verts = {
            vert for face in selected_faces for vert in face.verts
        }
        GeometryMixin._capture_deformation_state(
            self, context, bm, selected_verts
        )
        self._resize_dynamic_requires_boolean = True
        self._resize_requires_boolean = True
        GeometryMixin._recreate_hole_with_boolean(
            self,
            context,
            bm,
            selected_faces,
            preview_segments=preview_segments,
        )
        self._boolean_preview_last_update = time.monotonic()
        self._boolean_preview_pending = False
        build_overlay = getattr(self, "_build_cylinder_preview", None)
        if build_overlay is not None:
            build_overlay(context)

    def _preserved_blind_coordinates(self, surfaces):
        """Recover exterior-to-cap axial coordinates when Depth is unavailable."""
        fit = self._fit
        axis = fit["axis"]
        center = fit["center"]

        def coordinate(surface):
            points = surface.get("loop_world_points", ())
            if not points:
                points = [
                    self._object.matrix_world @ vert.co
                    for vert in surface.get("loop_verts", ())
                    if vert.is_valid
                ]
            if not points:
                return None
            return sum(
                (point - center).dot(axis) for point in points
            ) / len(points)

        exterior = [
            value
            for surface in surfaces
            if surface["closed"] and surface["exterior"]
            for value in (coordinate(surface),)
            if value is not None
        ]
        interior = [
            value
            for surface in surfaces
            if surface["closed"] and not surface["exterior"]
            for value in (coordinate(surface),)
            if value is not None
        ]
        if not exterior or not interior:
            return None
        _distance, outer, inner = max(
            (abs(inner - outer), outer, inner)
            for outer in exterior
            for inner in interior
        )
        minimum_separation = max(float(fit["radius"]) * 1.0e-6, 1.0e-9)
        return (
            (outer, inner)
            if abs(inner - outer) > minimum_separation
            else None
        )

    @staticmethod
    def _recover_reconstruction_rim(bm, surface, matrix_world):
        """Rediscover a support boundary after destructive cavity deletion."""
        reference_points = list(surface.get("loop_world_points", ()))
        if not reference_points:
            reference_points = [
                matrix_world @ vert.co
                for vert in surface.get("loop_verts", ())
                if vert.is_valid
            ]
        if len(reference_points) < 3:
            raise FairingError(
                "Could not isolate the exterior rim for reconstruction"
            )

        support_faces = surface.get(
            "support_patch_faces", surface.get("support_faces", ())
        )
        candidates = {
            edge
            for face in support_faces
            if face.is_valid
            for edge in face.edges
            if edge.is_valid and len(edge.link_faces) == 1
        }
        if not candidates:
            candidates = {
                edge
                for edge in bm.edges
                if edge.is_valid and not edge.hide and len(edge.link_faces) == 1
            }
        if not candidates:
            raise FairingError(
                "Could not isolate the exterior rim for reconstruction"
            )

        def distance_to_reference(edge):
            midpoint = matrix_world @ (
                (edge.verts[0].co + edge.verts[1].co) * 0.5
            )
            return min(
                (midpoint - point).length_squared
                for point in reference_points
            )

        seed = min(candidates, key=distance_to_reference)
        try:
            loop_verts, loop_edges = ordered_boundary_loop(
                seed, boundary_component(seed)
            )
        except FairingError as error:
            raise FairingError(
                "Could not isolate the exterior rim for reconstruction"
            ) from error
        return loop_verts, loop_edges

    def _recreate_hole_with_boolean(
        self, context, bm, selected_faces, preview_segments=None
    ):
        """Rebuild the old opening, then subtract the requested conic cutter."""
        obj = context.active_object
        shrinking = GeometryMixin._target_scale(self) < 1.0 - 1.0e-9
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
        backup = obj.data.copy()
        cutter_object = None
        cutter_mesh = None
        modifier_name = None
        try:
            unique_surfaces = list({
                id(surface): surface
                for surface in self._resize_surfaces_by_vertex.values()
            }.values())
            exterior_surfaces = [
                surface for surface in unique_surfaces
                if surface["exterior"] and surface["closed"]
            ]
            if not exterior_surfaces:
                raise FairingError(
                    "Boolean reconstruction needs a closed exterior rim"
                )
            blind_coordinates = (
                None
                if getattr(self, "_depth_fit", None) is not None
                else GeometryMixin._preserved_blind_coordinates(
                    self, unique_surfaces
                )
            )

            fit = self._fit
            selected = set(selected_faces)
            # The radial cavity envelope is deliberately a little generous so
            # it also removes imperfect wall/cap geometry.  On a curved outer
            # surface (most visibly a sphere), however, the center of a face
            # touching the rim can still fall just inside that envelope.  Those
            # faces define the accepted exterior boundary and must survive the
            # cleanup; otherwise their stored rim edges disappear before the
            # fairing pass can close the old opening.
            protected_faces = {
                face
                for surface in exterior_surfaces
                for face in surface.get(
                    "support_patch_faces", surface["support_faces"]
                )
                if face.is_valid
            }
            blocked_cavity_faces = selected | protected_faces

            # A support patch is intentionally shallow because analytic fitting
            # needs only the nearby surface. Deletion needs the opposite: walk
            # the complete blind-cap component bounded by the selected wall.
            # This removes dense, multi-ring bottoms whose inner rings lie
            # beyond both the support-patch depth and the wall-fit envelope.
            interior_faces = set()
            cavity_stack = [
                face
                for surface in unique_surfaces
                if not surface["exterior"] and surface["closed"]
                for face in surface["support_faces"]
                if face.is_valid and face not in blocked_cavity_faces
            ]
            while cavity_stack:
                face = cavity_stack.pop()
                if face in interior_faces or face in blocked_cavity_faces:
                    continue
                interior_faces.add(face)
                for edge in face.edges:
                    cavity_stack.extend(
                        neighbor
                        for neighbor in edge.link_faces
                        if (
                            neighbor.is_valid
                            and neighbor not in interior_faces
                            and neighbor not in blocked_cavity_faces
                        )
                    )
            # Redo starts from Blender's restored pre-operation mesh. Remove
            # every detected blind-cap support face explicitly before relying
            # on the fitted cavity envelope for less structured geometry. A
            # sloped or dished cap can extend axially beyond the wall fit, in
            # which case a centroid-only envelope test leaves it behind as a
            # floating bottom when the requested depth becomes a through hole.
            delete_faces = set(selected) | interior_faces
            axial_padding = max(fit["z_max"] - fit["z_min"], 1.0e-6) * 0.02
            for face in bm.faces:
                if face in selected or face in protected_faces:
                    continue
                point = obj.matrix_world @ face.calc_center_median()
                offset = point - fit["center"]
                coordinate = offset.dot(fit["axis"])
                if not (
                    fit["z_min"] - axial_padding
                    <= coordinate
                    <= fit["z_max"] + axial_padding
                ):
                    continue
                radial = offset - fit["axis"] * coordinate
                old_radius = GeometryMixin._fitted_radius_at(self, coordinate)
                if radial.length <= old_radius * 1.02:
                    delete_faces.add(face)

            bmesh.ops.delete(
                bm, geom=list(delete_faces), context='FACES'
            )
            bm.normal_update()
            self.reconstruction_result_method = ""
            self.reconstruction_result_fill_style = ""
            for surface in exterior_surfaces:
                loop_verts, loop_edges = (
                    GeometryMixin._recover_reconstruction_rim(
                        bm, surface, obj.matrix_world
                    )
                )
                hypothesis = surface["hypothesis"]
                requested_method = getattr(self, "fairing_method", 'AUTO')
                analytic_faces = [
                    face
                    for face in surface.get(
                        "support_patch_faces", surface["support_faces"]
                    )
                    if face.is_valid
                ]
                if requested_method != 'AUTO':
                    hypothesis = choose_hypothesis(
                        loop_verts,
                        loop_edges,
                        obj.matrix_world,
                        lambda faces: GeometryMixin._fit_cylinder(self, faces),
                        lambda face: GeometryMixin._world_normal(self, face),
                        support_faces=analytic_faces,
                        analytic_faces=analytic_faces,
                        forced_method=requested_method,
                    )
                fill_style = getattr(self, "fill_style", 'AUTO')
                prepared = prepare_fair_boundary(
                    obj,
                    bm,
                    loop_verts,
                    loop_edges,
                    hypothesis,
                    fill_style,
                )
                bm = prepared["bm"]
                loop_verts = prepared["loop_verts"]
                loop_edges = prepared["loop_edges"]
                # Preserve the hypothesis accepted from the intact support
                # patch. A second automatic cascade over only the surviving
                # post-delete rim can classify the same flat surface
                # differently and bend a planar reconstruction. Sphere and
                # cylinder are the exceptions: their boundary-inclusive
                # dissolve changes the analytic samples, so refit the same
                # forced model against the surviving support patch.
                if prepared["forced_method"] is not None:
                    analytic_faces = [
                        face
                        for face in surface.get(
                            "support_patch_faces", surface["support_faces"]
                        )
                        if face.is_valid
                    ]
                    hypothesis = choose_hypothesis(
                        loop_verts,
                        loop_edges,
                        obj.matrix_world,
                        lambda faces: GeometryMixin._fit_cylinder(self, faces),
                        lambda face: GeometryMixin._world_normal(self, face),
                        analytic_faces=analytic_faces,
                        forced_method=prepared["forced_method"],
                    )
                result = fair_boundary_loop(
                    bm,
                    loop_verts,
                    loop_edges,
                    obj.matrix_world,
                    hypothesis,
                    fill_style,
                    circle_loops=getattr(self, "circle_loops", 2),
                    grid_angle=getattr(self, "grid_angle", 0.0),
                    direction_samples=prepared["direction_samples"],
                )
                self.reconstruction_result_method = result["method"]
                self.reconstruction_result_fill_style = result["fill_style"]
                loop_verts = result.get("boundary_verts", loop_verts)
                loop_edges = result.get("boundary_edges", loop_edges)
                bm, _result = finalize_fair_patch(
                    obj,
                    bm,
                    result,
                    loop_verts,
                    loop_edges,
                    # This patch is an intermediate Boolean operand. Retain
                    # its radial/grid subdivisions so Exact Boolean has local
                    # faces to intersect; shrinking performs the expanded
                    # limited dissolve on the finished result instead.
                    final_simplify=False,
                )

            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=True, destructive=True
            )
            bpy.ops.object.mode_set(mode='OBJECT')

            cutter_vertices, cutter_faces = (
                GeometryMixin._boolean_cutter_geometry(
                    self,
                    obj,
                    blind_coordinates,
                    segment_override=preview_segments,
                )
            )
            cutter_mesh = bpy.data.meshes.new("Holey Tools Boolean Cutter")
            cutter_mesh.from_pydata(cutter_vertices, [], cutter_faces)
            cutter_mesh.validate(verbose=False)
            cutter_mesh.update()
            cutter_object = bpy.data.objects.new(
                "Holey Tools Boolean Cutter", cutter_mesh
            )
            context.collection.objects.link(cutter_object)
            cutter_object.matrix_world = obj.matrix_world.copy()
            cutter_object.display_type = 'WIRE'
            cutter_object.hide_render = True
            cutter_object.select_set(False)

            context.view_layer.objects.active = obj
            obj.select_set(True)
            modifier = obj.modifiers.new(
                name="Holey Tools Recreate Hole", type='BOOLEAN'
            )
            modifier_name = modifier.name
            modifier.operation = 'DIFFERENCE'
            modifier.solver = 'EXACT'
            modifier.object = cutter_object
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            modifier_name = None
            if getattr(self, "final_simplify", True):
                if shrinking:
                    GeometryMixin._cleanup_shrunk_boolean(
                        self, obj, cutter_vertices, cutter_faces
                    )
                if GeometryMixin._boolean_depth_unchanged(
                    self, blind_coordinates
                ):
                    GeometryMixin._cleanup_unchanged_boolean_base(
                        self, obj, cutter_vertices
                    )
            bpy.data.objects.remove(cutter_object, do_unlink=True)
            cutter_object = None
            if cutter_mesh.users == 0:
                bpy.data.meshes.remove(cutter_mesh)
            cutter_mesh = None
            bpy.data.meshes.remove(backup)
            backup = None
            if obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            self._resize_projection_ready = False
            self._resize_surfaces_by_vertex = {}
            self._resize_base_bvh = None
            return
        except Exception as error:
            if obj.mode == 'EDIT':
                bpy.ops.object.mode_set(mode='OBJECT')
            failed_mesh = obj.data
            obj.data = backup
            backup = None
            if failed_mesh.users == 0:
                bpy.data.meshes.remove(failed_mesh)
            if obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            self._resize_projection_ready = False
            if isinstance(error, FairingError):
                raise
            raise FairingError(f"Boolean hole reconstruction failed: {error}") from error
        finally:
            if modifier_name is not None:
                modifier = obj.modifiers.get(modifier_name)
                if modifier is not None:
                    obj.modifiers.remove(modifier)
            if cutter_object is not None:
                bpy.data.objects.remove(cutter_object, do_unlink=True)
            if cutter_mesh is not None and cutter_mesh.users == 0:
                bpy.data.meshes.remove(cutter_mesh)
            if backup is not None and backup.users == 0:
                bpy.data.meshes.remove(backup)

    def _resolve_resize_wall_faces(self, bm):
        """Return the fitted wall, recovering it when redo indices are stale.

        Blender normally restores the pre-operation mesh before executing an
        Adjust Last Operation change. Some modal/Boolean edit-mode sequences
        can instead expose the previous Boolean result briefly. Face indices
        from the original wall then refer to unrelated cap/support faces. Find
        the connected coaxial wall in that mesh so reconstruction remains
        transactional rather than cutting through stale topology.
        """
        bm.faces.ensure_lookup_table()
        fit = self._fit
        axis = fit["axis"].normalized()
        matrix_world = self._object.matrix_world
        fitted_radius = max(abs(float(fit.get("radius", 0.0))), 1.0e-6)
        target_radius = max(
            abs(float(getattr(self, "diameter", fitted_radius * 2.0))) * 0.5,
            1.0e-6,
        )
        axial_min = min(float(fit["z_min"]), float(fit["z_max"]))
        axial_max = max(float(fit["z_min"]), float(fit["z_max"]))
        axial_span = max(axial_max - axial_min, fitted_radius, 1.0e-6)
        # Modal direct previews may temporarily move the same indexed wall well
        # beyond both the fitted and final radii before confirmation.
        radial_limit = max(fitted_radius, target_radius) * 3.0
        radial_spread_limit = max(fitted_radius * 0.075, 1.0e-5)
        target_slope = abs(float(GeometryMixin._target_cone_slope(self)))
        expected_axial_normal = target_slope / math.sqrt(
            1.0 + target_slope * target_slope
        )

        def wall_measure(face):
            points = [matrix_world @ vert.co for vert in face.verts]
            if len(points) < 3:
                return None
            offsets = [point - fit["center"] for point in points]
            coordinates = [offset.dot(axis) for offset in offsets]
            if (
                max(coordinates) < axial_min - axial_span * 0.25
                or min(coordinates) > axial_max + axial_span * 0.25
            ):
                return None
            radii = [
                (offset - axis * coordinate).length
                for offset, coordinate in zip(offsets, coordinates)
            ]
            mean_radius = sum(radii) / len(radii)
            mean_coordinate = sum(coordinates) / len(coordinates)
            coordinate_variance = sum(
                (coordinate - mean_coordinate) ** 2
                for coordinate in coordinates
            )
            if coordinate_variance > 1.0e-12:
                local_slope = sum(
                    (coordinate - mean_coordinate) * (radius - mean_radius)
                    for coordinate, radius in zip(coordinates, radii)
                ) / coordinate_variance
                radial_residuals = [
                    radius - (
                        mean_radius
                        + local_slope * (coordinate - mean_coordinate)
                    )
                    for coordinate, radius in zip(coordinates, radii)
                ]
            else:
                radial_residuals = [radius - mean_radius for radius in radii]
            if (
                mean_radius <= 1.0e-7
                or mean_radius > radial_limit
                or max(radial_residuals) - min(radial_residuals)
                > radial_spread_limit
            ):
                return None
            center_offset = (
                matrix_world @ face.calc_center_median()
            ) - fit["center"]
            center_radial = center_offset - axis * center_offset.dot(axis)
            if center_radial.length <= 1.0e-7:
                return None
            normal = GeometryMixin._world_normal(self, face)
            if (
                abs(abs(normal.dot(axis)) - expected_axial_normal)
                > 0.25
            ):
                return None
            if abs(normal.dot(center_radial.normalized())) < 0.65:
                return None
            return mean_radius

        indexed_faces = []
        indices_valid = True
        for index in sorted(self._locked_indices):
            if index < 0 or index >= len(bm.faces):
                indices_valid = False
                break
            indexed_faces.append(bm.faces[index])
        expected_counts = (
            int(getattr(self, "fit_mesh_vertex_count", 0)),
            int(getattr(self, "fit_mesh_edge_count", 0)),
            int(getattr(self, "fit_mesh_face_count", 0)),
        )
        current_counts = (len(bm.verts), len(bm.edges), len(bm.faces))
        expected_signature = str(getattr(
            self, "fit_mesh_topology_signature", ""
        ))
        source_topology_known = bool(expected_signature) or all(
            count > 0 for count in expected_counts
        )
        source_topology_restored = False
        if expected_signature:
            source_topology_restored = (
                GeometryMixin._bmesh_topology_signature(bm)
                == expected_signature
            )
        elif all(count > 0 for count in expected_counts):
            # Compatibility with an Adjust Last Operation instance created
            # before connectivity signatures were added.
            source_topology_restored = current_counts == expected_counts
        # Face indices are authoritative on the source topology. In particular,
        # imperfect/missing-face walls need not satisfy the ideal coaxial test.
        # A zero count denotes redo data created before these guards existed.
        if (
            indices_valid
            and len(indexed_faces) >= 3
            and (source_topology_restored or not source_topology_known)
        ):
            return indexed_faces
        if (
            indices_valid
            and len(indexed_faces) >= 3
            and all(wall_measure(face) is not None for face in indexed_faces)
        ):
            return indexed_faces

        candidates = {
            face: measure
            for face in bm.faces
            if (measure := wall_measure(face)) is not None
        }
        components = []
        remaining = set(candidates)
        while remaining:
            seed = remaining.pop()
            component = {seed}
            stack = [seed]
            while stack:
                face = stack.pop()
                for edge in face.edges:
                    for neighbor in edge.link_faces:
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            component.add(neighbor)
                            stack.append(neighbor)
            components.append(component)
        components = [component for component in components if len(component) >= 3]
        if not components:
            if indices_valid and len(indexed_faces) >= 3:
                return indexed_faces
            raise FairingError("Could not recover the resized hole wall for redo")

        def component_score(component):
            coordinates = []
            radii = []
            for face in component:
                radii.append(candidates[face])
                for vert in face.verts:
                    point = matrix_world @ vert.co
                    coordinates.append((point - fit["center"]).dot(axis))
            ordered_radii = sorted(radii)
            median_radius = ordered_radii[len(ordered_radii) // 2]
            axial_coverage = max(coordinates) - min(coordinates)
            radius_error = min(
                abs(median_radius - fitted_radius),
                abs(median_radius - target_radius),
            )
            return (
                len(component),
                axial_coverage / axial_span,
                -radius_error / fitted_radius,
            )

        faces = sorted(max(components, key=component_score), key=lambda face: face.index)
        self._locked_indices = {face.index for face in faces}
        return faces

    def _apply_diameter_to_mesh(self, context, commit_boolean=False):
        """Rebuild affected vertices from the captured source geometry.

        Each world-space offset is split into axial and radial components. Radius
        follows the fitted linear cone model while diameter anchors the model at
        the top surface; depth scales coordinates away from that surface and the
        optional conic angle changes the model slope. Enlarged end rims project
        along the hole axis to the captured base mesh; reduced rims project to
        a virtual fair surface. The fitted end plane remains the fallback when
        neither projection intersects. Proportional weights blend each target
        with its original position before conversion back to object-local space.
        """
        obj = context.active_object
        if (
            not commit_boolean
            and getattr(self, "_boolean_preview_source", None) is not None
            and bool(getattr(self, "_drag_active", False))
            and time.monotonic()
            - float(getattr(self, "_boolean_preview_last_update", 0.0))
            < _BOOLEAN_PREVIEW_INTERVAL
        ):
            # Keep the previous Boolean mesh visible and let the inexpensive
            # GPU cutter follow every pointer event. The mouse-release handler
            # performs a trailing rebuild for the final drag value.
            self._boolean_preview_pending = True
            return
        restored_boolean_preview = False
        if getattr(self, "_boolean_preview_source", None) is not None:
            restored_boolean_preview = (
                GeometryMixin._replace_boolean_preview_with_source(self)
            )
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        faces = GeometryMixin._resolve_resize_wall_faces(self, bm)
        verts = {vert for face in faces for vert in face.verts}

        if not getattr(self, "_resize_projection_ready", False):
            GeometryMixin._build_resize_projection_state(self, bm, faces)
        if restored_boolean_preview:
            self._resize_dynamic_requires_boolean = True

        if not getattr(self, "_original_vertex_coords", None):
            GeometryMixin._capture_deformation_state(
                self, context, bm, verts
            )

        inverse = obj.matrix_world.inverted_safe()
        fit = self._fit
        target_scale = self._target_scale()
        depth_fit = getattr(self, "_depth_fit", None)
        depth_scale = self._target_depth_scale() if depth_fit else 1.0
        target_slope = GeometryMixin._target_cone_slope(self)
        fitted_slope = float(fit.get("slope", 0.0))

        base_requires_boolean = getattr(
            self,
            "_resize_base_requires_boolean",
            getattr(self, "_resize_requires_boolean", False),
        )
        dynamic_requires_boolean = bool(
            getattr(self, "_resize_dynamic_requires_boolean", False)
        )
        if (
            not base_requires_boolean
            and not dynamic_requires_boolean
            and not bool(fit.get("is_stud", False))
            and GeometryMixin._enlargement_reaches_nonplanar_support(self)
        ):
            dynamic_requires_boolean = True
        self._resize_dynamic_requires_boolean = dynamic_requires_boolean
        self._resize_requires_boolean = bool(
            base_requires_boolean or dynamic_requires_boolean
        )

        # Confirming untouched dimensions must be an exact geometry no-op. In
        # particular, do not flatten within-epsilon boundary variation onto the
        # best-fit planes. This path also restores the verbatim source geometry
        # when edited dimensions are returned to their fitted values.
        if math.isclose(
            target_scale, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-9
        ) and math.isclose(
            depth_scale, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-9
        ) and math.isclose(
            target_slope, fitted_slope, rel_tol=1.0e-6, abs_tol=1.0e-9
        ):
            for vertex_index, original in self._original_vertex_coords.items():
                if vertex_index < len(bm.verts):
                    bm.verts[vertex_index].co = original.copy()
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=True, destructive=False
            )
            if commit_boolean:
                GeometryMixin._discard_boolean_preview_source(self)
            return

        if getattr(self, "_resize_requires_boolean", False):
            if commit_boolean:
                try:
                    GeometryMixin._recreate_hole_with_boolean(
                        self, context, bm, faces
                    )
                finally:
                    GeometryMixin._discard_boolean_preview_source(self)
            else:
                # Restore direct-preview vertex movement before snapshotting a
                # pristine source, then rebuild a cheap 16-sided Boolean result
                # for the live modal preview. Confirmation restores that source
                # once more and performs the full-resolution reconstruction.
                for vertex_index, original in self._original_vertex_coords.items():
                    if vertex_index < len(bm.verts):
                        bm.verts[vertex_index].co = original.copy()
                bmesh.update_edit_mesh(
                    obj.data, loop_triangles=True, destructive=False
                )
                try:
                    GeometryMixin._update_boolean_preview(
                        self,
                        context,
                        source_is_current=restored_boolean_preview,
                    )
                except FairingError as error:
                    GeometryMixin._replace_boolean_preview_with_source(
                        self, consume=True
                    )
                    self.report({'WARNING'}, f"Boolean preview failed: {error}")
            return

        outer_plane = (
            (self._end_planes or ())[depth_fit["outer_plane_index"]]
            if depth_fit and not depth_fit.get("surface_depth", False)
            else None
        )
        inner_center = (
            fit["center"] + fit["axis"] * depth_fit["inner_coordinate"]
            if depth_fit
            else None
        )
        inner_half_space = (
            outer_plane["normal"].dot(inner_center - outer_plane["point"])
            if outer_plane is not None
            else 0.0
        )
        plane_by_vertex = {
            vertex_index: plane
            for plane in (self._end_planes or ())
            for vertex_index in plane["verts"]
        }
        selected_indices = {vert.index for vert in verts}
        proportional_weights = getattr(self, "_proportional_weights", None) or {
            index: 1.0 for index in selected_indices
        }
        for vertex_index, weight in proportional_weights.items():
            if vertex_index >= len(bm.verts):
                continue
            vert = bm.verts[vertex_index]
            original = self._original_vertex_coords.get(vertex_index, vert.co)
            world = obj.matrix_world @ original
            offset = world - fit["center"]
            axial_coordinate = offset.dot(fit["axis"])
            target_axial_coordinate = axial_coordinate
            if depth_fit:
                depth_affects_vertex = vertex_index in selected_indices
                if not depth_affects_vertex and abs(inner_half_space) > 1.0e-12:
                    vertex_half_space = outer_plane["normal"].dot(
                        world - outer_plane["point"]
                    )
                    depth_affects_vertex = (
                        vertex_half_space * inner_half_space > 0.0
                    )
                if depth_affects_vertex:
                    target_axial_coordinate = depth_fit["outer_coordinate"] + (
                        axial_coordinate - depth_fit["outer_coordinate"]
                    ) * depth_scale
            axial = fit["axis"] * axial_coordinate
            radial = offset - axial
            source_radius = GeometryMixin._fitted_radius_at(self, axial_coordinate)
            target_radius = GeometryMixin._target_radius_at(
                self, target_axial_coordinate
            )
            radial_scale = target_radius / source_radius
            world_target = (
                fit["center"]
                + fit["axis"] * target_axial_coordinate
                + radial * radial_scale
            )

            # Diameter edits treat exterior and blind-cap rims independently.
            # An enlarged exterior rim lands on the undeformed base mesh, but
            # an interior rim must remain on its own fitted cap surface instead
            # of accidentally hitting the exterior plane. Shrinking either rim
            # intersects its own virtual fairing hypothesis along the hole axis.
            # Depth translation is applied afterward so combined edits remain
            # coherent.
            projection_used = False
            surface = getattr(
                self, "_resize_surfaces_by_vertex", {}
            ).get(vertex_index)
            radius_changed = not math.isclose(
                target_radius,
                source_radius,
                rel_tol=1.0e-6,
                abs_tol=1.0e-9,
            )
            if surface is not None and radius_changed:
                depth_delta = fit["axis"] * (
                    target_axial_coordinate - axial_coordinate
                )
                projection_point = world_target - depth_delta
                if target_radius > source_radius and surface["exterior"]:
                    projected = GeometryMixin._project_bvh_along_axis(
                        getattr(self, "_resize_base_bvh", None),
                        getattr(self, "_resize_base_span", 0.0),
                        projection_point,
                        world,
                        fit["axis"],
                    )
                elif surface["hypothesis"].get("method") == "ORGANIC":
                    projected = GeometryMixin._project_bvh_along_axis(
                        surface.get("organic_bvh"),
                        surface.get("organic_span", 0.0),
                        projection_point,
                        world,
                        fit["axis"],
                    )
                else:
                    projected = GeometryMixin._project_hypothesis_along_axis(
                        surface["hypothesis"],
                        projection_point,
                        world,
                        fit["axis"],
                    )
                if projected is not None:
                    world_target = projected + depth_delta
                    projection_used = True

            # Follow the target cone generator to the translated end plane. For
            # cylinders the slope is zero, reducing exactly to an axial slide.
            plane = (
                plane_by_vertex.get(vertex_index)
                if vertex_index in selected_indices
                else None
            )
            if plane is not None and not projection_used:
                target_offset = world_target - fit["center"]
                target_axial = fit["axis"] * target_offset.dot(fit["axis"])
                target_radial = target_offset - target_axial
                radial_direction = (
                    target_radial.normalized()
                    if target_radial.length_squared > 1.0e-16
                    else fit["basis_x"]
                )
                generator = fit["axis"] + radial_direction * target_slope
                denominator = plane["normal"].dot(generator)
                if abs(denominator) > 1.0e-8:
                    plane_point = plane["point"]
                    if depth_fit:
                        plane_coordinate = plane["normal"].dot(
                            plane_point - fit["center"]
                        ) / denominator
                        target_plane_coordinate = depth_fit["outer_coordinate"] + (
                            plane_coordinate - depth_fit["outer_coordinate"]
                        ) * depth_scale
                        plane_point = plane_point + fit["axis"] * (
                            target_plane_coordinate - plane_coordinate
                        )
                    distance = plane["normal"].dot(
                        plane_point - world_target
                    ) / denominator
                    world_target += generator * distance
            blended_world = world.lerp(world_target, weight)
            vert.co = inverse @ blended_world

        if commit_boolean and target_scale < 1.0 - 1.0e-9:
            result = expanded_limited_dissolve(bm, faces)
            for face in bm.faces:
                face.select = False
            for face in result["faces"]:
                if face.is_valid:
                    face.select = True
            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=True, destructive=True
            )
        else:
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=True, destructive=False
            )
