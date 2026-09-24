# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Circular-cone fitting and mesh deformation for Resize Hole."""

import heapq
import math

import bmesh
import numpy as np

from mathutils import Vector
from mathutils.kdtree import KDTree

from .constants import (
    PARALLEL_PLANE_ANGLE_EPSILON,
    PLANE_FIT_ABSOLUTE_EPSILON,
    PLANE_FIT_RELATIVE_EPSILON,
)


_CONE_ANGLE_EPSILON = math.radians(0.1)
_CONE_RELATIVE_FIT_EPSILON = 0.05
_MAX_GENERATOR_ALIGNMENT = 0.985


class GeometryMixin:
    """Provide selection, fitting, and deformation algorithms to the operator.

    Cone fits and boundary planes are expressed in world space. A cylinder is
    represented as the zero-slope special case. Original
    BMesh vertex coordinates are retained in object-local space so every update
    can be rebuilt from the same undeformed geometry instead of accumulating
    floating-point error.
    """

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
            ) >= parallel_threshold,
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

    def _apply_diameter_to_mesh(self, context):
        """Rebuild affected vertices from the captured source geometry.

        Each world-space offset is split into axial and radial components. Radius
        follows the fitted linear cone model while diameter anchors the model at
        the top surface; depth scales coordinates away from that surface and the
        optional conic angle changes the model slope. Selected boundary vertices
        are subsequently moved along cone generators onto their translated end
        planes, preserving oblique cuts. Proportional weights blend each target
        with its original position before conversion back to object-local space.
        """
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        faces = [bm.faces[index] for index in sorted(self._locked_indices)]
        verts = {vert for face in faces for vert in face.verts}

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
            return

        outer_plane = (
            (self._end_planes or ())[depth_fit["outer_plane_index"]]
            if depth_fit
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

            # Follow the target cone generator to the translated end plane. For
            # cylinders the slope is zero, reducing exactly to an axial slide.
            plane = (
                plane_by_vertex.get(vertex_index)
                if vertex_index in selected_indices
                else None
            )
            if plane is not None:
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

        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
