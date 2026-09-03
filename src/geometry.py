# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cylinder fitting and mesh deformation for Resize Hole."""

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


class GeometryMixin:
    """Provide selection, fitting, and deformation algorithms to the operator.

    Cylinder fits and boundary planes are expressed in world space. Original
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
        return result

    def _fit_cylinder(self, faces):
        """Fit a world-space cylinder to wall faces and return its parameters.

        SVD of the area-weighted face normals finds the cylinder axis. Vertices
        are then projected onto the perpendicular plane for a least-squares
        circle fit. The configured minimum, average, or maximum radial distance
        supplies the working radius. Degenerate input raises ``ValueError``.

        The result contains the axis, radial basis, center, radius, and axial
        bounds used by both the mesh deformation and GPU preview.
        """
        matrix_world = self._object.matrix_world
        verts = {vert for face in faces for vert in face.verts}
        if len(verts) < 6 or len(faces) < 3:
            raise ValueError("The flood fill needs at least three faces and six vertices")

        points = np.asarray([matrix_world @ vert.co for vert in verts], dtype=np.float64)

        # Cylinder-wall normals span the radial plane. The right singular vector
        # with the smallest singular value is therefore the cylinder axis.
        weighted_normals = []
        for face in faces:
            normal = np.asarray(self._world_normal(face), dtype=np.float64)
            weighted_normals.append(normal * math.sqrt(max(face.calc_area(), 1.0e-12)))
        _u, _singular_values, vh = np.linalg.svd(
            np.asarray(weighted_normals), full_matrices=False
        )
        axis = Vector(vh[-1]).normalized()

        reference = Vector((0.0, 0.0, 1.0))
        if abs(axis.dot(reference)) > 0.9:
            reference = Vector((0.0, 1.0, 0.0))
        basis_x = axis.cross(reference).normalized()
        basis_y = axis.cross(basis_x).normalized()

        mean = Vector(points.mean(axis=0))
        relative = [Vector(point) - mean for point in points]
        x = np.asarray([point.dot(basis_x) for point in relative])
        y = np.asarray([point.dot(basis_y) for point in relative])
        z = np.asarray([point.dot(axis) for point in relative])

        # Algebraic 2D circle fit. NumPy's lstsq uses the same compiled SVD/LAPACK
        # path, avoiding a Python implementation of the matrix decomposition.
        system = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        rhs = x * x + y * y
        solution, _residuals, rank, _values = np.linalg.lstsq(system, rhs, rcond=None)
        if rank < 3:
            raise ValueError("The selected wall does not define a stable cylinder")

        center_x, center_y, constant = solution
        fitted_radius_squared = constant + center_x * center_x + center_y * center_y
        if fitted_radius_squared <= 1.0e-12:
            raise ValueError("The fitted cylinder radius is zero")

        center = mean + basis_x * center_x + basis_y * center_y
        radial_distances = np.hypot(x - center_x, y - center_y)
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
            raise ValueError("The fitted cylinder radius is zero")

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
            "z_min": float(z.min()),
            "z_max": float(z.max()),
            "is_stud": wall_orientation >= 0.0,
        }

    def _fit_boundary_planes(self, faces, cylinder_axis):
        """Fit world-space end planes to connected wall-boundary components.

        Boundary edges have exactly one face in the supplied region. Their
        connected vertex components are fitted independently using SVD. A
        component is ignored when its plane cannot be intersected reliably by
        lines parallel to ``cylinder_axis``, as with a longitudinal open edge.

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

        matrix_world = self._object.matrix_world
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

        # Both end faces of a manifold through-hole point away from the selected
        # cylindrical wall. A blind hole instead has one inner face whose normal
        # points back into that region. Studs reverse the outer-loop test and are
        # intentionally excluded from through-hole detection.
        if not is_stud and all(
            score is not None and score <= 1.0e-5 for score in into_scores
        ):
            return None

        scored_indices = [
            index for index, score in enumerate(outer_scores) if score is not None
        ]
        if len(scored_indices) == 2:
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

    def _target_scale(self):
        """Convert the requested diameter into a radial deformation scale.

        The legacy ``scale`` property remains a fallback for redo compatibility
        when no positive diameter has been assigned.
        """
        fitted_diameter = self._fit["radius"] * 2.0
        if fitted_diameter <= 1.0e-12:
            return 1.0
        if self.diameter <= 0.0:
            return self.scale
        return self.diameter / fitted_diameter

    def _apply_diameter_to_mesh(self, context):
        """Rebuild affected vertices from the captured source geometry.

        Each world-space offset is split into axial and radial components. The
        radial component follows the diameter; when depth adjustment is valid,
        the axial coordinate is scaled away from the fixed outer/model-side plane.
        Selected boundary vertices are subsequently moved parallel to the axis
        onto the corresponding translated plane so matching oblique cuts keep
        their angle. Proportional weights blend each target with its original
        position before conversion back to object-local coordinates.
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

        # Confirming untouched dimensions must be an exact geometry no-op. In
        # particular, do not flatten within-epsilon boundary variation onto the
        # best-fit planes. This path also restores the verbatim source geometry
        # when edited dimensions are returned to their fitted values.
        if math.isclose(
            target_scale, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-9
        ) and math.isclose(
            depth_scale, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-9
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
            world_target = (
                fit["center"]
                + fit["axis"] * target_axial_coordinate
                + radial * target_scale
            )

            # Radial scaling changes where an oblique end plane intersects the
            # cylinder. Slide boundary vertices parallel to the cylinder axis back
            # onto that fitted surface to preserve the original cut angle.
            plane = (
                plane_by_vertex.get(vertex_index)
                if vertex_index in selected_indices
                else None
            )
            if plane is not None:
                denominator = plane["normal"].dot(fit["axis"])
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
                    world_target += fit["axis"] * distance
            blended_world = world.lerp(world_target, weight)
            vert.co = inverse @ blended_world

        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
