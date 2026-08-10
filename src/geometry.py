# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cylinder fitting and mesh deformation for Resize Hole."""

import heapq
import math

import bmesh
import numpy as np

from mathutils import Vector
from mathutils.kdtree import KDTree


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
                view_matrix = context.region_data.view_matrix if projected else None
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
        estimator = getattr(self, "diameter_estimator", 'AVERAGE')
        if estimator == 'MINIMUM':
            radius = float(radial_distances.min())
        elif estimator == 'MAXIMUM':
            radius = float(radial_distances.max())
        else:
            radius = float(radial_distances.mean())
        if radius <= 1.0e-12:
            raise ValueError("The fitted cylinder radius is zero")
        return {
            "axis": axis,
            "basis_x": basis_x,
            "basis_y": basis_y,
            "center": center,
            "radius": radius,
            "z_min": float(z.min()),
            "z_max": float(z.max()),
        }

    def _fit_boundary_planes(self, faces, cylinder_axis):
        """Fit world-space end planes to connected wall-boundary components.

        Boundary edges have exactly one face in the supplied region. Their
        connected vertex components are fitted independently using SVD. A
        component is ignored when its plane cannot be intersected reliably by
        lines parallel to ``cylinder_axis``, as with a longitudinal open edge.

        Each result records the component's vertex indices, a point on its
        plane, and its normal for preserving skewed hole ends during resizing.
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
            components.append(component_verts)

        matrix_world = self._object.matrix_world
        planes = []
        for verts in components:
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

            # A valid cylinder end surface must intersect lines parallel to the
            # fitted axis. Skip longitudinal/open-patch boundaries.
            if abs(normal.dot(cylinder_axis)) < 1.0e-5:
                continue
            planes.append(
                {
                    "verts": sorted(vert.index for vert in verts),
                    "point": Vector(center),
                    "normal": normal,
                }
            )
        return planes

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

        Each world-space offset is split into axial and radial components, and
        only the radial component is scaled. Selected boundary vertices are
        subsequently moved parallel to the cylinder axis onto their fitted end
        plane so an oblique cut keeps its angle. Proportional weights blend each
        target with its original position before conversion back to object-local
        coordinates and the edit-mesh update.
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
            axial = fit["axis"] * offset.dot(fit["axis"])
            radial = offset - axial
            world_target = fit["center"] + axial + radial * target_scale

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
                    distance = plane["normal"].dot(
                        plane["point"] - world_target
                    ) / denominator
                    world_target += fit["axis"] * distance
            blended_world = world.lerp(world_target, weight)
            vert.co = inverse @ blended_world

        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
