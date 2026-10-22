# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""World-space circular-cone, end-plane fitting, and wall selection."""

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


def _fit_wall_normal_plane(normals, weights):
    """Fit the dominant normal plane without letting a stray rim face tilt it.

    Cylinder and cone normals both lie in one (possibly offset) plane. Only
    trim clearly separated outliers when at least 80% of BOTH faces and area
    agree. This is not permission to fit an arbitrary small patch of a sphere
    or discard half of a deliberately selected cylinder/chamfer combination.
    """
    keep = np.ones(len(normals), dtype=bool)

    def fit_axis(mask):
        mean = np.average(normals[mask], axis=0, weights=weights[mask])
        centered = (normals[mask] - mean) * np.sqrt(weights[mask])[:, None]
        _u, _singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        return vh[-1]

    axis = fit_axis(keep)
    minimum_count = max(6, math.ceil(len(normals) * .8))
    minimum_area = float(weights.sum()) * .8
    noise_floor = math.sin(_CONE_ANGLE_EPSILON)
    for _ in range(3):
        projections = normals @ axis
        offsets = np.abs(projections - np.median(projections[keep]))
        limit = max(noise_floor, 4.5 * float(np.median(offsets[keep])))
        candidate = keep & (offsets <= limit)
        rejected = keep & ~candidate
        if not rejected.any():
            break
        if (np.count_nonzero(candidate) < minimum_count
                or float(weights[candidate].sum()) < minimum_area):
            break
        noise = math.sqrt(float(np.average(offsets[candidate] ** 2, weights=weights[candidate])))
        if float(offsets[rejected].min()) <= 3.0 * max(noise, noise_floor):
            break
        keep = candidate
        axis = fit_axis(keep)
    return axis, keep


class _HoverRegion:
    """One connected seed component and its independently fitted extensions."""

    def __init__(self, faces):
        self.faces = frozenset(faces)
        self.triangles = None


class _HoverCache:
    """Read-only selection-session data; never reused for deformation or redo."""

    def __init__(self, bm, matrix_world):
        self.bm = bm
        self.matrix_world = matrix_world.copy()
        self.counts = (len(bm.verts), len(bm.edges), len(bm.faces))
        self.normal_matrix = matrix_world.to_3x3().inverted_safe().transposed()
        self.normals = {}
        self.adjacency = {}
        self.angle = None
        self.estimator = None
        self.regions = {}
        self.region_count = 0
        self.spatial_index = None
        self.centers = []
        self.visible_faces = tuple(face for face in bm.faces if not face.hide)

    def matches(self, bm, matrix_world):
        return (
            self.bm is bm and self.matrix_world == matrix_world
            and self.counts == (len(bm.verts), len(bm.edges), len(bm.faces))
        )

    def normal(self, face):
        normal = self.normals.get(face)
        if normal is None:
            normal = (self.normal_matrix @ face.normal).normalized()
            self.normals[face] = normal
        return normal

    def neighbors(self, face):
        neighbors = self.adjacency.get(face)
        if neighbors is None:
            normal = self.normal(face)
            neighbors = tuple(
                (other, normal.dot(self.normal(other)))
                for edge in face.edges for other in edge.link_faces
                if other is not face and not other.hide
            )
            self.adjacency[face] = neighbors
        return neighbors

    def nearby_faces(self, center, radius):
        if self.spatial_index is None:
            self.bm.faces.ensure_lookup_table()
            tree = KDTree(len(self.visible_faces))
            self.centers = [self.matrix_world @ face.calc_center_median() for face in self.visible_faces]
            for index, point in enumerate(self.centers):
                tree.insert(point, index)
            tree.balance()
            self.spatial_index = tree
        return (
            (self.visible_faces[index], self.centers[index])
            for _point, index, _distance in self.spatial_index.find_range(center, radius)
        )

    def remember(self, connected_faces, region):
        # Bound retained fitted regions/preview geometry while keeping the
        # reusable O(mesh) normal/adjacency/spatial data for this invocation.
        if self.region_count >= 32:
            self.regions.clear()
            self.region_count = 0
        for face in connected_faces:
            self.regions[face] = region
        self.region_count += 1


class FittingMixin:
    """Fit circular cones and their end planes without modifying the mesh."""

    def _world_normal(self, face):
        """Return a BMesh face normal transformed correctly into world space."""
        normal_matrix = self._object.matrix_world.to_3x3().inverted_safe().transposed()
        return (normal_matrix @ face.normal).normalized()

    def _get_hover_cache(self):
        """Cache only while SELECT owns a frozen, read-only edit mesh.

        Modal selection blocks geometry/visibility edits. A new invocation or
        leaving SELECT clears the cache; mesh/count/transform changes are also
        checked defensively. Non-modal/headless fitting never retains it.
        """
        if getattr(self, "_stage", None) != 'SELECT':
            return None
        bm = bmesh.from_edit_mesh(self._object.data)
        matrix_world = self._object.matrix_world
        cache = getattr(self, "_hover_cache", None)
        if cache is None or not cache.matches(bm, matrix_world):
            if cache is not None:
                self._bvh = None
                self._bvh_faces = ()
                self._hover_face_index = -1
            cache = _HoverCache(bm, matrix_world)
            self._hover_cache = cache
            self._hover_region = None
        estimator = getattr(self, "diameter_estimator", 'AVERAGE')
        if cache.angle != self.flood_angle or cache.estimator != estimator:
            cache.angle = self.flood_angle
            cache.estimator = estimator
            cache.regions.clear()
            cache.region_count = 0
            self._hover_face_index = -1
        return cache

    def _flood_fill(self, seed):
        """Collect the smooth, edge-connected face region containing ``seed``.

        A visible neighboring face is admitted when its world-space normal is
        within ``self.flood_angle`` of the current face normal. Comparing each
        pair of adjacent faces allows the fill to follow a curved cylinder. The
        method returns faces without changing Blender's actual mesh selection.
        """
        threshold = math.cos(self.flood_angle)
        if seed.hide:
            return set()
        cache = FittingMixin._get_hover_cache(self)
        if cache is not None and (region := cache.regions.get(seed)) is not None:
            self._hover_region = region
            return region.faces
        result = {seed}
        stack = [seed]
        normal_cache = {}
        normal_matrix = self._object.matrix_world.to_3x3().inverted_safe().transposed()

        def world_normal(face):
            normal = normal_cache.get(face)
            if normal is None:
                normal = (normal_matrix @ face.normal).normalized()
                normal_cache[face] = normal
            return normal

        while stack:
            face = stack.pop()
            neighbors = (
                cache.neighbors(face) if cache is not None else (
                    (neighbor, world_normal(face).dot(world_normal(neighbor)))
                    for edge in face.edges for neighbor in edge.link_faces
                    if neighbor is not face and not neighbor.hide
                )
            )
            for neighbor, dot in neighbors:
                if neighbor not in result and not neighbor.hide and dot >= threshold:
                    result.add(neighbor)
                    stack.append(neighbor)
        extended = FittingMixin._extend_flood_across_gaps(self, result)
        if cache is not None:
            region = _HoverRegion(extended)
            cache.remember(result, region)
            self._hover_region = region
            return region.faces
        return extended

    def _extend_flood_across_gaps(self, connected_faces):
        """Add disconnected wall remnants compatible with a provisional fit.

        A missing face breaks edge traversal even though the surfaces on either
        side still describe the same hole. Once the connected patch can define a
        cone, test other visible faces against its axial span, radial model, and
        expected wall normal. The deliberately tight tolerances bridge local
        defects without collecting end caps or remote coaxial features. Match
        the selected wall's inward/outward orientation, not just its tangent
        plane: an opposite-facing surface does not belong to this bore/stud.
        """
        result = set(connected_faces)
        if len(result) < 3:
            return result
        try:
            fit = FittingMixin._fit_cylinder(self, list(result))
        except (ValueError, np.linalg.LinAlgError):
            return result

        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        matrix_world = self._object.matrix_world
        axis = fit["axis"]
        center = fit["center"]
        slope = fit.get("slope", 0.0)
        orientation = 1.0 if fit['is_stud'] else -1.0
        axial_extent = max(fit["z_max"] - fit["z_min"], 1.0e-9)
        axial_tolerance = max(axial_extent * 0.08, fit["radius"] * 0.08)
        radial_tolerance = max(
            fit["radius"] * 0.08,
            fit.get("fit_relative_error", 0.0) * fit["radius"] * 3.0,
            1.0e-7,
        )
        normal_threshold = math.cos(math.radians(25.0))
        cache = FittingMixin._get_hover_cache(self)
        normal_matrix = matrix_world.to_3x3().inverted_safe().transposed()
        if cache is not None:
            low = fit["z_min"] - axial_tolerance
            high = fit["z_max"] + axial_tolerance
            radius = max(fit["radius"] + slope * low, fit["radius"] + slope * high) + radial_tolerance
            radius = math.hypot((high - low) * .5, max(0.0, radius))
            # KDTree uses native float coordinates. Widen the conservative
            # enclosing sphere slightly; the original exact tests remain below.
            query_center = center + axis * ((low + high) * .5)
            radius += max(radius * 1.0e-5, max(abs(value) for value in query_center) * 2.0e-7, 1.0e-6)
            candidates = cache.nearby_faces(query_center, radius)
        else:
            candidates = ((face, None) for face in bm.faces)

        for face, point in candidates:
            if face in result or face.hide:
                continue
            if point is None:
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
            expected_normal = (radial.normalized() - axis * slope).normalized() * orientation
            normal = cache.normal(face) if cache is not None else (normal_matrix @ face.normal).normalized()
            if normal.dot(expected_normal) < normal_threshold:
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
        A bounded minority of isolated normal-plane outliers is excluded before
        both axis and radius fitting, without altering the selected geometry.

        The historical method name is retained because it is part of the
        operator's internal redo/test surface.
        """
        matrix_world = self._object.matrix_world
        faces = [face for face in faces if face.is_valid and not face.hide]
        verts = {vert for face in faces for vert in face.verts}
        if len(verts) < 6 or len(faces) < 3:
            raise ValueError("The flood fill needs at least three faces and six vertices")

        face_points = []
        normals = []
        weights = []
        normal_matrix = matrix_world.to_3x3().inverted_safe().transposed()
        for face in faces:
            face_points.append(
                np.asarray(matrix_world @ face.calc_center_median(), dtype=np.float64)
            )
            normals.append(np.asarray((normal_matrix @ face.normal).normalized(), dtype=np.float64))
            weights.append(max(face.calc_area(), 1.0e-12))
        face_points = np.asarray(face_points)
        normals = np.asarray(normals)
        weights = np.asarray(weights)

        # Circular-cone normals terminate in a plane. Centering them before SVD
        # makes its normal the cone axis, including for a partial angular patch.
        axis_values, inliers = _fit_wall_normal_plane(normals, weights)
        outlier_count = int(np.count_nonzero(~inliers))
        if outlier_count:
            faces = [face for face, keep in zip(faces, inliers) if keep]
            face_points, normals, weights = face_points[inliers], normals[inliers], weights[inliers]
            verts = {vert for face in faces for vert in face.verts}
        points = np.asarray([matrix_world @ vert.co for vert in verts], dtype=np.float64)
        normal_mean = np.average(normals, axis=0, weights=weights)
        axis = Vector(axis_values).normalized()
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
            # Flat cylindrical walls can acquire uneven vertex samples at the
            # two ends. Their radius regression may look tapered even though
            # all wall normals remain perpendicular to the fitted axis.
            and abs(float(normal_mean @ np.asarray(axis))) > 1.0e-6
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
        # Face-plane clearance ignores collinear edge subdivisions introduced
        # by Boolean intersections. Vertex distances alone underestimate a
        # polygon's circumradius when some samples lie midway along its flats.
        perpendicular = normals - np.outer(normals @ np.asarray(axis), np.asarray(axis))
        perpendicular_lengths = np.linalg.norm(perpendicular, axis=1)
        usable_planes = perpendicular_lengths > 1.0e-8
        plane_distances = np.abs(np.einsum(
            'ij,ij->i', normals[usable_planes],
            face_points[usable_planes] - np.asarray(center),
        )) / perpendicular_lengths[usable_planes]
        inradius_estimates = (
            {'MINIMUM': float(plane_distances.min()),
             'AVERAGE': float(np.average(plane_distances, weights=weights[usable_planes])),
             'MAXIMUM': float(plane_distances.max())}
            if not is_cone and len(plane_distances) else None
        )
        return {
            "axis": axis,
            "basis_x": basis_x,
            "basis_y": basis_y,
            "center": center,
            "radius": radius,
            "radius_estimates": radius_estimates,
            "inradius_estimates": inradius_estimates,
            "slope": slope,
            "half_angle": math.atan(abs(slope)),
            "is_cone": is_cone,
            "fit_relative_error": relative_error,
            "fit_rank": int(axis_rank),
            "fit_outlier_count": outlier_count,
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
        faces = [face for face in faces if not face.hide]
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
        """Return fixed-side axial depth metadata for two end loops.

        Cylindrical depth is measured between each loop plane's intersection
        with the fitted axis, so non-parallel ends remain adjustable. Conic
        depth uses each loop's mean axial coordinate and tolerates a rough
        Boolean seam. The outer/model-side loop is selected from its adjacent
        face normal: for a hole that normal points away from the cylindrical
        region, while a stud reverses the test. Missing or ambiguous adjacent
        faces fall back to stable boundary-index ordering. Two outward-facing
        cylindrical ends identify a through-hole, which has no adjustable
        depth.
        """
        if len(planes) != 2:
            return None
        is_cone = bool(fit.get("is_cone", False))
        if not is_cone and any(
            plane.get("max_error", math.inf) > plane.get("tolerance", 0.0)
            for plane in planes
        ):
            return None

        axis = fit["axis"]
        center = fit["center"]
        coordinates = []
        for plane in planes:
            if is_cone:
                # A Boolean seam at a cone-to-bore transition can be slightly
                # corrugated or triangulated even though the fitted conic is
                # excellent. Conic truncation is an axial coordinate, so use
                # the boundary component's mean point directly instead of
                # rejecting the control because its fitted plane is imperfect.
                coordinates.append((plane["point"] - center).dot(axis))
                continue
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
        if is_cone:
            # A countersink's model-side/top end is its wider circle. Adjacent
            # normals at the small end may belong to the continuing through
            # bore and must not reverse this orientation; Exact Boolean relies
            # on ``outer_coordinate`` to extend the large cutter end outside
            # the reconstructed surface.
            outer_index = max(
                range(2),
                key=lambda index: (
                    fit["radius"]
                    + fit.get("slope", 0.0) * coordinates[index]
                ),
            )
        elif len(scored_indices) == 2 and not math.isclose(
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
