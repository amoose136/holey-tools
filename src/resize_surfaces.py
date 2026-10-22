# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Support-surface classification, axis projection, and resize-mode policy."""

import math

import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .boolean_bore import BooleanBoreMixin
from .fairing import FairingError, choose_hypothesis
from .fairing_pipeline import fair_hole_patch
from .fitting import FittingMixin
from .surface_projection import project_hypothesis_along_axis
from .visible_geometry import visible_bmesh_copy


class ResizeSurfaceMixin(FittingMixin, BooleanBoreMixin):
    """Fit immutable support surfaces and choose their deformation strategy."""

    @staticmethod
    def _surface_bvh(faces, matrix_world):
        """Build a world-space BVH and useful ray span from BMesh faces."""
        coordinates = []
        polygons = []
        for face in faces:
            if face.hide:
                continue
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
        selected = {face for face in selected_faces if not face.hide}
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
                        - ResizeSurfaceMixin._fitted_radius_at(self, coordinate)
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
                    - ResizeSurfaceMixin._fitted_radius_at(self, coordinate)
                ) > tolerance
            tested += 1
        # One genuine outward continuation is sufficient. Boolean-generated
        # cube faces can leave most rim edges supported by triangles containing
        # only boundary vertices, while just the corner-adjacent faces retain
        # vertices beyond the fitted radius. A blind cap has no such vertex.
        return tested > 0 and outward > 0

    def _is_cylindrical_bore_continuation(self, surface):
        """Recognize a coaxial cylinder attached to the cone's smaller end."""
        fit = self._fit
        if not fit.get("is_cone", False):
            return False
        hypothesis = surface.get("hypothesis", {})
        continuation_fit = (
            hypothesis.get("fit", {})
            if hypothesis.get("method") == "CONIC"
            else None
        )
        if continuation_fit is None:
            # Surface selection answers which model best reconstructs an open
            # patch, not whether the faces immediately beyond a countersink
            # form its coaxial through-bore. A short, finely tessellated bore
            # can marginally prefer Sphere despite near-zero axial normals.
            # Refit that local support explicitly for this junction test.
            support_faces = [
                face
                for face in surface.get(
                    "support_patch_faces", surface.get("support_faces", ())
                )
                if face.is_valid
            ]
            if not support_faces:
                return False
            try:
                continuation_fit = ResizeSurfaceMixin._fit_cylinder(
                    self, support_faces
                )
            except (ValueError, np.linalg.LinAlgError):
                return False
        continuation_axis = continuation_fit.get("axis")
        if continuation_axis is None or abs(
            Vector(continuation_axis).normalized().dot(fit["axis"].normalized())
        ) < math.cos(math.radians(2.0)):
            return False
        if abs(float(continuation_fit.get("slope", 0.0))) > math.tan(
            math.radians(0.4)
        ):
            return False
        if float(continuation_fit.get("fit_relative_error", math.inf)) > 0.01:
            return False
        normal_limit = math.sin(math.radians(0.4))
        if any(
            abs(ResizeSurfaceMixin._world_normal(self, face).dot(fit["axis"]))
            > normal_limit
            for face in surface.get("support_faces", ())
            if face.is_valid
        ):
            return False
        points = surface.get("loop_world_points", ())
        if not points:
            return False
        axis = fit["axis"].normalized()
        coordinate = sum(
            (point - fit["center"]).dot(axis) for point in points
        ) / len(points)
        endpoints = (float(fit["z_min"]), float(fit["z_max"]))
        smaller_coordinate = min(
            endpoints,
            key=lambda endpoint: ResizeSurfaceMixin._fitted_radius_at(
                self, endpoint
            ),
        )
        accepted = abs(coordinate - smaller_coordinate) <= max(
            abs(endpoints[1] - endpoints[0]) * 0.05,
            1.0e-6,
        )
        if accepted:
            surface["bore_fit"] = continuation_fit
            continuation_center = Vector(continuation_fit["center"])
            continuation_axis = Vector(
                continuation_fit["axis"]
            ).normalized()
            # Polygon face centers and Boolean edge splits lie inside the
            # actual faceted bore radius. Preserve its surviving corner radius,
            # not the smaller least-squares/mean chord radius, so the clearance
            # cutter fully contains the old coplanar wall.
            support_points = {
                vert
                for face in surface.get("support_faces", ())
                if face.is_valid
                for vert in face.verts
                if vert.is_valid
            }
            radial_lengths = []
            for vert in support_points:
                offset = self._object.matrix_world @ vert.co - continuation_center
                coordinate = offset.dot(continuation_axis)
                radial_lengths.append(
                    (offset - continuation_axis * coordinate).length
                )
            surface["bore_radius"] = max(
                radial_lengths or (float(continuation_fit["radius"]),)
            )
            # Scale temporary Boolean clearance to the cylinder fit noise.
            # Perfect analytic bores need only a microscopic offset, while a
            # prior Boolean may leave enough radial variation to require the
            # full half-percent safety envelope.
            relative_error = max(
                0.0,
                float(continuation_fit.get("fit_relative_error", 0.0)),
            )
            surface["bore_clearance_ratio"] = max(
                1.0e-4, min(5.0e-3, relative_error * 2.0)
            )
        return accepted

    def _cylindrical_bore_vertices(self, surfaces, selected_faces):
        """Flood the complete coaxial bore attached to a conic truncation."""
        selected = set(selected_faces)
        matrix_world = self._object.matrix_world
        result = set()
        for surface in surfaces:
            if not surface.get("bore_continuation", False):
                continue
            continuation_fit = surface.get(
                "bore_fit", surface["hypothesis"].get("fit", {})
            )
            axis = Vector(continuation_fit["axis"]).normalized()
            center = Vector(continuation_fit["center"])
            radius = float(continuation_fit["radius"])
            tolerance = max(abs(radius) * 0.02, 1.0e-6)
            normal_limit = math.sin(math.radians(2.0))

            def compatible(face):
                if face in selected or face.hide:
                    return False
                normal = ResizeSurfaceMixin._world_normal(self, face)
                if abs(normal.dot(axis)) > normal_limit:
                    return False
                for vert in face.verts:
                    point = matrix_world @ vert.co
                    offset = point - center
                    coordinate = offset.dot(axis)
                    radial = offset - axis * coordinate
                    expected = radius + float(
                        continuation_fit.get("slope", 0.0)
                    ) * coordinate
                    if abs(radial.length - expected) > tolerance:
                        return False
                return True

            stack = [
                face
                for face in surface.get("support_faces", ())
                if face.is_valid and compatible(face)
            ]
            visited = set()
            while stack:
                face = stack.pop()
                if face in visited or not compatible(face):
                    continue
                visited.add(face)
                result.update(vert.index for vert in face.verts)
                for edge in face.edges:
                    stack.extend(
                        neighbor
                        for neighbor in edge.link_faces
                        if neighbor not in visited
                    )
        return result

    def _fit_conic_continuation(self, surface, selected_faces):
        """Recognize the countersink at the small end of a cylindrical bore."""
        fit = self._fit
        if fit.get("is_cone", False) or not surface.get("closed", False):
            return False
        selected = set(selected_faces)
        samples = [
            face for face in surface["support_patch_faces"]
            if face.is_valid and face not in selected
        ]
        try:
            cone = ResizeSurfaceMixin._fit_cylinder(self, samples)
        except (ValueError, np.linalg.LinAlgError):
            return False
        if (
            not cone.get("is_cone", False)
            or float(cone.get("fit_relative_error", math.inf)) > 0.01
        ):
            return False
        axis = fit["axis"].normalized()
        cone_axis = cone["axis"].normalized()
        alignment = cone_axis.dot(axis)
        if abs(alignment) < math.cos(math.radians(2.0)):
            return False
        offset = cone["center"] - fit["center"]
        tolerance = max(float(fit["radius"]) * 0.02, 1.0e-6)
        if (offset - axis * offset.dot(axis)).length > tolerance:
            return False
        points = surface["loop_world_points"]
        inner = sum(
            (point - fit["center"]).dot(axis) for point in points
        ) / len(points)
        cone_inner = sum(
            (point - cone["center"]).dot(cone_axis) for point in points
        ) / len(points)
        slope = float(cone["slope"])
        small_end = min(
            (cone["z_min"], cone["z_max"]),
            key=lambda z: float(cone["radius"]) + slope * z,
        )
        if abs(cone_inner - small_end) > max(
            abs(cone["z_max"] - cone["z_min"]) * 0.05, tolerance
        ):
            return False

        matrix_world = self._object.matrix_world
        axial_normal = abs(slope) / math.sqrt(1.0 + slope * slope)

        def compatible(face):
            if face in selected or face.hide:
                return False
            normal = ResizeSurfaceMixin._world_normal(self, face)
            if abs(abs(normal.dot(cone_axis)) - axial_normal) > math.sin(
                math.radians(2.0)
            ):
                return False
            for vert in face.verts:
                offset = matrix_world @ vert.co - cone["center"]
                z = offset.dot(cone_axis)
                radius = float(cone["radius"]) + slope * z
                if radius <= 0.0 or abs(
                    (offset - cone_axis * z).length - radius
                ) > tolerance:
                    return False
            return True

        stack = [face for face in surface["support_faces"] if compatible(face)]
        cone_faces = set()
        while stack:
            face = stack.pop()
            if face in cone_faces or not compatible(face):
                continue
            cone_faces.add(face)
            for edge in face.edges:
                stack.extend(
                    neighbor for neighbor in edge.link_faces
                    if neighbor not in cone_faces
                )
        if not cone_faces:
            return False
        seam = set(surface["loop_verts"])
        outer = {
            vert for face in cone_faces for edge in face.edges
            if any(
                neighbor not in cone_faces and neighbor not in selected
                for neighbor in edge.link_faces
            )
            for vert in edge.verts if vert not in seam
        }
        if not outer:
            return False
        outer_points = [matrix_world @ vert.co for vert in outer]
        outer_coordinate = sum(
            (point - fit["center"]).dot(axis) for point in outer_points
        ) / len(outer_points)
        if abs(outer_coordinate - inner) <= tolerance:
            return False
        planes = ResizeSurfaceMixin._fit_boundary_planes(
            self, list(cone_faces), cone_axis, cone
        )
        outer_plane = next(
            (
                plane for plane in planes
                if set(plane["verts"]) == {vert.index for vert in outer}
            ),
            None,
        )
        surface.update(
            conic_fit=cone,
            conic_slope=slope * alignment,
            conic_inner_coordinate=inner,
            conic_outer_coordinate=outer_coordinate,
            conic_outer_plane=outer_plane,
            conic_outer_vertices={vert.index for vert in outer},
            conic_vertices={vert.index for face in cone_faces for vert in face.verts},
        )
        # Use the independently verified cone, even if the generic fair-surface
        # competition prefers a sphere or a plane on a narrow local band.
        surface["hypothesis"] = {"method": "CONIC", "fit": cone}
        return True

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
            else ResizeSurfaceMixin._estimate_angular_segments(self, selected)
        )
        base_faces = [
            face for face in bm.faces
            if face not in selected and not face.hide
        ]
        base_bvh, base_span = ResizeSurfaceMixin._surface_bvh(
            base_faces, matrix_world
        )
        self._resize_base_bvh = base_bvh
        self._resize_base_span = base_span
        self._resize_surfaces_by_vertex = {}
        self._resize_conic_surfaces_by_vertex = {}
        self._conic_bore_continuation = False
        self._conic_bore_radius = 0.0
        self._conic_bore_clearance_ratio = 1.0e-4

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
            ordered = ResizeSurfaceMixin._ordered_supported_loop(
                component, support_by_edge
            )
            if ordered is None:
                continue
            loop_verts, loop_edges, support_faces, closed = ordered
            support_patch_faces = ResizeSurfaceMixin._support_patch_faces(
                self, support_faces, selected, loop_edges
            )
            try:
                hypothesis = choose_hypothesis(
                    loop_verts,
                    loop_edges,
                    matrix_world,
                    lambda faces: ResizeSurfaceMixin._fit_cylinder(self, faces),
                    lambda face: ResizeSurfaceMixin._world_normal(self, face),
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
            surface["bore_continuation"] = (
                ResizeSurfaceMixin._is_cylindrical_bore_continuation(
                    self, surface
                )
            )
            surface["conic_continuation"] = ResizeSurfaceMixin._fit_conic_continuation(
                self, surface, selected
            )
            surface["exterior"] = (
                not surface["bore_continuation"]
                and (
                    surface["conic_continuation"]
                    or ResizeSurfaceMixin._support_extends_outward(
                        self, loop_verts, surface["support_faces"]
                    )
                )
            )
            if surface["conic_continuation"]:
                for index in surface["conic_vertices"]:
                    self._resize_conic_surfaces_by_vertex[index] = surface
            if hypothesis.get("method") == "ORGANIC" and closed:
                organic_bvh, organic_span = (
                    ResizeSurfaceMixin._build_virtual_organic_surface(
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
        }
        self._conic_bore_continuation = any(
            surface.get("bore_continuation", False)
            for surface in unique_surfaces.values()
        )
        bore_radii = [
            float(surface["bore_radius"])
            for surface in unique_surfaces.values()
            if surface.get("bore_continuation", False)
            and float(surface.get("bore_radius", 0.0)) > 1.0e-12
        ]
        if bore_radii:
            self._conic_bore_radius = sum(bore_radii) / len(bore_radii)
        bore_clearance_ratios = [
            float(surface["bore_clearance_ratio"])
            for surface in unique_surfaces.values()
            if surface.get("bore_continuation", False)
            and float(surface.get("bore_clearance_ratio", 0.0)) > 0.0
        ]
        if bore_clearance_ratios:
            self._conic_bore_clearance_ratio = max(bore_clearance_ratios)
        self._resize_bore_vertex_indices = (
            ResizeSurfaceMixin._cylindrical_bore_vertices(
                self, unique_surfaces.values(), selected
            )
        )
        # A cone may terminate on an exterior face rather than an existing
        # bore. Retain that original exit independently of the movable small
        # circle, including its axial extent on sloped/curved surfaces.
        self._conic_exit_coordinates = None
        self._conic_exit_vertex_indices = set()
        if self._fit.get("is_cone", False):
            endpoints = (
                [(plane["point"] - self._fit["center"]).dot(self._fit["axis"])
                 for plane in (self._end_planes or ())]
                if len(self._end_planes or ()) == 2
                else (self._fit["z_min"], self._fit["z_max"])
            )
            smaller = min(
                endpoints,
                key=lambda z: ResizeSurfaceMixin._fitted_radius_at(self, z),
            )
            tolerance = max(
                abs(self._fit["z_max"] - self._fit["z_min"]) * .05,
                1.0e-6,
            )
            for surface in unique_surfaces.values():
                points = surface["loop_world_points"]
                if not surface["exterior"] or not surface["closed"] or not points:
                    continue
                coordinates = [
                    (point - self._fit["center"]).dot(self._fit["axis"])
                    for point in points
                ]
                if abs(sum(coordinates) / len(coordinates) - smaller) <= tolerance:
                    self._conic_exit_coordinates = (
                        min(coordinates), max(coordinates)
                    )
                    self._conic_exit_vertex_indices = set(surface["verts"])
        self._resize_base_requires_boolean = (
            not bool(self._fit.get("is_stud", False))
            and any(
                surface["exterior"]
                and not surface.get("conic_continuation", False)
                and not (
                    surface["hypothesis"].get("method") == "PLANES"
                    and len(surface["hypothesis"].get("planes", ())) == 1
                )
                for surface in unique_surfaces.values()
            )
        )
        self._resize_dynamic_requires_boolean = False
        ResizeSurfaceMixin._resolve_resize_method(self)
        self._resize_projection_ready = True

    def _resolve_resize_method(self, check_transition=False):
        """Resolve the user's mode before using automatic or cached state."""
        method = getattr(self, "resize_method", 'AUTO')
        if method in {'SCALE', 'BOOLEAN'}:
            self._resize_dynamic_requires_boolean = False
            self._resize_requires_boolean = method == 'BOOLEAN'
            return self._resize_requires_boolean

        base_requires_boolean = getattr(
            self,
            "_resize_base_requires_boolean",
            getattr(self, "_resize_requires_boolean", False),
        )
        dynamic_requires_boolean = bool(
            getattr(self, "_resize_dynamic_requires_boolean", False)
        )
        if (
            check_transition
            and not base_requires_boolean
            and not dynamic_requires_boolean
            and not bool(self._fit.get("is_stud", False))
            and (
                ResizeSurfaceMixin._shortened_through_cone(self)
                or ResizeSurfaceMixin._conic_continuation_outgrown(self)
                or ResizeSurfaceMixin._enlargement_reaches_nonplanar_support(self)
            )
        ):
            dynamic_requires_boolean = True
        self._resize_dynamic_requires_boolean = dynamic_requires_boolean
        self._resize_requires_boolean = bool(
            base_requires_boolean or dynamic_requires_boolean
        )
        return self._resize_requires_boolean

    def _shortened_through_cone(self):
        """A shortened exposed cone needs a new cylindrical exit section."""
        return bool(
            getattr(self, "_conic_exit_coordinates", None) is not None
            and ResizeSurfaceMixin._uses_conic_dimensions(self)
            and ResizeSurfaceMixin._target_conic_truncated_depth(self)
            < self._depth_fit["initial"] - max(self._depth_fit["initial"] * 1.0e-6, 1.0e-8)
        )

    def _connect_scaled_conic_exit(self, bm, faces):
        """Insert a cylinder between a moved small circle and its fixed exit."""
        exit_indices = self._conic_exit_vertex_indices
        originals = self._original_vertex_coords
        matrix_world = self._object.matrix_world
        inverse = matrix_world.inverted_safe()
        fit = self._fit
        axis = fit["axis"].normalized()
        inner = ResizeSurfaceMixin._target_conic_inner_coordinate(self)
        replacements = {}
        source_edges = {edge for face in faces for edge in face.edges}
        source_verts = {vert for face in faces for vert in face.verts}
        exit_verts = [(index, bm.verts[index]) for index in exit_indices]
        for index, vert in exit_verts:
            offset = matrix_world @ originals[index] - fit["center"]
            radial = offset - axis * offset.dot(axis)
            target = (
                fit["center"] + axis * inner
                + radial.normalized()
                * ResizeSurfaceMixin._fixed_conic_bottom_radius(self)
            )
            replacement = bm.verts.new(inverse @ target)
            replacement.select_set(True)
            replacements[vert] = replacement
            vert.co = originals[index].copy()
        rim_pairs = []
        for face in list(faces):
            if not any(vert in replacements for vert in face.verts):
                continue
            for loop in face.loops:
                a, b = loop.vert, loop.link_loop_next.vert
                if a in replacements and b in replacements:
                    rim_pairs.append((a, b, face.material_index, face.smooth))
            new_face = bm.faces.new(
                [replacements.get(vert, vert) for vert in face.verts], face
            )
            new_face.select_set(True)
            bm.faces.remove(face)
        for a, b, material, smooth in rim_pairs:
            face = bm.faces.new((a, b, replacements[b], replacements[a]))
            face.material_index = material
            face.smooth = smooth
            face.select_set(True)
        for edge in source_edges:
            if edge.is_valid and not edge.link_faces:
                bm.edges.remove(edge)
        if ResizeSurfaceMixin._target_conic_truncated_depth(self) <= 1.0e-9:
            # A zero physical cone span is a straight through-cylinder, not
            # two coincident mouth rings joined by zero-area cone faces.
            bmesh.ops.remove_doubles(
                bm, verts=list(source_verts | set(replacements.values())),
                dist=max(
                    float(fit["radius"])
                    * inverse.to_3x3().median_scale * 1.0e-7,
                    1.0e-9,
                ),
            )
        bm.normal_update()

    def _conic_continuation_outgrown(self):
        """Require reconstruction if a resize would consume a whole cone/bore."""
        fit = self._fit
        for surface in {
            id(surface): surface
            for surface in getattr(self, "_resize_conic_surfaces_by_vertex", {}).values()
        }.values():
            inner = surface["conic_inner_coordinate"]
            outer = surface["conic_outer_coordinate"]
            radius_delta = (
                ResizeSurfaceMixin._target_radius_at(self, inner)
                - ResizeSurfaceMixin._fitted_radius_at(self, inner)
            )
            target_inner = inner + radius_delta / surface["conic_slope"]
            far = max((fit["z_min"], fit["z_max"]), key=lambda z: abs(z - inner))
            tolerance = max(float(fit["radius"]) * 1.0e-6, 1.0e-8)
            if (
                (target_inner - outer) * (inner - outer) <= tolerance * tolerance
                or (target_inner - far) * (inner - far) <= tolerance * tolerance
            ):
                return True
        return False

    def _conic_continuation_target(self, world, surface):
        """Redistribute a cone along its generators with its outer rim fixed."""
        fit = self._fit
        axis = fit["axis"].normalized()
        offset = world - fit["center"]
        z = offset.dot(axis)
        radial = offset - axis * z
        if radial.length <= 1.0e-12:
            return world
        slope = surface["conic_slope"]
        inner = surface["conic_inner_coordinate"]
        outer = surface["conic_outer_coordinate"]
        direction = axis + radial.normalized() * slope
        plane = surface.get("conic_outer_plane")
        if plane is not None:
            denominator = plane["normal"].dot(direction)
            if abs(denominator) > 1.0e-9:
                outer = z + plane["normal"].dot(plane["point"] - world) / denominator
        fraction = (z - outer) / (inner - outer)
        source_radius = ResizeSurfaceMixin._fitted_radius_at(self, inner)
        target_radius = ResizeSurfaceMixin._target_radius_at(self, inner)
        delta = (target_radius - source_radius) / slope
        return world + direction * (delta * fraction)

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
            normal = ResizeSurfaceMixin._world_normal(self, face)
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
                intersection = ResizeSurfaceMixin._project_bvh_along_axis(
                    surface.get("organic_bvh"),
                    surface.get("organic_span", 0.0),
                    axis_point,
                    reference,
                    axis,
                )
            else:
                intersection = ResizeSurfaceMixin._project_hypothesis_along_axis(
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
            if (
                surface["closed"]
                and not surface["exterior"]
                and not surface.get("bore_continuation", False)
            )
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
        copied = None
        try:
            if any(face.hide for face in bm.faces):
                copied, vertices, edges, faces = visible_bmesh_copy(bm)
                copied_loop_verts = [vertices[vert] for vert in loop_verts]
                copied_loop_edges = [edges[edge] for edge in loop_edges]
                copied_selected = [faces[face] for face in selected_faces if face in faces]
            else:
                copied = bm.copy()
                copied.verts.ensure_lookup_table()
                copied.edges.ensure_lookup_table()
                copied.faces.ensure_lookup_table()
                copied_loop_verts = [copied.verts[vert.index] for vert in loop_verts]
                copied_loop_edges = [copied.edges[edge.index] for edge in loop_edges]
                copied_selected = [copied.faces[face.index] for face in selected_faces]
            bmesh.ops.delete(
                copied, geom=copied_selected, context='FACES'
            )
            if any(
                not edge.is_valid or len(edge.link_faces) != 1
                for edge in copied_loop_edges
            ):
                return None, 0.0
            copied, result = fair_hole_patch(
                self._object, copied, copied_loop_verts, copied_loop_edges,
                {"method": "ORGANIC", "residual": 0.0},
                final_simplify=False,
                match_curvature=getattr(self, "grid_match_curvature", True),
            )
            return ResizeSurfaceMixin._surface_bvh(
                result["faces"], self._object.matrix_world
            )
        except (FairingError, IndexError, KeyError, ValueError, np.linalg.LinAlgError):
            return None, 0.0
        finally:
            if copied is not None:
                copied.free()

    @staticmethod
    def _project_bvh_along_axis(bvh, span, point, reference, axis):
        """Return the closest bidirectional axis-ray hit to ``reference``."""
        hit = ResizeSurfaceMixin._project_bvh_hit_along_axis(
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
        """Detect enlargement that needs local Boolean reconstruction."""
        if ResizeSurfaceMixin._target_scale(self) <= 1.0 + 1.0e-9:
            return False
        if (
            not ResizeSurfaceMixin._conic_continues_as_cylindrical_bore(self)
            and ResizeSurfaceMixin._enlargement_stretches_sparse_planar_support(self)
        ):
            return True
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
                source_radius = ResizeSurfaceMixin._fitted_radius_at(
                    self, coordinate
                )
                target_radius = ResizeSurfaceMixin._target_radius_at(
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
                hit = ResizeSurfaceMixin._project_bvh_hit_along_axis(
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

    def _enlargement_stretches_sparse_planar_support(self):
        """Reject direct scaling when a rim is attached to long planar fans.

        Moving the center vertices of a large triangulated face fan changes
        edges far away from the hole even though the represented surface stays
        planar. Once growth spans several source rim samples, reconstruct and
        cut the bore locally instead of stretching those long spokes.
        """
        fit = self._fit
        axis = fit["axis"].normalized()
        matrix_world = self._object.matrix_world
        for surface in {
            id(surface): surface
            for surface in getattr(
                self, "_resize_surfaces_by_vertex", {}
            ).values()
        }.values():
            hypothesis = surface.get("hypothesis", {})
            planes = hypothesis.get("planes", ())
            if (
                not surface.get("exterior")
                or not surface.get("closed")
                or hypothesis.get("method") != "PLANES"
                or len(planes) != 1
            ):
                continue
            loop_edges = [
                edge for edge in surface.get("loop_edges", ())
                if edge.is_valid
            ]
            points = surface.get("loop_world_points", ())
            if not loop_edges or not points:
                continue
            rim_lengths = sorted(
                (
                    matrix_world @ edge.verts[0].co
                    - matrix_world @ edge.verts[1].co
                ).length
                for edge in loop_edges
            )
            sample_length = rim_lengths[len(rim_lengths) // 2]
            if sample_length <= 1.0e-12:
                continue
            coordinates = [
                (point - fit["center"]).dot(axis) for point in points
            ]
            source_radius = sum(
                ((point - fit["center"]) - axis * coordinate).length
                for point, coordinate in zip(points, coordinates)
            ) / len(points)
            target_radius = ResizeSurfaceMixin._target_radius_at(
                self, sum(coordinates) / len(coordinates)
            )
            if target_radius - source_radius <= sample_length * 4.0:
                continue
            rim = set(surface.get("loop_verts", ()))
            has_long_spoke = any(
                (
                    matrix_world @ edge.verts[0].co
                    - matrix_world @ edge.verts[1].co
                ).length > sample_length * 4.0
                for face in surface.get("support_faces", ())
                if face.is_valid
                for edge in face.edges
                if (edge.verts[0] in rim) != (edge.verts[1] in rim)
            )
            if has_long_spoke:
                return True
        return False

    _project_hypothesis_along_axis = staticmethod(project_hypothesis_along_axis)
