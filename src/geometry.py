# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resize deformation, transactional reconstruction, and preview lifecycle."""

import hashlib
import heapq
import math
import time

import bmesh
import bpy

from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .constants import PARALLEL_PLANE_ANGLE_EPSILON
from .fairing import (
    FairingError,
    LIMITED_DISSOLVE_ANGLE,
    boundary_component,
    choose_hypothesis,
    expanded_limited_dissolve,
    ordered_boundary_loop,
)

from .fairing_pipeline import fair_hole_patch
from .mesh_state import restore_mesh_contents
from .boolean_transaction import BooleanTransaction
from .visible_geometry import detach_hidden_geometry, is_visible_vertex, update_visible_normals
from .resize_surfaces import ResizeSurfaceMixin
from .resize_state import invalidate_projection, reset_deformation


_BOOLEAN_PREVIEW_INTERVAL = 0.5


class GeometryMixin(ResizeSurfaceMixin):
    """Compose fitting, surface policy, bore handling, and mesh deformation.

    Cone fits and boundary planes are expressed in world space. A cylinder is
    represented as the zero-slope special case. Original BMesh vertex
    coordinates are retained in object-local space so every update
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
            ([vert.index for vert in face.verts] for face in bm.faces if not face.hide),
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
            (polygon.vertices for polygon in mesh.polygons if not polygon.hide),
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
        selected_verts = {vert for vert in selected_verts if is_visible_vertex(vert)}
        bm.verts.ensure_lookup_table()
        selected_verts.update(
            bm.verts[index]
            for index in getattr(self, "_resize_conic_surfaces_by_vertex", {})
            if index < len(bm.verts)
            and is_visible_vertex(bm.verts[index])
        )
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
                        if not is_visible_vertex(neighbor):
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
                    if vert in selected_verts or not is_visible_vertex(vert):
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


    def _boolean_cutter_geometry(
        self, obj, blind_coordinates=None, segment_override=None
    ):
        """Return local cutter vertices/faces for the requested hole dimensions."""
        self._boolean_bore_projection = None
        fit = self._fit
        axis = fit["axis"].normalized()
        segments = GeometryMixin._boolean_segment_count(self, segment_override)
        depth_fit = getattr(self, "_depth_fit", None)
        world_vertices = [obj.matrix_world @ vert.co for vert in obj.data.vertices]
        if not world_vertices:
            raise FairingError("The reconstructed base mesh is empty")
        axial_bounds = [
            (point - fit["center"]).dot(axis) for point in world_vertices
        ]
        bore_min = min(float(fit["z_min"]), float(fit["z_max"]))
        bore_max = max(float(fit["z_min"]), float(fit["z_max"]))
        span = max(bore_max - bore_min, fit["radius"], 1.0e-6)
        # Keep the cutter cap comfortably outside the reconstructed exterior.
        # Exact Boolean is much less reliable when the cap is coplanar with, or
        # only microscopically beyond, a large planar support face.
        outer_coordinate = (
            depth_fit["outer_coordinate"]
            if depth_fit
            else GeometryMixin._radius_coordinate(self)
        )
        target_outer_radius = max(
            GeometryMixin._boolean_radius_at(self, outer_coordinate, segments),
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
            coordinates = [outer - direction * margin, inner]
            exit_coordinates = getattr(self, "_conic_exit_coordinates", None)
            if exit_coordinates is not None and GeometryMixin._uses_conic_dimensions(self):
                # Only reach the original exit, not remote geometry elsewhere
                # on the same axis. Continue at the fixed truncation radius.
                continuation_end = (
                    max(exit_coordinates) + margin if direction > 0.0
                    else min(exit_coordinates) - margin
                )
                if (continuation_end - inner) * direction > 1.0e-9:
                    coordinates.append(continuation_end)
            elif GeometryMixin._conic_continues_as_cylindrical_bore(self):
                continuation_end = (
                    max(axial_bounds) + margin
                    if direction > 0.0
                    else min(axial_bounds) - margin
                )
                if (continuation_end - inner) * direction > 1.0e-9:
                    coordinates.append(continuation_end)
        elif blind_coordinates is not None:
            outer, inner = blind_coordinates
            direction = 1.0 if inner >= outer else -1.0
            coordinates = [outer - direction * margin, inner]
        else:
            mouth_coordinates = []
            for surface in {
                id(surface): surface
                for surface in getattr(
                    self, "_resize_surfaces_by_vertex", {}
                ).values()
            }.values():
                if not surface.get("exterior") or not surface.get("closed"):
                    continue
                hypothesis = surface.get("hypothesis", {})
                for point in surface.get("loop_world_points", ()):
                    offset = point - fit["center"]
                    coordinate = offset.dot(axis)
                    radial = offset - axis * coordinate
                    if radial.length_squared <= 1.0e-16:
                        continue
                    probe = (
                        fit["center"]
                        + axis * coordinate
                        + radial.normalized()
                        * GeometryMixin._boolean_radius_at(self, coordinate, segments)
                    )
                    projected = GeometryMixin._project_hypothesis_along_axis(
                        hypothesis, probe, point, axis
                    )
                    if projected is not None:
                        mouth_coordinates.append(
                            (projected - fit["center"]).dot(axis)
                        )
            if mouth_coordinates:
                bore_min = min(bore_min, min(mouth_coordinates))
                bore_max = max(bore_max, max(mouth_coordinates))
            # A through-hole cutter only needs to cross the two detected bore
            # mouths. Extending it across the complete object can cut a second,
            # remote part of a bracket that happens to lie on the same axis.
            coordinates = [
                bore_min - margin,
                bore_max + margin,
            ]

        if getattr(self, "_boolean_preserved_bore", None) is not None:
            return GeometryMixin._preserved_bore_cutter_geometry(
                self, obj, segments, coordinates[0]
            )
        inverse = obj.matrix_world.inverted_safe()
        vertices = []
        radii = [
            GeometryMixin._boolean_radius_at(self, coordinate, segments)
            for coordinate in coordinates[:2]
        ]
        if len(coordinates) > 2:
            # A countersink's cone ends at the requested truncation circle, but
            # its normal center bore continues at that circle's radius. Give
            # the complete junction and continuation temporary radial clearance
            # so no exact-radius ring is coplanar with the wall left by a prior
            # resize. The final result is projected back to the fitted radius.
            # The existing through-bore is commonly only approximately
            # cylindrical (and may already contain Boolean tessellation).
            # A microscopic offset can therefore still run through portions
            # of its old wall and Exact Boolean preserves overlapping faces.
            # Use enough temporary clearance to envelop that variation; the
            # generated wall is restored to ``radii[1]`` after subtraction.
            exact_bore_radius = radii[1]
            bore_clearance = max(
                exact_bore_radius
                * float(
                    getattr(self, "_conic_bore_clearance_ratio", 1.0e-4)
                ),
                1.0e-6,
            )
            expanded_bore_radius = exact_bore_radius + bore_clearance
            radii[1] = expanded_bore_radius
            radii.extend(
                [expanded_bore_radius] * (len(coordinates) - 2)
            )
            self._boolean_bore_projection = {
                "inner_coordinate": coordinates[1],
                "end_coordinate": coordinates[-1],
                "radius": exact_bore_radius,
                "expanded_radius": expanded_bore_radius,
                "segments": segments,
            }
        for coordinate, radius in zip(coordinates, radii):
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
        for ring_index in range(len(coordinates) - 1):
            start = ring_index * segments
            following_start = start + segments
            for index in range(segments):
                following = (index + 1) % segments
                faces.append((
                    start + index,
                    start + following,
                    following_start + following,
                    following_start + index,
                ))
        faces.append(tuple(reversed(range(segments))))
        last_start = (len(coordinates) - 1) * segments
        faces.append(tuple(range(last_start, last_start + segments)))
        if coordinates[1] < coordinates[0]:
            faces = [tuple(reversed(face)) for face in faces]
        return vertices, faces


    def _cleanup_faired_boolean_surface(
        self,
        obj,
        cutter_vertices,
        cutter_faces,
        fair_materials,
        simplify=True,
    ):
        """Simplify the tagged fair surface without touching cutter faces."""
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        fair_materials = dict(fair_materials)
        tagged_faces = {
            face for face in bm.faces
            if face.is_valid and not face.hide and face.material_index in fair_materials
        }
        world_cutter_vertices = [
            obj.matrix_world @ point for point in cutter_vertices
        ]
        cutter_bvh = BVHTree.FromPolygons(
            world_cutter_vertices, cutter_faces, all_triangles=False
        )
        if world_cutter_vertices:
            minimum = Vector(tuple(
                min(point[axis] for point in world_cutter_vertices)
                for axis in range(3)
            ))
            maximum = Vector(tuple(
                max(point[axis] for point in world_cutter_vertices)
                for axis in range(3)
            ))
            tolerance = max((maximum - minimum).length * 1.0e-5, 1.0e-7)
        else:
            tolerance = 1.0e-7

        def lies_on_cutter(face):
            points = [obj.matrix_world @ vert.co for vert in face.verts]
            points.append(obj.matrix_world @ face.calc_center_median())
            return all(
                (nearest := cutter_bvh.find_nearest(point)) is not None
                and nearest[0] is not None
                and nearest[3] <= tolerance
                for point in points
            )

        # Generic face attributes are interpolated by the Boolean. Newly made
        # cutter-wall/cap faces can therefore inherit the fair tag even though
        # they are not part of the reconstructed support surface.
        boolean_cutter_faces = {
            face for face in bm.faces if face.is_valid and not face.hide and lies_on_cutter(face)
        }
        fair_faces = tagged_faces - boolean_cutter_faces

        def restore_materials_and_slots():
            current_bm = bmesh.from_edit_mesh(obj.data)
            for face in current_bm.faces:
                restore_index = fair_materials.get(face.material_index)
                if restore_index is not None:
                    face.material_index = restore_index
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=False, destructive=False
            )
            for material_index in sorted(fair_materials, reverse=True):
                if material_index < len(obj.data.materials):
                    obj.data.materials.pop(index=material_index)

        if not simplify or not fair_faces:
            restore_materials_and_slots()
            return bool(fair_faces)

        previous_select_mode = tuple(
            bpy.context.tool_settings.mesh_select_mode
        )
        try:
            bpy.ops.mesh.select_mode(
                use_extend=False, use_expand=False, type='FACE'
            )
            for face in bm.faces:
                face.select = False
            for face in fair_faces:
                if face.is_valid:
                    face.select = True
            bm.select_flush_mode()
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=False, destructive=False
            )
            # The final region is exactly the complete reconstructed/fair
            # surface and one topological face ring on its exterior side.
            bpy.ops.mesh.select_more(use_face_step=True)
            bm = bmesh.from_edit_mesh(obj.data)
            for face in boolean_cutter_faces:
                if face.is_valid:
                    face.select = False
            bm.select_flush_mode()
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=False, destructive=False
            )
            # Convert that face region to edge selection immediately before
            # simplification so the old outer seam participates.
            bpy.ops.mesh.select_mode(
                use_extend=False, use_expand=False, type='EDGE'
            )
            bm = bmesh.from_edit_mesh(obj.data)
            fit = self._fit
            axis = fit["axis"].normalized()
            center = fit["center"]
            cutter_offsets = [point - center for point in world_cutter_vertices]
            cutter_coordinates = [
                offset.dot(axis) for offset in cutter_offsets
            ]
            cutter_radius = max(
                (
                    offset - axis * coordinate
                ).length
                for offset, coordinate in zip(
                    cutter_offsets, cutter_coordinates
                )
            )
            local_radius = max(cutter_radius, float(fit["radius"]))
            local_padding = max(local_radius * 0.25, tolerance * 10.0)

            def edge_is_local(edge):
                for vert in edge.verts:
                    offset = obj.matrix_world @ vert.co - center
                    coordinate = offset.dot(axis)
                    radial = offset - axis * coordinate
                    if (
                        radial.length > local_radius + local_padding
                        or coordinate
                        < min(cutter_coordinates) - local_padding
                        or coordinate
                        > max(cutter_coordinates) + local_padding
                    ):
                        return False
                return True

            # A topological one-face expansion can include an enormous planar
            # triangle whose far vertex belongs to another part of the model.
            # Keep the former rim and nearby Boolean spokes, but never let the
            # dissolve propagate along those non-local support edges.
            for edge in bm.edges:
                if edge.select and not edge_is_local(edge):
                    edge.select = False
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=False, destructive=False
            )
            bpy.ops.mesh.dissolve_limited(
                angle_limit=LIMITED_DISSOLVE_ANGLE,
                # The former hole rim is now the seam between the fair patch
                # and its selected outer ring. Boundary-inclusive dissolve is
                # required to remove those redundant rim vertices/spokes.
                use_dissolve_boundaries=True,
                # Boolean evaluation can leave split loop normals on every
                # coplanar fairing subdivision. NORMAL delimiting would retain
                # that data seam as the visible radial/concentric spider web.
                # The region and new opening are already explicitly bounded;
                # use geometric angle alone to retain genuine surface breaks.
                delimit=set(),
            )
        finally:
            bpy.context.tool_settings.mesh_select_mode = previous_select_mode
        restore_materials_and_slots()
        return True

    @staticmethod
    def _boolean_depth_unchanged(operator, blind_coordinates):
        """Return whether reconstruction retained the fitted blind depth."""
        depth_fit = getattr(operator, "_depth_fit", None)
        if depth_fit is None:
            return blind_coordinates is not None
        if GeometryMixin._conic_continues_as_cylindrical_bore(operator):
            return False
        initial = float(depth_fit.get("initial", 0.0))
        current = float(
            getattr(operator, "truncated_depth", initial)
            if GeometryMixin._uses_conic_dimensions(operator)
            else getattr(operator, "depth", initial)
        )
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
            if not face.is_valid or face.hide:
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
            update_visible_normals(bm)

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
        self._resize_wall_cache = None
        if obj.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        GeometryMixin._remove_live_boolean_preview(self)
        restore_mesh_contents(obj, source)
        if consume:
            self._boolean_preview_source = None
            self._boolean_preview_segments = 0
            self._boolean_preview_last_update = 0.0
            self._boolean_preview_pending = False
            reconstruction = getattr(
                self, "_boolean_reconstruction_source", None
            )
            self._boolean_reconstruction_source = None
            self._boolean_reconstruction_blind_coordinates = None
            if reconstruction is not None and reconstruction.users == 0:
                bpy.data.meshes.remove(reconstruction)
        if consume and source.users == 0:
            bpy.data.meshes.remove(source)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        invalidate_projection(self)
        self._resize_base_span = 0.0
        reset_deformation(self)
        GeometryMixin._release_boolean_fair_materials(self)
        return True

    def _release_boolean_fair_materials(self):
        """Release only our temporary tags once no operand mesh uses them."""
        retained = []
        for material in getattr(self, "_boolean_fair_materials", ()):
            if material.users == 0:
                bpy.data.materials.remove(material)
            else:
                retained.append(material)
        self._boolean_fair_materials = retained

    def _discard_boolean_preview_source(self):
        """Release the pristine source retained while previewing a Boolean."""
        self._resize_wall_cache = None
        GeometryMixin._remove_live_boolean_preview(self)
        source = getattr(self, "_boolean_preview_source", None)
        reconstruction = getattr(
            self, "_boolean_reconstruction_source", None
        )
        self._boolean_preview_source = None
        self._boolean_reconstruction_source = None
        self._boolean_reconstruction_blind_coordinates = None
        self._boolean_preview_segments = 0
        self._boolean_preview_last_update = 0.0
        self._boolean_preview_pending = False
        self._boolean_preserved_bore = None
        if source is not None and source.users == 0:
            bpy.data.meshes.remove(source)
        if reconstruction is not None and reconstruction.users == 0:
            bpy.data.meshes.remove(reconstruction)
        GeometryMixin._release_boolean_fair_materials(self)

    def _remove_live_boolean_preview(self):
        """Remove the reusable modal cutter and its unevaluated modifier."""
        obj = getattr(self, "_object", None)
        self._resize_fair_only_preview = False
        display_object = getattr(
            self, "_boolean_preview_display_object", None
        )
        self._boolean_preview_display_object = None
        modifier_name = getattr(self, "_boolean_preview_modifier_name", "")
        self._boolean_preview_modifier_name = ""
        modifier_owner = display_object or obj
        if modifier_owner is not None and modifier_name:
            try:
                modifier = modifier_owner.modifiers.get(modifier_name)
                if modifier is not None:
                    modifier_owner.modifiers.remove(modifier)
            except ReferenceError:
                pass
        if display_object is not None:
            try:
                display_mesh = display_object.data
                bpy.data.objects.remove(display_object, do_unlink=True)
                if (
                    display_mesh is not None
                    and display_mesh.users == 0
                    and (obj is None or display_mesh is not obj.data)
                ):
                    bpy.data.meshes.remove(display_mesh)
            except ReferenceError:
                pass
        if obj is not None:
            try:
                obj.hide_set(False)
            except ReferenceError:
                pass

        cutter = getattr(self, "_boolean_preview_cutter_object", None)
        self._boolean_preview_cutter_object = None
        if cutter is None:
            return
        try:
            cutter_mesh = cutter.data
            bpy.data.objects.remove(cutter, do_unlink=True)
            if cutter_mesh is not None and cutter_mesh.users == 0:
                bpy.data.meshes.remove(cutter_mesh)
        except ReferenceError:
            pass

    def _restore_boolean_reconstruction(self):
        """Restore the cached fair surface used by repeated live cutters."""
        self._resize_wall_cache = None
        source = getattr(self, "_boolean_reconstruction_source", None)
        obj = self._object
        if source is None or obj is None:
            return False
        if obj.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        restore_mesh_contents(obj, source)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        return True

    def _live_boolean_segments(self):
        """Use source resolution through 32 sides, otherwise preview at 32."""
        detected = max(3, min(512, int(
            getattr(self, "boolean_segments", 0)
            or getattr(self, "_resize_angular_segments", 0)
            or 32
        )))
        return detected if detected <= 32 else 32

    def _update_boolean_preview(
        self, context, preview_segments=None, source_is_current=False,
        fair_only=False,
    ):
        """Show a reversible, low-resolution Exact Boolean mesh in Edit Mode."""
        obj = context.active_object
        if preview_segments is None:
            preview_segments = GeometryMixin._live_boolean_segments(self)
        if getattr(self, "_boolean_preview_source", None) is None:
            # Edit-mode coordinates live in BMesh, not necessarily obj.data.
            # A preceding direct resize may have changed only that BMesh;
            # copying the stale datablock would resurrect the previous hole.
            obj.update_from_editmode()
            self._boolean_preview_source = obj.data.copy()
        self._boolean_preview_segments = preview_segments
        if not source_is_current:
            GeometryMixin._replace_boolean_preview_with_source(self)
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        # Restoring the pristine modal source can change element numbering when
        # the interaction entered Boolean mode after a direct preview or after
        # earlier topology edits. Never index that restored BMesh with the
        # previous mesh's face indices; validate them and rediscover the fitted
        # wall geometrically when its topology signature no longer matches.
        selected_faces = GeometryMixin._resolve_resize_wall_faces(self, bm)
        GeometryMixin._build_resize_projection_state(
            self, bm, selected_faces
        )
        selected_verts = {
            vert for face in selected_faces for vert in face.verts
        }
        GeometryMixin._capture_deformation_state(
            self, context, bm, selected_verts
        )
        self._resize_dynamic_requires_boolean = (
            getattr(self, "resize_method", 'AUTO') == 'AUTO'
        )
        GeometryMixin._resolve_resize_method(self)
        GeometryMixin._recreate_hole_with_boolean(
            self,
            context,
            bm,
            selected_faces,
            preview_segments=preview_segments,
            fair_only=fair_only,
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
            if (
                surface["closed"]
                and not surface["exterior"]
                and not surface.get("bore_continuation", False)
            )
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
        self, context, bm, selected_faces, preview_segments=None,
        fair_only=False,
    ):
        """Fair the old cavity, then cut again unless closing with fair_only."""
        self._resize_wall_cache = None
        obj = context.active_object
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
        obj.update_from_editmode()
        transaction = BooleanTransaction(obj)
        added_default_material_slot = False
        try:
            self._boolean_preserved_bore = None
            preserve_bore = (
                GeometryMixin._prepare_preserved_boolean_bore(
                    self, bm, selected_faces,
                    GeometryMixin._boolean_segment_count(self, preview_segments),
                ) if not fair_only else None
            )
            if fair_only:
                # Close the complete connected cavity, including a countersink
                # or its through-bore, rather than leaving a sealed inner shell.
                continuation_indices = set(getattr(self, "_resize_bore_vertex_indices", ()))
                continuation_indices.update(getattr(self, "_resize_conic_surfaces_by_vertex", {}))
                selected_faces = list(set(selected_faces) | {
                    face for face in bm.faces
                    if not face.hide and all(vert.index in continuation_indices for vert in face.verts)
                })
                GeometryMixin._build_resize_projection_state(self, bm, selected_faces)
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
                for surface in unique_surfaces
                if (
                    surface in exterior_surfaces
                    or surface.get("bore_continuation", False)
                )
                for face in surface.get(
                    "support_patch_faces", surface["support_faces"]
                )
                if face.is_valid
            }
            blocked_cavity_faces = selected | protected_faces
            if preserve_bore is not None:
                protected_faces.update(preserve_bore["faces"])
                blocked_cavity_faces.update(preserve_bore["faces"])

            # A support patch is intentionally shallow because analytic fitting
            # needs only the nearby surface. Deletion needs the opposite: walk
            # the complete blind-cap component bounded by the selected wall.
            # This removes dense, multi-ring bottoms whose inner rings lie
            # beyond both the support-patch depth and the wall-fit envelope.
            interior_faces = set()
            cavity_stack = [
                face
                for surface in unique_surfaces
                if (
                    not surface["exterior"]
                    and surface["closed"]
                    and not surface.get("bore_continuation", False)
                )
                for face in surface["support_faces"]
                if face.is_valid and not face.hide and face not in blocked_cavity_faces
            ]
            while cavity_stack:
                face = cavity_stack.pop()
                if face.hide or face in interior_faces or face in blocked_cavity_faces:
                    continue
                interior_faces.add(face)
                for edge in face.edges:
                    cavity_stack.extend(
                        neighbor
                        for neighbor in edge.link_faces
                        if (
                            neighbor.is_valid
                            and not neighbor.hide
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
                if face.hide or face in selected or face in protected_faces:
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
            if preserve_bore is not None:
                GeometryMixin._cap_preserved_boolean_bore(self, bm, preserve_bore)
            update_visible_normals(bm)
            fair_materials = {}
            self.reconstruction_result_method = ""
            self.reconstruction_result_fill_style = ""
            self.reconstruction_grid_edge_length = 0.0
            for surface in exterior_surfaces:
                loop_verts, loop_edges = (
                    GeometryMixin._recover_reconstruction_rim(
                        bm, surface, obj.matrix_world
                    )
                )
                if preserve_bore is not None:
                    # A preceding Boolean can leave pairs of nearly coincident
                    # mouth samples. Weld only that recovered boundary before
                    # filling it; a fan over zero-length edges is degenerate.
                    tolerance = max(abs(float(fit["radius"])) * 1.0e-6, 1.0e-7)
                    if any(
                        (obj.matrix_world @ edge.verts[0].co
                         - obj.matrix_world @ edge.verts[1].co).length <= tolerance
                        for edge in loop_edges
                    ):
                        local_scale = max(col.length for col in obj.matrix_world.to_3x3().col)
                        bmesh.ops.remove_doubles(
                            bm, verts=loop_verts,
                            dist=tolerance / max(local_scale, 1.0e-12),
                        )
                        update_visible_normals(bm)
                        loop_verts, loop_edges = GeometryMixin._recover_reconstruction_rim(
                            bm, surface, obj.matrix_world
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
                def refit_surface(verts, rim, forced_method):
                    # Keep the accepted model; only analytic pre-dissolve may
                    # require fresh samples from the surviving support faces.
                    return choose_hypothesis(
                        verts, rim, obj.matrix_world,
                        lambda faces: GeometryMixin._fit_cylinder(self, faces),
                        lambda face: GeometryMixin._world_normal(self, face),
                        analytic_faces=[face for face in analytic_faces if face.is_valid],
                        forced_method=forced_method,
                    )

                bm, _result = fair_hole_patch(
                    obj, bm, loop_verts, loop_edges, hypothesis, fill_style,
                    circle_loops=getattr(self, "circle_loops", 5),
                    match_curvature=getattr(self, "grid_match_curvature", True),
                    refit=refit_surface,
                    # Preserve subdivisions in the Boolean operand; simplify
                    # the tagged fair surface only after subtraction.
                    final_simplify=(fair_only and getattr(self, "final_simplify", True)),
                )
                self.reconstruction_result_method = _result.get("surface_method", _result["method"])
                self.reconstruction_result_fill_style = _result["fill_style"]
                self.reconstruction_grid_edge_length = _result.get("grid_estimated_edge_length", 0.0)
                support_materials = [
                    face.material_index
                    for face in analytic_faces
                    if face.is_valid
                ]
                restore_material = (
                    max(
                        set(support_materials),
                        key=support_materials.count,
                    )
                    if support_materials
                    else 0
                )
                if fair_only:
                    continue
                if not obj.data.materials:
                    # Untagged faces already use index zero even with no
                    # material slots. Reserve it before allocating a fair tag;
                    # otherwise all source faces masquerade as the fair patch.
                    obj.data.materials.append(None)
                    added_default_material_slot = True
                fair_material_index = len(obj.data.materials)
                # Empty slots are coalesced by Exact Boolean. Use a distinct
                # temporary material so source faces never acquire the fair
                # tag and accidentally enter reconstruction simplification.
                original_material = obj.data.materials[restore_material]
                material = (
                    original_material.copy() if original_material is not None
                    else bpy.data.materials.new("Holey Tools Fair Surface")
                )
                obj.data.materials.append(material)
                transaction.temporary_materials.append(material)
                fair_materials[fair_material_index] = restore_material
                for face in _result["faces"]:
                    if face.is_valid:
                        face.material_index = fair_material_index

            update_visible_normals(bm)
            if preserve_bore is not None:
                layer = bm.verts.layers.int.get(
                    self._boolean_preserved_bore["junction_layer_name"]
                )
                if layer is None:
                    raise FairingError("The preserved bore junction lost its topology marker")
                bm.verts.index_update()
                self._boolean_preserved_bore["junction_vertices"] = [
                    (vert.index, obj.matrix_world @ vert.co
                     - fit["center"] - fit["axis"] * self._boolean_preserved_bore["join"])
                    for vert in bm.verts if vert[layer]
                ]
                bm.verts.layers.int.remove(layer)
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=True, destructive=True
            )
            bpy.ops.object.mode_set(mode='OBJECT')

            # Native Boolean does not honor Edit Mode hide flags. Its operand
            # must contain only visible faces; hidden data is restored AFTER
            # solver cleanup, never sent through the Boolean or dissolve.
            transaction.hidden_geometry = detach_hidden_geometry(obj)
            if transaction.hidden_geometry is not None and preserve_bore is not None:
                self._boolean_preserved_bore["junction_vertices"] = [
                    (transaction.hidden_geometry.vertex_map[index], radial)
                    for index, radial in self._boolean_preserved_bore["junction_vertices"]
                    if index in transaction.hidden_geometry.vertex_map
                ]

            if fair_only:
                if preview_segments is not None:
                    transaction.preview_object = bpy.data.objects.new("Holey Tools Fair Preview", obj.data.copy())
                    context.collection.objects.link(transaction.preview_object)
                    transaction.preview_object.matrix_world = obj.matrix_world.copy()
                    transaction.preview_object.hide_render = True
                    transaction.preview_object.select_set(False)
                    restore_mesh_contents(obj, self._boolean_preview_source)
                elif transaction.hidden_geometry is not None:
                    transaction.hidden_geometry.restore(obj)
                bpy.ops.object.mode_set(mode='EDIT')
                if preview_segments is not None:
                    obj.hide_set(True)
                    self._boolean_preview_display_object = transaction.preview_object
                    transaction.release_preview()
                    self._resize_fair_only_preview = True
                self._resize_projection_ready = False
                return

            if (
                preview_segments is not None
                and getattr(
                    self, "_boolean_reconstruction_source", None
                ) is None
            ):
                self._boolean_reconstruction_source = obj.data.copy()
                self._boolean_reconstruction_blind_coordinates = (
                    tuple(blind_coordinates)
                    if blind_coordinates is not None
                    else None
                )

            cutter_vertices, cutter_faces = (
                GeometryMixin._boolean_cutter_geometry(
                    self,
                    obj,
                    blind_coordinates,
                    segment_override=preview_segments,
                )
            )
            transaction.cutter_mesh = bpy.data.meshes.new("Holey Tools Boolean Cutter")
            transaction.cutter_mesh.from_pydata(cutter_vertices, [], cutter_faces)
            transaction.cutter_mesh.validate(verbose=False)
            transaction.cutter_mesh.update()
            transaction.cutter_object = bpy.data.objects.new(
                "Holey Tools Boolean Cutter", transaction.cutter_mesh
            )
            context.collection.objects.link(transaction.cutter_object)
            transaction.cutter_object.matrix_world = obj.matrix_world.copy()
            transaction.cutter_object.display_type = 'WIRE'
            transaction.cutter_object.hide_render = True
            transaction.cutter_object.select_set(False)

            context.view_layer.objects.active = obj
            obj.select_set(True)
            transaction.modifier_owner = obj
            if preview_segments is not None:
                transaction.preview_object = bpy.data.objects.new(
                    "Holey Tools Boolean Preview", obj.data.copy()
                )
                context.collection.objects.link(transaction.preview_object)
                transaction.preview_object.matrix_world = obj.matrix_world.copy()
                transaction.preview_object.display_type = 'SOLID'
                transaction.preview_object.hide_render = True
                transaction.preview_object.select_set(False)
                transaction.modifier_owner = transaction.preview_object
            modifier = transaction.modifier_owner.modifiers.new(
                name="Holey Tools Recreate Hole", type='BOOLEAN'
            )
            transaction.modifier_name = modifier.name
            modifier.operation = 'DIFFERENCE'
            modifier.solver = 'EXACT'
            modifier.object = transaction.cutter_object
            if preview_segments is not None:
                # Keep the modal result non-destructive. Exact Boolean evaluation
                # still occurs in the viewport, but dragging only changes this
                # cutter mesh instead of repeatedly applying a modifier and
                # replacing the edited object's mesh datablock.
                modifier.show_viewport = True
                modifier.show_in_editmode = True
                modifier.show_on_cage = False
                transaction.cutter_object.hide_set(True)
                pristine_source = getattr(
                    self, "_boolean_preview_source", None
                )
                if pristine_source is None:
                    raise FairingError(
                        "Boolean preview lost its pristine source mesh"
                    )
                # The evaluated Boolean belongs exclusively to the display
                # object. Put an untouched source copy back on the real edit
                # object before returning to modal interaction so Blender's
                # undo/redo input can never become the reconstructed cap.
                restore_mesh_contents(obj, pristine_source)
                context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.mode_set(mode='EDIT')
                obj.hide_set(True)
                self._boolean_preview_display_object = transaction.preview_object
                self._boolean_preview_cutter_object = transaction.cutter_object
                self._boolean_preview_modifier_name = modifier.name
                self._boolean_fair_materials = (
                    list(getattr(self, "_boolean_fair_materials", ()))
                    + transaction.temporary_materials
                )
                transaction.release_preview()
                invalidate_projection(self)
                return
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            transaction.modifier_name = None
            if preview_segments is None:
                GeometryMixin._restore_boolean_bore_radius(self, obj)
                final_simplify = getattr(self, "final_simplify", True)
                GeometryMixin._cleanup_faired_boolean_surface(
                    self,
                    obj,
                    cutter_vertices,
                    cutter_faces,
                    fair_materials,
                    simplify=final_simplify,
                )
                if final_simplify and GeometryMixin._boolean_depth_unchanged(
                    self, blind_coordinates
                ):
                    GeometryMixin._cleanup_unchanged_boolean_base(
                        self, obj, cutter_vertices
                    )
                if added_default_material_slot:
                    obj.data.materials.pop(index=0)
                if transaction.hidden_geometry is not None:
                    if obj.mode == 'EDIT':
                        bpy.ops.object.mode_set(mode='OBJECT')
                    transaction.hidden_geometry.restore(obj)
            if obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            invalidate_projection(self)
            return
        except Exception as error:
            transaction.rollback()
            self._resize_projection_ready = False
            if isinstance(error, FairingError):
                raise
            raise FairingError(f"Boolean hole reconstruction failed: {error}") from error
        finally:
            transaction.close()

    def _apply_cached_boolean_preview(self, context, preview_segments):
        """Update the reusable cutter over the already-faired preview base."""
        obj = self._object
        display_object = getattr(
            self, "_boolean_preview_display_object", None
        )
        cutter_object = getattr(
            self, "_boolean_preview_cutter_object", None
        )
        modifier_name = getattr(self, "_boolean_preview_modifier_name", "")
        modifier = (
            display_object.modifiers.get(modifier_name)
            if display_object is not None and modifier_name
            else None
        )
        if (
            display_object is None
            or cutter_object is None
            or modifier is None
        ):
            GeometryMixin._remove_live_boolean_preview(self)
            if not GeometryMixin._restore_boolean_reconstruction(self):
                return False
            obj = self._object
            display_object = None
            cutter_object = None
            modifier = None

        if not GeometryMixin._update_preserved_bore_preview(
            self, display_object or obj, preview_segments
        ):
            # A deeper edit consumed the next source row: reclassify from the
            # pristine input instead of folding the retained cylinder over it.
            GeometryMixin._replace_boolean_preview_with_source(self, consume=True)
            GeometryMixin._update_boolean_preview(self, context, preview_segments,
                                                  source_is_current=True)
            return True

        blind_coordinates = getattr(
            self, "_boolean_reconstruction_blind_coordinates", None
        )
        cutter_vertices, cutter_faces = GeometryMixin._boolean_cutter_geometry(
            self,
            obj,
            blind_coordinates,
            segment_override=preview_segments,
        )
        if cutter_object is not None and modifier is not None:
            cutter_mesh = cutter_object.data
            cutter_mesh.clear_geometry()
            cutter_mesh.from_pydata(cutter_vertices, [], cutter_faces)
            cutter_mesh.validate(verbose=False)
            cutter_mesh.update()
            cutter_object.matrix_world = obj.matrix_world.copy()
        else:
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
            cutter_object.hide_set(True)
            cutter_object.select_set(False)
            context.view_layer.objects.active = obj
            obj.select_set(True)
            display_object = bpy.data.objects.new(
                "Holey Tools Boolean Preview", obj.data.copy()
            )
            context.collection.objects.link(display_object)
            display_object.matrix_world = obj.matrix_world.copy()
            display_object.display_type = 'SOLID'
            display_object.hide_render = True
            display_object.select_set(False)
            modifier = display_object.modifiers.new(
                name="Holey Tools Recreate Hole", type='BOOLEAN'
            )
            modifier.operation = 'DIFFERENCE'
            modifier.solver = 'EXACT'
            modifier.object = cutter_object
            modifier.show_viewport = True
            modifier.show_in_editmode = True
            modifier.show_on_cage = False
            pristine_source = getattr(self, "_boolean_preview_source", None)
            if pristine_source is None:
                GeometryMixin._remove_live_boolean_preview(self)
                raise FairingError(
                    "Boolean preview lost its pristine source mesh"
                )
            restore_mesh_contents(obj, pristine_source)
            context.view_layer.objects.active = obj
            obj.select_set(True)
            if obj.mode != 'EDIT':
                bpy.ops.object.mode_set(mode='EDIT')
            obj.hide_set(True)
            self._boolean_preview_display_object = display_object
            self._boolean_preview_cutter_object = cutter_object
            self._boolean_preview_modifier_name = modifier.name

        display_object.update_tag(refresh={'DATA'})
        self._boolean_preview_segments = preview_segments
        self._boolean_preview_last_update = time.monotonic()
        self._boolean_preview_pending = False
        build_overlay = getattr(self, "_build_cylinder_preview", None)
        if build_overlay is not None:
            build_overlay(context)
        return True

    def _resolve_resize_wall_faces(self, bm):
        """Reuse verified wall faces only during a coordinate-only modal drag.

        Redo and mesh restoration always perform the full connectivity check.
        During a drag we own mesh mutations; changing counts, mesh identity,
        selection, or restoring/rebuilding topology invalidates the cache.
        """
        cacheable = bool(getattr(self, "_drag_active", False))
        counts = (len(bm.verts), len(bm.edges), len(bm.faces))
        indices = frozenset(self._locked_indices)
        cached = getattr(self, "_resize_wall_cache", None)
        if cacheable and cached is not None:
            cached_bm, cached_counts, cached_indices, faces = cached
            if (
                cached_bm is bm and cached_counts == counts and cached_indices == indices
                and all(face.is_valid for face in faces)
            ):
                return faces
        faces = GeometryMixin._resolve_resize_wall_faces_uncached(self, bm)
        self._resize_wall_cache = (bm, counts, indices, faces) if cacheable else None
        return faces

    def _resolve_resize_wall_faces_uncached(self, bm):
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
            if face.hide:
                return None
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
            if not bm.faces[index].hide:
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
            if not face.hide and (measure := wall_measure(face)) is not None
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
        fair_only = GeometryMixin._fair_resize_requested(self)
        if fair_only:
            if not commit_boolean and getattr(self, "_resize_fair_only_preview", False):
                return
            GeometryMixin._replace_boolean_preview_with_source(self, consume=True)
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            for index, original in (getattr(self, "_original_vertex_coords", None) or {}).items():
                if index < len(bm.verts):
                    bm.verts[index].co = original.copy()
            faces = GeometryMixin._resolve_resize_wall_faces(self, bm)
            GeometryMixin._build_resize_projection_state(self, bm, faces)
            if commit_boolean:
                try:
                    GeometryMixin._recreate_hole_with_boolean(self, context, bm, faces, fair_only=True)
                finally:
                    GeometryMixin._discard_boolean_preview_source(self)
            else:
                bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
                try:
                    GeometryMixin._update_boolean_preview(self, context, fair_only=True)
                except FairingError as error:
                    GeometryMixin._replace_boolean_preview_with_source(self, consume=True)
                    self.report({'WARNING'}, f"Fair preview failed: {error}")
            return
        if getattr(self, "_resize_fair_only_preview", False):
            GeometryMixin._replace_boolean_preview_with_source(self, consume=True)
            self._resize_fair_only_preview = False
        if getattr(self, "_resize_scaled_exit_preview", False):
            GeometryMixin._replace_boolean_preview_with_source(self, consume=True)
            self._resize_scaled_exit_preview = False
        resize_method = getattr(self, "resize_method", 'AUTO')
        GeometryMixin._resolve_resize_method(self)
        if resize_method == 'SCALE':
            # A retained Boolean result must not bypass a new explicit Scaling
            # choice through the cached-preview or drag-throttle fast paths.
            GeometryMixin._replace_boolean_preview_with_source(
                self, consume=True
            )
        if (
            not commit_boolean
            and resize_method != 'SCALE'
            and getattr(self, "_boolean_preview_source", None) is not None
            and bool(getattr(self, "_drag_active", False))
            and time.monotonic()
            - float(getattr(self, "_boolean_preview_last_update", 0.0))
            < _BOOLEAN_PREVIEW_INTERVAL
        ):
            # Keep the previous Boolean mesh visible and let the inexpensive
            # GPU cutter follow every pointer event. The mouse-release handler
            # performs one trailing cutter update for the final drag value.
            self._boolean_preview_pending = True
            return
        if (
            not commit_boolean
            and resize_method != 'SCALE'
            and getattr(self, "_boolean_reconstruction_source", None)
            is not None
        ):
            try:
                GeometryMixin._apply_cached_boolean_preview(
                    self,
                    context,
                    GeometryMixin._live_boolean_segments(self),
                )
            except FairingError as error:
                GeometryMixin._replace_boolean_preview_with_source(
                    self, consume=True
                )
                self.report({'WARNING'}, f"Boolean preview failed: {error}")
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
        if restored_boolean_preview and resize_method == 'AUTO':
            self._resize_dynamic_requires_boolean = True

        if not getattr(self, "_original_vertex_coords", None):
            deformation_verts = set(verts)
            deformation_verts.update(
                bm.verts[index]
                for index in getattr(
                    self, "_resize_bore_vertex_indices", set()
                )
                if index < len(bm.verts)
            )
            GeometryMixin._capture_deformation_state(
                self, context, bm, deformation_verts
            )
        else:
            # Some callers capture the cylinder before fitting its adjacent
            # support. Include newly recognized cone vertices in that same
            # immutable deformation baseline.
            for index in getattr(self, "_resize_conic_surfaces_by_vertex", {}):
                if (
                    index < len(bm.verts)
                    and index not in self._original_vertex_coords
                ):
                    if getattr(self, "_proportional_weights", None) is None:
                        self._proportional_weights = {
                            key: 1.0 for key in self._original_vertex_coords
                        }
                    self._original_vertex_coords[index] = bm.verts[index].co.copy()
                    self._proportional_weights[index] = 1.0

        inverse = obj.matrix_world.inverted_safe()
        fit = self._fit
        target_scale = self._target_scale()
        depth_fit = getattr(self, "_depth_fit", None)
        depth_scale = self._target_depth_scale() if depth_fit else 1.0
        target_slope = GeometryMixin._target_cone_slope(self)
        fitted_slope = float(fit.get("slope", 0.0))

        GeometryMixin._resolve_resize_method(self, check_transition=True)

        # Confirming untouched dimensions must be an exact geometry no-op. In
        # particular, do not flatten within-epsilon boundary variation onto the
        # best-fit planes. This path also restores the verbatim source geometry
        # when edited dimensions are returned to their fitted values.
        cutter_resolution_changed = (
            getattr(self, "_resize_requires_boolean", False)
            and int(getattr(self, "boolean_segments", 0) or 0)
            != int(getattr(self, "fit_angular_segments", 0) or 0)
        )
        if resize_method != 'BOOLEAN' and not cutter_resolution_changed and math.isclose(
            target_scale, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-9
        ) and math.isclose(
            depth_scale, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-9
        ) and math.isclose(
            target_slope, fitted_slope, rel_tol=1.0e-6, abs_tol=1.0e-9
        ):
            for vertex_index, original in self._original_vertex_coords.items():
                if vertex_index < len(bm.verts):
                    bm.verts[vertex_index].co = original.copy()
            update_visible_normals(bm)
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
                # pristine source, then reconstruct once beneath a reusable
                # unapplied Boolean modifier for the live modal preview.
                # Confirmation restores that source once more and performs the
                # full-resolution reconstruction.
                for vertex_index, original in self._original_vertex_coords.items():
                    if vertex_index < len(bm.verts):
                        bm.verts[vertex_index].co = original.copy()
                update_visible_normals(bm)
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

        connect_exit = GeometryMixin._shortened_through_cone(self)
        if connect_exit and not commit_boolean:
            # A topology-changing Scaling preview needs the same pristine
            # source protection as Boolean preview, for drag/cancel/confirm.
            for index, original in self._original_vertex_coords.items():
                if index < len(bm.verts):
                    bm.verts[index].co = original.copy()
            update_visible_normals(bm)
            bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
            obj.update_from_editmode()
            self._boolean_preview_source = obj.data.copy()

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
        bore_indices = set(
            getattr(self, "_resize_bore_vertex_indices", set())
        ) - selected_indices
        proportional_weights = getattr(self, "_proportional_weights", None) or {
            index: 1.0 for index in selected_indices
        }
        for vertex_index, weight in proportional_weights.items():
            if vertex_index >= len(bm.verts):
                continue
            vert = bm.verts[vertex_index]
            if not is_visible_vertex(vert):
                continue
            original = self._original_vertex_coords.get(vertex_index, vert.co)
            world = obj.matrix_world @ original
            conic_surface = getattr(
                self, "_resize_conic_surfaces_by_vertex", {}
            ).get(vertex_index)
            if conic_surface is not None:
                world_target = (
                    world
                    if vertex_index in conic_surface["conic_outer_vertices"]
                    else GeometryMixin._conic_continuation_target(
                        self, world, conic_surface
                    )
                )
                vert.co = inverse @ world.lerp(world_target, weight)
                continue
            offset = world - fit["center"]
            axial_coordinate = offset.dot(fit["axis"])
            target_axial_coordinate = axial_coordinate
            if vertex_index in bore_indices and depth_fit:
                target_inner_coordinate = depth_fit["outer_coordinate"] + (
                    depth_fit["inner_coordinate"]
                    - depth_fit["outer_coordinate"]
                ) * depth_scale
                source_radius = GeometryMixin._fitted_radius_at(
                    self, depth_fit["inner_coordinate"]
                )
                target_radius = GeometryMixin._target_radius_at(
                    self, target_inner_coordinate
                )
                axial = fit["axis"] * axial_coordinate
                radial = offset - axial
                world_target = (
                    fit["center"]
                    + axial
                    + radial * (target_radius / source_radius)
                )
                vert.co = inverse @ world.lerp(world_target, weight)
                continue
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
            # Only blind caps follow the depth translation. The exterior
            # support stays fixed, even when its rim is not exactly normal to
            # the fitted axis (or has within-tolerance coordinate noise).
            projection_used = False
            surface = getattr(
                self, "_resize_surfaces_by_vertex", {}
            ).get(vertex_index)
            connecting_exit = (
                connect_exit and vertex_index in self._conic_exit_vertex_indices
            )
            fixed_exterior = (
                surface is not None and surface["exterior"] and not connecting_exit
            )
            radius_changed = not math.isclose(
                target_radius,
                source_radius,
                rel_tol=1.0e-6,
                abs_tol=1.0e-9,
            )
            if (
                surface is not None
                and (
                    radius_changed
                    or not math.isclose(depth_scale, 1.0, rel_tol=1.0e-6)
                )
                and not surface.get("bore_continuation", False)
                and not connecting_exit
            ):
                depth_delta = fit["axis"] * (
                    target_axial_coordinate - axial_coordinate
                )
                projection_point = world_target - depth_delta
                if target_radius > source_radius and surface["exterior"]:
                    hypothesis = surface["hypothesis"]
                    if (
                        hypothesis.get("method") == "PLANES"
                        and len(hypothesis.get("planes", ())) == 1
                    ):
                        # The dynamic mode check has already established that
                        # this enlarged rim remains on its accepted planar
                        # collar. Intersect that plane directly: a bidirectional
                        # whole-mesh ray can miss a triangulation boundary and
                        # incorrectly choose the opposite side of the model.
                        projected = (
                            GeometryMixin._project_hypothesis_along_axis(
                                hypothesis,
                                projection_point,
                                world,
                                fit["axis"],
                            )
                        )
                    else:
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
                    world_target = (
                        projected
                        if fixed_exterior
                        else projected + depth_delta
                    )
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
                    if depth_fit and plane is not outer_plane and not fixed_exterior:
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

        # The next wall flood/fitting pass reads BMesh face normals. Keep them
        # consistent with the new coordinates, especially when a cone becomes
        # a cylinder and its wall must join the continuing bore selection.
        update_visible_normals(bm)
        if connect_exit:
            self._resize_wall_cache = None
            GeometryMixin._connect_scaled_conic_exit(self, bm, faces)
            self._resize_scaled_exit_preview = not commit_boolean
            bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
            if commit_boolean:
                GeometryMixin._discard_boolean_preview_source(self)
            return

        if (
            commit_boolean
            and target_scale < 1.0 - 1.0e-9
            and getattr(self, "scaling_final_simplify", False)
        ):
            result = expanded_limited_dissolve(bm, faces)
            for face in bm.faces:
                face.select = False
            for face in result["faces"]:
                if face.is_valid:
                    face.select = True
            update_visible_normals(bm)
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=True, destructive=True
            )
        else:
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=True, destructive=False
            )
