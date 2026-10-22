# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Local Boolean junctions that preserve an existing cylindrical bore."""

import math

import bmesh
import bpy
from mathutils import Vector

from .resize_dimensions import ResizeDimensionsMixin


class BooleanBoreMixin(ResizeDimensionsMixin):
    """Build, cap, and update a cone-to-bore junction without recutting its wall."""

    def _boolean_segment_count(self, override=None):
        """Resolve the same clamped cutter resolution for setup and lofting."""
        return max(3, min(512, int(
            override
            or getattr(self, "boolean_segments", 0)
            or getattr(self, "_resize_angular_segments", 0)
            or len(self._locked_indices)
            or 32
        )))

    def _prepare_preserved_boolean_bore(self, bm, selected_faces, segments):
        """Keep a compatible existing bore out of the cone's Boolean cut.

        After the old cone is removed, cap its bore at the requested junction.
        A cutter matching that cap then opens it into the new cone, tapering
        inward immediately below it so the surviving cylinder is never recut.
        Incompatible/consumed bore rows retain the general reconstruction path.
        """
        self._boolean_preserved_bore = None
        if not BooleanBoreMixin._conic_continues_as_cylindrical_bore(self):
            return None
        surfaces = {
            id(surface): surface
            for surface in self._resize_surfaces_by_vertex.values()
            if surface.get("bore_continuation", False)
        }
        if len(surfaces) != 1:
            return None
        surface = next(iter(surfaces.values()))
        rim = list(surface["loop_verts"])
        if len(rim) < 3 or any(not vert.is_valid for vert in rim):
            return None
        fit = self._fit
        axis = fit["axis"].normalized()
        center = fit["center"]
        matrix = self._object.matrix_world
        inner = BooleanBoreMixin._target_conic_inner_coordinate(self)
        direction = math.copysign(
            1.0, self._depth_fit["inner_coordinate"] - self._depth_fit["outer_coordinate"]
        )
        radius = BooleanBoreMixin._fixed_conic_bottom_radius(self)
        adapter = max(radius * .002, 1.0e-6) if segments != len(rim) else 0.0
        join = inner + direction * adapter
        rim_set = set(rim)
        bm.verts.ensure_lookup_table()
        bore_verts = {
            bm.verts[index]
            for index in self._resize_bore_vertex_indices
            if index < len(bm.verts)
        }
        other_verts = bore_verts - rim_set
        if not other_verts:
            return None
        remaining = min(
            ((matrix @ vert.co - center).dot(axis) - join) * direction
            for vert in other_verts
        )
        if remaining <= max(radius * 1.0e-5, 1.0e-7):
            return None
        # Radial sorting retains the source polygon's phase, nonuniform samples,
        # and dissolvable rim vertices, instead of assuming angle zero matches it.
        samples = []
        for vert in rim:
            offset = matrix @ vert.co - center
            radial = offset - axis * offset.dot(axis)
            if radial.length <= radius * .5:
                return None
            angle = math.atan2(radial.dot(fit["basis_y"]), radial.dot(fit["basis_x"])) % math.tau
            samples.append((angle, vert, radial))
        samples.sort(key=lambda sample: sample[0])
        gaps = [(samples[(i + 1) % len(samples)][0] - sample[0]) % math.tau
                for i, sample in enumerate(samples)]
        if min(gaps) <= 1.0e-7 or max(gaps) >= math.pi:
            return None
        selected = set(selected_faces)
        bore_faces = {
            face for face in bm.faces
            if not face.hide and face not in selected and all(vert in bore_verts for vert in face.verts)
        }
        if not bore_faces:
            return None
        return {
            "samples": samples, "faces": bore_faces,
            "inner": inner, "join": join, "direction": direction,
            "tip_span": min(remaining * .05, max(radius * .002, 1.0e-6)),
            "limit": join + direction * remaining,
        }

    def _cap_preserved_boolean_bore(self, bm, state):
        """Close the moved bore with a temporary, cutter-matched solid cap."""
        fit = self._fit
        inverse = self._object.matrix_world.inverted_safe()
        for _angle, vert, radial in state["samples"]:
            vert.co = inverse @ (fit["center"] + fit["axis"] * state["join"] + radial)
        rim = [sample[1] for sample in state["samples"]]
        # The cap faces into the retained bore (the solid's outward direction).
        cap = bm.faces.new(rim if state["direction"] > 0.0 else list(reversed(rim)))
        cap.material_index = next(iter(state["faces"])).material_index
        bm.verts.index_update()
        rim_indices = [vert.index for vert in rim]
        # Creating custom data reallocates BMVert storage; reacquire wrappers
        # before touching the new layer rather than retaining stale references.
        layer = bm.verts.layers.int.new("_holey_bore_junction")
        bm.verts.ensure_lookup_table()
        for index in rim_indices:
            bm.verts[index][layer] = 1
        bm.normal_update()
        # Store values only: fairing can refresh BMesh wrappers, and modal
        # previews retain this state after the original source is put back.
        self._boolean_preserved_bore = {
            key: state[key] for key in ("inner", "join", "direction", "tip_span", "limit")
        }
        self._boolean_preserved_bore["samples"] = [
            (angle, radial.copy()) for angle, _vert, radial in state["samples"]
        ]
        self._boolean_preserved_bore["junction_layer_name"] = layer.name

    def _update_preserved_bore_preview(self, obj, segments):
        """Move only the cached operand's cap/rim when conic depth changes."""
        state = getattr(self, "_boolean_preserved_bore", None)
        if state is None:
            return True
        inner = BooleanBoreMixin._target_conic_inner_coordinate(self)
        radius = BooleanBoreMixin._fixed_conic_bottom_radius(self)
        adapter = max(radius * .002, 1.0e-6) if segments != len(state["samples"]) else 0.0
        join = inner + state["direction"] * adapter
        remaining = (state["limit"] - join) * state["direction"]
        if remaining <= max(radius * 1.0e-5, 1.0e-7):
            return False
        if inner == state["inner"] and join == state["join"]:
            # Angle-only edits change the cutter, not the retained bore cap.
            # Rebuilding identical junction coordinates through world/local
            # arithmetic introduces last-bit drift against the cutter polygon.
            # At zero slope those coplanar facets can make Exact Boolean drop
            # wall faces, even though a fresh cylindrical cut is well formed.
            return True
        state.update(inner=inner, join=join,
                     tip_span=min(remaining * .05, max(radius * .002, 1.0e-6)))
        inverse = obj.matrix_world.inverted_safe()
        meshes = [obj.data, getattr(self, "_boolean_reconstruction_source", None)]
        for mesh in meshes:
            if mesh is None:
                continue
            for index, radial in state["junction_vertices"]:
                mesh.vertices[index].co = inverse @ (
                    self._fit["center"] + self._fit["axis"] * join + radial
                )
            mesh.update()
        return True

    def _preserved_bore_cutter_geometry(self, obj, segments, outer):
        """Loft the cone into the original bore polygon, not through its walls."""
        state = self._boolean_preserved_bore
        fit = self._fit
        samples = state["samples"]
        slope = BooleanBoreMixin._target_cone_slope(self)
        inner, join = state["inner"], state["join"]
        direction = state["direction"]
        if segments == len(samples):
            cone_samples = samples
        else:
            phase = samples[0][0]
            radius = BooleanBoreMixin._fixed_conic_bottom_radius(self)
            cone_samples = [
                (phase + math.tau * i / segments,
                 (fit["basis_x"] * math.cos(phase + math.tau * i / segments)
                  + fit["basis_y"] * math.sin(phase + math.tau * i / segments)) * radius)
                for i in range(segments)
            ]
        rings = [
            [(angle, fit["center"] + fit["axis"] * outer
              + radial.normalized() * max(1.0e-9, radial.length + slope * (outer - inner)))
             for angle, radial in cone_samples],
            [(angle, fit["center"] + fit["axis"] * inner + radial)
             for angle, radial in cone_samples],
        ]
        if segments != len(samples):
            # Different requested resolutions transition in one short strip at
            # the junction; no cuts propagate down the original cylinder.
            phase = cone_samples[0][0]
            ordered = sorted(samples, key=lambda sample: (sample[0] - phase) % math.tau)
            rings.append([
                (phase + (angle - phase) % math.tau,
                 fit["center"] + fit["axis"] * join + radial)
                for angle, radial in ordered
            ])
        last = rings[-1]
        tip = join + direction * state["tip_span"]
        rings.append([
            (angle, fit["center"] + fit["axis"] * tip
             + (point - fit["center"] - fit["axis"] * join) * .98)
            for angle, point in last
        ])
        vertices, faces = [], []
        inverse = obj.matrix_world.inverted_safe()
        starts = []
        for ring in rings:
            starts.append(len(vertices))
            vertices.extend(inverse @ point for _angle, point in ring)
        for ring_index in range(len(rings) - 1):
            a, b = rings[ring_index:ring_index + 2]
            a_start, b_start = starts[ring_index:ring_index + 2]
            i = j = 0
            while i < len(a) or j < len(b):
                next_a = a[i + 1][0] if i + 1 < len(a) else a[0][0] + math.tau
                next_b = b[j + 1][0] if j + 1 < len(b) else b[0][0] + math.tau
                if i >= len(a):
                    next_a = math.inf
                if j >= len(b):
                    next_b = math.inf
                av, bv = a_start + i % len(a), b_start + j % len(b)
                if abs(next_a - next_b) <= 1.0e-7:
                    faces.append((av, a_start + (i + 1) % len(a), b_start + (j + 1) % len(b), bv))
                    i += 1
                    j += 1
                elif next_a < next_b:
                    faces.append((av, a_start + (i + 1) % len(a), bv))
                    i += 1
                else:
                    faces.append((av, b_start + (j + 1) % len(b), bv))
                    j += 1
        faces.append(tuple(reversed(range(len(rings[0])))))
        faces.append(tuple(range(starts[-1], len(vertices))))
        if direction < 0.0:
            faces = [tuple(reversed(face)) for face in faces]
        return vertices, faces

    def _restore_boolean_bore_radius(self, obj):
        """Project a clearance-cut continuation back to its fitted radius."""
        projection = getattr(self, "_boolean_bore_projection", None)
        if not projection:
            return
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        fit = self._fit
        axis = fit["axis"].normalized()
        center = fit["center"]
        inverse = obj.matrix_world.inverted_safe()
        inner = float(projection["inner_coordinate"])
        end = float(projection["end_coordinate"])
        direction = 1.0 if end >= inner else -1.0
        radius = float(projection["radius"])
        expanded = float(projection["expanded_radius"])
        axial_tolerance = max(abs(end - inner) * 1.0e-7, 1.0e-7)
        clearance = max(expanded - radius, 1.0e-12)
        radial_tolerance = max(clearance * 2.0, 1.0e-7)
        changed = False
        projected_vertices = set()
        for vert in bm.verts:
            if vert.hide:
                continue
            world = obj.matrix_world @ vert.co
            offset = world - center
            coordinate = offset.dot(axis)
            if (
                (coordinate - inner) * direction < -axial_tolerance
                or (end - coordinate) * direction < -axial_tolerance
            ):
                continue
            radial = offset - axis * coordinate
            radial_length = radial.length
            if (
                radial_length <= 1.0e-12
                # Exact-radius vertices belong to the preserved source wall;
                # scaling those would undersize the bore. Only solver output
                # displaced toward the clearance cutter is normalized.
                or radial_length <= radius + clearance * 0.25
                or radial_length > expanded + radial_tolerance
            ):
                continue
            world = center + axis * coordinate + radial * (radius / expanded)
            vert.co = inverse @ world
            changed = True
            projected_vertices.add(vert)
        if changed:
            # Exact Boolean may split the same intersection into multiple
            # vertices on its clearance wall. Projection deliberately puts
            # those vertices back at the identical fitted radius, so weld the
            # resulting coincidences immediately. Without this local weld a
            # second resize leaves zero-length edges and duplicate boundary
            # edges where the new cone rejoins the already rebuilt bore.
            weld_distance = max(abs(radius) * 1.0e-6, 1.0e-7)
            weld_vertices = set(projected_vertices)
            for vert in tuple(projected_vertices):
                if vert.is_valid:
                    weld_vertices.update(
                        neighbor
                        for edge in vert.link_edges
                        for neighbor in edge.verts
                        if neighbor.is_valid
                    )
            bmesh.ops.remove_doubles(
                bm,
                verts=[vert for vert in weld_vertices if vert.is_valid],
                dist=weld_distance,
            )
            bm.normal_update()
            bmesh.update_edit_mesh(
                obj.data, loop_triangles=True, destructive=True
            )

    def _conic_continues_as_cylindrical_bore(self):
        """Detect a normal coaxial bore attached to a conic truncation circle."""
        if not BooleanBoreMixin._uses_conic_dimensions(self):
            return False
        depth_fit = self._depth_fit
        inner_coordinate = float(depth_fit["inner_coordinate"])
        axis = self._fit["axis"].normalized()
        tolerance = max(float(depth_fit["initial"]) * 0.05, 1.0e-6)
        surfaces = list({
            id(surface): surface
            for surface in getattr(
                self, "_resize_surfaces_by_vertex", {}
            ).values()
        }.values())
        if not surfaces:
            return bool(
                getattr(self, "_conic_bore_continuation", False)
            )
        for surface in surfaces:
            if not surface.get("closed", False):
                continue
            points = surface.get("loop_world_points", ())
            if not points:
                continue
            coordinate = sum(
                (point - self._fit["center"]).dot(axis)
                for point in points
            ) / len(points)
            if abs(coordinate - inner_coordinate) > tolerance:
                continue
            if surface.get("bore_continuation", False):
                return True
            hypothesis = surface.get("hypothesis", {})
            if hypothesis.get("method") != "CONIC":
                continue
            continuation_fit = hypothesis.get("fit", {})
            continuation_axis = continuation_fit.get("axis")
            if continuation_axis is None:
                continue
            if abs(Vector(continuation_axis).normalized().dot(axis)) < math.cos(
                math.radians(2.0)
            ):
                continue
            if abs(float(continuation_fit.get("slope", 0.0))) > math.tan(
                math.radians(0.4)
            ):
                continue
            return True
        return False
