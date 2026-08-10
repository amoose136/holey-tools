# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resize Hole modal operator and runtime lifecycle."""

import json
import math

import bmesh
import bpy
import numpy as np

from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .constants import ADDON_VERSION, DIAMETER_ESTIMATOR_ITEMS
from .geometry import GeometryMixin
from .gpu_preview import GpuPreviewMixin


_modal_active = False
_draw_handles = set()
_persistent_preview = None


class _PersistentPreview:
    """Own a post-modal GPU preview without retaining an Operator RNA object.

    Blender may invalidate the Python wrapper for a completed operator while its
    Redo panel is still active. This plain Python object copies only the drawing
    data needed to keep the cylinder visible and follows the last-operation stack
    so it can remove itself when Resize Hole is no longer the redoable action.
    """

    def __init__(self, operator, context):
        """Copy preview state, register a draw handler, and begin lifetime checks."""
        self._fill_coords = list(operator._fill_coords or ())
        self._line_coords = list(operator._line_coords or ())
        self._region = operator._region
        self._region_data = operator._region_data
        self._stage = 'CYLINDER'
        self._accepts_numeric_input = False
        self._numeric_input = operator._numeric_input
        self._draw_error_reported = False
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_preview, (), 'WINDOW', 'POST_PIXEL'
        )
        _draw_handles.add(self._draw_handle)
        self._watch_last_operation(context)

    def update_from(self, operator, context):
        """Refresh copied geometry and label text after a Redo-panel execution."""
        self._fill_coords = list(operator._fill_coords or ())
        self._line_coords = list(operator._line_coords or ())
        self._stage = 'CYLINDER'
        self._numeric_input = operator._numeric_input
        if context.area is not None:
            context.area.tag_redraw()

    def remove(self):
        """Remove this preview's draw handler if it is still registered."""
        if self._draw_handle is None:
            return
        try:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
        except ValueError:
            pass
        _draw_handles.discard(self._draw_handle)
        self._draw_handle = None

    def _watch_last_operation(self, context):
        """Poll Blender's last operation and retire the preview when it changes.

        The timer deliberately captures only ordinary context objects and this
        plain Python preview. It never dereferences the completed operator whose
        RNA wrapper Blender is free to remove.
        """
        window_manager = context.window_manager
        area = context.area

        def watch():
            global _persistent_preview
            if self._draw_handle is None:
                return None
            try:
                operators = window_manager.operators
                if operators and operators[-1].bl_idname in {
                    "mesh.holey_tools_resize_hole",
                    "MESH_OT_holey_tools_resize_hole",
                }:
                    return 0.25
            except ReferenceError:
                pass
            self.remove()
            if _persistent_preview is self:
                _persistent_preview = None
            if area is not None:
                area.tag_redraw()
            return None

        bpy.app.timers.register(watch, first_interval=0.25)

    def _project_triangles(self, region, region_data):
        """Delegate triangle projection to the shared GPU-preview implementation."""
        return GpuPreviewMixin._project_triangles(
            self, region, region_data
        )

    def _project_lines(self, region, region_data):
        """Delegate line projection to the shared GPU-preview implementation."""
        return GpuPreviewMixin._project_lines(
            self, region, region_data
        )

    def _draw_preview(self):
        """Delegate the guarded Blender draw callback to ``GpuPreviewMixin``."""
        GpuPreviewMixin._draw_preview(self)

    def _draw_preview_impl(self):
        """Delegate preview rendering to ``GpuPreviewMixin``."""
        GpuPreviewMixin._draw_preview_impl(self)

    def _draw_dimension_label(self, region, projected_points, color):
        """Delegate the non-editable post-modal dimension label drawing."""
        GpuPreviewMixin._draw_dimension_label(
            self, region, projected_points, color
        )


class MESH_OT_holey_tools_resize_hole(
    GeometryMixin,
    GpuPreviewMixin,
    bpy.types.Operator,
):
    """Select one cylindrical wall, fit it, and resize it interactively.

    The modal operator has two stages. SELECT ray-casts beneath the pointer and
    displays a non-destructive flood fill. A click locks that region, fits the
    cylinder, and enters CYLINDER, where typing or dragging updates the diameter.
    Confirmation hands plain drawing data to ``_PersistentPreview`` so Blender's
    Redo panel can continue changing the completed operation safely.
    """

    bl_idname = "mesh.holey_tools_resize_hole"
    bl_label = "Resize Hole"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}
    bl_description = (
        "Hover to flood-fill a cylindrical wall, click to fit a cylinder, "
        "then click again to apply"
    )

    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Internal radius scale relative to the fitted cylinder",
        default=1.0,
        min=0.01,
        max=10.0,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    diameter: bpy.props.FloatProperty(
        name="Diameter",
        description="Physical world-space diameter of the resized hole",
        default=0.0,
        min=0.0,
        subtype='DISTANCE',
        unit='LENGTH',
    )
    flood_angle: bpy.props.FloatProperty(
        name="Flood Angle",
        description="Maximum angle between neighboring faces in the hover flood fill",
        subtype='ANGLE',
        default=math.radians(60.0),
        min=math.radians(0.1),
        max=math.radians(89.9),
    )
    diameter_estimator: bpy.props.EnumProperty(
        name="Diameter Estimate",
        description="How to estimate diameter when the selected wall has multiple radii",
        items=DIAMETER_ESTIMATOR_ITEMS,
        default='AVERAGE',
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    fit_valid: bpy.props.BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    fit_face_indices: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_axis: bpy.props.FloatVectorProperty(
        size=3, options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_basis_x: bpy.props.FloatVectorProperty(
        size=3, options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_basis_y: bpy.props.FloatVectorProperty(
        size=3, options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_center: bpy.props.FloatVectorProperty(
        size=3, options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_radius: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_z_min: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_z_max: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_end_planes: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'}
    )

    _draw_handle = None
    _fill_batch = None
    _line_batch = None
    _fill_coords = None
    _line_coords = None
    _draw_error_reported = False
    _bvh = None
    _hover_face_index = -1
    _candidate_indices = None
    _locked_indices = None
    _initial_selection = None
    _original_vertex_coords = None
    _proportional_weights = None
    _object = None
    _region = None
    _region_data = None
    _mouse_entered_view = False
    _fit = None
    _end_planes = None
    _stage = 'SELECT'
    _numeric_input = ""
    _numeric_reference = ""
    _numeric_replace = True
    _drag_active = False
    _drag_moved = False
    _drag_start_x = 0
    _drag_start_diameter = 0.0
    _gizmo_rect = None

    @classmethod
    def poll(cls, context):
        """Allow invocation only for a mesh in Edit Mode inside a 3D viewport."""
        obj = context.active_object
        return (
            obj is not None
            and obj.type == 'MESH'
            and context.mode == 'EDIT_MESH'
            and context.area is not None
            and context.area.type == 'VIEW_3D'
        )

    def draw(self, context):
        """Draw the Redo-panel UI and initially activate its diameter field."""
        row = self.layout.row()
        row.activate_init = True
        row.prop(self, "diameter")

    @staticmethod
    def _tag_redraw(context):
        """Request another draw from the current area when one is available."""
        if context.area is not None:
            context.area.tag_redraw()

    def _clear_batches(self, context):
        """Discard all preview geometry and request a viewport redraw."""
        self._fill_batch = None
        self._line_batch = None
        self._fill_coords = None
        self._line_coords = None
        self._tag_redraw(context)

    def _finish_modal(self, context):
        """Release transient modal resources and restore viewport UI state."""
        global _modal_active
        self._remove_draw_handler()
        self._clear_batches(context)
        self._bvh = None
        self._object = None
        self._region = None
        self._region_data = None
        self._mouse_entered_view = False
        _modal_active = False
        context.workspace.status_text_set(None)
        context.window.cursor_modal_restore()

    def _remove_draw_handler(self):
        """Idempotently unregister this operator's View3D draw handler."""
        if self._draw_handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            except ValueError:
                pass
            _draw_handles.discard(self._draw_handle)
            self._draw_handle = None

    def _end_modal_with_persistent_preview(self, context):
        """Leave modal interaction while preserving its cylinder overlay.

        The live draw handler is removed before its plain coordinate data is
        copied into a new persistent preview. The caller then executes and
        completes the Blender operator, making its properties available to Redo.
        """
        global _modal_active, _persistent_preview
        self._bvh = None
        self._mouse_entered_view = False
        _modal_active = False
        self._remove_draw_handler()
        if _persistent_preview is not None:
            _persistent_preview.remove()
        _persistent_preview = _PersistentPreview(self, context)
        context.workspace.status_text_set(None)
        context.window.cursor_modal_restore()

    def _restore_initial_selection(self):
        """Restore the face selection captured when the modal session began."""
        if self._object is None or self._initial_selection is None:
            return
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        for face in bm.faces:
            face.select = face.index in self._initial_selection
        bmesh.update_edit_mesh(self._object.data, destructive=False)

    def _restore_original_geometry(self):
        """Restore captured object-local vertex positions after cancellation."""
        if self._object is None or not self._original_vertex_coords:
            return
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.verts.ensure_lookup_table()
        for index, coordinate in self._original_vertex_coords.items():
            if index < len(bm.verts):
                bm.verts[index].co = coordinate
        bmesh.update_edit_mesh(self._object.data, loop_triangles=True, destructive=False)

    def _set_status_text(self, context):
        """Describe controls and current values for the active modal stage."""
        if self._stage == 'SELECT':
            angle = math.degrees(self.flood_angle)
            count = len(self._candidate_indices or ())
            candidate_status = (
                f"Flood: {count} face{'s' if count != 1 else ''}"
                if count
                else "No face under cursor"
            )
            context.workspace.status_text_set(
                f"Holey Tools {ADDON_VERSION}  •  {candidate_status}  •  "
                f"[ / ] Flood Angle: {angle:.1f}°  •  "
                "Click: apply and adjust  •  Esc: cancel"
            )
        else:
            context.workspace.status_text_set(
                f"Diameter: {self._numeric_input}  •  Type to replace  •  "
                "Drag handle: adjust  •  Ctrl: invert 0.1 mm snap  •  Shift: fine  •  "
                "MMB/Wheel: navigate  •  "
                "Click/Enter: apply  •  Esc: cancel"
            )

    def _rebuild_hover(self, context):
        """Repeat flood filling from the current seed after an angle change."""
        if self._hover_face_index < 0:
            return
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        if self._hover_face_index >= len(bm.faces):
            return
        faces = self._flood_fill(bm.faces[self._hover_face_index])
        self._candidate_indices = {face.index for face in faces}
        self._build_selection_preview(context, faces)
        self._set_status_text(context)

    def _set_hover_seed(self, context, face):
        """Flood from ``face`` and publish the resulting selection preview."""
        faces = self._flood_fill(face)
        self._hover_face_index = face.index
        self._candidate_indices = {candidate.index for candidate in faces}
        self._build_selection_preview(context, faces)
        self._set_status_text(context)

    def _change_flood_angle(self, context, direction, fine_adjustment=False):
        """Adjust the flood threshold, clamp it, and rebuild the hover preview.

        Normal bracket presses move by five degrees; Shift requests a one-degree
        step for finer control.
        """
        step = math.radians(1.0 if fine_adjustment else 5.0)
        minimum = math.radians(0.1)
        maximum = math.radians(89.9)
        self.flood_angle = min(maximum, max(minimum, self.flood_angle + direction * step))
        self._rebuild_hover(context)
        self._set_status_text(context)

    def _update_hover(self, context, event):
        """Ray-cast the pointer and update the non-destructive face flood.

        Window coordinates are converted explicitly to the View3D WINDOW region,
        then the world-space view ray is transformed into object-local space for
        the BMesh BVH. Leaving the viewport clears an established hover, while an
        initial selected-face preview remains until the pointer first enters it.
        """
        if self._stage != 'SELECT' or self._object.mode != 'EDIT':
            return

        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        if self._bvh is None:
            self._bvh = BVHTree.FromBMesh(bm)

        # Modal operators launched from the N-panel retain that UI region in
        # context. Convert window coordinates explicitly into the View3D WINDOW
        # region so hover picking works immediately after leaving the sidebar.
        mouse = (
            event.mouse_x - self._region.x,
            event.mouse_y - self._region.y,
        )
        if not (0 <= mouse[0] < self._region.width and 0 <= mouse[1] < self._region.height):
            # Invoking from the sidebar begins outside the WINDOW region. Keep an
            # initial selected-face preview until the cursor has actually entered
            # the viewport; after that, leaving the viewport clears hover normally.
            if self._mouse_entered_view:
                self._hover_face_index = -1
                self._candidate_indices = set()
                self._clear_batches(context)
                self._set_status_text(context)
            return
        self._mouse_entered_view = True
        origin_world = view3d_utils.region_2d_to_origin_3d(
            self._region, self._region_data, mouse
        )
        direction_world = view3d_utils.region_2d_to_vector_3d(
            self._region, self._region_data, mouse
        )

        inverse = self._object.matrix_world.inverted_safe()
        origin_local = inverse @ origin_world
        direction_local = (inverse.to_3x3() @ direction_world).normalized()
        _location, _normal, face_index, _distance = self._bvh.ray_cast(
            origin_local, direction_local
        )

        if face_index is None or face_index < 0 or face_index >= len(bm.faces):
            self._hover_face_index = -1
            self._candidate_indices = set()
            self._clear_batches(context)
            self._set_status_text(context)
            return

        face = bm.faces[face_index]
        if face.hide:
            self._hover_face_index = -1
            self._candidate_indices = set()
            self._clear_batches(context)
            self._set_status_text(context)
            return

        if face.index == self._hover_face_index:
            return

        self._set_hover_seed(context, face)

    def _store_fit_for_redo(self):
        """Serialize the fitted cylinder and end planes into RNA properties.

        Blender preserves operator properties for undo and Redo-panel execution,
        but it does not preserve arbitrary Python attributes reliably. Vectors
        use hidden RNA vector properties and variable-length plane data is stored
        as compact JSON.
        """
        fit = self._fit
        self.scale = 1.0
        self.diameter = fit["radius"] * 2.0
        self.fit_valid = True
        self.fit_face_indices = ",".join(str(index) for index in sorted(self._locked_indices))
        self.fit_axis = fit["axis"]
        self.fit_basis_x = fit["basis_x"]
        self.fit_basis_y = fit["basis_y"]
        self.fit_center = fit["center"]
        self.fit_radius = fit["radius"]
        self.fit_z_min = fit["z_min"]
        self.fit_z_max = fit["z_max"]
        self.fit_end_planes = json.dumps(
            [
                {
                    "verts": plane["verts"],
                    "point": list(plane["point"]),
                    "normal": list(plane["normal"]),
                }
                for plane in (self._end_planes or ())
            ],
            separators=(",", ":"),
        )

    def _restore_fit_for_redo(self):
        """Reconstruct transient fit state from hidden RNA redo properties.

        Returns ``True`` when the required face and cylinder data was available.
        Malformed optional boundary-plane JSON is treated as an empty plane list
        so the core resize operation can still run.
        """
        if not self.fit_valid or not self.fit_face_indices:
            return False
        try:
            self._locked_indices = {
                int(index) for index in self.fit_face_indices.split(",") if index
            }
        except ValueError:
            return False
        self._fit = {
            "axis": Vector(self.fit_axis),
            "basis_x": Vector(self.fit_basis_x),
            "basis_y": Vector(self.fit_basis_y),
            "center": Vector(self.fit_center),
            "radius": self.fit_radius,
            "z_min": self.fit_z_min,
            "z_max": self.fit_z_max,
        }
        try:
            self._end_planes = [
                {
                    "verts": plane["verts"],
                    "point": Vector(plane["point"]),
                    "normal": Vector(plane["normal"]),
                }
                for plane in json.loads(self.fit_end_planes or "[]")
            ]
        except (TypeError, ValueError, KeyError):
            self._end_planes = []
        self._stage = 'CYLINDER'
        return True

    @staticmethod
    def _scene_unit_info(context):
        """Return the scene unit system and a safe Blender-to-display scale."""
        settings = context.scene.unit_settings
        system = settings.system
        scale = settings.scale_length if system != 'NONE' else 1.0
        return system, max(scale, 1.0e-12)

    def _format_diameter(self, context):
        """Format the internal world-space diameter using the scene's units."""
        system, scale = self._scene_unit_info(context)
        if system == 'NONE':
            return f"{self.diameter:.6g}"
        return bpy.utils.units.to_string(
            system,
            'LENGTH',
            self.diameter * scale,
            precision=6,
            split_unit=False,
            compatible_unit=True,
        )

    def _parse_diameter(self, context, text):
        """Parse user-entered dimension text into Blender world-space units.

        Explicit unit suffixes and the previous formatted unit are handled by
        Blender's unit parser. Unitless scenes accept a plain floating-point
        value, and a decimal comma is normalized for convenience.
        """
        system, scale = self._scene_unit_info(context)
        normalized = text.strip().replace(',', '.')
        if not normalized:
            raise ValueError("Empty dimension")
        if system == 'NONE':
            return float(normalized)
        value = bpy.utils.units.to_value(
            system,
            'LENGTH',
            normalized,
            str_ref_unit=self._numeric_reference,
        )
        return value / scale

    def _begin_numeric_input(self, context):
        """Seed direct entry from the fitted diameter and select it for replacement."""
        self._numeric_reference = self._format_diameter(context)
        self._numeric_input = self._numeric_reference
        self._numeric_replace = True
        self._set_status_text(context)

    def _update_numeric_preview(self, context):
        """Apply valid positive input immediately to the mesh and GPU cylinder.

        Incomplete or invalid text remains visible for continued editing but does
        not disturb the last valid geometry.
        """
        try:
            value = self._parse_diameter(context, self._numeric_input)
        except (TypeError, ValueError):
            self._set_status_text(context)
            return
        if value <= 0.0:
            self._set_status_text(context)
            return
        self.diameter = value
        self._build_cylinder_preview(context)
        self._apply_diameter_to_mesh(context)
        self._set_status_text(context)

    def _handle_numeric_event(self, context, event):
        """Consume keyboard events that edit the live diameter expression.

        Ctrl-A selects the whole logical field, Backspace removes input, and
        printable number/unit characters trigger immediate parsing and resizing.
        Returns whether the modal event was consumed.
        """
        if event.value != 'PRESS':
            return False

        if event.ctrl and event.type == 'A':
            self._numeric_replace = True
            self._set_status_text(context)
            return True

        if event.type == 'BACK_SPACE':
            if self._numeric_replace:
                self._numeric_input = ""
                self._numeric_replace = False
            else:
                self._numeric_input = self._numeric_input[:-1]
            self._update_numeric_preview(context)
            return True

        character = event.ascii
        if not character and event.type.startswith('NUMPAD_'):
            suffix = event.type.removeprefix('NUMPAD_')
            if suffix.isdigit() and len(suffix) == 1:
                character = suffix
            elif suffix == 'PERIOD':
                character = '.'

        if not character or not character.isprintable() or event.ctrl or event.alt:
            return False

        if self._numeric_replace:
            if character not in "0123456789.,":
                return False
            self._numeric_input = character
            self._numeric_replace = False
        else:
            allowed = "0123456789.,abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ \'\"/"
            if character not in allowed:
                return False
            self._numeric_input += character

        self._update_numeric_preview(context)
        return True

    def _begin_diameter_drag(self, event):
        """Capture the pointer and diameter that anchor a drag adjustment."""
        self._drag_active = True
        self._drag_moved = False
        self._drag_start_x = event.mouse_x
        self._drag_start_diameter = self.diameter

    def _update_diameter_drag(self, context, event):
        """Convert horizontal drag distance into a live exponential resize.

        Shift lowers sensitivity. Snapping follows Blender's global snap toggle,
        while Ctrl temporarily inverts that setting, and uses physical 0.1 mm
        increments independently of the scene's displayed unit scale.
        """
        delta = event.mouse_x - self._drag_start_x
        if abs(delta) >= 3:
            self._drag_moved = True
        divisor = 1000.0 if event.shift else 200.0
        diameter = self._drag_start_diameter * (2.0 ** (delta / divisor))
        snap_enabled = bool(context.scene.tool_settings.use_snap) != bool(event.ctrl)
        if snap_enabled:
            # Blender stores length in scene units. Snap the physical result to
            # tenths of a millimeter regardless of the displayed unit system.
            _system, scale = self._scene_unit_info(context)
            snap_step = 0.0001 / scale
            diameter = round(diameter / snap_step) * snap_step
            diameter = max(snap_step, diameter)
        self.diameter = max(1.0e-9, diameter)
        self._numeric_reference = self._format_diameter(context)
        self._numeric_input = self._numeric_reference
        self._numeric_replace = True
        self._build_cylinder_preview(context)
        self._apply_diameter_to_mesh(context)
        self._set_status_text(context)

    def _lock_selection_and_fit(self, context, build_preview=True):
        """Commit the hover region and initialize diameter-editing state.

        The method fits the cylinder and boundary planes, makes the candidate
        faces Blender's real selection, captures original/proportional vertex
        state, serializes the fit for Redo, and enters CYLINDER. It returns
        ``False`` without changing stages when no stable fit can be produced.
        """
        if not self._candidate_indices:
            return False

        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        faces = [bm.faces[index] for index in sorted(self._candidate_indices)]
        try:
            fit = self._fit_cylinder(faces)
        except (ValueError, np.linalg.LinAlgError) as error:
            self.report({'WARNING'}, str(error))
            return False

        for face in bm.faces:
            face.select = face.index in self._candidate_indices
        bmesh.update_edit_mesh(self._object.data, destructive=False)

        self._locked_indices = set(self._candidate_indices)
        self._fit = fit
        self._end_planes = self._fit_boundary_planes(faces, fit["axis"])
        selected_verts = {vert for face in faces for vert in face.verts}
        self._capture_deformation_state(context, bm, selected_verts)
        self._store_fit_for_redo()
        self._stage = 'CYLINDER'
        self._begin_numeric_input(context)
        if build_preview:
            self._build_cylinder_preview(context)
        self._set_status_text(context)
        return True

    def execute(self, context):
        """Apply the stored diameter during confirmation or Redo execution.

        Transient fit state is restored from RNA properties when necessary. If a
        persistent overlay exists, its geometry and label are refreshed from the
        newly executed value without retaining this operator wrapper.
        """
        global _persistent_preview
        if (self._fit is None or not self._locked_indices) and not self._restore_fit_for_redo():
            self.report({'WARNING'}, "Choose a cylindrical wall first")
            return {'CANCELLED'}

        MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(self, context)

        # Rebuild on the current operator wrapper, then copy only plain Python GPU
        # data into the persistent preview. No RNA operator survives past redo.
        if _persistent_preview is not None:
            # Redo-panel edits update the RNA diameter property directly rather
            # than passing through modal numeric input. Keep the overlay text in
            # sync with the value Blender just executed.
            self._numeric_input = self._format_diameter(context)
            self._numeric_reference = self._numeric_input
            self._build_cylinder_preview(context)
            _persistent_preview.update_from(self, context)
        return {'FINISHED'}

    def modal(self, context, event):
        """Route View3D events through selection and cylinder-editing stages.

        Escape restores geometry and selection, viewport-navigation events pass
        through to Blender, and confirmation transfers the overlay before normal
        operator execution. Pointer dragging can begin with the selecting click
        or with any later press on the on-screen gizmo.
        """
        if context.area is None or context.area.type != 'VIEW_3D':
            self._restore_original_geometry()
            self._restore_initial_selection()
            self._finish_modal(context)
            return {'CANCELLED'}

        if event.type == 'MOUSEMOVE':
            if self._stage == 'SELECT':
                self._update_hover(context, event)
            elif self._stage == 'CYLINDER' and self._drag_active:
                self._update_diameter_drag(context, event)
            elif self._stage == 'CYLINDER':
                cursor = (
                    'SCROLL_X'
                    if self._event_over_drag_gizmo(event)
                    else 'CROSSHAIR'
                )
                context.window.cursor_modal_set(cursor)

        elif (
            self._stage == 'SELECT'
            and event.type in {'LEFT_BRACKET', 'RIGHT_BRACKET'}
            and event.value == 'PRESS'
        ):
            direction = -1 if event.type == 'LEFT_BRACKET' else 1
            self._change_flood_angle(context, direction, event.shift)
            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._restore_original_geometry()
            self._restore_initial_selection()
            self._finish_modal(context)
            return {'CANCELLED'}

        elif (
            self._stage == 'CYLINDER'
            and event.type == 'LEFTMOUSE'
            and event.value == 'RELEASE'
            and self._drag_active
        ):
            self._drag_active = False
            return {'RUNNING_MODAL'}

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._stage == 'SELECT':
                # A click can arrive without a preceding mouse-move event (notably
                # just after invoking from the sidebar), so pick once at click time.
                self._update_hover(context, event)
                if not self._lock_selection_and_fit(context, build_preview=True):
                    return {'RUNNING_MODAL'}
                self._begin_diameter_drag(event)
                return {'RUNNING_MODAL'}
            if self._event_over_drag_gizmo(event):
                self._begin_diameter_drag(event)
                return {'RUNNING_MODAL'}
            self._end_modal_with_persistent_preview(context)
            return self.execute(context)

        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if self._stage == 'CYLINDER':
                self._end_modal_with_persistent_preview(context)
                return self.execute(context)

        elif self._stage == 'CYLINDER' and self._handle_numeric_event(context, event):
            return {'RUNNING_MODAL'}

        if event.type in {
            'MIDDLEMOUSE',
            'WHEELUPMOUSE',
            'WHEELDOWNMOUSE',
            'WHEELINMOUSE',
            'WHEELOUTMOUSE',
        }:
            return {'PASS_THROUGH'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        """Initialize modal state, drawing, picking, and the first hover preview.

        Every invocation starts at scale one with no fitted diameter. Existing
        persistent drawing is retired, source selection is captured, and the
        View3D WINDOW region is saved because panel invocation supplies a sidebar
        context. A selected face seeds an immediate preview when available.
        """
        global _modal_active, _persistent_preview
        if not self.poll(context):
            self.report({'WARNING'}, "Use Resize Hole in Mesh Edit Mode")
            return {'CANCELLED'}

        self._object = context.active_object

        # A prior fitted preview belongs to the previous last operation.
        if _persistent_preview is not None:
            _persistent_preview.remove()
            _persistent_preview = None
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        self._bvh = BVHTree.FromBMesh(bm)
        self._initial_selection = {face.index for face in bm.faces if face.select}
        self._original_vertex_coords = None
        self._proportional_weights = None
        self._candidate_indices = set()
        self._locked_indices = set()
        self._hover_face_index = -1
        self._fit = None
        self._end_planes = None
        self.scale = 1.0
        self.diameter = 0.0
        self.fit_valid = False
        self.fit_face_indices = ""
        self.fit_end_planes = ""
        self._stage = 'SELECT'
        self._numeric_input = ""
        self._numeric_reference = ""
        self._numeric_replace = True
        self._drag_active = False
        self._drag_moved = False
        self._gizmo_rect = None
        self._mouse_entered_view = False
        _modal_active = True

        self._region = next(
            (region for region in context.area.regions if region.type == 'WINDOW'),
            None,
        )
        self._region_data = context.space_data.region_3d
        if self._region is None or self._region_data is None:
            _modal_active = False
            self.report({'WARNING'}, "Could not find the active 3D viewport")
            return {'CANCELLED'}

        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_preview, (), 'WINDOW', 'POST_PIXEL'
        )
        _draw_handles.add(self._draw_handle)
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)

        # A panel button cannot provide a mesh-space hover coordinate. Seed the
        # first GPU preview from the active selected face (or any selected face),
        # then let mouse hover take over once it enters the viewport.
        selected_faces = [face for face in bm.faces if face.select and not face.hide]
        active_face = bm.faces.active
        if active_face is not None and active_face in selected_faces:
            self._set_hover_seed(context, active_face)
        elif selected_faces:
            self._set_hover_seed(context, selected_faces[0])

        self._update_hover(context, event)
        self._set_status_text(context)
        return {'RUNNING_MODAL'}


def cleanup_runtime():
    """Remove every runtime draw handler during extension unregistration.

    Tracking handlers independently of operator RNA objects makes cleanup safe
    after script reloads and after Blender has invalidated completed operators.
    """
    global _modal_active, _persistent_preview
    if _persistent_preview is not None:
        _persistent_preview.remove()
        _persistent_preview = None
    for handle in tuple(_draw_handles):
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
        except ValueError:
            pass
        _draw_handles.discard(handle)
    _modal_active = False
