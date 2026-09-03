# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Modal face-group extraction for the generated axis Geometry Nodes group."""

import math
import time

import bmesh
import bpy

from bpy_extras import view3d_utils
from mathutils.bvhtree import BVHTree

from . import operator as resize_runtime
from .axis_nodes import ensure_axis_node_group
from .constants import (
    ADDON_VERSION,
    AXIS_FACE_ATTRIBUTE,
    FLOOD_ANGLE_HUD_SECONDS,
)
from .geometry import GeometryMixin
from .gpu_preview import GpuPreviewMixin


def assign_axis_face_group(obj, face_indices, attribute_name):
    """Replace an integer face-domain group with ``face_indices``.

    Integer storage is used because BMesh exposes it directly in Edit Mode.
    Geometry Nodes' Boolean Named Attribute socket converts its zero/one values
    to a selection without requiring a mode switch or a duplicate mesh.
    """
    attribute_name = attribute_name.strip()
    if not attribute_name:
        raise ValueError("Face Group cannot be empty")

    mesh = obj.data
    existing = mesh.attributes.get(attribute_name)
    if existing is not None and (
        existing.domain != 'FACE' or existing.data_type not in {'INT', 'BOOLEAN'}
    ):
        raise ValueError(
            f"Attribute '{attribute_name}' already exists and is not a face group"
        )

    selected = set(face_indices)
    if obj.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(mesh)
        bm.faces.ensure_lookup_table()
        layer = bm.faces.layers.int.get(attribute_name)
        if layer is None:
            if existing is not None:
                raise ValueError(
                    f"Attribute '{attribute_name}' cannot be edited as an integer face group"
                )
            layer = bm.faces.layers.int.new(attribute_name)
        for face in bm.faces:
            is_selected = face.index in selected
            face[layer] = int(is_selected)
            face.select = is_selected
        bmesh.update_edit_mesh(mesh, destructive=False)
        return

    if existing is None:
        existing = mesh.attributes.new(
            name=attribute_name, type='INT', domain='FACE'
        )
    for polygon, value in zip(mesh.polygons, existing.data):
        is_selected = polygon.index in selected
        value.value = int(is_selected)
        polygon.select = is_selected
    mesh.update()


class MESH_OT_holey_tools_extract_axis(
    GeometryMixin,
    GpuPreviewMixin,
    bpy.types.Operator,
):
    """Flood-fill faces and publish them for Geometry Nodes axis extraction."""

    bl_idname = "mesh.holey_tools_extract_axis"
    bl_label = "Extract Axis"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}
    bl_description = (
        "Flood-fill a region, store it as a named face group, and create a "
        "Geometry Nodes principal-axis group"
    )

    flood_angle: bpy.props.FloatProperty(
        name="Flood Angle",
        description="Maximum angle between neighboring faces in the hover flood fill",
        subtype='ANGLE',
        default=math.radians(60.0),
        min=math.radians(0.1),
        max=math.radians(89.9),
    )
    attribute_name: bpy.props.StringProperty(
        name="Face Group",
        description="Face-domain named attribute read by the generated node group",
        default=AXIS_FACE_ATTRIBUTE,
    )
    face_indices: bpy.props.StringProperty(
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
    _initial_selection = None
    _object = None
    _region = None
    _region_data = None
    _mouse_entered_view = False
    _stage = 'SELECT'
    _flood_angle_hud_state = None
    _mouse_position = None
    _angle_numeric_input = ""
    _angle_numeric_replace = True

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

    def draw(self, _context):
        """Expose the reusable face-group name in Adjust Last Operation."""
        self.layout.prop(self, "attribute_name")

    @staticmethod
    def _tag_redraw(context):
        if context.area is not None:
            context.area.tag_redraw()

    def _clear_batches(self, context):
        self._fill_batch = None
        self._line_batch = None
        self._fill_coords = None
        self._line_coords = None
        self._tag_redraw(context)

    def _remove_draw_handler(self):
        if self._draw_handle is None:
            return
        try:
            bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
        except ValueError:
            pass
        resize_runtime._draw_handles.discard(self._draw_handle)
        self._draw_handle = None

    def _finish_modal(self, context):
        self._remove_draw_handler()
        self._clear_batches(context)
        self._bvh = None
        self._object = None
        self._region = None
        self._region_data = None
        self._mouse_entered_view = False
        resize_runtime._modal_active = False
        context.workspace.status_text_set(None)
        context.window.cursor_modal_restore()

    def _restore_initial_selection(self):
        if self._object is None or self._initial_selection is None:
            return
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        for face in bm.faces:
            face.select = face.index in self._initial_selection
        bmesh.update_edit_mesh(self._object.data, destructive=False)

    def _set_status_text(self, context):
        angle = math.degrees(self.flood_angle)
        count = len(self._candidate_indices or ())
        candidate = (
            f"Flood: {count} face{'s' if count != 1 else ''}"
            if count
            else "No face under cursor"
        )

        def draw_status(header, _context):
            row = header.layout.row(align=True)
            row.alignment = 'LEFT'
            row.label(text=f"Holey Tools {ADDON_VERSION}  •  Extract Axis  •  {candidate}  •")
            row.label(text="", icon='EVENT_LEFTBRACKET')
            row.label(text="/")
            row.label(text="", icon='EVENT_RIGHTBRACKET')
            row.label(
                text=f"Flood Angle: {angle:.1f}°  •  "
                "Type: set angle  •  Click/Enter: create face group  •  Esc: cancel"
            )

        context.workspace.status_text_set(draw_status)

    def _set_hover_seed(self, context, face):
        faces = self._flood_fill(face)
        self._hover_face_index = face.index
        self._candidate_indices = {candidate.index for candidate in faces}
        self._build_selection_preview(context, faces)
        self._set_status_text(context)

    def _rebuild_hover(self, context):
        if self._hover_face_index < 0:
            return
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        if self._hover_face_index >= len(bm.faces):
            return
        self._set_hover_seed(context, bm.faces[self._hover_face_index])

    def _show_flood_angle_hud(self, context):
        state = self._flood_angle_hud_state
        state["until"] = time.monotonic() + FLOOD_ANGLE_HUD_SECONDS
        self._tag_redraw(context)
        if state["timer_running"]:
            return
        state["timer_running"] = True
        area = context.area

        def expire():
            remaining = state["until"] - time.monotonic()
            if remaining > 0.0:
                return remaining
            state["timer_running"] = False
            try:
                if area is not None:
                    area.tag_redraw()
            except ReferenceError:
                pass
            return None

        bpy.app.timers.register(expire, first_interval=FLOOD_ANGLE_HUD_SECONDS)

    def _change_flood_angle(self, context, direction, fine_adjustment=False):
        step = math.radians(1.0 if fine_adjustment else 5.0)
        self.flood_angle = min(
            math.radians(89.9),
            max(math.radians(0.1), self.flood_angle + direction * step),
        )
        self._angle_numeric_input = f"{math.degrees(self.flood_angle):.1f}"
        self._angle_numeric_replace = True
        context.scene.holey_tools_flood_angle = self.flood_angle
        self._show_flood_angle_hud(context)
        self._rebuild_hover(context)

    def _handle_flood_angle_numeric_event(self, context, event):
        if event.value != 'PRESS':
            return False
        if event.ctrl and event.type == 'A':
            self._angle_numeric_replace = True
            return True
        if event.type == 'BACK_SPACE':
            if self._angle_numeric_replace:
                self._angle_numeric_input = ""
                self._angle_numeric_replace = False
            else:
                self._angle_numeric_input = self._angle_numeric_input[:-1]
            self._update_flood_angle_numeric_preview(context)
            return True

        character = event.ascii
        if not character and event.type.startswith('NUMPAD_'):
            suffix = event.type.removeprefix('NUMPAD_')
            if suffix.isdigit() and len(suffix) == 1:
                character = suffix
            elif suffix == 'PERIOD':
                character = '.'
        if not character or character not in "0123456789.," or event.ctrl or event.alt:
            return False
        if self._angle_numeric_replace:
            self._angle_numeric_input = character
            self._angle_numeric_replace = False
        else:
            self._angle_numeric_input += character
        self._update_flood_angle_numeric_preview(context)
        return True

    def _update_flood_angle_numeric_preview(self, context):
        try:
            degrees = float(self._angle_numeric_input.replace(',', '.'))
        except ValueError:
            self._set_status_text(context)
            return
        degrees = min(89.9, max(0.1, degrees))
        self.flood_angle = math.radians(degrees)
        context.scene.holey_tools_flood_angle = self.flood_angle
        self._show_flood_angle_hud(context)
        self._rebuild_hover(context)

    def _update_hover(self, context, event):
        if self._object.mode != 'EDIT':
            return
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        if self._bvh is None:
            self._bvh = BVHTree.FromBMesh(bm)
        mouse = (
            event.mouse_x - self._region.x,
            event.mouse_y - self._region.y,
        )
        if not (
            0 <= mouse[0] < self._region.width
            and 0 <= mouse[1] < self._region.height
        ):
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
        if (
            face_index is None
            or face_index < 0
            or face_index >= len(bm.faces)
            or bm.faces[face_index].hide
        ):
            self._hover_face_index = -1
            self._candidate_indices = set()
            self._clear_batches(context)
            self._set_status_text(context)
            return
        if face_index != self._hover_face_index:
            self._set_hover_seed(context, bm.faces[face_index])

    def _confirm_candidate(self, context):
        if not self._candidate_indices:
            self.report({'WARNING'}, "Choose a face region first")
            return {'RUNNING_MODAL'}
        self.face_indices = ",".join(
            str(index) for index in sorted(self._candidate_indices)
        )
        self._finish_modal(context)
        return self.execute(context)

    def execute(self, context):
        """Write the face group and ensure the principal-axis node group exists."""
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            self.report({'WARNING'}, "Extract Axis needs an active mesh")
            return {'CANCELLED'}
        try:
            indices = {
                int(value) for value in self.face_indices.split(',') if value
            }
        except ValueError:
            self.report({'WARNING'}, "Stored face selection is invalid")
            return {'CANCELLED'}
        if not indices:
            self.report({'WARNING'}, "Choose a face region first")
            return {'CANCELLED'}
        try:
            assign_axis_face_group(obj, indices, self.attribute_name)
            node_group = ensure_axis_node_group(self.attribute_name.strip())
        except (RuntimeError, TypeError, ValueError) as error:
            self.report({'WARNING'}, str(error))
            return {'CANCELLED'}
        self.report(
            {'INFO'},
            f"Stored '{self.attribute_name.strip()}' for node group '{node_group.name}'",
        )
        return {'FINISHED'}

    def modal(self, context, event):
        """Run the same hover, flood-angle, navigation, and cancel workflow."""
        if context.area is None or context.area.type != 'VIEW_3D':
            self._restore_initial_selection()
            self._finish_modal(context)
            return {'CANCELLED'}

        if hasattr(event, "mouse_x"):
            self._mouse_position = (
                event.mouse_x - self._region.x,
                event.mouse_y - self._region.y,
            )
            state = self._flood_angle_hud_state
            if (
                event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}
                and state
                and time.monotonic() < state["until"]
            ):
                self._tag_redraw(context)

        if resize_runtime._is_view_navigation_event(event.type) or event.type == 'Z':
            return {'PASS_THROUGH'}
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._update_hover(context, event)
        elif (
            event.type in {'LEFT_BRACKET', 'RIGHT_BRACKET'}
            and event.value == 'PRESS'
        ):
            direction = -1 if event.type == 'LEFT_BRACKET' else 1
            self._change_flood_angle(context, direction, event.shift)
            return {'RUNNING_MODAL'}
        elif self._handle_flood_angle_numeric_event(context, event):
            return {'RUNNING_MODAL'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._restore_initial_selection()
            self._finish_modal(context)
            return {'CANCELLED'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._update_hover(context, event)
            return self._confirm_candidate(context)
        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            return self._confirm_candidate(context)
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        """Initialize picking and seed the preview from an existing selection."""
        if not self.poll(context):
            self.report({'WARNING'}, "Use Extract Axis in Mesh Edit Mode")
            return {'CANCELLED'}
        self._object = context.active_object
        if resize_runtime._persistent_preview is not None:
            resize_runtime._persistent_preview.remove()
            resize_runtime._persistent_preview = None
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        self._bvh = BVHTree.FromBMesh(bm)
        self._initial_selection = {face.index for face in bm.faces if face.select}
        self._candidate_indices = set()
        self._hover_face_index = -1
        self._stage = 'SELECT'
        self._angle_numeric_input = f"{math.degrees(self.flood_angle):.1f}"
        self._angle_numeric_replace = True
        self._mouse_entered_view = False
        self._flood_angle_hud_state = {"until": 0.0, "timer_running": False}
        resize_runtime._modal_active = True

        self._region = next(
            (region for region in context.area.regions if region.type == 'WINDOW'),
            None,
        )
        self._region_data = context.space_data.region_3d
        if self._region is None or self._region_data is None:
            resize_runtime._modal_active = False
            self.report({'WARNING'}, "Could not find the active 3D viewport")
            return {'CANCELLED'}
        self._mouse_position = (
            event.mouse_x - self._region.x,
            event.mouse_y - self._region.y,
        )
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_preview, (), 'WINDOW', 'POST_PIXEL'
        )
        resize_runtime._draw_handles.add(self._draw_handle)
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)

        selected_faces = [face for face in bm.faces if face.select and not face.hide]
        active_face = bm.faces.active
        if active_face is not None and active_face in selected_faces:
            self._set_hover_seed(context, active_face)
        elif selected_faces:
            self._set_hover_seed(context, selected_faces[0])
        self._update_hover(context, event)
        self._set_status_text(context)
        return {'RUNNING_MODAL'}
