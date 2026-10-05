# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Modal non-manifold loop selection and adaptive hole fairing operator."""

import bmesh
import bpy

from bpy_extras import view3d_utils
from mathutils import Vector

from . import operator as resize_runtime
from .constants import ADDON_VERSION
from .fairing import (
    FairingError,
    _limited_dissolve_analytic_boundary,
    _limited_dissolve_cylinder_patch,
    boundary_component,
    choose_hypothesis,
    fair_boundary_loop,
    finalize_fair_patch,
    ordered_boundary_loop,
    prepare_fair_boundary,
)
from .geometry import GeometryMixin
from .gpu_preview import GpuPreviewMixin, _ui_scale


_METHOD_LABELS = {
    'PLANES': "N-plane intersection",
    'CONIC': "circular cone/cylinder",
    'SPHERE': "sphere",
    'ORGANIC': "organic curvature",
}

_FILL_LABELS = {
    'PLANE': "center fan",
    'CYLINDER': "axial collapse",
    'GRID': "grid",
    'CIRCLE': "circular rings",
}


class MESH_OT_holey_tools_fair_hole(
    GeometryMixin,
    GpuPreviewMixin,
    bpy.types.Operator,
):
    """Select one non-manifold boundary loop and fill it adaptively."""

    bl_idname = "mesh.holey_tools_fair_hole"
    bl_label = "Fair Hole"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}
    bl_description = (
        "Select a non-manifold edge loop and fill it using plane, conic, "
        "sphere, or organic curvature fairing"
    )

    edge_indices: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    fairing_method: bpy.props.EnumProperty(
        name="Surface",
        description="Surface model used to fair the selected opening",
        items=(
            ('AUTO', "Auto", "Choose the lowest-complexity accepted fit"),
            ('PLANES', "N-Plane", "Use the adjacent planar regions"),
            (
                'CYLINDER',
                "Cylinder",
                "Force a cylindrical fit even when its residual is not lowest",
            ),
            (
                'SPHERE',
                "Sphere",
                "Force a spherical fit even when its residual is not lowest",
            ),
            (
                'ORGANIC',
                "Organic",
                "Use the thin-plate surface continuation",
            ),
        ),
        default='AUTO',
        options={'SKIP_SAVE'},
    )
    fill_style: bpy.props.EnumProperty(
        name="Fill",
        description="Topology used inside the selected opening",
        items=(
            (
                'AUTO',
                "Auto",
                "Use the topology associated with the surface fit",
            ),
            ('GRID', "Grid", "Clip a source-aligned grid to the opening"),
            ('CIRCLE', "Circle", "Use concentric rings and a center fan"),
        ),
        default='AUTO',
        options={'SKIP_SAVE'},
    )
    circle_loops: bpy.props.IntProperty(
        name="Circle Loops",
        description="Number of concentric quad loops before the center fan",
        default=2,
        min=0,
        max=64,
        options={'SKIP_SAVE'},
    )
    grid_angle: bpy.props.FloatProperty(
        name="Grid Angle Offset",
        description="Rotate the grid relative to its inferred source alignment",
        default=0.0,
        subtype='ANGLE',
        soft_min=-1.5707963267948966,
        soft_max=1.5707963267948966,
        options={'SKIP_SAVE'},
    )
    final_simplify: bpy.props.BoolProperty(
        name="Final Simplify",
        description=(
            "Expand the completed patch selection and run Limited Dissolve"
        ),
        default=True,
        options={'SKIP_SAVE'},
    )
    result_method: bpy.props.StringProperty(
        name="Method", default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    result_fill_style: bpy.props.StringProperty(
        name="Fill Result", default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    result_residual: bpy.props.FloatProperty(
        name="Fit Residual", default=0.0, options={'HIDDEN', 'SKIP_SAVE'}
    )

    _draw_handle = None
    _fill_batch = None
    _line_batch = None
    _fill_coords = None
    _line_coords = None
    _draw_error_reported = False
    _candidate_indices = None
    _candidate_valid = False
    _hover_edge_index = -1
    _initial_selection = None
    _object = None
    _region = None
    _region_data = None
    _stage = 'FAIR_LOOP'
    _mouse_position = None

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None
            and obj.type == 'MESH'
            and context.mode == 'EDIT_MESH'
            and context.area is not None
            and context.area.type == 'VIEW_3D'
        )

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "fairing_method")
        layout.prop(self, "fill_style")
        effective_fill = (
            self.fill_style
            if self.fill_style != 'AUTO'
            else self.result_fill_style
        )
        if effective_fill == 'CIRCLE':
            layout.prop(self, "circle_loops")
        elif effective_fill == 'GRID':
            layout.prop(self, "grid_angle")
        layout.prop(self, "final_simplify")
        layout.separator()
        label = _METHOD_LABELS.get(self.result_method, self.result_method or "—")
        layout.label(text=f"Result: {label}")
        fill_label = _FILL_LABELS.get(
            self.result_fill_style, self.result_fill_style or "—"
        )
        layout.label(text=f"Topology: {fill_label}")
        if self.result_method in {'PLANES', 'CONIC', 'SPHERE'}:
            layout.label(text=f"Residual: {self.result_residual:.6g}")

    @staticmethod
    def _tag_redraw(context):
        if context.area is not None:
            context.area.tag_redraw()

    def _clear_preview(self, context):
        self._fill_coords = None
        self._line_coords = None
        self._fill_batch = None
        self._line_batch = None
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
        self._clear_preview(context)
        self._object = None
        self._region = None
        self._region_data = None
        resize_runtime._modal_active = False
        context.workspace.status_text_set(None)
        context.window.cursor_modal_restore()

    def _capture_selection(self, bm):
        self._initial_selection = {
            "verts": {vert.index for vert in bm.verts if vert.select},
            "edges": {edge.index for edge in bm.edges if edge.select},
            "faces": {face.index for face in bm.faces if face.select},
        }

    def _restore_selection(self):
        if self._object is None or self._initial_selection is None:
            return
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        for vert in bm.verts:
            vert.select = vert.index in self._initial_selection["verts"]
        for edge in bm.edges:
            edge.select = edge.index in self._initial_selection["edges"]
        for face in bm.faces:
            face.select = face.index in self._initial_selection["faces"]
        bmesh.update_edit_mesh(self._object.data, destructive=False)

    @staticmethod
    def _point_segment_distance_squared(point, first, second):
        delta = second - first
        length_squared = delta.length_squared
        if length_squared <= 1.0e-12:
            return (point - first).length_squared
        factor = max(0.0, min(1.0, (point - first).dot(delta) / length_squared))
        return (point - first - delta * factor).length_squared

    def _edge_under_mouse(self, bm, mouse):
        """Pick the nearest projected one-face boundary edge under the pointer."""
        best = None
        best_distance = (14.0 * _ui_scale()) ** 2
        matrix_world = self._object.matrix_world
        for edge in bm.edges:
            if edge.hide or len(edge.link_faces) != 1:
                continue
            first = view3d_utils.location_3d_to_region_2d(
                self._region,
                self._region_data,
                matrix_world @ edge.verts[0].co,
            )
            second = view3d_utils.location_3d_to_region_2d(
                self._region,
                self._region_data,
                matrix_world @ edge.verts[1].co,
            )
            if first is None or second is None:
                continue
            distance = self._point_segment_distance_squared(mouse, first, second)
            if distance <= best_distance:
                best = edge
                best_distance = distance
        return best

    def _set_candidate(self, context, seed_edge):
        component = boundary_component(seed_edge)
        self._hover_edge_index = seed_edge.index if seed_edge else -1
        self._candidate_indices = {edge.index for edge in component}
        try:
            _verts, ordered_edges = ordered_boundary_loop(seed_edge, component)
            valid = True
        except FairingError:
            ordered_edges = sorted(component, key=lambda edge: edge.index)
            valid = False
        matrix_world = self._object.matrix_world
        self._fill_coords = None
        self._line_coords = [
            (
                matrix_world @ edge.verts[0].co,
                matrix_world @ edge.verts[1].co,
            )
            for edge in ordered_edges
        ]
        self._fill_batch = None
        self._line_batch = None
        self._candidate_valid = valid
        self._draw_error_reported = False
        self._set_status_text(context)
        self._tag_redraw(context)

    def _update_hover(self, context, event):
        if self._object.mode != 'EDIT':
            return
        mouse = Vector((
            event.mouse_x - self._region.x,
            event.mouse_y - self._region.y,
        ))
        self._mouse_position = tuple(mouse)
        if not (
            0.0 <= mouse.x < self._region.width
            and 0.0 <= mouse.y < self._region.height
        ):
            return
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.edges.ensure_lookup_table()
        edge = self._edge_under_mouse(bm, mouse)
        if edge is None:
            if self._hover_edge_index != -1:
                self._hover_edge_index = -1
                self._candidate_indices = set()
                self._candidate_valid = False
                self._clear_preview(context)
                self._set_status_text(context)
            return
        if edge.index != self._hover_edge_index:
            self._set_candidate(context, edge)

    def _set_status_text(self, context):
        count = len(self._candidate_indices or ())
        if not count:
            candidate = "Hover a non-manifold edge loop"
        elif self._candidate_valid:
            candidate = f"Valid closed loop: {count} edges"
        else:
            candidate = f"Invalid boundary component: {count} edges"
        context.workspace.status_text_set(
            f"Holey Tools {ADDON_VERSION}  •  Fair Hole  •  {candidate}  •  "
            "Click/Enter: fair  •  MMB/Wheel/Trackpad: navigate  •  Esc: cancel"
        )

    def _confirm_candidate(self, context):
        if not self._candidate_indices or not self._candidate_valid:
            self.report({'WARNING'}, "Choose one simple closed non-manifold loop")
            return {'RUNNING_MODAL'}
        self.edge_indices = ",".join(
            str(index) for index in sorted(self._candidate_indices)
        )
        self._finish_modal(context)
        return self.execute(context)

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or obj.mode != 'EDIT':
            self.report({'WARNING'}, "Fair Hole needs an active mesh in Edit Mode")
            return {'CANCELLED'}
        self._object = obj
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=False)
        backup = obj.data.copy()
        try:
            indices = {
                int(value) for value in self.edge_indices.split(',') if value
            }
            edges = {bm.edges[index] for index in indices}
            loop_verts, loop_edges = ordered_boundary_loop(edges=edges)

            def fit_hypothesis(forced_method=None):
                return choose_hypothesis(
                    loop_verts,
                    loop_edges,
                    obj.matrix_world,
                    lambda faces: GeometryMixin._fit_cylinder(self, faces),
                    lambda face: GeometryMixin._world_normal(self, face),
                    forced_method=(
                        forced_method
                        or getattr(self, "fairing_method", 'AUTO')
                    ),
                )

            hypothesis = fit_hypothesis()
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
            hypothesis = fit_hypothesis(prepared["forced_method"])
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
            loop_verts = result.get("boundary_verts", loop_verts)
            loop_edges = result.get("boundary_edges", loop_edges)
            bm, result = finalize_fair_patch(
                obj,
                bm,
                result,
                loop_verts,
                loop_edges,
                final_simplify=getattr(self, "final_simplify", True),
            )
        except (FairingError, IndexError, RuntimeError, ValueError) as error:
            if obj.mode == 'EDIT':
                bpy.ops.object.mode_set(mode='OBJECT')
            failed_mesh = obj.data
            obj.data = backup
            backup = None
            if failed_mesh.users == 0:
                bpy.data.meshes.remove(failed_mesh)
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'}, str(error))
            return {'CANCELLED'}

        for vert in bm.verts:
            vert.select = False
        for edge in bm.edges:
            edge.select = False
        for face in bm.faces:
            face.select = False
        for face in result["faces"]:
            face.select = True
        for vert in result["verts"]:
            vert.select = True
        self.result_method = result["method"]
        self.result_fill_style = result["fill_style"]
        self.result_residual = result["residual"]
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
        bpy.data.meshes.remove(backup)
        label = _METHOD_LABELS[result["method"]]
        self.report({'INFO'}, f"Filled hole using {label} fairing")
        return {'FINISHED'}

    def modal(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self._restore_selection()
            self._finish_modal(context)
            return {'CANCELLED'}
        if resize_runtime._is_view_navigation_event(event.type) or event.type == 'Z':
            return {'PASS_THROUGH'}
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._update_hover(context, event)
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._restore_selection()
            self._finish_modal(context)
            return {'CANCELLED'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._update_hover(context, event)
            return self._confirm_candidate(context)
        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            return self._confirm_candidate(context)
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if not self.poll(context):
            self.report({'WARNING'}, "Use Fair Hole in Mesh Edit Mode")
            return {'CANCELLED'}
        if resize_runtime._modal_active:
            self.report({'WARNING'}, "Another Holey Tools operation is already active")
            return {'CANCELLED'}
        self._object = context.active_object
        if resize_runtime._persistent_preview is not None:
            resize_runtime._persistent_preview.remove()
            resize_runtime._persistent_preview = None
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        self._capture_selection(bm)
        self._candidate_indices = set()
        self._candidate_valid = False
        self._hover_edge_index = -1
        self._stage = 'FAIR_LOOP'
        self._region = next(
            (region for region in context.area.regions if region.type == 'WINDOW'),
            None,
        )
        self._region_data = context.space_data.region_3d
        if self._region is None or self._region_data is None:
            self.report({'WARNING'}, "Could not find the active 3D viewport")
            return {'CANCELLED'}
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_preview, (), 'WINDOW', 'POST_PIXEL'
        )
        resize_runtime._draw_handles.add(self._draw_handle)
        resize_runtime._modal_active = True
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)

        selected = [
            edge for edge in bm.edges
            if edge.select and not edge.hide and len(edge.link_faces) == 1
        ]
        if selected:
            self._set_candidate(context, selected[0])
        self._update_hover(context, event)
        self._set_status_text(context)
        return {'RUNNING_MODAL'}
