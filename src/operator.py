# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resize Hole modal operator and runtime lifecycle."""

import json
import math
import time

import bmesh
import bpy
import numpy as np

from bpy_extras import view3d_utils
from mathutils import Vector

from .constants import (
    ADDON_VERSION,
    DIAMETER_ESTIMATOR_ITEMS,
    FLOOD_ANGLE_HUD_SECONDS,
    RESIZE_METHOD_ITEMS,
    VIEW_NAVIGATION_EVENTS,
)
from .fairing import FairingError
from .geometry import GeometryMixin, _BOOLEAN_PREVIEW_INTERVAL
from .mesh_state import restore_mesh_contents
from .resize_state import reset_mesh_derived_state
from .modal_session import begin_modal_session, end_modal_session, cancel_modal_sessions
from .visible_geometry import visible_bvh, update_visible_normals
from .gpu_preview import GpuPreviewMixin, _ui_scale
from .preview_cache import clear_draw_cache
from .resize_dimensions import (
    _cylinder_inradius_factor,
    # Retain this import path for existing headless clients and regression tests.
    _conic_depth_preserving_bottom as _conic_depth_preserving_bottom,
    _update_conic_angle,
    _update_conic_depth,
    _update_conic_truncated_depth,
    _update_diameter_estimator,
)


_modal_active = False
_draw_handles = set()
_persistent_preview = None
_DRAG_PREVIEW_INTERVAL = 1.0 / 30.0
_drag_timers = []


def _is_view_navigation_event(event_type):
    """Return whether Blender should receive an event for viewport navigation."""
    return event_type in VIEW_NAVIGATION_EVENTS or event_type.startswith('NDOF_')


def _update_resize_method(operator, context):
    """Reflect an explicit redo choice without re-running automatic selection."""
    # Blender also invokes RNA updates on the argument-only OperatorProperties
    # wrapper, which cannot own our transient Python preview attributes.
    if not isinstance(operator, bpy.types.OperatorProperties):
        GeometryMixin._resolve_resize_method(operator)
        operator._boolean_preview_pending = False
    if context.area is not None:
        context.area.tag_redraw()


class _PersistentPreview:
    """Own a post-modal GPU preview without retaining an Operator RNA object.

    Blender may invalidate the Python wrapper for a completed operator while its
    Redo panel is still active. This plain Python owner retains the drawing data
    needed to keep the cylinder visible plus a canonical source-mesh snapshot,
    and follows the last-operation stack so it can release both when Resize Hole
    is no longer the redoable action.
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
        self._diameter_text = operator._diameter_text
        self._depth_text = operator._depth_text
        self._truncated_depth_text = operator._truncated_depth_text
        self._conic_angle_text = operator._conic_angle_text
        self._active_dimension = operator._active_dimension
        self.depth_valid = operator.depth_valid
        self.angle_valid = operator.angle_valid
        self._draw_error_reported = False
        self._source_object_name = operator._object.name
        (
            self._source_mesh_name,
            self._source_mesh,
            self._source_bmesh,
            self._source_materials,
        ) = self._capture_canonical_source(operator)
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_preview, (), 'WINDOW', 'POST_PIXEL'
        )
        _draw_handles.add(self._draw_handle)
        self._watch_last_operation(context)

    @staticmethod
    def _capture_canonical_source(operator):
        """Copy the pre-preview mesh used as the baseline for every redo."""
        operator._object.update_from_editmode()
        source_mesh = operator._object.data
        canonical_mesh = source_mesh.copy()
        # A direct (non-Boolean) modal preview deforms the real edit mesh so the
        # user can see every drag immediately.  The redo baseline must predate
        # that preview; otherwise each Adjust Last Operation execution starts
        # from an already resized hole and compounds both radius and depth.
        # Boolean previews keep the real object pristine, so applying the same
        # captured coordinates is harmless there.
        original_coords = getattr(operator, "_original_vertex_coords", None) or {}
        for vertex_index, coordinate in original_coords.items():
            if vertex_index < len(canonical_mesh.vertices):
                canonical_mesh.vertices[vertex_index].co = coordinate
        canonical_mesh.update()
        canonical_bmesh = bmesh.new()
        canonical_bmesh.from_mesh(canonical_mesh)
        return (
            source_mesh.name,
            canonical_mesh,
            canonical_bmesh,
            list(source_mesh.materials),
        )

    def update_from(self, operator, context):
        """Refresh copied geometry and label text after a Redo-panel execution."""
        self._fill_coords = list(operator._fill_coords or ())
        self._line_coords = list(operator._line_coords or ())
        self._stage = 'CYLINDER'
        self._numeric_input = operator._numeric_input
        self._diameter_text = operator._diameter_text
        self._depth_text = operator._depth_text
        self._truncated_depth_text = operator._truncated_depth_text
        self._conic_angle_text = operator._conic_angle_text
        self._active_dimension = operator._active_dimension
        self.depth_valid = operator.depth_valid
        self.angle_valid = operator.angle_valid
        if context.area is not None:
            context.area.tag_redraw()

    def remove(self):
        """Remove drawing state and release the canonical redo source."""
        clear_draw_cache(self)
        if self._draw_handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            except ValueError:
                pass
            _draw_handles.discard(self._draw_handle)
            self._draw_handle = None
        source_mesh = getattr(self, "_source_mesh", None)
        self._source_mesh = None
        if source_mesh is not None:
            try:
                if source_mesh.users == 0:
                    bpy.data.meshes.remove(source_mesh)
            except ReferenceError:
                pass
        source_bmesh = getattr(self, "_source_bmesh", None)
        self._source_bmesh = None
        if source_bmesh is not None and source_bmesh.is_valid:
            source_bmesh.free()

    def restore_source(self, obj):
        """Restore the modal input before every confirmation or redo execute."""
        if obj is None or obj.name != self._source_object_name:
            return False
        source_bmesh = getattr(self, "_source_bmesh", None)
        if source_bmesh is None or not source_bmesh.is_valid:
            return False

        # Exact-Boolean modal previews hide the real edit object while a
        # separate display object shows the unapplied modifier result. A hidden
        # object cannot pass object.mode_set's poll even when it is assigned as
        # the view-layer active object, so make the source visible and active
        # before either mode transition.
        obj.hide_set(False)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        restore_mesh_contents(obj, source_bmesh, self._source_materials)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        return True

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
    """Select one circular wall, fit it, and resize it interactively.

    The modal operator has two stages. SELECT ray-casts beneath the pointer and
    displays a non-destructive flood fill. A click locks that region, fits the
    cone, and enters CYLINDER, where typing or dragging updates dimensions.
    Confirmation hands plain drawing data to ``_PersistentPreview`` so Blender's
    Redo panel can continue changing the completed operation safely.
    """

    bl_idname = "mesh.holey_tools_resize_hole"
    bl_label = "Resize Hole"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}
    bl_description = (
        "Hover to flood-fill a circular wall, click to fit its cone or cylinder, "
        "then click again to apply. Ctrl/Cmd-click to use selected wall faces instead"
    )

    scale: bpy.props.FloatProperty(
        name="Scale",
        description="Internal radius scale relative to the fitted wall",
        default=1.0,
        min=0.01,
        max=10.0,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    diameter: bpy.props.FloatProperty(
        name="Diameter",
        description="World-space diameter; for a cylindrical hole, the circle fitting inside its polygonal wall",
        default=0.0,
        min=0.0,
        subtype='DISTANCE',
        unit='LENGTH',
    )
    depth: bpy.props.FloatProperty(
        name="Depth",
        description=(
            "For cones, distance from the exterior rim plane to the theoretical "
            "tip; otherwise the distance between the hole ends"
        ),
        default=0.0,
        min=0.0,
        subtype='DISTANCE',
        unit='LENGTH',
        update=_update_conic_depth,
    )
    truncated_depth: bpy.props.FloatProperty(
        name="Truncated Depth",
        description=(
            "Axial distance from the exterior rim plane to the actual truncated "
            "end of the selected conic region"
        ),
        default=0.0,
        min=0.0,
        subtype='DISTANCE',
        unit='LENGTH',
        update=_update_conic_truncated_depth,
    )
    conic_angle: bpy.props.FloatProperty(
        name="Conic Angle",
        description=(
            "Full included angle at the theoretical cone tip; edits pivot "
            "around the bottom truncation circle. Zero removes the taper "
            "while preserving the connected bore"
        ),
        default=0.0,
        min=0.0,
        max=math.radians(178.0),
        subtype='ANGLE',
        update=_update_conic_angle,
    )
    resize_method: bpy.props.EnumProperty(
        name="Resize Method",
        description="Choose direct scaling, Exact Boolean reconstruction, or automatic selection",
        items=RESIZE_METHOD_ITEMS,
        default='AUTO',
        update=_update_resize_method,
    )
    flood_angle: bpy.props.FloatProperty(
        name="Flood Angle",
        description="Maximum angle between neighboring faces in the hover flood fill",
        subtype='ANGLE',
        default=math.radians(25.0),
        min=math.radians(0.1),
        max=math.radians(89.9),
    )
    diameter_estimator: bpy.props.EnumProperty(
        name="Diameter Estimate",
        description="How to estimate diameter when the selected wall has multiple radii",
        items=DIAMETER_ESTIMATOR_ITEMS,
        default='AVERAGE',
        update=_update_diameter_estimator,
        options={'SKIP_SAVE'},
    )
    fairing_method: bpy.props.EnumProperty(
        name="Reconstruction Fit",
        description="Surface fit used to reconstruct the mesh before Boolean cutting",
        items=(
            ('AUTO', "Auto Fit", "Use the accepted automatic fit"),
            ('PLANES', "N-Plane", "Fit one to three adjacent planes"),
            ('CYLINDER', "Cylinder", "Force a zero-slope circular-cylinder fit"),
            ('SPHERE', "Sphere", "Force a spherical fit"),
            ('ORGANIC', "Organic", "Use thin-plate surface continuation"),
        ),
        default='AUTO',
    )
    fill_style: bpy.props.EnumProperty(
        name="Reconstruction Topology",
        description="Topology used to reconstruct the old opening",
        items=(
            ('AUTO', "Auto Topology", "Choose topology from the selected fit"),
            ('GRID', "Grid", "Fit a source-aligned quad patch to the opening"),
            ('CIRCLE', "Circle", "Use concentric rings and a center fan"),
        ),
        default='AUTO',
    )
    circle_loops: bpy.props.IntProperty(
        name="Circle Loops",
        description="Concentric reconstruction loops before the center fan",
        default=5,
        min=0,
        max=64,
    )
    grid_match_curvature: bpy.props.BoolProperty(
        name="Match Curvature (C2)",
        description="Estimate a curvature-matching ideal surface for the Manhattan grid patch",
        default=True,
    )
    final_simplify: bpy.props.BoolProperty(
        name="Final Simplify",
        description="Simplify the resized Boolean result after reconstruction",
        default=True,
    )
    scaling_final_simplify: bpy.props.BoolProperty(
        name="Final Simplify",
        description=(
            "Expand the directly scaled wall selection and run Limited Dissolve "
            "after committing a smaller hole"
        ),
        default=False,
    )
    boolean_segments: bpy.props.IntProperty(
        name="Boolean Segments",
        description="Number of radial segments in the final Boolean cutter",
        default=32,
        min=3,
        max=512,
        soft_max=128,
    )
    reconstruction_result_method: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    reconstruction_result_fill_style: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    reconstruction_grid_edge_length: bpy.props.FloatProperty(
        name="Estimated Edge Size", default=0.0,
        subtype='DISTANCE', unit='LENGTH', options={'HIDDEN', 'SKIP_SAVE'},
    )
    fit_valid: bpy.props.BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    conic_dimensions_updating: bpy.props.BoolProperty(
        default=False, options={'HIDDEN', 'SKIP_SAVE'}
    )
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
    fit_radius_minimum: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_radius_average: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_radius_maximum: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_slope: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_radius_coordinate: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_is_cone: bpy.props.BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_z_min: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_z_max: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_is_stud: bpy.props.BoolProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_angular_segments: bpy.props.IntProperty(
        default=0, min=0, options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_mesh_vertex_count: bpy.props.IntProperty(
        default=0, min=0, options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_mesh_edge_count: bpy.props.IntProperty(
        default=0, min=0, options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_mesh_face_count: bpy.props.IntProperty(
        default=0, min=0, options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_mesh_topology_signature: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_end_planes: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    depth_valid: bpy.props.BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    angle_valid: bpy.props.BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    fit_depth_initial: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_depth_outer: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_depth_inner: bpy.props.FloatProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_depth_outer_plane: bpy.props.IntProperty(options={'HIDDEN', 'SKIP_SAVE'})
    fit_conic_depth_initial: bpy.props.FloatProperty(
        options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_truncated_depth_initial: bpy.props.FloatProperty(
        options={'HIDDEN', 'SKIP_SAVE'}
    )
    fit_conic_bottom_radius: bpy.props.FloatProperty(
        default=0.0, options={'HIDDEN', 'SKIP_SAVE'}
    )
    conic_angle_previous: bpy.props.FloatProperty(
        default=0.0, options={'HIDDEN', 'SKIP_SAVE'}
    )
    conic_depth_previous: bpy.props.FloatProperty(
        default=0.0, options={'HIDDEN', 'SKIP_SAVE'}
    )
    conic_truncated_depth_previous: bpy.props.FloatProperty(
        default=0.0, options={'HIDDEN', 'SKIP_SAVE'}
    )

    _draw_handle = None
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
    _depth_fit = None
    _resize_projection_ready = False
    _resize_requires_boolean = False
    _resize_dynamic_requires_boolean = False
    _resize_surfaces_by_vertex = None
    _resize_bore_vertex_indices = None
    _resize_base_bvh = None
    _resize_base_span = 0.0
    _resize_angular_segments = 0
    _boolean_preview_source = None
    _boolean_reconstruction_source = None
    _boolean_reconstruction_blind_coordinates = None
    _boolean_preview_display_object = None
    _boolean_preview_cutter_object = None
    _boolean_preview_modifier_name = ""
    _boolean_preview_segments = 0
    _boolean_preview_last_update = 0.0
    _boolean_preview_pending = False
    _stage = 'SELECT'
    _numeric_input = ""
    _numeric_reference = ""
    _numeric_replace = True
    _diameter_text = ""
    _depth_text = ""
    _truncated_depth_text = ""
    _conic_angle_text = ""
    _active_dimension = 'DIAMETER'
    _drag_active = False
    _drag_moved = False
    _drag_start_x = 0
    _drag_start_y = 0
    _drag_start_value = 0.0
    _gizmo_rect = None
    _gizmo_rects = None
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

    def draw(self, context):
        """Draw dimensions available in the Redo panel for the fitted wall."""
        conic_dimensions = bool(self.angle_valid and self.depth_valid)
        if not conic_dimensions:
            self.layout.prop(self, "diameter_estimator")
        if conic_dimensions:
            row = self.layout.row()
            row.activate_init = True
            row.prop(self, "depth")
            self.layout.prop(self, "truncated_depth")
            self.layout.prop(self, "conic_angle")
        else:
            row = self.layout.row()
            row.activate_init = True
            row.prop(self, "diameter")
            if self.depth_valid:
                self.layout.prop(self, "depth")
            if self.angle_valid:
                self.layout.prop(self, "conic_angle")
        self.layout.prop(self, "resize_method", expand=True)
        boolean_active = (
            self.resize_method == 'BOOLEAN'
            or (
                self.resize_method == 'AUTO'
                and getattr(self, "_resize_requires_boolean", False)
            )
        )
        fair_only = GeometryMixin._fair_resize_requested(self)
        if boolean_active or fair_only:
            box = self.layout.box()
            box.label(text="Reconstruction: Fair Hole" if fair_only else "Reconstruction: Exact Boolean", icon='MOD_BOOLEAN')
            if not fair_only:
                box.prop(self, "boolean_segments")
            box.prop(self, "fairing_method")
            box.prop(self, "fill_style")
            effective_fill = (
                self.fill_style
                if self.fill_style != 'AUTO'
                else self.reconstruction_result_fill_style
            )
            if effective_fill == 'CIRCLE':
                box.prop(self, "circle_loops")
            elif effective_fill == 'GRID' and self.reconstruction_result_method not in {'PLANES', 'CONIC', 'SPHERE'}:
                box.prop(self, "grid_match_curvature")
            box.prop(self, "final_simplify")
            if self.reconstruction_result_method:
                box.label(
                    text=f"Fit result: {self.reconstruction_result_method.title()}"
                )
            if self.reconstruction_result_fill_style:
                box.label(
                    text=(
                        "Topology result: "
                        f"{self.reconstruction_result_fill_style.title()}"
                    )
                )
        else:
            box = self.layout.box()
            box.label(text="Resize: Scaling", icon='FULLSCREEN_ENTER')
            box.prop(self, "scaling_final_simplify")

    @staticmethod
    def _tag_redraw(context):
        """Request another draw from the current area when one is available."""
        if context.area is not None:
            context.area.tag_redraw()

    def _clear_batches(self, context):
        """Discard all preview geometry and request a viewport redraw."""
        clear_draw_cache(self)
        self._fill_coords = None
        self._line_coords = None
        self._tag_redraw(context)

    def _finish_modal(self, context):
        """Release transient modal resources and restore viewport UI state."""
        global _modal_active
        end_modal_session(self)
        MESH_OT_holey_tools_resize_hole._remove_drag_timer(self, context)
        self._resize_wall_cache = None
        self._hover_cache = self._hover_region = self._hover_preview_region = None
        self._remove_draw_handler()
        self._clear_batches(context)
        self._bvh = None
        self._bvh_faces = ()
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
        """Leave modal interaction while preserving its cone/cylinder overlay.

        The live draw handler is removed before its plain coordinate data is
        copied into a new persistent preview. The caller then executes and
        completes the Blender operator, making its properties available to Redo.
        """
        global _modal_active, _persistent_preview
        end_modal_session(self)
        MESH_OT_holey_tools_resize_hole._remove_drag_timer(self, context)
        self._resize_wall_cache = None
        self._hover_cache = self._hover_region = self._hover_preview_region = None
        self._bvh = None
        self._bvh_faces = ()
        self._mouse_entered_view = False
        _modal_active = False
        self._remove_draw_handler()
        clear_draw_cache(self)
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
        if getattr(self, "_boolean_preview_source", None) is not None:
            GeometryMixin._replace_boolean_preview_with_source(
                self, consume=True
            )
            return
        if self._object is None or not self._original_vertex_coords:
            return
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.verts.ensure_lookup_table()
        for index, coordinate in self._original_vertex_coords.items():
            if index < len(bm.verts):
                bm.verts[index].co = coordinate
        update_visible_normals(bm)
        bmesh.update_edit_mesh(self._object.data, loop_triangles=True, destructive=False)

    def _modal_dimensions_changed(self):
        """Return whether the modal session differs from its fitted baseline.

        The baseline is captured only after fitting has initialized every
        user-facing dimension. Comparing the values, rather than mouse motion,
        also treats a drag or numeric edit that returns to its starting value as
        unchanged.
        """
        initial = getattr(self, "_modal_initial_dimensions", None)
        if initial is None:
            # A missing baseline indicates a non-modal or partially restored
            # execution. Never suppress real work in that defensive case.
            return True
        names = {'diameter'}
        if getattr(self, "depth_valid", False):
            names.add('depth')
        if getattr(self, "angle_valid", False):
            names.add('conic_angle')
            if getattr(self, "depth_valid", False):
                names.add('truncated_depth')
                names.discard('diameter')
        for name in names:
            baseline = initial[name]
            current = float(getattr(self, name))
            tolerance = max(abs(baseline), abs(current), 1.0) * 1.0e-9
            if not math.isclose(
                current,
                baseline,
                rel_tol=1.0e-9,
                abs_tol=tolerance,
            ):
                return True
        return False

    def _cancel_unchanged_modal(self, context):
        """Cancel a fitted interaction that has not changed any dimension."""
        if MESH_OT_holey_tools_resize_hole._modal_dimensions_changed(self):
            return False
        self._restore_original_geometry()
        self._restore_initial_selection()
        self._runtime_cancelled = True
        self._finish_modal(context)
        return True

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

            def draw_status(header, _context):
                row = header.layout.row(align=True)
                row.alignment = 'LEFT'
                row.label(text=f"Holey Tools {ADDON_VERSION}  •  {candidate_status}  •")
                row.label(text="", icon='EVENT_LEFTBRACKET')
                row.label(text="/")
                row.label(text="", icon='EVENT_RIGHTBRACKET')
                row.label(
                    text=f"Flood Angle: {angle:.1f}°  •  "
                    "Type: set angle  •  Click: apply and adjust  •  Esc: cancel"
                )

            context.workspace.status_text_set(draw_status)
        else:
            active = self._active_dimension.replace('_', ' ').title()
            if self.angle_valid and self.depth_valid:
                dimensions = (
                    f"Depth: {self._depth_text}  •  "
                    f"Truncated Depth: {self._truncated_depth_text}  •  "
                    f"Included Angle: {self._conic_angle_text}"
                )
            else:
                dimensions = f"Diameter: {self._diameter_text}"
            if self.depth_valid and not self.angle_valid:
                dimensions += f"  •  Depth: {self._depth_text}"
            if self.angle_valid and not self.depth_valid:
                dimensions += f"  •  Conic Angle: {self._conic_angle_text}"
            if (
                self.resize_method == 'BOOLEAN'
                or (
                    self.resize_method == 'AUTO'
                    and getattr(self, "_resize_requires_boolean", False)
                )
            ):
                dimensions += "  •  Rebuild: Exact Boolean"
            step_hint = (
                "+/-: 1° (Shift: 0.1°)"
                if self._active_dimension == 'ANGLE'
                else "+/-: 1 mm (Shift: 0.1 mm)"
            )
            context.workspace.status_text_set(
                f"{dimensions}  •  Active: {active}  •  Type to replace  •  "
                f"{step_hint}  •  "
                "Touch handle: select/adjust  •  Ctrl: invert 0.1 mm snap  •  "
                "Shift: fine  •  "
                "MMB/Wheel/Trackpad/3D Mouse: navigate  •  "
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
        self._set_hover_seed(context, bm.faces[self._hover_face_index])

    def _set_hover_seed(self, context, face):
        """Flood from ``face`` and publish the resulting selection preview."""
        self._hover_region = None
        faces = self._flood_fill(face)
        self._hover_face_index = face.index
        region = getattr(self, "_hover_region", None)
        if (
            region is not None and region is getattr(self, "_hover_preview_region", None)
            and getattr(self, "_fill_coords", None) is not None
        ):
            return
        self._candidate_indices = {candidate.index for candidate in faces}
        if region is not None and region.triangles is not None:
            self._fill_coords = region.triangles
            self._line_coords = None
            self._draw_error_reported = False
            self._tag_redraw(context)
        else:
            self._build_selection_preview(context, faces)
            if region is not None:
                region.triangles = self._fill_coords
        self._hover_preview_region = region
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
        self._angle_numeric_input = f"{math.degrees(self.flood_angle):.1f}"
        self._angle_numeric_replace = True
        context.scene.holey_tools_flood_angle = self.flood_angle
        self._show_flood_angle_hud(context)
        self._rebuild_hover(context)
        self._set_status_text(context)

    def _handle_flood_angle_numeric_event(self, context, event):
        """Consume direct degree entry while choosing the flood selection."""
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
        """Apply valid direct degree input to the selection and angle HUD."""
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
        self._set_status_text(context)

    def _show_flood_angle_hud(self, context):
        """Show the flood-angle HUD and schedule one redraw after it expires."""
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
        GeometryMixin._get_hover_cache(self)
        if self._bvh is None:
            self._bvh, self._bvh_faces = visible_bvh(bm)
        if self._bvh is None:
            return

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

        if face_index is None or face_index < 0 or face_index >= len(self._bvh_faces):
            self._hover_face_index = -1
            self._candidate_indices = set()
            self._clear_batches(context)
            self._set_status_text(context)
            return

        face = self._bvh_faces[face_index]
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
        # Dimension callbacks are meaningful only after every fitted baseline
        # has been serialized. A reused operator may still carry a true flag.
        self.fit_valid = False
        if fit.get("is_cone", False):
            # A conic frustum's two end circles define physical truncation even
            # when the smaller one continues into a normal cylindrical bore.
            self._depth_fit = GeometryMixin._fit_adjustable_depth(
                self, self._end_planes or (), fit
            )
        elif getattr(self, "_resize_requires_boolean", False):
            self._depth_fit = GeometryMixin._fit_reconstruction_depth(self)
        else:
            self._depth_fit = GeometryMixin._fit_adjustable_depth(
                self, self._end_planes or (), fit
            )
        self.depth_valid = self._depth_fit is not None
        fit["radius_coordinate"] = (
            self._depth_fit["outer_coordinate"] if self._depth_fit else 0.0
        )
        self.scale = 1.0
        self.fit_angular_segments = max(
            3, int(getattr(self, "_resize_angular_segments", 0) or 32)
        )
        inradius_factor = _cylinder_inradius_factor(self)
        inradius_estimates = fit.get("inradius_estimates")
        if inradius_factor < 1.0 and inradius_estimates:
            # Preserve the internal vertex-radius convention for deformation;
            # use actual wall-plane clearance for the public hole diameter.
            fit["radius_estimates"] = {
                key: value / inradius_factor
                for key, value in inradius_estimates.items()
            }
            fit["radius"] = fit["radius_estimates"].get(
                getattr(self, "diameter_estimator", 'AVERAGE'),
                fit["radius_estimates"]['AVERAGE'],
            )
        self.diameter = GeometryMixin._fitted_radius_at(
            self, fit["radius_coordinate"]
        ) * 2.0 * _cylinder_inradius_factor(self)
        self.angle_valid = bool(fit.get("is_cone", False))
        self.conic_angle = float(fit.get("half_angle", 0.0)) * 2.0
        self.fit_face_indices = ",".join(str(index) for index in sorted(self._locked_indices))
        self.fit_axis = fit["axis"]
        self.fit_basis_x = fit["basis_x"]
        self.fit_basis_y = fit["basis_y"]
        self.fit_center = fit["center"]
        self.fit_radius = fit["radius"]
        estimates = fit["radius_estimates"]
        self.fit_radius_minimum = estimates['MINIMUM']
        self.fit_radius_average = estimates['AVERAGE']
        self.fit_radius_maximum = estimates['MAXIMUM']
        self.fit_slope = fit.get("slope", 0.0)
        self.fit_radius_coordinate = fit["radius_coordinate"]
        self.fit_is_cone = fit.get("is_cone", False)
        self.fit_z_min = fit["z_min"]
        self.fit_z_max = fit["z_max"]
        self.fit_is_stud = fit.get("is_stud", False)
        source_mesh = self._object.data
        self.fit_mesh_vertex_count = len(source_mesh.vertices)
        self.fit_mesh_edge_count = len(source_mesh.edges)
        self.fit_mesh_face_count = len(source_mesh.polygons)
        self.fit_mesh_topology_signature = (
            GeometryMixin._object_topology_signature(self._object)
        )
        self.boolean_segments = self.fit_angular_segments
        self.fit_end_planes = json.dumps(
            [
                {
                    "verts": plane["verts"],
                    "point": list(plane["point"]),
                    "normal": list(plane["normal"]),
                    "max_error": plane.get("max_error"),
                    "tolerance": plane.get("tolerance"),
                    "adjacent_normal": (
                        list(plane["adjacent_normal"])
                        if plane.get("adjacent_normal") is not None
                        else None
                    ),
                }
                for plane in (self._end_planes or ())
            ],
            separators=(",", ":"),
        )
        if self.depth_valid:
            self.fit_depth_initial = self._depth_fit["initial"]
            self.fit_depth_outer = self._depth_fit["outer_coordinate"]
            self.fit_depth_inner = self._depth_fit["inner_coordinate"]
            self.fit_depth_outer_plane = self._depth_fit["outer_plane_index"]
            if self.angle_valid and abs(float(fit.get("slope", 0.0))) > 1.0e-12:
                self.truncated_depth = self._depth_fit["initial"]
                self.fit_truncated_depth_initial = self.truncated_depth
                self.fit_conic_bottom_radius = float(
                    getattr(self, "_conic_bore_radius", 0.0)
                    or GeometryMixin._fitted_radius_at(
                        self, self._depth_fit["inner_coordinate"]
                    )
                )
                self.depth = (
                    GeometryMixin._fitted_radius_at(
                        self, self._depth_fit["outer_coordinate"]
                    )
                    / abs(float(fit["slope"]))
                )
                self.fit_conic_depth_initial = self.depth
            else:
                self.depth = self._depth_fit["initial"]
                self.truncated_depth = 0.0
                self.fit_truncated_depth_initial = 0.0
                self.fit_conic_depth_initial = 0.0
                self.fit_conic_bottom_radius = 0.0
        else:
            self.depth = 0.0
            self.truncated_depth = 0.0
            self.fit_depth_initial = 0.0
            self.fit_depth_outer = 0.0
            self.fit_depth_inner = 0.0
            self.fit_depth_outer_plane = 0
            self.fit_truncated_depth_initial = 0.0
            self.fit_conic_depth_initial = 0.0
            self.fit_conic_bottom_radius = 0.0
        self.conic_depth_previous = self.depth
        self.conic_truncated_depth_previous = self.truncated_depth
        self.conic_angle_previous = self.conic_angle
        self.fit_valid = True

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
            "radius_estimates": {
                'MINIMUM': self.fit_radius_minimum,
                'AVERAGE': self.fit_radius_average,
                'MAXIMUM': self.fit_radius_maximum,
            },
            "slope": self.fit_slope,
            "half_angle": math.atan(abs(self.fit_slope)),
            "is_cone": self.fit_is_cone,
            "radius_coordinate": self.fit_radius_coordinate,
            "z_min": self.fit_z_min,
            "z_max": self.fit_z_max,
            "is_stud": self.fit_is_stud,
        }
        try:
            self._end_planes = [
                {
                    "verts": plane["verts"],
                    "point": Vector(plane["point"]),
                    "normal": Vector(plane["normal"]),
                    "max_error": plane.get("max_error", math.inf),
                    "tolerance": plane.get("tolerance", 0.0),
                    "adjacent_normal": (
                        Vector(plane["adjacent_normal"])
                        if plane.get("adjacent_normal") is not None
                        else None
                    ),
                }
                for plane in json.loads(self.fit_end_planes or "[]")
            ]
        except (TypeError, ValueError, KeyError):
            self._end_planes = []
        if (
            self.depth_valid
            and self.fit_depth_initial > 1.0e-12
            and (
                self.fit_depth_outer_plane == -1
                or (
                    len(self._end_planes) == 2
                    and self.fit_depth_outer_plane in {0, 1}
                )
            )
        ):
            surface_depth = self.fit_depth_outer_plane == -1
            self._depth_fit = {
                "initial": self.fit_depth_initial,
                "outer_coordinate": self.fit_depth_outer,
                "inner_coordinate": self.fit_depth_inner,
                "outer_plane_index": self.fit_depth_outer_plane,
                "inner_plane_index": (
                    -1
                    if surface_depth
                    else 1 - self.fit_depth_outer_plane
                ),
                "is_stud": self.fit_is_stud,
                "surface_depth": surface_depth,
            }
        else:
            self._depth_fit = None
            self.depth_valid = False
        self.angle_valid = bool(self.fit_is_cone)
        self._resize_angular_segments = max(
            3, int(self.fit_angular_segments or 32)
        )
        self._stage = 'CYLINDER'
        return True

    @staticmethod
    def _scene_unit_info(context):
        """Return the scene unit system and a safe Blender-to-display scale."""
        settings = context.scene.unit_settings
        system = settings.system
        scale = settings.scale_length if system != 'NONE' else 1.0
        return system, max(scale, 1.0e-12)

    def _format_dimension(self, context, value):
        """Format one internal world-space length using the scene's units."""
        system, scale = self._scene_unit_info(context)
        if system == 'NONE':
            return f"{value:.6g}"
        return bpy.utils.units.to_string(
            system,
            'LENGTH',
            value * scale,
            precision=6,
            split_unit=False,
            compatible_unit=True,
        )

    def _format_diameter(self, context):
        """Format the current diameter for numeric input and overlays."""
        return self._format_dimension(context, self.diameter)

    def _format_depth(self, context):
        """Format the current depth for numeric input and overlays."""
        return self._format_dimension(context, self.depth)

    def _format_truncated_depth(self, context):
        """Format the physical axial span of a selected conic frustum."""
        return self._format_dimension(context, self.truncated_depth)

    def _format_conic_angle(self):
        """Format the full included conic tip angle for input and overlays."""
        return f"{math.degrees(self.conic_angle):.6g}°"

    def _parse_dimension(self, context, text):
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

    def _parse_diameter(self, context, text):
        """Retain the diameter parser name for compatibility with older redo state."""
        return self._parse_dimension(context, text)

    @staticmethod
    def _parse_angle(text):
        """Parse a user-facing conic angle in degrees."""
        normalized = text.strip().lower().replace(',', '.')
        normalized = normalized.removesuffix("degrees").removesuffix("degree")
        normalized = normalized.removesuffix("deg").removesuffix("°").strip()
        if not normalized:
            raise ValueError("Empty angle")
        return math.radians(float(normalized))

    def _active_dimension_value(self):
        """Return the value controlled by the most recently touched handle."""
        active = getattr(self, "_active_dimension", 'DIAMETER')
        if active == 'DEPTH':
            return self.depth
        if active == 'TRUNCATED_DEPTH':
            return self.truncated_depth
        if active == 'ANGLE':
            return self.conic_angle
        return self.diameter

    def _set_active_dimension_value(self, value):
        """Assign the most recently touched dimension."""
        active = getattr(self, "_active_dimension", 'DIAMETER')
        if active == 'DEPTH':
            self.depth = value
        elif active == 'TRUNCATED_DEPTH':
            self.truncated_depth = value
        elif active == 'ANGLE':
            self.conic_angle = min(math.radians(178.0), max(0.0, value))
        else:
            self.diameter = value

    def _sync_dimension_text(self, context):
        """Refresh both overlay values while preserving active partial input."""
        active = getattr(self, "_active_dimension", 'DIAMETER')
        self._diameter_text = (
            self._numeric_input
            if active == 'DIAMETER'
            else self._format_diameter(context)
        )
        self._depth_text = (
            self._numeric_input
            if active == 'DEPTH'
            else self._format_depth(context)
        ) if self.depth_valid else ""
        self._truncated_depth_text = (
            self._numeric_input
            if active == 'TRUNCATED_DEPTH'
            else self._format_truncated_depth(context)
        ) if self.angle_valid and self.depth_valid else ""
        self._conic_angle_text = (
            self._numeric_input
            if active == 'ANGLE'
            else self._format_conic_angle()
        ) if getattr(self, "angle_valid", False) else ""

    def _activate_dimension(self, context, dimension):
        """Make a handle the target for subsequent typing, stepping, and drag."""
        if dimension == 'DEPTH' and not self.depth_valid:
            dimension = 'DIAMETER'
        if dimension == 'TRUNCATED_DEPTH' and not (
            self.angle_valid and self.depth_valid
        ):
            dimension = 'DIAMETER'
        if dimension == 'ANGLE' and not self.angle_valid:
            dimension = 'DIAMETER'
        self._active_dimension = dimension
        if dimension == 'DEPTH':
            self._numeric_reference = self._format_depth(context)
        elif dimension == 'TRUNCATED_DEPTH':
            self._numeric_reference = self._format_truncated_depth(context)
        elif dimension == 'ANGLE':
            self._numeric_reference = self._format_conic_angle()
        else:
            self._numeric_reference = self._format_diameter(context)
        self._numeric_input = self._numeric_reference
        self._numeric_replace = True
        self._sync_dimension_text(context)
        self._set_status_text(context)
        self._tag_redraw(context)

    def _begin_numeric_input(self, context):
        """Default direct entry to apex depth for cones, otherwise diameter."""
        self._activate_dimension(
            context,
            'DEPTH' if self.angle_valid and self.depth_valid else 'DIAMETER',
        )

    def _update_numeric_preview(self, context):
        """Apply valid input immediately to the mesh and GPU cone/cylinder.

        Incomplete or invalid text remains visible for continued editing but does
        not disturb the last valid geometry.
        """
        try:
            value = (
                self._parse_angle(self._numeric_input)
                if self._active_dimension == 'ANGLE'
                else self._parse_dimension(context, self._numeric_input)
            )
        except (TypeError, ValueError):
            self._sync_dimension_text(context)
            self._set_status_text(context)
            return
        if value < 0.0:
            self._sync_dimension_text(context)
            self._set_status_text(context)
            return
        self._set_active_dimension_value(value)
        self._sync_dimension_text(context)
        self._build_cylinder_preview(context)
        self._apply_diameter_to_mesh(context)
        self._set_status_text(context)

    def _step_dimension(self, context, direction, fine_adjustment=False):
        """Step the active dimension by one millimeter, or a tenth with Shift."""
        if self._active_dimension == 'ANGLE':
            step = math.radians(0.1 if fine_adjustment else 1.0)
        else:
            _system, scale = self._scene_unit_info(context)
            millimeters = 0.1 if fine_adjustment else 1.0
            step = millimeters * 0.001 / scale
        minimum = 0.0
        self._set_active_dimension_value(
            max(minimum, self._active_dimension_value() + direction * step)
        )
        self._numeric_reference = (
            self._format_conic_angle()
            if self._active_dimension == 'ANGLE'
            else self._format_dimension(context, self._active_dimension_value())
        )
        self._numeric_input = self._numeric_reference
        self._numeric_replace = True
        self._sync_dimension_text(context)
        self._build_cylinder_preview(context)
        self._apply_diameter_to_mesh(context)
        self._set_status_text(context)

    def _handle_dimension_step_event(self, context, event):
        """Consume main-keyboard or numpad step shortcuts for the active value."""
        if event.value != 'PRESS':
            return False
        if event.type in {'PLUS', 'EQUAL', 'NUMPAD_PLUS'}:
            direction = 1
        elif event.type in {'MINUS', 'NUMPAD_MINUS'}:
            direction = -1
        else:
            return False
        self._step_dimension(context, direction, event.shift)
        return True

    def _handle_numeric_event(self, context, event):
        """Consume keyboard events that edit the active dimension expression.

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

    def _begin_dimension_drag(self, context, event, dimension):
        """Activate a handle and capture the pointer/value anchoring its drag."""
        self._activate_dimension(context, dimension)
        self._drag_active = True
        self._drag_moved = False
        self._drag_start_x = event.mouse_x
        self._drag_start_y = event.mouse_y
        self._drag_start_value = self._active_dimension_value()
        self._drag_input_value = None
        self._drag_preview_pending = False
        self._drag_preview_last_update = None
        self._resize_wall_cache = None
        MESH_OT_holey_tools_resize_hole._remove_drag_timer(self, context)
        manager = getattr(context, "window_manager", None)
        if manager is not None and hasattr(manager, "event_timer_add"):
            self._drag_preview_timer = manager.event_timer_add(
                _DRAG_PREVIEW_INTERVAL, window=context.window
            )
            _drag_timers.append((manager, self._drag_preview_timer))

    def _remove_drag_timer(self, context):
        """Release the drag wakeup on release, cancellation, or confirmation."""
        timer = getattr(self, "_drag_preview_timer", None)
        self._drag_preview_timer = None
        self._drag_preview_pending = False
        if timer is not None:
            context.window_manager.event_timer_remove(timer)
            _drag_timers[:] = [
                (manager, active) for manager, active in _drag_timers
                if active != timer
            ]

    def _flush_drag_preview(self, context, force=False):
        """Apply only the latest dimensions, leaving time for input and drawing."""
        if not getattr(self, "_drag_preview_pending", False):
            # A Boolean update may have been deferred by its slower solver
            # throttle. Catch up even if the pointer stops while held down.
            if not getattr(self, "_boolean_preview_pending", False):
                return
            if (
                not force
                and time.monotonic() - getattr(self, "_boolean_preview_last_update", 0.0)
                < _BOOLEAN_PREVIEW_INTERVAL
            ):
                return
        last_update = getattr(self, "_drag_preview_last_update", None)
        if (
            not force and last_update is not None
            and time.monotonic() - last_update < _DRAG_PREVIEW_INTERVAL
        ):
            return
        self._drag_preview_pending = False
        self._apply_diameter_to_mesh(context)
        # Measure from completion, not start: a slow evaluation must not cause
        # the next queued mouse event to immediately launch another evaluation.
        self._drag_preview_last_update = time.monotonic()

    def _update_dimension_drag(self, context, event):
        """Convert handle-aligned drag distance into a live exponential resize.

        Shift lowers sensitivity. Snapping follows Blender's global snap toggle,
        while Ctrl temporarily inverts that setting, and uses physical 0.1 mm
        increments independently of the scene's displayed unit scale.
        """
        ui_scale = _ui_scale()
        active = self._active_dimension
        if active in {'DEPTH', 'TRUNCATED_DEPTH'}:
            # Window-space Y increases upward, while dragging the depth handle
            # downward should make the hole deeper.
            delta = self._drag_start_y - event.mouse_y
        else:
            delta = event.mouse_x - self._drag_start_x
        if abs(delta) >= 3.0 * max(1.0, ui_scale):
            self._drag_moved = True
        if not self._drag_moved:
            # Selection-click jitter and snapping must not edit a fitted value
            # before the pointer has actually begun a drag.
            return
        divisor = (1000.0 if event.shift else 200.0) * ui_scale
        if active == 'ANGLE':
            angle_divisor = (20.0 if event.shift else 4.0) * ui_scale
            value = self._drag_start_value + math.radians(delta / angle_divisor)
            value = min(math.radians(178.0), max(0.0, value))
        else:
            value = self._drag_start_value * (2.0 ** (delta / divisor))
            if self._drag_start_value == 0.0:
                key = active.lower()
                baseline = (getattr(self, "_modal_initial_dimensions", None) or {}).get(key, 1.0e-3)
                value = max(0.0, baseline * (2.0 ** (delta / divisor) - 1.0))
        snap_enabled = bool(context.scene.tool_settings.use_snap) != bool(event.ctrl)
        if snap_enabled and delta != 0.0:
            if active == 'ANGLE':
                snap_step = math.radians(0.1)
            else:
                # Blender stores length in scene units. Snap the physical result
                # to tenths of a millimeter in every displayed unit system.
                _system, scale = self._scene_unit_info(context)
                snap_step = 0.0001 / scale
            value = round(value / snap_step) * snap_step
            value = max(0.0 if active == 'ANGLE' else snap_step, value)
        value = max(0.0, value)
        current_value = self._active_dimension_value()
        if (
            value == getattr(self, "_drag_input_value", None)
            and current_value == getattr(self, "_drag_stored_value", None)
        ) or math.isclose(value, current_value, rel_tol=1.0e-12, abs_tol=1.0e-12):
            return
        self._set_active_dimension_value(value)
        # RNA stores FloatProperty values at lower precision than the drag
        # calculation. Compare repeated input to its last stored value too.
        self._drag_input_value = value
        self._drag_stored_value = self._active_dimension_value()
        self._numeric_reference = (
            self._format_conic_angle()
            if active == 'ANGLE'
            else self._format_dimension(context, self._active_dimension_value())
        )
        self._numeric_input = self._numeric_reference
        self._numeric_replace = True
        self._sync_dimension_text(context)
        self._build_cylinder_preview(context)
        self._drag_preview_pending = True
        MESH_OT_holey_tools_resize_hole._flush_drag_preview(self, context)
        self._set_status_text(context)

    def _lock_selection_and_fit(self, context, build_preview=True):
        """Commit the candidate wall region and initialize dimension-editing state.

        The method fits the circular cone and boundary planes, makes the candidate
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
        self._end_planes = self._fit_boundary_planes(faces, fit["axis"], fit)
        self._resize_projection_ready = False
        self._resize_dynamic_requires_boolean = False
        self._build_resize_projection_state(bm, faces)
        selected_verts = {vert for face in faces for vert in face.verts}
        bm.verts.ensure_lookup_table()
        selected_verts.update(
            bm.verts[index]
            for index in getattr(
                self, "_resize_bore_vertex_indices", set()
            )
            if index < len(bm.verts)
        )
        self._capture_deformation_state(context, bm, selected_verts)
        self._store_fit_for_redo()
        self._modal_initial_dimensions = {
            "diameter": float(self.diameter),
            "depth": float(self.depth),
            "truncated_depth": float(self.truncated_depth),
            "conic_angle": float(self.conic_angle),
        }
        self._stage = 'CYLINDER'
        self._hover_cache = self._hover_region = self._hover_preview_region = None
        self._begin_numeric_input(context)
        if build_preview:
            self._build_cylinder_preview(context)
        self._set_status_text(context)
        return True

    def _bind_execute_object(self, active_object):
        """Replace stale redo-time object and mesh references when necessary."""
        # Blender may keep this operator wrapper for Adjust Last Operation after
        # invalidating the Object RNA cached by the modal invocation. Always
        # bind geometry work to the current context object. If it is not the
        # same live RNA object, every transient BMesh/BVH reference belongs to
        # the old mesh and must be rebuilt from the serialized redo properties.
        try:
            previous_pointer = (
                self._object.as_pointer() if self._object is not None else 0
            )
        except ReferenceError:
            previous_pointer = 0
        object_changed = previous_pointer != active_object.as_pointer()
        if object_changed:
            GeometryMixin._discard_boolean_preview_source(self)
        self._object = active_object
        if object_changed:
            MESH_OT_holey_tools_resize_hole._reset_execute_transients(self)
        return object_changed

    @staticmethod
    def _reset_execute_transients(operator):
        """Discard mesh-derived state before rebuilding a Redo execution."""
        reset_mesh_derived_state(operator)
        GeometryMixin._resolve_resize_method(operator)

    def execute(self, context):
        """Apply stored dimensions during confirmation or Redo execution.

        Transient fit state is restored from RNA properties when necessary. If a
        persistent overlay exists, its geometry and label are refreshed from the
        newly executed value without retaining this operator wrapper.
        """
        global _persistent_preview
        active_object = context.active_object
        if (
            active_object is None
            or active_object.type != 'MESH'
            or context.mode != 'EDIT_MESH'
        ):
            self.report({'WARNING'}, "Use Resize Hole in Mesh Edit Mode")
            return {'CANCELLED'}

        if _persistent_preview is not None:
            _persistent_preview.restore_source(active_object)

        object_changed = MESH_OT_holey_tools_resize_hole._bind_execute_object(
            self, active_object
        )
        if (
            not object_changed
            and getattr(self, "_boolean_preview_source", None) is not None
        ):
            GeometryMixin._replace_boolean_preview_with_source(
                self, consume=True
            )
        MESH_OT_holey_tools_resize_hole._reset_execute_transients(self)
        if not MESH_OT_holey_tools_resize_hole._restore_fit_for_redo(self):
            self.report({'WARNING'}, "Choose a cylindrical or conical wall first")
            return {'CANCELLED'}

        try:
            MESH_OT_holey_tools_resize_hole._apply_diameter_to_mesh(
                self, context, commit_boolean=True
            )
        except FairingError as error:
            self.report({'WARNING'}, str(error))
            return {'CANCELLED'}

        # Rebuild on the current operator wrapper, then copy only plain Python GPU
        # data into the persistent preview. No RNA operator survives past redo.
        if _persistent_preview is not None:
            # Redo-panel edits update RNA properties directly rather than passing
            # through modal input. Keep both overlay values synchronized.
            if (
                (self._active_dimension == 'DEPTH' and not self.depth_valid)
                or (
                    self._active_dimension == 'TRUNCATED_DEPTH'
                    and not (self.angle_valid and self.depth_valid)
                )
                or (self._active_dimension == 'ANGLE' and not self.angle_valid)
            ):
                self._active_dimension = (
                    'DEPTH'
                    if self.angle_valid and self.depth_valid
                    else 'DIAMETER'
                )
            self._numeric_input = (
                self._format_conic_angle()
                if self._active_dimension == 'ANGLE'
                else self._format_dimension(context, self._active_dimension_value())
            )
            self._numeric_reference = self._numeric_input
            self._sync_dimension_text(context)
            self._build_cylinder_preview(context)
            _persistent_preview.update_from(self, context)
        return {'FINISHED'}

    def cancel(self, context):
        """Rollback an abandoned modal operation, including extension cleanup."""
        if getattr(self, '_runtime_cancelled', False):
            return
        self._runtime_cancelled = True
        try:
            self._restore_original_geometry()
            self._restore_initial_selection()
        finally:
            try:
                GeometryMixin._discard_boolean_preview_source(self)
            finally:
                self._finish_modal(context)

    def modal(self, context, event):
        """Route View3D events through selection and cylinder-editing stages.

        Escape restores geometry and selection, viewport-navigation events pass
        through to Blender, and confirmation transfers the overlay before normal
        operator execution. Pointer dragging can begin with the selecting click
        or with any later press on the on-screen gizmo.
        """
        if getattr(self, '_runtime_cancelled', False):
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            MESH_OT_holey_tools_resize_hole.cancel(self, context)
            return {'CANCELLED'}

        region = getattr(self, "_region", None)
        if region is not None and hasattr(event, "mouse_x"):
            self._mouse_position = (
                event.mouse_x - region.x,
                event.mouse_y - region.y,
            )
            hud_state = getattr(self, "_flood_angle_hud_state", None)
            if (
                event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}
                and hud_state
                and time.monotonic() < hud_state["until"]
            ):
                self._tag_redraw(context)

        # Navigation must be offered to Blender before text input or tool-state
        # handling. Trackpad gestures and NDOF events are continuous event
        # streams, rather than aliases for the mouse wheel or middle button.
        if _is_view_navigation_event(event.type) or event.type == 'Z':
            return {'PASS_THROUGH'}

        if event.type == 'TIMER':
            timer = getattr(self, "_drag_preview_timer", None)
            if (
                self._stage == 'CYLINDER'
                and self._drag_active
                and timer is not None
                and getattr(event, "timer", None) == timer
            ):
                MESH_OT_holey_tools_resize_hole._flush_drag_preview(self, context)
            return {'PASS_THROUGH'}

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            if self._stage == 'SELECT':
                self._update_hover(context, event)
            elif self._stage == 'CYLINDER' and self._drag_active:
                self._update_dimension_drag(context, event)
            elif self._stage == 'CYLINDER':
                hovered_dimension = self._dimension_gizmo_at_event(event)
                cursor = {
                    'DIAMETER': 'SCROLL_X',
                    'DEPTH': 'SCROLL_Y',
                    'ANGLE': 'SCROLL_X',
                }.get(hovered_dimension, 'CROSSHAIR')
                context.window.cursor_modal_set(cursor)

        elif (
            self._stage == 'SELECT'
            and event.type in {'LEFT_BRACKET', 'RIGHT_BRACKET'}
            and event.value == 'PRESS'
        ):
            direction = -1 if event.type == 'LEFT_BRACKET' else 1
            self._change_flood_angle(context, direction, event.shift)
            return {'RUNNING_MODAL'}

        elif (
            self._stage == 'SELECT'
            and self._handle_flood_angle_numeric_event(context, event)
        ):
            return {'RUNNING_MODAL'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            MESH_OT_holey_tools_resize_hole.cancel(self, context)
            return {'CANCELLED'}

        elif (
            self._stage == 'CYLINDER'
            and event.type == 'LEFTMOUSE'
            and event.value == 'RELEASE'
            and self._drag_active
        ):
            self._drag_active = False
            # Dropping the drag flag bypasses the Boolean throttle as well.
            self._drag_preview_pending = bool(
                getattr(self, "_drag_preview_pending", False)
                or getattr(self, "_boolean_preview_pending", False)
            )
            MESH_OT_holey_tools_resize_hole._flush_drag_preview(self, context, force=True)
            MESH_OT_holey_tools_resize_hole._remove_drag_timer(self, context)
            self._resize_wall_cache = None
            return {'RUNNING_MODAL'}

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._stage == 'SELECT':
                # A click can arrive without a preceding mouse-move event (notably
                # just after invoking from the sidebar), so pick once at click time.
                self._update_hover(context, event)
                if not self._lock_selection_and_fit(context, build_preview=True):
                    return {'RUNNING_MODAL'}
                self._begin_dimension_drag(
                    context, event, self._active_dimension
                )
                return {'RUNNING_MODAL'}
            touched_dimension = self._dimension_gizmo_at_event(event)
            if touched_dimension is not None:
                self._begin_dimension_drag(context, event, touched_dimension)
                return {'RUNNING_MODAL'}
            if self._cancel_unchanged_modal(context):
                return {'CANCELLED'}
            self._end_modal_with_persistent_preview(context)
            return self.execute(context)

        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            if self._stage == 'CYLINDER':
                if self._cancel_unchanged_modal(context):
                    return {'CANCELLED'}
                self._end_modal_with_persistent_preview(context)
                return self.execute(context)

        elif (
            self._stage == 'CYLINDER'
            and self._handle_dimension_step_event(context, event)
        ):
            return {'RUNNING_MODAL'}

        elif self._stage == 'CYLINDER' and self._handle_numeric_event(context, event):
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        """Initialize modal state, drawing, picking, and the first hover preview.

        Every invocation starts at scale one with no fitted diameter. Existing
        persistent drawing is retired, source selection is captured, and the
        View3D WINDOW region is saved because panel invocation supplies a sidebar
        context. A selected face seeds an immediate preview when available;
        Ctrl/Cmd invocation instead locks the exact visible face selection.
        """
        global _modal_active, _persistent_preview
        if not self.poll(context):
            self.report({'WARNING'}, "Use Resize Hole in Mesh Edit Mode")
            return {'CANCELLED'}

        self._object = context.active_object
        self._runtime_cancelled = False
        self._hover_cache = self._hover_region = self._hover_preview_region = None
        bm = bmesh.from_edit_mesh(self._object.data)
        bm.faces.ensure_lookup_table()
        use_selection = bool(getattr(event, "ctrl", False) or getattr(event, "oskey", False))
        selected_faces = [face for face in bm.faces if face.select and not face.hide]
        if use_selection and not selected_faces:
            self.report({'WARNING'}, "Select cylindrical or conical wall faces before Ctrl/Cmd-clicking Resize Hole")
            return {'CANCELLED'}

        # A prior fitted preview belongs to the previous last operation.
        if _persistent_preview is not None:
            _persistent_preview.remove()
            _persistent_preview = None
        self._bvh, self._bvh_faces = (None, ()) if use_selection else visible_bvh(bm)
        self._initial_selection = {face.index for face in bm.faces if face.select}
        reset_mesh_derived_state(self)
        self._candidate_indices = set()
        self._hover_face_index = -1
        self._modal_initial_dimensions = None
        self._boolean_preview_source = None
        self._boolean_reconstruction_source = None
        self._boolean_reconstruction_blind_coordinates = None
        self._boolean_preview_display_object = None
        self._boolean_preview_cutter_object = None
        self._boolean_preview_modifier_name = ""
        self._boolean_preview_segments = 0
        self._boolean_preview_last_update = 0.0
        self._boolean_preview_pending = False
        self.fit_valid = False
        self.scale = 1.0
        self.diameter = 0.0
        self.depth = 0.0
        self.truncated_depth = 0.0
        self.conic_depth_previous = 0.0
        self.conic_truncated_depth_previous = 0.0
        self.conic_angle_previous = 0.0
        self.conic_angle = 0.0
        # Keep the mode supplied by the sidebar button (or an explicit caller).
        self.scaling_final_simplify = False
        self.fit_angular_segments = 0
        self.boolean_segments = 32
        self.depth_valid = False
        self.angle_valid = False
        self.fit_is_stud = False
        self.fit_is_cone = False
        self.fit_slope = 0.0
        self.fit_radius_coordinate = 0.0
        self.fit_depth_initial = 0.0
        self.fit_depth_outer = 0.0
        self.fit_depth_inner = 0.0
        self.fit_depth_outer_plane = 0
        self.fit_conic_depth_initial = 0.0
        self.fit_truncated_depth_initial = 0.0
        self.fit_conic_bottom_radius = 0.0
        self.fit_face_indices = ""
        self.fit_end_planes = ""
        self.reconstruction_result_method = ""
        self.reconstruction_result_fill_style = ""
        self.reconstruction_grid_edge_length = 0.0
        self._stage = 'SELECT'
        self._numeric_input = ""
        self._numeric_reference = ""
        self._numeric_replace = True
        self._diameter_text = ""
        self._depth_text = ""
        self._truncated_depth_text = ""
        self._conic_angle_text = ""
        self._active_dimension = 'DIAMETER'
        self._angle_numeric_input = f"{math.degrees(self.flood_angle):.1f}"
        self._angle_numeric_replace = True
        self._drag_active = False
        self._drag_preview_timer = None
        self._drag_preview_pending = False
        self._drag_preview_last_update = None
        self._resize_wall_cache = None
        self._drag_moved = False
        self._gizmo_rect = None
        self._gizmo_rects = {}
        self._mouse_entered_view = False
        self._flood_angle_hud_state = {"until": 0.0, "timer_running": False}
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

        self._mouse_position = (
            event.mouse_x - self._region.x,
            event.mouse_y - self._region.y,
        )

        if use_selection:
            self._candidate_indices = {face.index for face in selected_faces}
            if not self._lock_selection_and_fit(context, build_preview=False):
                self._restore_initial_selection()
                _modal_active = False
                context.workspace.status_text_set(None)
                return {'CANCELLED'}

        begin_modal_session(self, context)
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_preview, (), 'WINDOW', 'POST_PIXEL'
        )
        _draw_handles.add(self._draw_handle)
        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)

        # A panel button cannot provide a mesh-space hover coordinate. Seed the
        # first GPU preview from the active selected face (or any selected face),
        # then let mouse hover take over once it enters the viewport.
        if use_selection:
            self._build_cylinder_preview(context)
        else:
            active_face = bm.faces.active
            if active_face is not None and active_face in selected_faces:
                self._set_hover_seed(context, active_face)
            elif selected_faces:
                self._set_hover_seed(context, selected_faces[0])
            self._update_hover(context, event)
        self._set_status_text(context)
        return {'RUNNING_MODAL'}


def cleanup_runtime():
    """Cancel live sessions and release preview, timer, and drawing resources.

    Live operators are owned only until cancellation/confirmation. Completed
    operators are never retained; their redo drawing has its separate owner.
    """
    global _modal_active, _persistent_preview
    cancel_modal_sessions()
    for manager, timer in _drag_timers:
        try:
            manager.event_timer_remove(timer)
        except (ReferenceError, ValueError):
            pass
    _drag_timers.clear()
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
