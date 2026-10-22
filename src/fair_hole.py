# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Modal non-manifold loop selection and adaptive hole fairing operator."""

import bmesh
import bpy

from bpy_extras import view3d_utils
from mathutils import Vector

from . import operator as resize_runtime
from .constants import ADDON_VERSION, DEVELOPMENT_CONTROLS
from .fairing import (
    FairingError,
    # Compatibility exports used by existing headless clients.
    _limited_dissolve_analytic_boundary as _limited_dissolve_analytic_boundary,
    _limited_dissolve_cylinder_patch as _limited_dissolve_cylinder_patch,
    boundary_component,
    choose_hypothesis,
    ordered_boundary_loop,
)
from .fairing_pipeline import fair_hole_patch
from .geometry import GeometryMixin
from .gpu_preview import GpuPreviewMixin, _ui_scale
from .preview_cache import clear_draw_cache
from .mesh_state import restore_mesh_contents
from .modal_session import begin_modal_session, end_modal_session


_METHOD_LABELS = {
    'GRID_PREP': "grid boundary preparation",
    'GRID_ROWS': "refined-edge row patches",
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
    """Select a boundary for staged quad row fill or analytic fairing."""

    bl_idname = "mesh.holey_tools_fair_hole"
    bl_label = "Fair Hole"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}
    bl_description = (
        "Select an open boundary for inspectable grid preparation and row fill, "
        "or choose Auto/Circle for surface fairing"
    )

    edge_indices: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    fairing_method: bpy.props.EnumProperty(
        name="Fit",
        description="Surface fit used to fair the selected opening",
        items=(
            ('AUTO', "Auto Fit", "Choose the lowest-complexity accepted fit"),
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
        name="Topology",
        description="Topology used inside the selected opening",
        items=(
            (
                'AUTO',
                "Auto Topology",
                "Choose topology from the selected fit",
            ),
            ('GRID', "Grid", "Inspect boundary preparation and fill refined quad rows"),
            ('CIRCLE', "Circle", "Use concentric rings and a center fan"),
        ),
        default='AUTO',
        options={'SKIP_SAVE'},
    )
    grid_step: bpy.props.EnumProperty(
        name="Stop After",
        description="Cumulative grid preparation or row-fill stage to inspect",
        items=(
            ('DISSOLVE', "1. Dissolve Two-Edge Verts", "Remove boundary vertices with only two connected edges"),
            ('PENTAGONS', "2. Repair Pentagon Corners", "Also intersect and merge exposed pentagon corners"),
            ('SLIDE', "3. Slide and Refine T Points", "Align border vertices, then run three T-only refinement passes before triangle completion"),
            ('REFINE', "4. Refined Edge", "Complete triangle corners and one fixed post-mirror pass"),
            ('ROWS', "5. Fill Quad Rows", "Fill the refined edge row by row with local count transitions and surface projection"),
        ),
        default='ROWS',
        options={'SKIP_SAVE'},
    )
    grid_fill_rows: bpy.props.IntProperty(
        name="Rows to Fill",
        description="Stop after this many rows for inspection; zero fills the complete patch",
        default=0, min=0, max=256, options={'SKIP_SAVE'},
    )
    grid_relaxation_iterations: bpy.props.IntProperty(
        name="Interior Relaxation",
        description="Relax interior and unpinned refined-rim vertices on the frozen fitted surface; zero disables smoothing",
        default=20, min=0, max=100, options={'SKIP_SAVE'},
    )
    grid_relaxation_strength: bpy.props.FloatProperty(
        name="Relaxation Strength", description="Fraction of each neighbor-average move, with backtracking to prevent folds",
        default=.5, min=0., max=1., options={'SKIP_SAVE'},
    )
    grid_match_curvature: bpy.props.BoolProperty(
        name="Match Curvature (C2)",
        description="Estimate rim slopes and curvature from exterior faces; matching is approximate, with 3D bending fallback for folded height charts",
        default=True, options={'SKIP_SAVE'},
    )
    result_grid_rows_filled: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_rows_total: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_mutations: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_corner_transitions: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_triangles: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_row_method: bpy.props.StringProperty(default="", options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_side_counts: bpy.props.StringProperty(default="", options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_template_size: bpy.props.StringProperty(default="", options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_cuts: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_trimmed_cells: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_relaxation_passes: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_analysis_error: bpy.props.StringProperty(default="", options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_surface_continuity: bpy.props.StringProperty(default="", options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_refinement_counts: bpy.props.StringProperty(default="", options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_t_refinement_counts: bpy.props.StringProperty(default="", options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_dissolved: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_merged: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_collapsed_merged: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_repaired: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_skipped: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_slid: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_bowties: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_mirrored: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_notch_caps: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    result_grid_step3_skipped: bpy.props.IntProperty(default=0, options={'HIDDEN', 'SKIP_SAVE'})
    circle_loops: bpy.props.IntProperty(
        name="Circle Loops",
        description="Number of concentric quad loops before the center fan",
        default=5,
        min=0,
        max=64,
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
        name="Topology Result", default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    result_grid_layout: bpy.props.StringProperty(
        name="Grid Layout", default="", options={'HIDDEN', 'SKIP_SAVE'}
    )
    result_grid_edge_length: bpy.props.FloatProperty(
        name="Estimated Edge Size", default=0.0,
        subtype='DISTANCE', unit='LENGTH', options={'HIDDEN', 'SKIP_SAVE'},
    )
    result_residual: bpy.props.FloatProperty(
        name="Fit Residual", default=0.0, options={'HIDDEN', 'SKIP_SAVE'}
    )

    _draw_handle = None
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

    def draw(self, context):
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
        elif effective_fill == 'GRID' and not DEVELOPMENT_CONTROLS:
            if self.result_method not in {'PLANES', 'CONIC', 'SPHERE'}:
                layout.prop(self, "grid_match_curvature")
        elif effective_fill == 'GRID':
            layout.prop(self, "grid_step")
            layout.label(text="Row fill" if self.grid_step == 'ROWS' else "Boundary preparation — hole stays open")
            layout.label(text=f"Dissolved vertices: {self.result_grid_dissolved}")
            if self.grid_step in {'PENTAGONS', 'SLIDE', 'REFINE', 'ROWS'}:
                layout.label(text=f"Ring-face vertices merged (0.001): {self.result_grid_merged}")
                layout.label(text=f"Collapsed corner edges merged: {self.result_grid_collapsed_merged}")
                layout.label(text=f"Repaired pentagons: {self.result_grid_repaired}")
                layout.label(text=f"Skipped pentagons: {self.result_grid_skipped}")
            if self.grid_step in {'SLIDE', 'REFINE', 'ROWS'}:
                layout.label(text=f"Slid vertices: {self.result_grid_slid}")
                layout.label(text=f"Pre-mirror T moves: {self.result_grid_t_refinement_counts}")
            if self.grid_step in {'REFINE', 'ROWS'}:
                layout.label(text=f"Repaired bowties: {self.result_grid_bowties}")
                layout.label(text=f"Completed triangles: {self.result_grid_mirrored}")
                layout.label(text=f"Single-cell notch caps: {self.result_grid_notch_caps}")
                layout.label(text=f"Skipped slide/triangle cases: {self.result_grid_step3_skipped}")
                layout.label(text=f"Post-mirror moves: {self.result_grid_refinement_counts}")
                layout.label(text=f"U1 / V1 / U2 / V2: {self.result_grid_side_counts or '—'}")
                layout.label(text=f"Grid template: {self.result_grid_template_size or '—'}")
                if self.result_grid_analysis_error:
                    layout.label(text=self.result_grid_analysis_error, icon='ERROR')
            if self.grid_step == 'ROWS':
                layout.prop(self, "grid_fill_rows")
                layout.prop(self, "grid_relaxation_iterations")
                if self.grid_relaxation_iterations:
                    layout.prop(self, "grid_relaxation_strength")
                if self.result_method not in {'PLANES', 'CONIC', 'SPHERE'}:
                    layout.prop(self, "grid_match_curvature")
                continuation = {
                    'ANALYTIC': 'Selected analytic fit',
                    'C2_ESTIMATED': 'C2 continuation (estimated)',
                    'INTRINSIC_BENDING': '3D bending continuation (folded chart)',
                }.get(self.result_grid_surface_continuity, 'Thin plate')
                layout.label(text=f"Surface: {continuation}")
                layout.label(text="Layout: Manhattan Grid")
                layout.label(text=f"Rows filled: {self.result_grid_rows_filled} / {self.result_grid_rows_total}")
                layout.label(text=f"Mutation patches: {self.result_grid_mutations}")
                layout.label(text=f"Corner transitions: {self.result_grid_corner_transitions}")
                layout.label(text=f"Transition triangles: {self.result_grid_triangles}")
                layout.label(text=f"Required boundary cuts: {self.result_grid_cuts}")
                layout.label(text=f"Encroached template cells: {self.result_grid_trimmed_cells}")
        if effective_fill != 'GRID':
            layout.prop(self, "final_simplify")
        if self.result_grid_surface_continuity == 'INTRINSIC_BENDING' and (
                effective_fill != 'GRID' or not DEVELOPMENT_CONTROLS):
            layout.label(text="Surface: 3D bending continuation (folded chart)")
        layout.separator()
        label = _METHOD_LABELS.get(self.result_method, self.result_method or "—")
        layout.label(text=f"Fit: {label}")
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
        clear_draw_cache(self)
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
        end_modal_session(self)
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
            if edge.hide or len(edge.link_faces) != 1 or edge.link_faces[0].hide:
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
            "Click/Enter: apply  •  MMB/Wheel/Trackpad: navigate  •  Esc: cancel"
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
        # Include current Edit Mode coordinates/topology in the rollback mesh.
        # A boundary merge can invalidate a tiny opening, so a stale object-mode
        # snapshot must not discard edits made before invoking this operation.
        obj.update_from_editmode()
        backup = obj.data.copy()
        try:
            indices = {
                int(value) for value in self.edge_indices.split(',') if value
            }
            edges = {bm.edges[index] for index in indices}
            loop_verts, loop_edges = ordered_boundary_loop(edges=edges)

            def fit_hypothesis(verts, rim, forced_method=None):
                return choose_hypothesis(
                    verts, rim, obj.matrix_world,
                    lambda faces: GeometryMixin._fit_cylinder(self, faces),
                    lambda face: GeometryMixin._world_normal(self, face),
                    forced_method=forced_method or getattr(self, "fairing_method", 'AUTO'),
                )

            fill_style = getattr(self, "fill_style", 'AUTO')
            hypothesis = fit_hypothesis(loop_verts, loop_edges)
            bm, result = fair_hole_patch(
                obj, bm, loop_verts, loop_edges, hypothesis, fill_style,
                circle_loops=getattr(self, "circle_loops", 5),
                final_simplify=getattr(self, "final_simplify", True),
                refit=fit_hypothesis,
                grid_step=getattr(self, "grid_step", 'ROWS') if DEVELOPMENT_CONTROLS else 'ROWS',
                fill_rows=getattr(self, "grid_fill_rows", 0) if DEVELOPMENT_CONTROLS else 0,
                relaxation_iterations=getattr(self, "grid_relaxation_iterations", 20) if DEVELOPMENT_CONTROLS else 20,
                relaxation_strength=getattr(self, "grid_relaxation_strength", .5) if DEVELOPMENT_CONTROLS else .5,
                match_curvature=getattr(self, "grid_match_curvature", True),
            )
        except (FairingError, IndexError, RuntimeError, ValueError) as error:
            if obj.mode == 'EDIT':
                bpy.ops.object.mode_set(mode='OBJECT')
            restore_mesh_contents(obj, backup)
            bpy.data.meshes.remove(backup)
            backup = None
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
            face.select_set(True)
        for vert in result["verts"]:
            vert.select = True
        if result["method"] in {'GRID_PREP', 'GRID_ROWS'}:
            for edge in result["boundary_edges"]:
                edge.select_set(True)
        self.result_grid_dissolved = result.get("dissolved", 0)
        self.result_grid_merged = result.get("merged", 0)
        self.result_grid_collapsed_merged = result.get("collapsed_merged", 0)
        self.result_grid_repaired = result.get("repaired", 0)
        self.result_grid_skipped = result.get("skipped", 0)
        self.result_grid_slid = result.get("slid", 0)
        self.result_grid_bowties = result.get("bowties", 0)
        self.result_grid_mirrored = result.get("mirrored", 0)
        self.result_grid_notch_caps = result.get("notch_caps", 0)
        self.result_grid_step3_skipped = result.get("slide_skipped", 0) + result.get("triangle_skipped", 0)
        self.result_grid_refinement_counts = " / ".join(str(count) for count in result.get("refinement_counts", []))
        self.result_grid_t_refinement_counts = " / ".join(str(count) for count in result.get("t_refinement_counts", []))
        self.result_grid_rows_filled = result.get("rows_filled", 0)
        self.result_grid_rows_total = result.get("rows_total", 0)
        self.result_grid_mutations = result.get("mutation_patches", 0)
        self.result_grid_corner_transitions = result.get("corner_transitions", 0)
        self.result_grid_triangles = result.get("transition_triangles", 0)
        self.result_grid_row_method = result.get("row_layout_method", "")
        self.result_grid_side_counts = " / ".join(map(str, result.get("side_counts", ())))
        self.result_grid_template_size = " × ".join(map(str, result.get("grid_size", ())))
        self.result_grid_cuts = result.get("grid_cut_count", 0)
        self.result_grid_trimmed_cells = result.get("grid_trimmed_cells", 0)
        self.result_grid_relaxation_passes = result.get("relaxation_passes", 0)
        self.result_grid_analysis_error = result.get("grid_analysis_error", "")
        self.result_grid_surface_continuity = result.get("surface_continuity", "")
        self.result_method = result.get("surface_method", result["method"])
        self.result_fill_style = result["fill_style"]
        self.result_grid_layout = result.get("grid_layout", "")
        self.result_grid_edge_length = result.get("grid_estimated_edge_length", 0.0)
        self.result_residual = result["residual"]
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
        bpy.data.meshes.remove(backup)
        label = _METHOD_LABELS[result["method"]]
        if result["method"] == 'GRID_PREP':
            step = {'DISSOLVE': 1, 'PENTAGONS': 2, 'SLIDE': 3, 'REFINE': 4}[result['grid_step']]
            self.report({'INFO'}, (
                f"Grid step {step}: "
                f"{result['dissolved']} vertices dissolved, {result['repaired']} pentagons repaired, "
                f"{result.get('merged', 0)} ring-face vertices merged, "
                f"{result.get('collapsed_merged', 0)} collapsed corner edges merged, "
                f"{result.get('slid', 0)} vertices slid, {result.get('bowties', 0)} bowties repaired, "
                f"{result.get('mirrored', 0)} triangles mirrored, "
                f"{sum(result.get('t_refinement_counts', []))} pre-mirror T moves, "
                f"{sum(result.get('refinement_counts', []))} post-mirror refinement moves; hole left open"
            ))
        elif result["method"] == 'GRID_ROWS':
            self.report({'INFO'}, (
                f"Grid rows: {result['rows_filled']} / {result['rows_total']}, "
                f"{result['mutation_patches']} local mutation patches, "
                f"{result['transition_triangles']} transition triangles"
            ))
        else:
            self.report({'INFO'}, f"Filled hole using {label} fairing")
        return {'FINISHED'}

    def cancel(self, context):
        if getattr(self, '_runtime_cancelled', False):
            return
        self._runtime_cancelled = True
        try:
            self._restore_selection()
        finally:
            self._finish_modal(context)

    def modal(self, context, event):
        if getattr(self, '_runtime_cancelled', False):
            return {'CANCELLED'}
        if context.area is None or context.area.type != 'VIEW_3D':
            MESH_OT_holey_tools_fair_hole.cancel(self, context)
            return {'CANCELLED'}
        if resize_runtime._is_view_navigation_event(event.type) or event.type == 'Z':
            return {'PASS_THROUGH'}
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._update_hover(context, event)
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            MESH_OT_holey_tools_fair_hole.cancel(self, context)
            return {'CANCELLED'}
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            self._update_hover(context, event)
            return self._confirm_candidate(context)
        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            return self._confirm_candidate(context)
        return {'RUNNING_MODAL'}

    def _reset_invocation_defaults(self, scene):
        self.fill_style = scene.holey_tools_fair_fill
        self.fairing_method = scene.holey_tools_fair_surface
        self.grid_step = 'ROWS'
        self.grid_fill_rows = 0
        self.circle_loops = 5

    def invoke(self, context, event):
        if not self.poll(context):
            self.report({'WARNING'}, "Use Fair Hole in Mesh Edit Mode")
            return {'CANCELLED'}
        if resize_runtime._modal_active:
            self.report({'WARNING'}, "Another Holey Tools operation is already active")
            return {'CANCELLED'}
        # Blender remembers prior redo values even for SKIP_SAVE properties.
        # Each new interactive invocation takes only the sidebar preselection.
        self._reset_invocation_defaults(context.scene)
        self._runtime_cancelled = False
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
        begin_modal_session(self, context)

        selected = [
            edge for edge in bm.edges
            if edge.select and not edge.hide and len(edge.link_faces) == 1 and not edge.link_faces[0].hide
        ]
        if selected:
            self._set_candidate(context, selected[0])
        self._update_hover(context, event)
        self._set_status_text(context)
        return {'RUNNING_MODAL'}
