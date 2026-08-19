# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Screen-space GPU previews and the diameter drag gizmo."""

import math
import time
import traceback
from numbers import Integral

import blf
import bpy
import gpu

from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix
from mathutils.geometry import tessellate_polygon

from .constants import ADDON_VERSION, DIAMETER_SYMBOL, FALLBACK_RETOPOLOGY_COLOR


def _ui_scale():
    """Return Blender's DPI-aware scale for custom interface elements."""
    try:
        return max(0.25, float(bpy.context.preferences.system.ui_scale))
    except (AttributeError, TypeError, ValueError):
        return 1.0


def _line_width(base_width=2.0):
    """Return a DPI-aware GPU line width using Blender's pixel recommendation."""
    try:
        pixel_size = max(1.0, float(bpy.context.preferences.system.pixel_size))
    except (AttributeError, TypeError, ValueError):
        pixel_size = 1.0
    return base_width * pixel_size


def _viewport_size(region):
    """Return the active framebuffer dimensions, with a region-size fallback."""
    try:
        _x, _y, width, height = gpu.state.viewport_get()
    except (AttributeError, TypeError, ValueError):
        width, height = region.width, region.height
    return max(1, width), max(1, height)


def _face_retopology_colors():
    """Return theme-derived fill and outline colors for preview overlays.

    Blender's Face Retopology color supplies the RGB and starting alpha. The
    fill alpha is strengthened for visibility, while the outline receives a
    higher minimum opacity. A stock-theme fallback keeps drawing functional if
    the theme property is unavailable.
    """
    try:
        theme_color = tuple(
            bpy.context.preferences.themes[0].view_3d.face_retopology
        )
    except (AttributeError, IndexError, TypeError):
        theme_color = FALLBACK_RETOPOLOGY_COLOR
    preview_alpha = min(1.0, max(0.15, theme_color[3] * 2.5))
    fill = (theme_color[0], theme_color[1], theme_color[2], preview_alpha)
    # Outlines share the theme RGB but remain legible against the stronger fill.
    line = (fill[0], fill[1], fill[2], max(fill[3], 0.85))
    return fill, line


def _convex_hull_lines(points):
    """Return line endpoints around a 2D point cloud's convex hull.

    The monotonic-chain hull is recomputed from projected cylinder vertices on
    each draw. Its outline therefore follows the visible cylinder silhouette as
    the camera moves, without exposing longitudinal segment lines.
    """
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 2:
        return []

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    lines = []
    for index, point in enumerate(hull):
        following = hull[(index + 1) % len(hull)]
        lines.extend(((point[0], point[1], 0.0), (following[0], following[1], 0.0)))
    return lines


class GpuPreviewMixin:
    """Build and draw selection, cylinder, label, and drag-handle overlays.

    Preview geometry is retained in world space and projected during each draw
    callback so it tracks viewport navigation. Drawing occurs in POST_PIXEL
    screen space with depth testing disabled, keeping the tint visible through
    the edited mesh regardless of the viewport X-Ray setting.
    """

    def _triangles_for_faces(self, faces):
        """Tessellate BMesh faces into world-space preview triangles.

        ``tessellate_polygon`` has returned either vertex indices or coordinates
        across supported Blender versions, so both representations are accepted.
        The source mesh and its selection state are not modified.
        """
        matrix_world = self._object.matrix_world
        triangles = []
        for face in faces:
            polygon = [matrix_world @ loop.vert.co for loop in face.loops]
            for triangle in tessellate_polygon([polygon]):
                if triangle and isinstance(triangle[0], Integral):
                    triangles.append(tuple(polygon[index].copy() for index in triangle))
                else:
                    triangles.append(tuple(point.copy() for point in triangle))
        return triangles

    def _build_selection_preview(self, context, faces):
        """Replace the overlay geometry with a filled flood-selection preview.

        Cached GPU batches are invalidated because their coordinates no longer
        match, error throttling is reset, and all relevant viewport regions are
        tagged for redraw.
        """
        triangles = self._triangles_for_faces(faces)
        self._fill_coords = triangles
        self._line_coords = None
        self._fill_batch = None
        self._line_batch = None
        self._draw_error_reported = False
        self._tag_redraw(context)

    def _build_cylinder_preview(self, context):
        """Build a world-space cylinder for the current fitted diameter.

        The wall is stored as filled triangles. Only the two cap circumferences
        are stored as fixed lines; the camera-dependent wall silhouette is added
        after projection during drawing. Segment count scales with the selected
        region but remains bounded to control preview cost.
        """
        fit = self._fit
        radius = fit["radius"] * self._target_scale()
        segments = max(32, min(128, len(self._locked_indices) * 4))
        bottom = []
        top = []

        for index in range(segments):
            angle = math.tau * index / segments
            radial = (
                fit["basis_x"] * math.cos(angle)
                + fit["basis_y"] * math.sin(angle)
            ) * radius
            bottom.append(fit["center"] + fit["axis"] * fit["z_min"] + radial)
            top.append(fit["center"] + fit["axis"] * fit["z_max"] + radial)

        triangles = []
        lines = []
        for index in range(segments):
            following = (index + 1) % segments
            triangles.append((bottom[index], bottom[following], top[following]))
            triangles.append((bottom[index], top[following], top[index]))
            lines.append((bottom[index], bottom[following]))
            lines.append((top[index], top[following]))

        self._fill_coords = triangles
        self._line_coords = lines
        self._fill_batch = None
        self._line_batch = None
        self._draw_error_reported = False
        self._tag_redraw(context)

    def _project_triangles(self, region, region_data):
        """Project complete world-space fill triangles into region pixels.

        Triangles containing an unprojectable point are omitted. The returned
        flat coordinate sequence is ready for a ``TRIS`` GPU batch.
        """
        projected = []
        coords = self._fill_coords or ()
        for source_triangle in coords:
            triangle = [
                view3d_utils.location_3d_to_region_2d(region, region_data, point)
                for point in source_triangle
            ]
            if all(point is not None for point in triangle):
                projected.extend((point.x, point.y, 0.0) for point in triangle)
        return projected

    def _project_lines(self, region, region_data):
        """Project complete world-space line segments into region pixels.

        Lines containing an unprojectable endpoint are omitted. The returned
        flat coordinate sequence is ready for a ``LINES`` GPU batch.
        """
        projected = []
        coords = self._line_coords or ()
        for source_line in coords:
            line = [
                view3d_utils.location_3d_to_region_2d(region, region_data, point)
                for point in source_line
            ]
            if all(point is not None for point in line):
                projected.extend((point.x, point.y, 0.0) for point in line)
        return projected

    def _draw_preview(self):
        """Run the draw callback while reporting a repeated failure only once.

        Exceptions must not escape a Blender draw handler. A successful preview
        rebuild resets ``_draw_error_reported`` so a later, distinct failure can
        still be diagnosed in the console.
        """
        try:
            self._draw_preview_impl()
        except Exception as error:
            if not self._draw_error_reported:
                print(f"Holey Tools {ADDON_VERSION}: GPU preview failed: {error!r}")
                traceback.print_exc()
                self._draw_error_reported = True

    def _draw_preview_impl(self):
        """Project and render the active preview in the current View3D region.

        Pixel coordinates are converted explicitly to normalized-device
        coordinates and drawn under identity matrices for consistent Blender
        5.2 backend behavior. GPU state is restored in ``finally`` before the
        optional cylinder dimension label is drawn.
        """
        region = bpy.context.region
        region_data = bpy.context.region_data
        if region is None or region_data is None:
            return

        # SpaceView3D draw handlers run for every 3D viewport. Restrict this
        # overlay to the WINDOW region where the operator was invoked so split
        # views and other windows do not receive a duplicate preview.
        target_region = getattr(self, "_region", None)
        if target_region is not None:
            try:
                if region.as_pointer() != target_region.as_pointer():
                    return
            except ReferenceError:
                return

        if not self._fill_coords:
            if self._stage == 'SELECT':
                _fill_color, line_color = _face_retopology_colors()
                self._draw_flood_angle_label(region, line_color)
            return

        projected_triangles = self._project_triangles(region, region_data)
        if not projected_triangles:
            return

        width = max(1, region.width)
        height = max(1, region.height)

        def to_ndc(point):
            return (
                2.0 * point[0] / width - 1.0,
                2.0 * point[1] / height - 1.0,
                0.0,
            )

        # Explicit normalized-device coordinates plus identity GPU matrices avoid
        # relying on Blender's implicit POST_PIXEL matrix state, which differs
        # between viewport backends in 5.2.
        label_points = projected_triangles
        projected_lines = self._project_lines(region, region_data)
        if self._stage == 'CYLINDER':
            projected_lines.extend(_convex_hull_lines(projected_triangles))

        triangles = [to_ndc(point) for point in projected_triangles]

        fill_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        line_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        fill_batch = batch_for_shader(
            fill_shader, 'TRIS', {"pos": triangles}
        )
        projected_lines = [to_ndc(point) for point in projected_lines]
        line_batch = (
            batch_for_shader(line_shader, 'LINES', {"pos": projected_lines})
            if projected_lines
            else None
        )

        # POST_PIXEL executes after the 3D scene. Screen-space triangles therefore
        # remain visible independently of mesh depth and the viewport X-Ray setting.
        fill_color, line_color = _face_retopology_colors()
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
        gpu.state.face_culling_set('NONE')
        try:
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_identity()
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                fill_shader.uniform_float("color", fill_color)
                fill_batch.draw(fill_shader)

                if line_batch is not None:
                    line_shader.uniform_float(
                        "viewportSize", _viewport_size(region)
                    )
                    line_shader.uniform_float("lineWidth", _line_width())
                    line_shader.uniform_float("color", line_color)
                    line_batch.draw(line_shader)

        finally:
            gpu.state.face_culling_set('NONE')
            gpu.state.depth_mask_set(True)
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('NONE')

        if self._stage == 'CYLINDER':
            self._draw_dimension_label(region, label_points, line_color)
        elif self._stage == 'SELECT':
            self._draw_flood_angle_label(region, line_color)

    def _draw_flood_angle_label(self, region, color):
        """Draw a transient angle readout and wedge beside the mouse cursor."""
        state = getattr(self, "_flood_angle_hud_state", None)
        if not state or time.monotonic() >= state["until"]:
            return

        font_id = 0
        ui_scale = _ui_scale()
        text = f"Flood Angle: {math.degrees(self.flood_angle):.1f}°"
        blf.size(font_id, 16.0 * ui_scale)
        width, height = blf.dimensions(font_id, text)

        mouse_x, mouse_y = getattr(
            self,
            "_mouse_position",
            (region.width * 0.5, region.height * 0.5),
        )
        margin = 8.0 * ui_scale
        cursor_gap = 18.0 * ui_scale
        wedge_gap = 7.0 * ui_scale
        wedge_radius = 38.0 * ui_scale
        block_width = max(width, wedge_radius * 2.0)
        block_height = height + wedge_gap + wedge_radius * 2.0

        block_x = mouse_x + cursor_gap
        if block_x + block_width > region.width - margin:
            block_x = mouse_x - cursor_gap - block_width
        block_x = min(max(margin, block_x), region.width - block_width - margin)

        block_y = mouse_y + cursor_gap
        if block_y + block_height > region.height - margin:
            block_y = mouse_y - cursor_gap - block_height
        block_y = min(max(margin, block_y), region.height - block_height - margin)

        text_x = block_x + (block_width - width) * 0.5
        pivot = (
            block_x + block_width * 0.5,
            block_y + height + wedge_gap + wedge_radius,
        )
        self._draw_flood_angle_wedge(
            region,
            pivot,
            wedge_radius,
            self.flood_angle,
            color,
        )

        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 5, 0.0, 0.0, 0.0, 0.9)
        shadow_offset = max(1, round(ui_scale))
        blf.shadow_offset(font_id, shadow_offset, -shadow_offset)
        blf.color(font_id, color[0], color[1], color[2], 1.0)
        blf.position(font_id, text_x, block_y, 0.0)
        blf.draw(font_id, text)
        blf.disable(font_id, blf.SHADOW)

    @staticmethod
    def _draw_flood_angle_wedge(region, pivot, radius, angle, color):
        """Draw a normal threshold with green go and red no-go sectors."""
        viewport_width = max(1, region.width)
        viewport_height = max(1, region.height)

        def to_ndc(point):
            return (
                2.0 * point[0] / viewport_width - 1.0,
                2.0 * point[1] / viewport_height - 1.0,
                0.0,
            )

        def sector(start, end, segments=12):
            points = [
                (
                    pivot[0] + radius * math.cos(start + (end - start) * i / segments),
                    pivot[1] + radius * math.sin(start + (end - start) * i / segments),
                )
                for i in range(segments + 1)
            ]
            triangles = []
            for index in range(segments):
                triangles.extend((pivot, points[index], points[index + 1]))
            return [to_ndc(point) for point in triangles]

        green_coords = sector(-angle, angle)
        red_coords = sector(-math.pi * 0.5, -angle) + sector(angle, math.pi * 0.5)
        green_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        red_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        green_batch = batch_for_shader(green_shader, 'TRIS', {"pos": green_coords})
        red_batch = batch_for_shader(red_shader, 'TRIS', {"pos": red_coords})

        upper = (
            pivot[0] + radius * math.cos(angle),
            pivot[1] + radius * math.sin(angle),
        )
        lower = (
            pivot[0] + radius * math.cos(angle),
            pivot[1] - radius * math.sin(angle),
        )
        reference = (pivot[0] - radius, pivot[1])
        line_coords = [
            to_ndc(point)
            for point in (pivot, upper, pivot, lower, pivot, reference)
        ]
        line_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        line_batch = batch_for_shader(line_shader, 'LINES', {"pos": line_coords})

        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        gpu.state.face_culling_set('NONE')
        try:
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_identity()
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))
                red_shader.uniform_float("color", (0.9, 0.12, 0.08, 0.12))
                red_batch.draw(red_shader)
                green_shader.uniform_float("color", (0.12, 0.8, 0.25, 0.18))
                green_batch.draw(green_shader)
                line_shader.uniform_float("viewportSize", _viewport_size(region))
                line_shader.uniform_float("lineWidth", _line_width())
                line_shader.uniform_float("color", color)
                line_batch.draw(line_shader)
        finally:
            gpu.state.face_culling_set('NONE')
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('NONE')

    def _draw_dimension_label(self, region, projected_points, color):
        """Draw the live diameter text, interaction hint, and optional gizmo.

        The block is placed beside the projected cylinder while remaining within
        the region bounds. During direct numeric editing it also records the drag
        handle's pixel hit rectangle; after handoff to Blender's Redo panel only
        the value and updated hint remain.
        """
        text = getattr(self, "_numeric_input", "").strip()
        if not text or not projected_points:
            if getattr(self, "_accepts_numeric_input", True):
                self._gizmo_rect = None
            return

        font_id = 0
        editing = getattr(self, "_accepts_numeric_input", True)
        ui_scale = _ui_scale()
        display_text = f"{DIAMETER_SYMBOL} {text}"
        hint_text = (
            "Type to change"
            if editing
            else "Change in Redo panel"
        )
        blf.size(font_id, 16.0 * ui_scale)
        width, height = blf.dimensions(font_id, display_text)
        blf.size(font_id, 12.0 * ui_scale)
        hint_width, hint_height = blf.dimensions(font_id, hint_text)
        gizmo_width = 68.0 * ui_scale
        block_width = max(width, hint_width, gizmo_width if editing else 0.0)
        xs = [point[0] for point in projected_points]
        ys = [point[1] for point in projected_points]
        margin = 8.0 * ui_scale
        label_gap = 14.0 * ui_scale
        text_gap = 4.0 * ui_scale
        x = max(xs) + label_gap
        if x + block_width > region.width - margin:
            x = max(margin, min(xs) - block_width - label_gap)
        total_height = hint_height + text_gap + height
        minimum_y = (30.0 if editing else 8.0) * ui_scale
        y = min(
            region.height - total_height - margin,
            max(minimum_y, (min(ys) + max(ys)) * 0.5),
        )

        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 5, 0.0, 0.0, 0.0, 0.9)
        shadow_offset = max(1, round(ui_scale))
        blf.shadow_offset(font_id, shadow_offset, -shadow_offset)
        blf.size(font_id, 12.0 * ui_scale)
        blf.color(font_id, color[0], color[1], color[2], 0.8)
        blf.position(font_id, x, y + height + text_gap, 0.0)
        blf.draw(font_id, hint_text)
        blf.size(font_id, 16.0 * ui_scale)
        blf.color(font_id, color[0], color[1], color[2], 1.0)
        blf.position(font_id, x, y, 0.0)
        blf.draw(font_id, display_text)
        blf.disable(font_id, blf.SHADOW)

        if editing:
            gizmo_y = y - 14.0 * ui_scale
            self._gizmo_rect = (
                x - 7.0 * ui_scale,
                gizmo_y - 10.0 * ui_scale,
                x + gizmo_width + 7.0 * ui_scale,
                gizmo_y + 10.0 * ui_scale,
            )
            GpuPreviewMixin._draw_drag_gizmo(
                region, x, gizmo_y, gizmo_width, color, ui_scale
            )

    @staticmethod
    def _draw_drag_gizmo(region, x, y, width, color, ui_scale=1.0):
        """Draw a screen-space bidirectional diameter drag handle.

        Pixel endpoints are converted to normalized-device coordinates and
        rendered under identity matrices, matching the main preview pipeline.
        This method draws only; event hit testing uses ``self._gizmo_rect``.
        """
        left = x
        right = x + width
        inset = 8.0 * ui_scale
        arrow = 6.0 * ui_scale
        tick = 4.0 * ui_scale
        pixel_lines = (
            (left + inset, y), (right - inset, y),
            (left, y), (left + arrow, y + arrow),
            (left, y), (left + arrow, y - arrow),
            (right, y), (right - arrow, y + arrow),
            (right, y), (right - arrow, y - arrow),
            (left + inset, y - tick), (left + inset, y + tick),
            (right - inset, y - tick), (right - inset, y + tick),
        )
        viewport_width = max(1, region.width)
        viewport_height = max(1, region.height)
        coords = [
            (
                2.0 * point[0] / viewport_width - 1.0,
                2.0 * point[1] / viewport_height - 1.0,
                0.0,
            )
            for point in pixel_lines
        ]
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        gpu.state.blend_set('ALPHA')
        try:
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_identity()
                gpu.matrix.load_projection_matrix(Matrix.Identity(4))
                shader.uniform_float(
                    "viewportSize", _viewport_size(region)
                )
                shader.uniform_float("lineWidth", _line_width())
                shader.uniform_float("color", color)
                batch.draw(shader)
        finally:
            gpu.state.blend_set('NONE')

    def _event_over_drag_gizmo(self, event):
        """Return whether a View3D event lies inside the active gizmo rectangle."""
        rect = self._gizmo_rect
        if rect is None:
            return False
        region = getattr(self, "_region", None)
        if region is not None:
            x = event.mouse_x - region.x
            y = event.mouse_y - region.y
        else:
            x = event.mouse_region_x
            y = event.mouse_region_y
        return (
            rect[0] <= x <= rect[2]
            and rect[1] <= y <= rect[3]
        )
