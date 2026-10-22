# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Invocation-owned screen-space projection, shaders, and bounded GPU batches."""

from dataclasses import dataclass

import bpy
import gpu
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader


def convex_hull_lines(points):
    """Return a projected silhouette without longitudinal cylinder edges."""
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 2:
        return []

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower, upper = [], []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return [(point[0], point[1], 0.0)
            for index, first in enumerate(hull)
            for point in (first, hull[(index + 1) % len(hull)])]


def project_elements(elements, region, region_data, points=None):
    """Project each unique point once; omit elements with clipped endpoints."""
    points = {} if points is None else points
    projected = []
    for element in elements or ():
        target = []
        for point in element:
            key = tuple(point)
            if key not in points:
                pixel = view3d_utils.location_3d_to_region_2d(region, region_data, point)
                points[key] = None if pixel is None else (pixel.x, pixel.y, 0.0)
            target.append(points[key])
        if all(point is not None for point in target):
            projected.extend(target)
    return projected


@dataclass
class ProjectedPreview:
    fill_batch: object
    line_batch: object
    label_points: list


class PreviewDrawCache:
    """Retain only the latest projection and one batch per named HUD element.

    Geometry builders replace their world-space lists, never mutate them after
    publication. Strong references let identity detect replacement without
    hashing all vertices or risking reuse of a retired list's Python ID.
    Nothing here retains a Blender operator, scene object, or BMesh.
    """

    def __init__(self, window_pointer=None):
        self.window_pointer = window_pointer
        self._shaders = {}
        self._hud_batches = {}
        self._geometry_key = None
        self._fill_coords = None
        self._line_coords = None
        self._projected = None

    def shader(self, name):
        if name not in self._shaders:
            self._shaders[name] = gpu.shader.from_builtin(name)
        return self._shaders[name]

    def batch(self, slot, shader_name, primitive, coords):
        """A slot is a fixed HUD role, not a growing history of coordinates."""
        key = (shader_name, primitive, tuple(coords))
        previous = self._hud_batches.get(slot)
        if previous is None or previous[0] != key:
            batch = batch_for_shader(self.shader(shader_name), primitive, {"pos": coords})
            self._hud_batches[slot] = (key, batch)
        return self._hud_batches[slot][1]

    def geometry(self, fill_coords, line_coords, stage, region, region_data):
        """Rebuild screen-space batches only for geometry/camera/region changes."""
        key = (region.as_pointer(), region.width, region.height, stage,
               tuple(value for row in region_data.perspective_matrix for value in row))
        if (self._projected is not None and key == self._geometry_key
                and fill_coords is self._fill_coords and line_coords is self._line_coords):
            return self._projected

        points = {}
        triangles = project_elements(fill_coords, region, region_data, points)
        lines = project_elements(line_coords, region, region_data, points)
        if stage == 'CYLINDER':
            lines.extend(convex_hull_lines(triangles))
        width, height = max(1, region.width), max(1, region.height)

        def to_ndc(coords):
            return [(2.0 * x / width - 1.0, 2.0 * y / height - 1.0, 0.0)
                    for x, y, _z in coords]

        fill_batch = (batch_for_shader(self.shader('UNIFORM_COLOR'), 'TRIS',
                                      {"pos": to_ndc(triangles)}) if triangles else None)
        line_batch = (batch_for_shader(self.shader('POLYLINE_UNIFORM_COLOR'), 'LINES',
                                      {"pos": to_ndc(lines)}) if lines else None)
        result = ProjectedPreview(fill_batch, line_batch, triangles)
        # Publish only a complete build: transient GPU failures can be retried.
        self._fill_coords, self._line_coords = fill_coords, line_coords
        self._geometry_key, self._projected = key, result
        return result


def draw_cache(owner):
    """Keep GPU resources on their drawing owner and inside its window context."""
    window = bpy.context.window
    pointer = window.as_pointer() if window is not None else None
    cache = getattr(owner, '_gpu_preview_cache', None)
    if cache is None or cache.window_pointer != pointer:
        cache = owner._gpu_preview_cache = PreviewDrawCache(pointer)
    return cache


def clear_draw_cache(owner):
    """Drop all GPU and projected geometry references when an overlay retires."""
    owner._gpu_preview_cache = None
