# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scoped invalidation of mesh-derived resize state, not RNA or resources.

These helpers only discard Python caches. Callers must release owned Blender
objects/meshes separately before clearing any native resource references.
"""


def invalidate_projection(operator):
    """Drop topology-dependent surface maps without changing the fitted policy."""
    operator._resize_projection_ready = False
    operator._resize_surfaces_by_vertex = {}
    operator._resize_conic_surfaces_by_vertex = {}
    operator._resize_bore_vertex_indices = set()
    operator._resize_base_bvh = None


def reset_deformation(operator):
    """Forget coordinates and weights after restoring source mesh contents."""
    operator._resize_wall_cache = None
    operator._original_vertex_coords = None
    operator._proportional_weights = None


def reset_mesh_derived_state(operator):
    """Start an invocation/redo fit without resetting user choices or RNA fits."""
    invalidate_projection(operator)
    reset_deformation(operator)
    operator._fit = None
    operator._hover_cache = None
    operator._hover_region = None
    operator._hover_preview_region = None
    operator._end_planes = None
    operator._depth_fit = None
    operator._locked_indices = set()
    operator._resize_requires_boolean = False
    operator._resize_base_requires_boolean = False
    operator._resize_dynamic_requires_boolean = False
    operator._resize_base_span = 0.0
    operator._resize_angular_segments = 0
