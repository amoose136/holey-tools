# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy

from .operator import MESH_OT_holey_tools_resize_hole, cleanup_runtime
from .ui import (
    VIEW3D_PT_holey_tools,
    register_properties,
    unregister_properties,
)


bl_info = {
    "name": "Holey Tools",
    "author": "Amos Manneschmidt",
    "version": (0, 9, 3),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Holey Tools",
    "description": "Precision tools for editing holes in mesh objects",
    "category": "Mesh",
}


_CLASSES = (
    MESH_OT_holey_tools_resize_hole,
    VIEW3D_PT_holey_tools,
)


def register():
    """Register Holey Tools properties, operators, and panels with Blender."""
    register_properties()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    """Release runtime resources and unregister Holey Tools from Blender."""
    cleanup_runtime()
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    unregister_properties()
