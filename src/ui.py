# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Holey Tools sidebar UI and scene settings."""

import bpy

from .constants import ADDON_VERSION, DIAMETER_ESTIMATOR_ITEMS


class VIEW3D_PT_holey_tools(bpy.types.Panel):
    bl_label = "Holey Tools"
    bl_idname = "VIEW3D_PT_holey_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Holey Tools"

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"Version {ADDON_VERSION}")
        layout.prop(
            context.scene,
            "holey_tools_diameter_estimator",
            text="Diameter Estimate",
        )
        operator = layout.operator("mesh.holey_tools_resize_hole")
        operator.diameter_estimator = context.scene.holey_tools_diameter_estimator
        layout.label(text="[ / ] adjusts flood angle")
        layout.label(text="Click wall: apply and adjust")


def register_properties():
    bpy.types.Scene.holey_tools_diameter_estimator = bpy.props.EnumProperty(
        name="Diameter Estimate",
        description="How to estimate diameter when the selected wall has multiple radii",
        items=DIAMETER_ESTIMATOR_ITEMS,
        default='AVERAGE',
    )


def unregister_properties():
    if hasattr(bpy.types.Scene, "holey_tools_diameter_estimator"):
        del bpy.types.Scene.holey_tools_diameter_estimator
