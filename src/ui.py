# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Holey Tools sidebar UI and scene settings."""

import math

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
        diameter_estimator = layout.split(factor=0.65, align=True)
        diameter_estimator.label(text="Diameter Estimate:")
        diameter_estimator.prop(
            context.scene,
            "holey_tools_diameter_estimator",
            text="",
        )
        operator = layout.operator("mesh.holey_tools_resize_hole")
        operator.diameter_estimator = context.scene.holey_tools_diameter_estimator
        operator.flood_angle = context.scene.holey_tools_flood_angle
        flood_angle_hint = layout.row(align=True)
        flood_angle_hint.alignment = 'LEFT'
        flood_angle_hint.label(text="", icon='EVENT_LEFTBRACKET')
        flood_angle_hint.label(text="/")
        flood_angle_hint.label(text="", icon='EVENT_RIGHTBRACKET')
        flood_angle_hint.label(
            text=(
                "Flood Angle: "
                f"{math.degrees(context.scene.holey_tools_flood_angle):.1f}°"
            )
        )
        layout.label(text="Click wall: apply and adjust")

def register_properties():
    bpy.types.Scene.holey_tools_flood_angle = bpy.props.FloatProperty(
        name="Flood Angle",
        description="Maximum angle between neighboring faces in the hover flood fill",
        subtype='ANGLE',
        default=math.radians(60.0),
        min=math.radians(0.1),
        max=math.radians(89.9),
    )
    bpy.types.Scene.holey_tools_diameter_estimator = bpy.props.EnumProperty(
        name="Diameter Estimate",
        description="How to estimate diameter when the selected wall has multiple radii",
        items=DIAMETER_ESTIMATOR_ITEMS,
        default='AVERAGE',
    )


def unregister_properties():
    if hasattr(bpy.types.Scene, "holey_tools_flood_angle"):
        del bpy.types.Scene.holey_tools_flood_angle
    if hasattr(bpy.types.Scene, "holey_tools_diameter_estimator"):
        del bpy.types.Scene.holey_tools_diameter_estimator
