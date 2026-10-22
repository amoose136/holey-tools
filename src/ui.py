# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Holey Tools sidebar UI and scene settings."""

import math

import bpy

from .constants import DIAMETER_ESTIMATOR_ITEMS, RESIZE_METHOD_ITEMS


class VIEW3D_PT_holey_tools(bpy.types.Panel):
    bl_label = "Holey Tools"
    bl_idname = "VIEW3D_PT_holey_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Holey Tools"

    def draw(self, context):
        layout = self.layout
        diameter_estimator = layout.split(factor=0.65, align=True)
        diameter_label = diameter_estimator.row()
        diameter_label.alignment = 'RIGHT'
        diameter_label.label(text="Diameter Estimate:")
        diameter_estimator.prop(
            context.scene,
            "holey_tools_diameter_estimator",
            text="",
        )
        resize_mode = layout.split(factor=0.65, align=True)
        mode_label = resize_mode.row()
        mode_label.alignment = 'RIGHT'
        mode_label.label(text="Mode:")
        resize_mode.prop(context.scene, "holey_tools_resize_method", text="")
        operator = layout.operator("mesh.holey_tools_resize_hole")
        operator.diameter_estimator = context.scene.holey_tools_diameter_estimator
        operator.resize_method = context.scene.holey_tools_resize_method
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
        layout.label(text="Ctrl/Cmd-click Resize: use selection")

        layout.separator()
        fair_row = layout.split(factor=0.5, align=True)
        fair = fair_row.operator("mesh.holey_tools_fair_hole", icon='MOD_SMOOTH')
        defaults = fair_row.column(align=True)
        defaults.prop(context.scene, "holey_tools_fair_surface", text="")
        defaults.prop(context.scene, "holey_tools_fair_fill", text="")
        fair.fairing_method = context.scene.holey_tools_fair_surface
        fair.fill_style = context.scene.holey_tools_fair_fill
        layout.label(text="Click boundary: fit and fair")


def register_properties():
    bpy.types.Scene.holey_tools_resize_method = bpy.props.EnumProperty(
        name="Resize Mode",
        description="Mode used by the next Resize Hole invocation",
        items=RESIZE_METHOD_ITEMS,
        default='AUTO',
    )
    bpy.types.Scene.holey_tools_fair_fill = bpy.props.EnumProperty(
        name="Fair Topology", description="Topology used by the next Fair Hole invocation",
        items=(('AUTO', "Auto Topology", "Choose topology from the selected fit"),
               ('GRID', "Grid Fill", "Require the Manhattan grid patch"),
               ('CIRCLE', "Circle Fill", "Use concentric rings")), default='AUTO',
    )
    bpy.types.Scene.holey_tools_fair_surface = bpy.props.EnumProperty(
        name="Fair Fit", description="Fit used by the next Fair Hole invocation",
        items=(('AUTO', "Auto Fit", "Choose an accepted surface fit"),
               ('PLANES', "N-Plane", "Use adjacent planes"),
               ('CYLINDER', "Cylinder", "Force a cylinder"),
               ('SPHERE', "Sphere", "Force a sphere"),
               ('ORGANIC', "Organic", "Use smooth surface continuation")), default='AUTO',
    )
    bpy.types.Scene.holey_tools_flood_angle = bpy.props.FloatProperty(
        name="Flood Angle",
        description="Maximum angle between neighboring faces in the hover flood fill",
        subtype='ANGLE',
        default=math.radians(25.0),
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
    for name in ("holey_tools_fair_fill", "holey_tools_fair_surface", "holey_tools_resize_method"):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
    if hasattr(bpy.types.Scene, "holey_tools_flood_angle"):
        del bpy.types.Scene.holey_tools_flood_angle
    if hasattr(bpy.types.Scene, "holey_tools_diameter_estimator"):
        del bpy.types.Scene.holey_tools_diameter_estimator
