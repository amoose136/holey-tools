# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared metadata and user-facing option definitions."""

import math

ADDON_VERSION = "0.9.4"
DIAMETER_SYMBOL = "\N{LATIN CAPITAL LETTER O WITH STROKE}"
FLOOD_ANGLE_HUD_SECONDS = 3.0
AXIS_FACE_ATTRIBUTE = "holey_axis_region"
AXIS_NODE_GROUP_NAME = "Holey Tools Extract Axis"

# End loops qualify for depth adjustment when their maximum point-to-plane
# residual is small at both an absolute and component-relative scale. A separate
# angular epsilon records whether the accepted planes are effectively parallel.
PLANE_FIT_ABSOLUTE_EPSILON = 1.0e-6
PLANE_FIT_RELATIVE_EPSILON = 1.0e-5
PARALLEL_PLANE_ANGLE_EPSILON = math.radians(0.1)

VIEW_NAVIGATION_EVENTS = frozenset({
    'MIDDLEMOUSE',
    'MOUSEROTATE',
    'MOUSESMARTZOOM',
    'TRACKPADPAN',
    'TRACKPADZOOM',
    'WHEELUPMOUSE',
    'WHEELDOWNMOUSE',
    'WHEELINMOUSE',
    'WHEELOUTMOUSE',
    'WHEELLEFTMOUSE',
    'WHEELRIGHTMOUSE',
})

# Copied from Blender 5.2's stock settings. This fallback is used if the
# ThemeView3D.face_retopology property becomes unavailable in a future release.
FALLBACK_RETOPOLOGY_COLOR = (0.3137, 0.7843, 1.0, 0.0588)

DIAMETER_ESTIMATOR_ITEMS = (
    (
        'MINIMUM',
        "Minimum",
        "Use the smallest fitted radius, useful when the selection includes a chamfer",
    ),
    (
        'AVERAGE',
        "Average",
        "Use the average fitted radius across the selected wall",
    ),
    (
        'MAXIMUM',
        "Maximum",
        "Use the largest fitted radius across the selected wall",
    ),
)
