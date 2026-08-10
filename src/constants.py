# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared metadata and user-facing option definitions."""

ADDON_VERSION = "0.9.0"

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
