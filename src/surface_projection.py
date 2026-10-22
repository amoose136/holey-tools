# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared analytic surface intersections for resize and grid fairing."""

import math
import numpy as np


def sphere_axis_projector(hypothesis, center, x, y, axis):
    """Freeze a vectorized sphere/chart intersection for relaxation trials.

    Keep the same nearest axis-line root and discriminant tolerance as the
    scalar projector. Float64 chart arithmetic avoids rounding every tiny
    finite-difference trial through a separate float32 mathutils.Vector.
    """
    origin, basis = np.array(center, dtype=float), np.array((x, y), dtype=float)
    direction = np.array(axis.normalized(), dtype=float)
    direction /= np.linalg.norm(direction)
    sphere_center = np.array(hypothesis['center'], dtype=float)
    radius_squared = float(hypothesis['radius']) ** 2

    def project(queries):
        points = origin + np.atleast_2d(queries) @ basis
        relative = points - sphere_center
        linear = relative @ direction
        constant = np.sum(relative*relative, axis=1) - radius_squared
        discriminant = linear*linear - constant
        tolerance = np.maximum((linear*linear + np.abs(constant))*1e-10, 1e-12)
        root = np.sqrt(np.maximum(0., discriminant))
        first, second = -linear-root, -linear+root
        distance = np.where(np.abs(first) <= np.abs(second), first, second)
        result = points + distance[:, None]*direction
        result[discriminant < -tolerance] = np.nan
        return result
    return project


def project_hypothesis_along_axis(hypothesis, point, reference, axis):
    """Intersect an axis line with a virtual plane/sphere/conic surface."""
    method = hypothesis.get("method")
    axis = axis.normalized()
    candidates = []
    if method == "PLANES":
        for plane in hypothesis["planes"]:
            denominator = plane["normal"].dot(axis)
            if abs(denominator) <= 1.0e-9:
                continue
            distance = plane["normal"].dot(
                plane["point"] - point
            ) / denominator
            candidates.append(point + axis * distance)
    elif method == "SPHERE":
        relative = point - hypothesis["center"]
        linear = relative.dot(axis)
        constant = relative.length_squared - hypothesis["radius"] ** 2
        discriminant = linear * linear - constant
        tolerance = max(
            (linear * linear + abs(constant)) * 1.0e-10,
            1.0e-12,
        )
        if discriminant >= -tolerance:
            root = math.sqrt(max(0.0, discriminant))
            candidates.extend((
                point + axis * (-linear - root),
                point + axis * (-linear + root),
            ))
    elif method == "CONIC":
        fit = hypothesis["fit"]
        surface_axis = fit["axis"].normalized()
        relative = point - fit["center"]
        axial = relative.dot(surface_axis)
        ray_axial = axis.dot(surface_axis)
        radial = relative - surface_axis * axial
        ray_radial = axis - surface_axis * ray_axial
        slope = float(fit.get("slope", 0.0))
        radius = float(fit["radius"]) + slope * axial
        radius_rate = slope * ray_axial
        quadratic = ray_radial.length_squared - radius_rate * radius_rate
        linear = 2.0 * (
            radial.dot(ray_radial) - radius * radius_rate
        )
        constant = radial.length_squared - radius * radius
        if abs(quadratic) <= 1.0e-12:
            if abs(linear) > 1.0e-12:
                candidates.append(point + axis * (-constant / linear))
        else:
            discriminant = linear * linear - 4.0 * quadratic * constant
            tolerance = max(
                (
                    linear * linear
                    + abs(4.0 * quadratic * constant)
                ) * 1.0e-10,
                1.0e-12,
            )
            if discriminant >= -tolerance:
                root = math.sqrt(max(0.0, discriminant))
                candidates.extend((
                    point + axis * ((-linear - root) / (2.0 * quadratic)),
                    point + axis * ((-linear + root) / (2.0 * quadratic)),
                ))
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (candidate - reference).length)
