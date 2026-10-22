# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resize dimension policies and coupled conic controls, independent of Blender."""

import math


def _cylinder_inradius_factor(operator, segments=None):
    """Convert vertex radius to usable radius for a polygonal cylindrical bore.

    Keep legacy fits without source resolution, cones, and studs unchanged.
    Clearance does not depend on whether the bore has a blind end or usable
    depth controls. Source resolution is immutable across redo; an explicit
    segment count describes only a generated cutter, never the measurement.
    """
    fit = getattr(operator, "_fit", None) or {}
    # Only a serialized source baseline opts into the inside-diameter
    # convention. A transient estimate alone must not reinterpret a legacy
    # caller's diameter halfway through projection setup.
    source_segments = int(getattr(operator, "fit_angular_segments", 0) or 0)
    if (
        source_segments < 3
        or fit.get("is_cone", getattr(operator, "fit_is_cone", False))
        or fit.get("is_stud", getattr(operator, "fit_is_stud", False))
    ):
        return 1.0
    count = source_segments if segments is None else max(3, int(segments))
    return math.cos(math.pi / count)


def _update_diameter_estimator(operator, _context):
    """Switch the fitted baseline and diameter when Redo changes estimator."""
    if not operator.fit_valid:
        return
    radius = {
        'MINIMUM': operator.fit_radius_minimum,
        'AVERAGE': operator.fit_radius_average,
        'MAXIMUM': operator.fit_radius_maximum,
    }.get(operator.diameter_estimator, operator.fit_radius_average)
    if radius <= 1.0e-12:
        return
    operator.fit_radius = radius
    coordinate = getattr(operator, "fit_radius_coordinate", 0.0)
    slope = getattr(operator, "fit_slope", 0.0)
    operator.diameter = (
        max(1.0e-9, radius + slope * coordinate)
        * 2.0 * _cylinder_inradius_factor(operator)
    )
    if (
        getattr(operator, "fit_is_cone", False)
        and getattr(operator, "depth_valid", False)
        and abs(slope) > 1.0e-12
    ):
        operator.conic_dimensions_updating = True
        try:
            operator.depth = max(
                1.0e-9, operator.diameter * 0.5 / abs(slope)
            )
            operator.conic_depth_previous = operator.depth
            operator.fit_conic_depth_initial = operator.depth
        finally:
            operator.conic_dimensions_updating = False
    fit = getattr(operator, "_fit", None)
    if fit is not None:
        fit["radius"] = radius


def _conic_depth_preserving_bottom(
    depth, truncated_depth, previous_angle, new_angle
):
    """Return apex depth after pivoting a cone around its bottom circle."""
    previous_slope = math.tan(max(0.0, float(previous_angle)) * 0.5)
    new_slope = math.tan(max(0.0, float(new_angle)) * 0.5)
    if previous_slope <= 1.0e-12 or new_slope <= 1.0e-12:
        return float(depth)
    bottom_radius = max(
        0.0, float(depth) - float(truncated_depth)
    ) * previous_slope
    return float(truncated_depth) + bottom_radius / new_slope


def _update_conic_angle(operator, _context):
    """Pivot angle edits around the existing conic truncation circle."""
    new_angle = float(operator.conic_angle)
    if getattr(operator, "conic_dimensions_updating", False):
        operator.conic_angle_previous = new_angle
        return
    previous_angle = float(
        getattr(operator, "conic_angle_previous", new_angle)
    )
    if _conic_dimensions_are_coupled(operator):
        operator.conic_dimensions_updating = True
        try:
            slope = math.tan(max(0.0, new_angle) * .5)
            bottom_radius = float(getattr(operator, "fit_conic_bottom_radius", 0.0))
            if slope > 1.0e-12 and bottom_radius > 0.0 and operator.depth > 0.0:
                # At zero angle the apex is at infinity, not on the axis at
                # zero depth. Keep the finite controls while cylindrical, then
                # recover the apex from the stored bore radius when reopening
                # the countersink; the preceding zero slope carries no radius.
                operator.depth = float(operator.truncated_depth) + bottom_radius / slope
            else:
                operator.depth = _conic_depth_preserving_bottom(
                    operator.depth,
                    operator.truncated_depth,
                    previous_angle,
                    new_angle,
                )
            operator.conic_depth_previous = operator.depth
        finally:
            operator.conic_dimensions_updating = False
    operator.conic_angle_previous = new_angle


def _conic_dimensions_are_coupled(operator):
    """Return whether redo dimension edits describe an adjustable frustum."""
    return (
        getattr(operator, "fit_valid", False)
        and getattr(operator, "fit_is_cone", False)
        and getattr(operator, "depth_valid", False)
        and getattr(operator, "angle_valid", False)
    )


def _conic_apex_span(operator, previous_depth, previous_truncated):
    """Retain the small-circle radius, including recovery from zero depth."""
    span = max(0.0, previous_depth - previous_truncated)
    if previous_depth == 0.0:
        slope = math.tan(float(getattr(operator, "conic_angle", 0.0)) * .5)
        if slope > 1.0e-12:
            span = float(getattr(operator, "fit_conic_bottom_radius", 0.0)) / slope
    return span


def _set_conic_depths(operator, depth, truncated):
    """Assign coupled RNA values once, without recursive dimension updates."""
    operator.conic_dimensions_updating = True
    try:
        operator.depth = depth
        operator.truncated_depth = truncated
        operator.conic_depth_previous = depth
        operator.conic_truncated_depth_previous = truncated
    finally:
        operator.conic_dimensions_updating = False


def _update_conic_depth(operator, _context):
    """Move the physical truncation with the apex-depth control.

    At a fixed cone angle, the lower-circle radius is proportional to
    ``depth - truncated_depth``. Moving both dimensions by the same amount
    therefore changes the cone's extent without silently resizing the bore it
    joins.
    """
    new_depth = float(operator.depth)
    previous_depth = float(
        getattr(operator, "conic_depth_previous", new_depth)
    )
    if getattr(operator, "conic_dimensions_updating", False):
        operator.conic_depth_previous = new_depth
        return
    if _conic_dimensions_are_coupled(operator):
        previous_truncated = float(
            getattr(
                operator,
                "conic_truncated_depth_previous",
                operator.truncated_depth,
            )
        )
        fixed_span = _conic_apex_span(operator, previous_depth, previous_truncated)
        truncated = new_depth - fixed_span
        if new_depth == 0.0:
            # Zero is an explicit close/fair request, not an invalid cone span.
            truncated = 0.0
        elif truncated < 0.0:
            new_depth = fixed_span
            truncated = 0.0
        _set_conic_depths(operator, new_depth, truncated)
    else:
        operator.conic_depth_previous = new_depth


def _update_conic_truncated_depth(operator, _context):
    """Move apex depth with the physical conic truncation control."""
    new_truncated = float(operator.truncated_depth)
    previous_truncated = float(
        getattr(
            operator,
            "conic_truncated_depth_previous",
            new_truncated,
        )
    )
    if getattr(operator, "conic_dimensions_updating", False):
        operator.conic_truncated_depth_previous = new_truncated
        return
    if _conic_dimensions_are_coupled(operator):
        previous_depth = float(
            getattr(operator, "conic_depth_previous", operator.depth)
        )
        fixed_span = _conic_apex_span(operator, previous_depth, previous_truncated)
        new_depth = new_truncated + fixed_span
        _set_conic_depths(operator, new_depth, new_truncated)
    else:
        operator.conic_truncated_depth_previous = new_truncated


class ResizeDimensionsMixin:
    """Evaluate requested cylinder/cone dimensions against their fitted baseline."""

    def _target_depth_scale(self):
        """Return the axial scale requested by the optional depth property."""
        depth_fit = getattr(self, "_depth_fit", None)
        if not depth_fit or depth_fit["initial"] <= 1.0e-12:
            return 1.0
        requested = (
            ResizeDimensionsMixin._target_conic_truncated_depth(self)
            if ResizeDimensionsMixin._uses_conic_dimensions(self)
            else float(self.depth)
        )
        return max(requested, 1.0e-9) / depth_fit["initial"]

    def _uses_conic_dimensions(self):
        """Return whether apex depth, truncation depth, and tip angle are usable."""
        return bool(
            getattr(self, "angle_valid", False)
            and getattr(self, "depth_valid", False)
            and getattr(self, "_depth_fit", None)
            and self._fit.get("is_cone", False)
        )

    def _radius_coordinate(self):
        """Return the axial coordinate at which Diameter is measured."""
        depth_fit = getattr(self, "_depth_fit", None)
        if depth_fit:
            return depth_fit["outer_coordinate"]
        return float(self._fit.get("radius_coordinate", 0.0))

    def _fitted_radius_at(self, coordinate):
        """Evaluate the fitted cone radius at one axial coordinate."""
        fit = self._fit
        return max(
            1.0e-9,
            float(fit["radius"]) + float(fit.get("slope", 0.0)) * coordinate,
        )

    def _target_cone_slope(self):
        """Return the signed slope requested by the optional angle property."""
        fit_slope = float(self._fit.get("slope", 0.0))
        if not getattr(self, "angle_valid", False):
            return fit_slope
        sign = -1.0 if fit_slope < 0.0 else 1.0
        return sign * math.tan(
            min(
                math.radians(89.0),
                max(0.0, float(self.conic_angle) * 0.5),
            )
        )

    def _target_conic_inner_coordinate(self):
        """Return the requested axial coordinate of the small conic circle."""
        depth_fit = getattr(self, "_depth_fit", None)
        if not depth_fit:
            return None
        outer = float(depth_fit["outer_coordinate"])
        inner = float(depth_fit["inner_coordinate"])
        direction = 1.0 if inner >= outer else -1.0
        truncated_depth = ResizeDimensionsMixin._target_conic_truncated_depth(self)
        return outer + direction * truncated_depth

    def _fixed_conic_bottom_radius(self):
        """Return the fitted small-circle radius retained by conic controls."""
        stored = float(getattr(self, "fit_conic_bottom_radius", 0.0))
        if stored > 1.0e-12:
            return stored
        depth_fit = getattr(self, "_depth_fit", None)
        if depth_fit:
            return ResizeDimensionsMixin._fitted_radius_at(
                self, float(depth_fit["inner_coordinate"])
            )
        return 0.0

    def _target_conic_truncated_depth(self):
        """Resolve the physical span without depending on RNA callback order.

        Truncated Depth directly places the small circle. Angle edits retain
        that placement. A lone Depth edit derives the span from the fixed bore
        radius, while coupled Depth/Truncated Depth callbacks produce the same
        result through either value.
        """
        depth = max(0.0, float(getattr(self, "depth", 0.0)))
        slope = abs(ResizeDimensionsMixin._target_cone_slope(self))
        bottom_radius = ResizeDimensionsMixin._fixed_conic_bottom_radius(self)
        truncated = min(
            max(0.0, float(getattr(self, "truncated_depth", 0.0))),
            depth,
        )
        depth_fit = getattr(self, "_depth_fit", None)
        if depth_fit:
            initial_truncated = float(
                getattr(
                    self,
                    "fit_truncated_depth_initial",
                    depth_fit.get("initial", truncated),
                )
            )
            fitted_slope = abs(float(self._fit.get("slope", 0.0)))
            initial_depth = float(
                getattr(
                    self,
                    "fit_conic_depth_initial",
                    initial_truncated + (
                        bottom_radius / fitted_slope
                        if fitted_slope > 1.0e-12
                        else 0.0
                    ),
                )
            )
            initial_angle = 2.0 * math.atan(fitted_slope)
            tolerance = max(
                abs(initial_depth), abs(initial_truncated), 1.0
            ) * 1.0e-8
            angle_changed = not math.isclose(
                float(getattr(self, "conic_angle", initial_angle)),
                initial_angle,
                rel_tol=1.0e-8,
                # Fitted slopes and the stored angle take slightly different
                # floating-point paths; sub-micro-radian drift is not an edit.
                abs_tol=1.0e-7,
            )
            truncated_changed = not math.isclose(
                truncated,
                initial_truncated,
                rel_tol=1.0e-8,
                abs_tol=tolerance,
            )
            depth_changed = not math.isclose(
                depth,
                initial_depth,
                rel_tol=1.0e-8,
                abs_tol=tolerance,
            )
            if angle_changed or truncated_changed or not depth_changed:
                return truncated
        if slope > 1.0e-12 and bottom_radius > 1.0e-12:
            return max(0.0, depth - bottom_radius / slope)
        return truncated

    def _target_radius_at(self, coordinate):
        """Evaluate the requested cone radius.

        Adjustable conics are anchored at their originally fitted small circle.
        This makes the connected bore diameter a geometry-level invariant, not
        merely a consequence of UI callbacks updating two fields in lockstep.
        """
        anchor = ResizeDimensionsMixin._radius_coordinate(self)
        if ResizeDimensionsMixin._uses_conic_dimensions(self):
            inner_coordinate = ResizeDimensionsMixin._target_conic_inner_coordinate(
                self
            )
            bottom_radius = ResizeDimensionsMixin._fixed_conic_bottom_radius(self)
            if inner_coordinate is not None and bottom_radius > 1.0e-12:
                return max(
                    1.0e-9,
                    bottom_radius
                    + ResizeDimensionsMixin._target_cone_slope(self)
                    * (float(coordinate) - inner_coordinate),
                )
            anchor_radius = max(
                1.0e-9,
                float(self.depth)
                * abs(ResizeDimensionsMixin._target_cone_slope(self)),
            )
        else:
            anchor_radius = (
                max(float(self.diameter), 1.0e-9) * 0.5
                / _cylinder_inradius_factor(self)
                if float(self.diameter) > 0.0
                else ResizeDimensionsMixin._fitted_radius_at(self, anchor) * float(self.scale)
            )
        return max(
            1.0e-9,
            anchor_radius
            + ResizeDimensionsMixin._target_cone_slope(self) * (coordinate - anchor),
        )

    def _preview_radius_at(self, coordinate):
        """Show the usable inside circle, independently of cutter resolution."""
        return (
            ResizeDimensionsMixin._target_radius_at(self, coordinate)
            * _cylinder_inradius_factor(self)
        )

    def _boolean_radius_at(self, coordinate, segments):
        """Keep the requested inside diameter when cutter sides change."""
        return (
            ResizeDimensionsMixin._target_radius_at(self, coordinate)
            * _cylinder_inradius_factor(self)
            / _cylinder_inradius_factor(self, segments)
        )

    def _target_scale(self):
        """Convert the requested diameter into a radial deformation scale.

        The legacy ``scale`` property remains a fallback for redo compatibility
        when no positive diameter has been assigned.
        """
        fitted_diameter = ResizeDimensionsMixin._fitted_radius_at(
            self, ResizeDimensionsMixin._radius_coordinate(self)
        ) * 2.0 * _cylinder_inradius_factor(self)
        if fitted_diameter <= 1.0e-12:
            return 1.0
        if ResizeDimensionsMixin._uses_conic_dimensions(self):
            target_radius = ResizeDimensionsMixin._target_radius_at(
                self, ResizeDimensionsMixin._radius_coordinate(self)
            )
            return target_radius * 2.0 / fitted_diameter
        if self.diameter <= 0.0:
            return self.scale
        return self.diameter / fitted_diameter

    def _fair_resize_requested(self):
        """Distinguish explicit zero dimensions from legacy unset defaults."""
        diameter_set = getattr(self, "fit_valid", False)
        is_set = getattr(self, "is_property_set", None)
        if is_set is not None:
            diameter_set = diameter_set or is_set("diameter")
        # Diameter is hidden and is not an independent conic control. In
        # particular, a stale/default zero here must not turn the zero-angle
        # cylindrical limit into a request to close the connected through-hole.
        diameter_set = diameter_set and not _conic_dimensions_are_coupled(self)
        return bool(
            (diameter_set and float(getattr(self, "diameter", -1.0)) == 0.0)
            or (getattr(self, "depth_valid", False) and float(self.depth) == 0.0)
        )
