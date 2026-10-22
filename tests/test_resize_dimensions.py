"""Dimension regressions runnable with ordinary Python, without Blender."""

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace


# Load the pure module without executing the Blender extension entry point.
MODULE_PATH = Path(__file__).resolve().parents[1] / 'src' / 'resize_dimensions.py'
spec = importlib.util.spec_from_file_location('holey_resize_dimensions', MODULE_PATH)
dimensions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dimensions)
Dimensions = dimensions.ResizeDimensionsMixin


def cone():
    return SimpleNamespace(
        _fit={'radius': 2.0, 'slope': -1.0, 'is_cone': True},
        _depth_fit={'initial': 1.0, 'outer_coordinate': 0.0, 'inner_coordinate': -1.0},
        fit_valid=True,
        fit_is_cone=True,
        depth_valid=True,
        angle_valid=True,
        diameter=4.0,
        scale=1.0,
        depth=2.0,
        truncated_depth=1.0,
        conic_angle=math.pi / 2.0,
        conic_angle_previous=math.pi / 2.0,
        conic_depth_previous=2.0,
        conic_truncated_depth_previous=1.0,
        fit_conic_bottom_radius=1.0,
        fit_conic_depth_initial=2.0,
        fit_truncated_depth_initial=1.0,
    )


class ResizeDimensionsTests(unittest.TestCase):
    def test_polygonal_blind_bore_uses_inside_diameter(self):
        for source_count in (3, 4, 6, 8, 32, 128):
            factor = math.cos(math.pi / source_count)
            op = SimpleNamespace(
                _fit={'radius': 2.0, 'slope': 0.0, 'is_cone': False, 'is_stud': False},
                _depth_fit={'initial': 1.0, 'outer_coordinate': 0.0, 'inner_coordinate': -1.0},
                fit_angular_segments=source_count, depth_valid=True,
                diameter=4.0 * factor, scale=1.0, angle_valid=False,
            )
            self.assertAlmostEqual(Dimensions._target_scale(op), 1.0)
            self.assertAlmostEqual(Dimensions._target_radius_at(op, 0.0), 2.0)
            self.assertAlmostEqual(Dimensions._preview_radius_at(op, 0.0), 2.0 * factor)
            for target_count in (3, 4, 8, 16, 32, 128):
                op.boolean_segments = target_count
                cutter_radius = Dimensions._boolean_radius_at(op, 0.0, target_count)
                self.assertAlmostEqual(cutter_radius * math.cos(math.pi / target_count), op.diameter / 2)
                self.assertAlmostEqual(Dimensions._preview_radius_at(op, 0.0), op.diameter / 2)
            op.diameter *= 1.2
            self.assertAlmostEqual(Dimensions._target_scale(op), 1.2)
            self.assertAlmostEqual(Dimensions._target_radius_at(op, -1.0), 2.4)

    def test_polygon_correction_leaves_other_dimension_conventions_unchanged(self):
        for fit, depth in (({'is_cone': True}, {'initial': 1}),
                           ({'is_stud': True}, {'initial': 1})):
            op = SimpleNamespace(_fit=fit, _depth_fit=depth, fit_angular_segments=4)
            self.assertEqual(dimensions._cylinder_inradius_factor(op), 1.0)

    def test_through_bore_keeps_correction_without_depth_controls(self):
        op = SimpleNamespace(
            _fit={'radius': 1.0, 'slope': 0.0, 'is_cone': False, 'is_stud': False},
            _depth_fit=None, depth_valid=False, fit_angular_segments=32,
            diameter=2.0, scale=1.0, angle_valid=False,
        )
        for count in (3, 4, 8, 32, 128):
            self.assertAlmostEqual(Dimensions._preview_radius_at(op, 0), 1.0)
            self.assertAlmostEqual(
                Dimensions._boolean_radius_at(op, 0, count) * math.cos(math.pi / count), 1.0)

    def assert_bottom(self, operator, truncated):
        inner = Dimensions._target_conic_inner_coordinate(operator)
        self.assertAlmostEqual(inner, -truncated)
        self.assertAlmostEqual(Dimensions._target_radius_at(operator, inner), 1.0)

    def test_depth_moves_truncation_without_resizing_bore(self):
        operator = cone()
        operator.depth = 2.4
        dimensions._update_conic_depth(operator, None)
        self.assertAlmostEqual(operator.truncated_depth, 1.4)
        self.assertFalse(operator.conic_dimensions_updating)
        self.assert_bottom(operator, 1.4)

    def test_truncation_moves_apex_without_resizing_bore(self):
        operator = cone()
        operator.truncated_depth = .6
        dimensions._update_conic_truncated_depth(operator, None)
        self.assertAlmostEqual(operator.depth, 1.6)
        self.assert_bottom(operator, .6)

    def test_angle_pivots_around_the_fixed_bottom(self):
        for angle in (40, 90, 110, 160):
            with self.subTest(angle=angle):
                operator = cone()
                operator.conic_angle = math.radians(angle)
                dimensions._update_conic_angle(operator, None)
                self.assertAlmostEqual(operator.depth, 1 + 1 / math.tan(operator.conic_angle / 2))
                self.assert_bottom(operator, 1.0)

    def test_zero_depth_closes_and_positive_depth_recovers(self):
        operator = cone()
        operator.depth = 0.0
        dimensions._update_conic_depth(operator, None)
        self.assertEqual(operator.truncated_depth, 0.0)
        self.assertTrue(Dimensions._fair_resize_requested(operator))
        operator.depth = 2.3
        dimensions._update_conic_depth(operator, None)
        self.assertFalse(Dimensions._fair_resize_requested(operator))
        self.assert_bottom(operator, 1.3)

    def test_zero_truncation_is_not_a_close_request(self):
        operator = cone()
        operator.truncated_depth = 0.0
        dimensions._update_conic_truncated_depth(operator, None)
        self.assertAlmostEqual(operator.depth, 1.0)
        self.assertFalse(Dimensions._fair_resize_requested(operator))
        self.assert_bottom(operator, 0.0)

    def test_zero_angle_retains_bore_and_can_reopen_cone(self):
        operator = cone()
        # A hidden/default diameter is not a close request for conic controls.
        operator.diameter = 0.0
        operator.conic_angle = 0.0
        dimensions._update_conic_angle(operator, None)
        self.assertFalse(Dimensions._fair_resize_requested(operator))
        self.assertEqual(operator.depth, 2.0)
        self.assert_bottom(operator, 1.0)
        for coordinate in (-3.0, -1.0, 0.0, 1.0):
            self.assertAlmostEqual(Dimensions._target_radius_at(operator, coordinate), 1.0)
        operator.conic_angle = math.radians(60)
        dimensions._update_conic_angle(operator, None)
        self.assertAlmostEqual(operator.depth, 1.0 + 1.0 / math.tan(math.radians(30)))
        self.assert_bottom(operator, 1.0)
        # Explicit Depth zero remains a distinct request to close the cavity.
        operator.depth = 0.0
        dimensions._update_conic_depth(operator, None)
        self.assertTrue(Dimensions._fair_resize_requested(operator))

    def test_lone_redo_depth_edit_does_not_require_a_callback(self):
        operator = cone()
        operator.depth = 2.4
        self.assert_bottom(operator, 1.4)

    def test_explicit_zero_is_distinct_from_an_unset_default(self):
        operator = SimpleNamespace(fit_valid=False, diameter=0.0, depth_valid=False)
        self.assertFalse(Dimensions._fair_resize_requested(operator))
        operator.is_property_set = lambda name: name == 'diameter'
        self.assertTrue(Dimensions._fair_resize_requested(operator))

    def test_cylinder_diameter_overrides_legacy_scale(self):
        operator = SimpleNamespace(
            _fit={'radius': 1.0, 'slope': 0.0, 'is_cone': False},
            _depth_fit=None, diameter=3.0, scale=7.0, angle_valid=False,
        )
        self.assertAlmostEqual(Dimensions._target_scale(operator), 1.5)
        self.assertAlmostEqual(Dimensions._target_radius_at(operator, 100.0), 1.5)
        operator.diameter = 0.0
        self.assertAlmostEqual(Dimensions._target_scale(operator), 7.0)

    def test_unfitted_dimensions_are_not_coupled(self):
        operator = cone()
        operator.fit_valid = False
        operator.depth = 3.0
        dimensions._update_conic_depth(operator, None)
        self.assertEqual(operator.truncated_depth, 1.0)
        self.assertEqual(operator.conic_depth_previous, 3.0)

    def test_reentrant_callback_does_not_reapply_the_edit(self):
        operator = cone()
        operator.conic_dimensions_updating = True
        operator.depth = 3.0
        dimensions._update_conic_depth(operator, None)
        self.assertTrue(operator.conic_dimensions_updating)
        self.assertEqual(operator.truncated_depth, 1.0)
        self.assertEqual(operator.conic_depth_previous, 3.0)

    def test_assignment_failure_always_releases_callback_guard(self):
        class RejectTruncation(SimpleNamespace):
            def __setattr__(self, name, value):
                if name == 'truncated_depth':
                    raise RuntimeError('Simulated RNA assignment failure')
                super().__setattr__(name, value)

        operator = RejectTruncation()
        with self.assertRaises(RuntimeError):
            dimensions._set_conic_depths(operator, 2.3, 1.3)
        self.assertFalse(operator.conic_dimensions_updating)


if __name__ == '__main__':
    unittest.main(argv=[__file__], verbosity=2)
