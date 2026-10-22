"""Scoped cache resets run without Blender and preserve user-owned state."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


path = Path(__file__).resolve().parents[1] / 'src' / 'resize_state.py'
spec = importlib.util.spec_from_file_location('holey_resize_state', path)
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


class ResizeStateTests(unittest.TestCase):
    def test_projection_invalidation_keeps_fit_and_dimension_policy(self):
        op = SimpleNamespace(
            _fit=object(), _depth_fit=object(), _original_vertex_coords=object(),
            _resize_base_span=17, _resize_angular_segments=32,
            _resize_requires_boolean=True, _resize_base_requires_boolean=True,
        )
        retained = vars(op).copy()
        state.invalidate_projection(op)
        for key, value in retained.items():
            self.assertIs(getattr(op, key), value)
        self.assertFalse(op._resize_projection_ready)
        self.assertEqual(op._resize_surfaces_by_vertex, {})
        self.assertEqual(op._resize_conic_surfaces_by_vertex, {})
        self.assertEqual(op._resize_bore_vertex_indices, set())
        self.assertIsNone(op._resize_base_bvh)

    def test_deformation_reset_leaves_surface_maps_alone(self):
        op = SimpleNamespace(_resize_surfaces_by_vertex=object(), _fit=object())
        surfaces, fit = op._resize_surfaces_by_vertex, op._fit
        state.reset_deformation(op)
        self.assertIs(op._resize_surfaces_by_vertex, surfaces)
        self.assertIs(op._fit, fit)
        self.assertIsNone(op._original_vertex_coords)
        self.assertIsNone(op._proportional_weights)
        self.assertIsNone(op._resize_wall_cache)

    def test_full_reset_keeps_rna_choices_and_native_resource_owners(self):
        retained = dict(
            diameter=1.2, depth=2.4, resize_method='SCALE', boolean_segments=24,
            fit_valid=True, fit_face_indices='1,2,3', fit_end_planes='[]',
            _object=object(), _boolean_preview_source=object(),
            _boolean_preview_display_object=object(), _modal_session=object(),
        )
        op = SimpleNamespace(**retained)
        state.reset_mesh_derived_state(op)
        for key, value in retained.items():
            self.assertIs(getattr(op, key), value)
        self.assertIsNone(op._fit)
        self.assertIsNone(op._depth_fit)
        self.assertIsNone(op._hover_cache)
        self.assertFalse(op._resize_requires_boolean)
        self.assertFalse(op._resize_base_requires_boolean)
        self.assertFalse(op._resize_dynamic_requires_boolean)
        self.assertEqual(op._resize_angular_segments, 0)

    def test_mutable_caches_are_fresh_per_reset_and_operator(self):
        first, second = SimpleNamespace(), SimpleNamespace()
        state.reset_mesh_derived_state(first)
        state.reset_mesh_derived_state(second)
        for key in ('_locked_indices', '_resize_surfaces_by_vertex',
                    '_resize_conic_surfaces_by_vertex', '_resize_bore_vertex_indices'):
            self.assertIsNot(getattr(first, key), getattr(second, key))
        previous = first._locked_indices
        state.reset_mesh_derived_state(first)
        self.assertIsNot(previous, first._locked_indices)


if __name__ == '__main__':
    unittest.main()
