"""Curvature matching must not turn a narrow Suzanne corner into a spike."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fairing import _organic_grid_frame
from src.grid_patcher import _inside_boundary, prepare_grid_steps
from src.ideal_surface import (_interior_samples, capture_curvature_surface,
                               HeightSurface, fit_thin_plate_surface)
from src.fairing import _organic_depth_samples
from src.face_curvature import capture_face_samples
from src.grid_quality import improve_projected_shapes
from src.row_patcher import fill_refined_rows
from tests.blender_support import edit_mesh

data = json.loads(Path(__file__).with_name('fixtures').joinpath('suzanne_triangle_grid_hole.json').read_text())
profile = json.loads(Path(__file__).with_name('fixtures').joinpath('suzanne_curvature_profile.json').read_text())
reference_mesh = json.loads(Path(__file__).with_name('fixtures').joinpath('suzanne_reference_surface.json').read_text())

for scale in (.01, 1., 100.):
    obj, bm = edit_mesh('Suzanne curvature overshoot', data['vertices'], data['faces'])
    obj.matrix_world = Matrix.Scale(scale, 4) @ Matrix(data['matrix'])
    bm.normal_update()
    loop = [bm.verts[i] for i in data['boundary']]
    edges = [bm.edges.get((v, loop[(i+1) % len(loop)])) for i, v in enumerate(loop)]
    prepared = prepare_grid_steps(bm, loop, edges, obj.matrix_world, 'REFINE', refinement_passes=1)
    loop, edges = prepared['boundary_verts'], prepared['boundary_edges']
    center, axis, x, y, points, _ = _organic_grid_frame(loop, edges, obj.matrix_world)
    surface = capture_curvature_surface(loop, obj.matrix_world, center, x, y, axis)
    assert surface is not None  # Keep useful C2 matching, not an unconditional fallback.
    polygon = [Vector(p) for p in points]
    p = np.asarray(points)
    queries = np.asarray([(u, v) for u in np.linspace(p[:, 0].min(), p[:, 0].max(), 25)
                          for v in np.linspace(p[:, 1].min(), p[:, 1].max(), 25)
                          if _inside_boundary(Vector((u, v)), polygon)])
    source = set(loop)
    for _ in range(2):
        source |= {e.other_vert(v) for v in list(source) for e in v.link_edges}
    source_heights = np.asarray([(obj.matrix_world @ v.co - center).dot(axis) for v in source])
    lower, upper = source_heights.min() - .02 * scale, source_heights.max() + .02 * scale
    heights = surface(queries)
    assert np.isfinite(heights).all()
    assert heights.min() > lower and heights.max() < upper, (scale, heights.min(), heights.max())
    rim_heights = [(obj.matrix_world @ v.co - center).dot(axis) for v in loop]
    assert np.max(np.abs(surface(points) - rim_heights)) < 1e-6 * scale
    # The previous height-range check allowed a broad hump of about 0.15 units
    # beneath the original surface. Compare the actual missing local profile.
    reference = [obj.matrix_world @ Vector(p) - center for p in profile['vertices']]
    local_queries = [(p.dot(x), p.dot(y)) for p in reference]
    reference_heights = np.asarray([p.dot(axis) for p in reference])
    error = surface(local_queries) - reference_heights
    assert np.sqrt(np.mean(error * error)) < .015 * scale, (scale, error.min(), error.max())
    assert np.max(np.abs(error)) < .03 * scale
    # Check the WHOLE opening, not just the central height profile. The r^5
    # fit passed that profile but pushed a different brow region outward by
    # ~0.045 units. A nearest-surface metric also avoids choosing a wrong ray
    # intersection where the original brow turns away from the height chart.
    tree = BVHTree.FromPolygons([obj.matrix_world @ Vector(p) for p in reference_mesh['vertices']],
                               reference_mesh['faces'])
    def signed_distances(world_points):
        result = []
        for point in world_points:
            nearest, normal, _face, _distance = tree.find_nearest(Vector(point))
            result.append((Vector(point) - nearest).dot(normal))
        return np.asarray(result)
    full_queries = _interior_samples(points, resolution=25)
    full_heights = surface(full_queries)
    world_samples = [center + x * q[0] + y * q[1] + axis * h
                     for q, h in zip(full_queries, full_heights)]
    signed = signed_distances(world_samples)
    assert signed.max() < .025 * scale, (scale, signed.min(), signed.max())
    assert np.sqrt(np.mean(signed * signed)) < .016 * scale
    # Tangents must actually follow the refined source faces more closely,
    # without trading away the no-bulge reference profile above.
    samples = capture_face_samples(loop, obj.matrix_world, center, x, y, axis)
    primary = samples['primary']
    face_queries, target = samples['points'][primary], samples['gradients'][primary]
    weights = (samples['areas'] * samples['confidence'])[primary]
    def normal_error(field):
        gradients = np.column_stack((field(face_queries, (0,)), field(face_queries, (1,))))
        normals = np.column_stack((-gradients, np.ones(len(gradients))))
        expected = np.column_stack((-target, np.ones(len(target))))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        angles = np.arccos(np.clip(np.sum(normals * expected, axis=1), -1., 1.))
        return np.sqrt(np.average(angles * angles, weights=weights))
    with patch('src.face_curvature.capture_face_samples', return_value=None):
        vertex_only = capture_curvature_surface(loop, obj.matrix_world, center, x, y, axis)
    assert vertex_only is not None
    assert normal_error(surface) < .9 * normal_error(vertex_only)

    bm.verts.index_update()
    bm.edges.index_update()
    loop_ids, edge_ids = [v.index for v in loop], [e.index for e in edges]
    outputs = []
    for curvature, rows in ((False, 0), (True, 0), (True, 2)):
        trial = bm.copy()
        trial.verts.ensure_lookup_table()
        trial.edges.ensure_lookup_table()
        original = {v: v.co.copy() for v in trial.verts}
        result = fill_refined_rows(trial, [trial.verts[i] for i in loop_ids],
                                   [trial.edges[i] for i in edge_ids], obj.matrix_world,
                                   match_curvature=curvature, fill_rows=rows)
        assert result['surface_continuity'] == 'INTRINSIC_BENDING'
        output = np.asarray([tuple(obj.matrix_world @ v.co) for v in result['verts']])
        if curvature:
            h = (output - np.asarray(center)) @ np.asarray(axis)
            assert h.min() > lower and h.max() < upper
            assert signed_distances(output).max() < .025 * scale
        trial_rim = set(result['boundary_verts'])
        assert all(v.is_valid and v.co == co for v, co in original.items() if v not in trial_rim)
        assert result['relaxed_rim_vertices'] > 0
        assert all(len(f.verts) == 4 and f.calc_area() > 1e-14 for f in result['faces'])
        outputs.append(output)
        trial.free()
    assert outputs[0].shape == outputs[1].shape
    # Neither toggle may force an unstable height chart on this folded rim.
    assert np.max(np.linalg.norm(outputs[0] - outputs[1], axis=1)) < 1e-6 * scale
    assert all(np.min(np.linalg.norm(outputs[1] - point, axis=1)) < 1e-6 * scale for point in outputs[2])

    # The brow spike was a twisted projected QUAD, not necessarily a point
    # far from the reference surface. It occurred with either height fitter
    # and survived 100 flat relaxation passes. Compare actual cell warpage,
    # preserving the complete rim and topology, at the screenshot settings.
    # Isolate the interior-only solve here; test_rim_relaxation separately
    # covers the newly movable rim and its attached exterior faces.
    def interior_only(*args, **kwargs):
        kwargs['movable_boundary'] = False
        return improve_projected_shapes(*args, **kwargs)
    def warpage(result):
        maximum = 0.
        for face in result['faces']:
            p = [obj.matrix_world @ v.co for v in face.verts]
            normal = (p[1]-p[0]).cross(p[2]-p[0]).normalized()
            maximum = max(maximum, abs((p[3]-p[0]).dot(normal)))
        return maximum
    for curvature in (False, True):
        # Retain the old scalar-fit warpage regression independently of the
        # intrinsic fallback; the new fit-only tests cover the folded chart.
        field = (surface if curvature else fit_thin_plate_surface(
            *_organic_depth_samples([], loop, obj.matrix_world, center, x, y, axis)))
        height_surface = HeightSurface(center, x, y, axis, field,
                                       'C2_ESTIMATED' if curvature else 'THIN_PLATE')
        runs = []
        for surface_relaxation in (False, True):
            trial = bm.copy()
            trial.verts.ensure_lookup_table()
            trial.edges.ensure_lookup_table()
            original = {v: v.co.copy() for v in trial.verts}
            options = dict(relaxation_iterations=100, relaxation_strength=1., match_curvature=curvature)
            with patch('src.ideal_surface.capture_organic_surface', return_value=height_surface), (
                    patch('src.grid_quality.improve_projected_shapes', return_value=0) if not surface_relaxation else patch(
                    'src.grid_quality.improve_projected_shapes', side_effect=interior_only)):
                result = fill_refined_rows(trial, [trial.verts[i] for i in loop_ids],
                                           [trial.edges[i] for i in edge_ids], obj.matrix_world, **options)
            trial_rim = set(result['boundary_verts'])
            assert all(v.co == co for v, co in original.items() if v not in trial_rim)
            assert result['grid_cut_count'] == 0
            assert all(len(f.verts) == 4 for f in result['faces'])
            runs.append((warpage(result), len(result['faces']), len(result['verts'])))
            if surface_relaxation:
                queries = [(p.dot(x), p.dot(y)) for v in result['verts']
                           for p in [obj.matrix_world @ v.co - center]]
                if curvature:
                    expected = surface(queries)
                    actual = [(obj.matrix_world @ v.co - center).dot(axis) for v in result['verts']]
                    # BMesh positions and chart dot products are float32;
                    # the steep fitted collar amplifies their rounding.
                    assert np.max(np.abs(expected - actual)) < 2e-6 * scale
            trial.free()
        assert runs[0][1:] == runs[1][1:]
        assert runs[1][0] < .6 * runs[0][0], (scale, curvature, runs)
        assert runs[1][0] < .045 * scale

print('HOLEY_TOOLS_CURVATURE_OVERSHOOT_OK')
