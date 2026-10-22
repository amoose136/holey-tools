"""A folded brow needs a frozen 3D fit, independently of output topology."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fairing import _organic_projection_frame, fair_boundary_loop
from src.ideal_surface import capture_organic_surface
from src.intrinsic_surface import _ears, _delaunay
from tests.blender_support import edit_mesh

fixtures = Path(__file__).with_name('fixtures')
data = json.loads((fixtures / 'suzanne_triangle_grid_hole.json').read_text())
reference = json.loads((fixtures / 'suzanne_reference_surface.json').read_text())

# Concave and collinear stations are retained in the private parameter domain;
# the fit must not introduce cuts in the actual mesh boundary.
polygon = np.asarray(((0., 0.), (1., 0.), (2., 0.), (2., 2.), (1., 2.),
                      (1., 1.), (0., 1.)))
triangles = _delaunay(_ears(polygon), polygon)
assert len(triangles) == len(polygon) - 2
area = sum(abs(np.linalg.det(np.column_stack((polygon[b]-polygon[a], polygon[c]-polygon[a]))))*.5
           for a, b, c in triangles)
assert abs(area - 3.) < 1e-12
owners = {}
for triangle in triangles:
    for a, b in zip(triangle, triangle[1:] + triangle[:1]):
        edge = tuple(sorted((a, b)))
        owners[edge] = owners.get(edge, 0) + 1
assert {edge for edge, count in owners.items() if count == 1} == {
    tuple(sorted((i, (i+1) % len(polygon)))) for i in range(len(polygon))}

for scale in (.01, 1., 100.):
    obj, bm = edit_mesh('Folded organic fit', data['vertices'], data['faces'])
    obj.matrix_world = Matrix.Scale(scale, 4) @ Matrix(data['matrix'])
    bm.normal_update()
    loop = [bm.verts[i] for i in data['boundary']]
    edges = [bm.edges.get((v, loop[(i+1) % len(loop)])) for i, v in enumerate(loop)]
    center, axis, x, y, points = _organic_projection_frame(loop, obj.matrix_world)
    captured = capture_organic_surface(loop, obj.matrix_world, center, x, y, axis)
    assert captured.continuity == 'INTRINSIC_BENDING'
    before = {v: v.co.copy() for v in bm.verts}
    topology = (len(bm.verts), len(bm.edges), len(bm.faces))
    samples = np.asarray(points) * .5
    expected = captured.evaluate_chart(samples)
    assert np.isfinite(expected).all()
    assert topology == (len(bm.verts), len(bm.edges), len(bm.faces))
    assert all(v.co == co for v, co in before.items())
    # No BMesh handles/normals can leak into the frozen fit.
    for v in bm.verts:
        v.co.z += 3. * scale
    assert np.array_equal(captured.evaluate_chart(samples), expected)
    for v, co in before.items():
        v.co = co
    bm.normal_update()
    tree = BVHTree.FromPolygons([obj.matrix_world @ Vector(p) for p in reference['vertices']],
                               reference['faces'])
    def errors(vertices):
        signed, distances = [], []
        for v in vertices:
            point = obj.matrix_world @ v.co
            nearest, normal, _face, distance = tree.find_nearest(point)
            signed.append((point-nearest).dot(normal) / scale)
            distances.append(distance / scale)
        return max(signed), max(distances)
    # No relaxation: this regression measures the surface rather than a
    # topology optimizer concealing a bad fit. Capture must precede creation.
    def capture_before_construction(*args, **kwargs):
        assert topology == (len(bm.verts), len(bm.edges), len(bm.faces))
        assert all(v.co == co for v, co in before.items())
        return captured
    with patch('src.ideal_surface.capture_organic_surface', side_effect=capture_before_construction) as capture:
        result = fair_boundary_loop(bm, loop, edges, obj.matrix_world,
                                    dict(method='ORGANIC', residual=0.), 'CIRCLE', circle_loops=37)
    assert capture.call_count == 1
    assert result['surface_continuity'] == 'INTRINSIC_BENDING'
    assert all(v.co == co for v, co in before.items())
    assert all(edge.is_valid for edge in edges)
    outward, distance = errors(result['verts'])
    assert outward < .025 and distance < .04, (scale, outward, distance)
    world = np.asarray([obj.matrix_world @ v.co for v in result['verts']])
    membership_error = np.max(np.linalg.norm(captured.project(world)-world, axis=1)) / scale
    # BMesh coordinates and BVH queries round independently to float32;
    # narrow reference triangles amplify the nearest-point rounding slightly.
    assert membership_error < 1e-5, (scale, membership_error)
    print('Folded Circle37 error:', scale, outward, distance)
    if scale == 1.:
        # Negative control: the former Circle height solve reproduces the
        # reported protrusion, even with exactly the same output topology.
        trial = bm.copy()
        trial.verts.ensure_lookup_table()
        # Delete only the generated patch from this copy, preserving source.
        bm.verts.index_update()
        bmesh.ops.delete(trial, geom=[trial.verts[v.index] for v in result['verts']], context='VERTS')
        trial.verts.ensure_lookup_table()
        source_loop = [trial.verts[v.index] for v in loop]
        source_edges = [trial.edges.get((v, source_loop[(i+1) % len(source_loop)]))
                        for i, v in enumerate(source_loop)]
        with patch('src.ideal_surface.capture_organic_surface', return_value=None):
            legacy = fair_boundary_loop(trial, source_loop, source_edges, obj.matrix_world,
                                       dict(method='ORGANIC', residual=0.), 'CIRCLE', circle_loops=37)
        assert errors(legacy['verts'])[0] > .05
        trial.free()

print('HOLEY_TOOLS_ORGANIC_SURFACE_OK')
