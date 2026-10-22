"""A boundary closed by native limited dissolve is a completed N-plane repair."""

import math
from pathlib import Path
import sys
from unittest.mock import patch

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src
import src.fair_hole as fair_operator
import src.fairing_pipeline as pipeline
from src.fairing import FairingError, _rediscover_boundary, ordered_boundary_loop
from tests.blender_support import edit_mesh


def fixture(remote=False):
    count = 12
    vertices, faces = [], []
    for offset in ((0, 0, 0), (10, 0, 0)) if remote else ((0, 0, 0),):
        base = len(vertices)
        vertices.extend((offset[0] + radius * math.cos(math.tau * i / count),
                         offset[1] + radius * math.sin(math.tau * i / count), z)
                        for radius, z in ((1, 0), (2, 0), (2, -1)) for i in range(count))
        faces.extend((base + a * count + i, base + a * count + (i + 1) % count,
                      base + b * count + (i + 1) % count, base + b * count + i)
                     for a, b in ((1, 0), (2, 1)) for i in range(count))
        faces.append(tuple(reversed(range(base + 2 * count, base + 3 * count))))
    obj, bm = edit_mesh('Dissolve-completed planar opening', vertices, faces)
    bm.normal_update()
    edges = [bm.edges.get((bm.verts[i], bm.verts[(i + 1) % count])) for i in range(count)]
    loop, edges = ordered_boundary_loop(edges=edges)
    remote_faces = {face for face in bm.faces if all(vert.co.x > 5 for vert in face.verts)}
    return obj, bm, loop, edges, remote_faces


sphere = dict(method='SPHERE', center=Vector((0, 0, -1)), radius=1., residual=0.)
cylinder = dict(method='CONIC', fit=dict(is_cone=False), residual=0.)

for hypothesis, style in ((sphere, 'CIRCLE'), (cylinder, 'CYLINDER')):
    for simplify in (False, True):
        for remote in (False, True):
            obj, bm, loop, edges, remote_faces = fixture(remote)
            untouched = {face: tuple(tuple(vert.co) for vert in face.verts) for face in remote_faces}
            mode = tuple(bpy.context.tool_settings.mesh_select_mode)
            with patch.object(pipeline, 'fair_boundary_loop', side_effect=AssertionError('Do not fill an already closed rim')):
                bm, result = pipeline.fair_hole_patch(
                    obj, bm, loop, edges, hypothesis, style,
                    final_simplify=simplify,
                    refit=lambda *_args: (_ for _ in ()).throw(AssertionError('No rim remains to refit')),
                )
            assert result['method'] == 'PLANES' and result['fill_style'] == 'PLANE'
            assert result['boundary_dissolved'] and result['residual'] == 0
            assert not result['boundary_verts'] and not result['boundary_edges']
            assert result['faces'] and all(face.is_valid for face in result['faces'])
            assert all(face.calc_area() > 1e-14 for face in bm.faces)
            assert all(not edge.is_boundary for edge in bm.edges if any(v.co.x < 5 for v in edge.verts))
            assert all(face.is_valid and tuple(tuple(v.co) for v in face.verts) == points
                       for face, points in untouched.items())
            assert not (set(result['faces']) & remote_faces)
            assert tuple(bpy.context.tool_settings.mesh_select_mode) == mode
            if remote:
                assert sum(edge.is_boundary for edge in bm.edges) == 12

# Erasing the local surface is still a failure, not proof of successful closure.
# In particular, do not rediscover and cap the separate opening as a fallback.
obj, bm, loop, edges, remote_faces = fixture(True)
points = [vert.co.copy() for vert in loop]
support = {face for edge in edges for face in edge.link_faces}
bmesh.ops.delete(bm, geom=list({v for f in bm.faces if f not in remote_faces for v in f.verts}), context='VERTS')
try:
    _rediscover_boundary(bm, points, support, original_verts=loop, original_edges=edges)
except FairingError as error:
    assert 'support surface' in str(error)
else:
    raise AssertionError('Deleted support geometry is not a completed planar repair')
assert sum(edge.is_boundary for edge in bm.edges) == 12

# Exercise the real operator's result reporting/selection with a native dissolve.
# Substitute only the incoming hypothesis to reach analytic pre-simplification
# on this planar fixture; do not mock the dissolve or completion logic.
obj, bm, loop, edges, _remote_faces = fixture()
bm.edges.index_update()
indices = ','.join(str(edge.index) for edge in edges)
src.register()
try:
    window = bpy.context.window
    area = next(area for area in window.screen.areas if area.type == 'VIEW_3D')
    region = next(region for region in area.regions if region.type == 'WINDOW')
    with bpy.context.temp_override(window=window, area=area, region=region), patch.object(
        fair_operator, 'choose_hypothesis', return_value=sphere
    ):
        assert bpy.ops.mesh.holey_tools_fair_hole(edge_indices=indices, fill_style='CIRCLE') == {'FINISHED'}
    bm = bmesh.from_edit_mesh(obj.data)
    assert not any(edge.is_boundary for edge in bm.edges)
    assert any(face.select for face in bm.faces)
finally:
    src.unregister()

print('HOLEY_TOOLS_DISSOLVE_COMPLETION_OK')
