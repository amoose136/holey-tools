"""Run with Blender --background --factory-startup --python tests/test_fairing_pipeline.py."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src
import src.fairing_pipeline as pipeline
import src.geometry as geometry
import src.fair_hole as fair_hole
from src.constants import DEVELOPMENT_CONTROLS
from src.fair_hole import MESH_OT_holey_tools_fair_hole
from src.fairing import FairingError, ordered_boundary_loop, resolved_fill_style


def fixture(grid=True):
    if bpy.context.object and bpy.context.object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bm = bmesh.new()
    if grid:
        coordinates = {(i, j): bm.verts.new((i * .15, j * .15,
                          math.sqrt(1. - (i * .15) ** 2 - (j * .15) ** 2)))
                       for i in range(-3, 4) for j in range(-3, 4)}
        for i in range(-3, 3):
            for j in range(-3, 3):
                if -1 <= i < 1 and -1 <= j < 1:
                    continue
                bm.faces.new([coordinates[i, j], coordinates[i + 1, j],
                              coordinates[i + 1, j + 1], coordinates[i, j + 1]])
        for vert in list(bm.verts):
            if not vert.link_edges:
                bm.verts.remove(vert)
        edges = [e for e in bm.edges if len(e.link_faces) == 1 and
                 all(abs(v.co.x) <= .151 and abs(v.co.y) <= .151 for v in e.verts)]
    else:
        # A circular strip is not a four-sided source grid. It must fall back
        # without retaining vertices/faces created by failed preparation.
        rings = [[bm.verts.new((radius * math.cos(math.tau * i / 12),
                               radius * math.sin(math.tau * i / 12), math.sqrt(1. - radius ** 2)))
                  for i in range(12)] for radius in (.3, .5, .7)]
        for first, second in zip(rings, rings[1:]):
            for i in range(12):
                j = (i + 1) % 12
                bm.faces.new([first[i], second[i], second[j], first[j]])
        edges = [bm.edges.get((rings[0][i], rings[0][(i + 1) % 12])) for i in range(12)]
    bm.normal_update()
    bm.edges.index_update()
    indices = [e.index for e in edges]
    mesh = bpy.data.meshes.new('pipeline_fixture')
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.edges.ensure_lookup_table()
    loop, edges = ordered_boundary_loop(edges=[bm.edges[i] for i in indices])
    return obj, bm, loop, edges


sphere = dict(method='SPHERE', center=Vector(), radius=1., residual=0.)
assert geometry.fair_hole_patch is fair_hole.fair_hole_patch is pipeline.fair_hole_patch
assert resolved_fill_style(sphere) == 'GRID'
assert resolved_fill_style(dict(method='ORGANIC')) == 'GRID'
assert resolved_fill_style(dict(method='PLANES')) == 'PLANE'
assert resolved_fill_style(dict(method='CONIC', fit=dict(is_cone=False))) == 'CYLINDER'

obj, bm, loop, edges = fixture()
bm, result = pipeline.fair_hole_patch(obj, bm, loop, edges, sphere, final_simplify=False)
assert result['method'] == 'GRID_ROWS' and result['fill_style'] == 'GRID'
assert len(result['refinement_counts']) == 1
assert result['rows_filled'] == result['rows_total']
assert all(len(f.verts) == 4 for f in result['faces'])
assert all(len(e.link_faces) == 2 for v in result['boundary_verts'] for e in v.link_edges)

obj, bm, loop, edges = fixture(False)
bm, result = pipeline.fair_hole_patch(obj, bm, loop, edges, sphere, final_simplify=False)
assert result['fill_style'] == 'CIRCLE' and result['grid_fallback_reason']
assert len(set(result['verts']) - set(result['boundary_verts'])) == 5 * len(result['boundary_verts']) + 1
assert all(abs(v.co.length - 1.) < 1e-6 for v in result['verts'])
assert all(len(e.link_faces) == 2 for e in result['boundary_edges'])

# Deliberately corrupt the trial before rejecting it: the analytic preparation
# must receive exactly the original live geometry and source identities.
obj, bm, loop, edges = fixture()
original = {v: v.co.copy() for v in bm.verts}
source_faces = {f: tuple(f.verts) for f in bm.faces}
prepare_grid, prepare_analytic = pipeline.prepare_grid_steps, pipeline.prepare_fair_boundary
calls = []
def reject_grid(trial, *args, **kwargs):
    assert trial is not bm
    assert kwargs['refinement_passes'] == 1
    trial.verts.ensure_lookup_table()
    trial.verts[0].co += Vector((9., 8., 7.))
    trial.verts.new((100., 100., 100.))
    raise FairingError('injected failure after preparation')
def check_original(obj, live, *args, **kwargs):
    assert live is bm and len(live.verts) == len(original)
    assert all(v.is_valid and v.co == co for v, co in original.items())
    assert all(f.is_valid and tuple(f.verts) == vertices for f, vertices in source_faces.items())
    calls.append('circle')
    return prepare_analytic(obj, live, *args, **kwargs)
try:
    pipeline.prepare_grid_steps = reject_grid
    pipeline.prepare_fair_boundary = check_original
    bm, result = pipeline.fair_hole_patch(obj, bm, loop, edges, sphere, final_simplify=False)
    assert calls == ['circle'] and result['fill_style'] == 'CIRCLE'
finally:
    pipeline.prepare_grid_steps, pipeline.prepare_fair_boundary = prepare_grid, prepare_analytic

# An exception can have an empty message. It is still a failed grid probe,
# distinct from None (success), and its trial must always be freed.
obj, bm, loop, edges = fixture()
trials = []
def reject_without_message(trial, *args, **kwargs):
    trials.append(trial)
    trial.verts.new((100., 100., 100.))
    raise FairingError()
try:
    pipeline.prepare_grid_steps = reject_without_message
    bm, result = pipeline.fair_hole_patch(
        obj, bm, loop, edges, sphere, final_simplify=False
    )
    assert result['fill_style'] == 'CIRCLE'
    assert result['grid_fallback_reason'] == ''
    assert len(trials) == 1 and not trials[0].is_valid
finally:
    pipeline.prepare_grid_steps = prepare_grid

# Defaults, preselection reset, and development-only UI controls.
obj, bm, loop, edges = fixture()
next(face for face in bm.faces if not any(edge in edges for edge in face.edges)).hide = True
with patch.object(pipeline, 'visible_bmesh_copy', side_effect=ValueError('copy failed')):
    assert pipeline._probe_grid_patch(obj, bm, edges, {}) == 'copy failed'
# Mapping failure also frees an already allocated copy before Auto fallback.
trial = bmesh.new()
with patch.object(pipeline, 'visible_bmesh_copy', return_value=(trial, {}, {}, {})):
    assert pipeline._probe_grid_patch(obj, bm, edges, {}) is not None
assert not trial.is_valid

# Defaults, preselection reset, and development-only UI controls.
src.register()
props = bpy.ops.mesh.holey_tools_fair_hole.get_rna_type().properties
assert props['fill_style'].default == props['fairing_method'].default == 'AUTO'
assert props['fairing_method'].name == 'Fit'
assert props['fairing_method'].enum_items['AUTO'].name == 'Auto Fit'
assert props['fill_style'].name == 'Topology'
assert props['fill_style'].enum_items['AUTO'].name == 'Auto Topology'
assert props['circle_loops'].default == 5 and props['grid_match_curvature'].default
assert 'grid_refinement_passes' not in props and 'grid_backend' not in props
assert 'grid_angle' not in props and 'grid_density' not in props
resize = bpy.ops.mesh.holey_tools_resize_hole.get_rna_type().properties
assert resize['circle_loops'].default == 5 and resize['grid_match_curvature'].default
assert 'grid_angle' not in resize and 'grid_density' not in resize
scene = bpy.context.scene
assert scene.holey_tools_fair_fill == scene.holey_tools_fair_surface == 'AUTO'
assert bpy.types.Scene.bl_rna.properties['holey_tools_fair_surface'].name == 'Fair Fit'
assert bpy.types.Scene.bl_rna.properties['holey_tools_fair_fill'].name == 'Fair Topology'
assert not DEVELOPMENT_CONTROLS
assert not hasattr(scene, 'holey_tools_development_mode')
op = SimpleNamespace(fill_style='CIRCLE', fairing_method='CYLINDER', grid_step='DISSOLVE',
                     grid_fill_rows=2, circle_loops=1)
MESH_OT_holey_tools_fair_hole._reset_invocation_defaults(op, scene)
assert (op.fill_style, op.fairing_method, op.grid_step, op.grid_fill_rows, op.circle_loops) == ('AUTO', 'AUTO', 'ROWS', 0, 5)
scene.holey_tools_fair_fill, scene.holey_tools_fair_surface = 'GRID', 'SPHERE'
MESH_OT_holey_tools_fair_hole._reset_invocation_defaults(op, scene)
assert (op.fill_style, op.fairing_method) == ('GRID', 'SPHERE')

class Layout:
    def __init__(self): self.controls, self.labels = [], []
    def prop(self, obj, name): self.controls.append(name)
    def label(self, **kwargs): self.labels.append(kwargs['text'])
    def separator(self): pass
layout = Layout()
op = SimpleNamespace(layout=layout, fill_style='AUTO', result_fill_style='GRID',
                     result_method='GRID_ROWS', result_grid_surface_continuity='INTRINSIC_BENDING')
MESH_OT_holey_tools_fair_hole.draw(op, bpy.context)
assert layout.controls == ['fairing_method', 'fill_style', 'grid_match_curvature']
assert 'Surface: 3D bending continuation (folded chart)' in layout.labels
layout = Layout()
op = SimpleNamespace(layout=layout, fill_style='GRID', result_fill_style='GRID',
                     result_method='SPHERE', result_residual=.01, result_grid_surface_continuity='ANALYTIC')
MESH_OT_holey_tools_fair_hole.draw(op, bpy.context)
assert layout.controls == ['fairing_method', 'fill_style']
layout = Layout()
op = SimpleNamespace(layout=layout, fill_style='CIRCLE', result_fill_style='CIRCLE',
                     result_method='ORGANIC', result_grid_surface_continuity='INTRINSIC_BENDING')
MESH_OT_holey_tools_fair_hole.draw(op, bpy.context)
assert layout.controls == ['fairing_method', 'fill_style', 'circle_loops', 'final_simplify']
assert 'Surface: 3D bending continuation (folded chart)' in layout.labels

# Even stale development redo values cannot leave a partial production patch.
obj, bm, loop, edges = fixture()
bm.edges.index_update()
op = SimpleNamespace(edge_indices=','.join(str(e.index) for e in edges),
                     fill_style='GRID', grid_step='DISSOLVE', grid_fill_rows=1,
                     fairing_method='ORGANIC',
                     grid_relaxation_iterations=0, report=lambda *_args: None)
assert MESH_OT_holey_tools_fair_hole.execute(op, bpy.context) == {'FINISHED'}
assert op.result_method == 'ORGANIC'
assert op.result_grid_rows_filled == op.result_grid_rows_total
assert op.result_grid_relaxation_passes > 0
src.unregister()
print('HOLEY_TOOLS_FAIRING_PIPELINE_OK', flush=True)
