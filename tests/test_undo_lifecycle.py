"""Run in a separate Blender process: the old restore segfaults native undo."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.operator import MESH_OT_holey_tools_resize_hole as Resize
from src.geometry import GeometryMixin
from src.fairing import FairingError
from tests.blender_support import capture_redo_source


bpy.context.preferences.edit.use_global_undo = True
segments = 32
rings = ((5, 0), (2, 0), (1, -1), (1, -3), (5, -3))
vertices = [(r * math.cos(math.tau * i / segments),
             r * math.sin(math.tau * i / segments), z)
            for r, z in rings for i in range(segments)]
faces = []
for first, second, reverse in ((0, 1, False), (0, 4, True), (4, 3, True),
                               (1, 2, False), (2, 3, False)):
    for i in range(segments):
        j = (i + 1) % segments
        face = (first * segments + i, first * segments + j,
                second * segments + j, second * segments + i)
        faces.append(tuple(reversed(face)) if reverse else face)
mesh = bpy.data.meshes.new('Undo countersink')
mesh.from_pydata(vertices, (), faces)
mesh['undo_identity_marker'] = 42
mesh.uv_layers.new(name='Source UV')
weights = mesh.attributes.new('source_weight', 'FLOAT', 'POINT')
for index, item in enumerate(weights.data):
    item.value = index / len(vertices)
obj = bpy.data.objects.new('Undo countersink', mesh)
bpy.context.collection.objects.link(obj)
for selected in bpy.context.selected_objects:
    selected.select_set(False)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(mesh)
bm.normal_update()
bm.faces.ensure_lookup_table()


def attach_methods(op):
    op._world_normal = lambda face: Resize._world_normal(op, face)
    op._target_scale = lambda: Resize._target_scale(op)
    op._target_depth_scale = lambda: Resize._target_depth_scale(op)
    op.report = lambda _level, message: (_ for _ in ()).throw(AssertionError(message))
    return op


baseline = attach_methods(SimpleNamespace(
    _object=obj, resize_method='AUTO', diameter_estimator='AVERAGE',
    fit_valid=False, scale=1.0, scaling_final_simplify=False,
))
wall = list(bm.faces[96:128])
baseline._locked_indices = {face.index for face in wall}
baseline._fit = Resize._fit_cylinder(baseline, wall)
baseline._end_planes = Resize._fit_boundary_planes(baseline, wall, baseline._fit['axis'], baseline._fit)
Resize._build_resize_projection_state(baseline, bm, wall)
Resize._store_fit_for_redo(baseline)
serialized = {key: value for key, value in vars(baseline).items()
              if not key.startswith('_') and key != 'report'}
owner = capture_redo_source(baseline)


def fresh_state(obj, method, truncated):
    op = attach_methods(SimpleNamespace(**serialized))
    op._object = obj
    op.resize_method = method
    op.truncated_depth = truncated
    op.depth = truncated + op.fit_conic_bottom_radius
    assert Resize._restore_fit_for_redo(op)
    return op


# Modal Boolean/Scaling transitions, including cached preview recovery and
# zero-depth fairing displays, may replace contents but never the real mesh ID.
for method, truncated, fair_only in (
    ('BOOLEAN', .8, False), ('BOOLEAN', 1.2, False), ('AUTO', 0.0, True),
):
    obj = bpy.context.active_object
    assert owner.restore_source(obj)
    mesh_pointer = obj.data.as_pointer()
    op = fresh_state(obj, method, truncated)
    if fair_only:
        op.depth = 0.0
    Resize._apply_diameter_to_mesh(op, bpy.context)
    assert obj.data.as_pointer() == mesh_pointer
    assert op._boolean_preview_display_object is not None
    assert op._boolean_preview_display_object.data != obj.data
    if not fair_only:
        Resize._apply_cached_boolean_preview(op, bpy.context, 32)
        assert obj.data.as_pointer() == mesh_pointer
        # Recreate a missing display from the cached operand, without ever
        # lending the real mesh ID to the replacement display object.
        Resize._remove_live_boolean_preview(op)
        Resize._apply_cached_boolean_preview(op, bpy.context, 32)
        assert obj.data.as_pointer() == mesh_pointer
        assert op._boolean_preview_display_object.data != obj.data
    Resize._replace_boolean_preview_with_source(op, consume=True)
    Resize._discard_boolean_preview_source(op)
    assert obj.data.as_pointer() == mesh_pointer
assert owner.restore_source(obj)

# Failure after cavity deletion restores the same real mesh, not a new ID.
mesh_pointer = obj.data.as_pointer()
signature = Resize._object_topology_signature(obj)
op = fresh_state(obj, 'BOOLEAN', .8)
with patch.object(GeometryMixin, '_boolean_cutter_geometry', side_effect=FairingError('rollback probe')):
    try:
        Resize._apply_diameter_to_mesh(op, bpy.context, commit_boolean=True)
    except FairingError as error:
        assert str(error) == 'rollback probe'
    else:
        raise AssertionError('Expected reconstruction failure')
assert obj.data.as_pointer() == mesh_pointer
assert Resize._object_topology_signature(obj) == signature

# Failure after native subtraction must also release the cutter/material tags
# and roll back attributes and geometry without replacing the real mesh ID.
before_resources = tuple({item.as_pointer() for item in collection}
                         for collection in (bpy.data.objects, bpy.data.meshes, bpy.data.materials))
op = fresh_state(obj, 'BOOLEAN', .8)
with patch.object(GeometryMixin, '_cleanup_faired_boolean_surface', side_effect=RuntimeError('post-Boolean rollback probe')):
    try:
        Resize._apply_diameter_to_mesh(op, bpy.context, commit_boolean=True)
    except FairingError as error:
        assert 'post-Boolean rollback probe' in str(error)
    else:
        raise AssertionError('Expected post-Boolean failure')
assert obj.mode == 'EDIT' and obj.data.as_pointer() == mesh_pointer
assert Resize._object_topology_signature(obj) == signature
assert before_resources == tuple({item.as_pointer() for item in collection}
                                 for collection in (bpy.data.objects, bpy.data.meshes, bpy.data.materials))


class MESH_OT_holey_undo_transition(bpy.types.Operator):
    bl_idname = 'mesh.holey_undo_transition'
    bl_label = 'Holey native undo regression'
    bl_options = {'REGISTER', 'UNDO'}

    method: bpy.props.EnumProperty(items=[(key, key, '') for key in ('SCALE', 'BOOLEAN', 'AUTO')])
    truncated: bpy.props.FloatProperty(default=1.0)

    def execute(self, context):
        obj = context.active_object
        mesh_pointer = obj.data.as_pointer()
        assert owner.restore_source(obj)
        assert obj.data.as_pointer() == mesh_pointer
        assert obj.data['undo_identity_marker'] == 42
        assert 'Source UV' in obj.data.uv_layers
        restored = bmesh.from_edit_mesh(obj.data)
        layer = restored.verts.layers.float['source_weight']
        restored.verts.index_update()
        assert all(abs(vert[layer] - vert.index / len(vertices)) < 1.0e-6 for vert in restored.verts)
        op = fresh_state(obj, self.method, self.truncated)
        assert Resize.execute(op, context) == {'FINISHED'}
        assert obj.data.as_pointer() == mesh_pointer
        current = bmesh.from_edit_mesh(obj.data)
        assert all(edge.is_manifold for edge in current.edges), ('execute', self.method, self.truncated)
        return {'FINISHED'}


bpy.utils.register_class(MESH_OT_holey_undo_transition)
try:
    bpy.ops.ed.undo_push(message='Before conic resize')
    for index, method in enumerate(('BOOLEAN', 'SCALE', 'AUTO') * 4):
        bpy.ops.mesh.holey_undo_transition(method=method, truncated=(.72, .84, 1.08, 1.22)[index % 4])
        # Scripts need explicit checkpoints to exercise Blender's native stack.
        bpy.ops.ed.undo_push(message='After conic resize')
        bpy.ops.ed.undo()
        bpy.ops.ed.redo()
        current = bmesh.from_edit_mesh(bpy.context.active_object.data)
        assert all(edge.is_manifold for edge in current.edges), (
            'redo', index, method, len(current.verts), len(current.faces)
        )
        assert all(face.calc_area() > 1.0e-8 for face in current.faces)
finally:
    owner.remove()
    bpy.utils.unregister_class(MESH_OT_holey_undo_transition)

print('HOLEY_TOOLS_UNDO_LIFECYCLE_OK')
