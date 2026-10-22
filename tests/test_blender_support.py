"""Shared fixtures preserve supplied topology and native redo snapshot identity."""

import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.blender_support import (
    bind_operator_methods, capture_redo_source, edit_mesh, rna_defaults,
)


class Base:
    @staticmethod
    def static(value):
        return value + 1

    def method(self):
        return 'base'


class Derived(Base):
    def method(self):
        return self.marker


operator = SimpleNamespace(marker='derived', resize_method='SCALE')
assert bind_operator_methods(operator, Derived) is operator
assert operator.static(4) == 5 and operator.method() == 'derived'
assert operator.resize_method == 'SCALE'
assert not hasattr(operator, 'diameter'), 'Method binding must not invent RNA state'
first, second = rna_defaults(), rna_defaults()
first['diameter'] = 12
assert second['diameter'] != first['diameter']

vertices, faces = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [(0, 1, 2, 3)]
obj, bm = edit_mesh('Support fixture', vertices, faces)
assert bpy.context.active_object == obj and bpy.context.selected_objects == [obj]
assert [tuple(v.co) for v in bm.verts] == vertices
assert [tuple(v.index for v in f.verts) for f in bm.faces] == faces
pointer = obj.data.as_pointer()
original = bm.verts[0].co.copy()
bm.verts[0].co.x += 2
source = capture_redo_source(SimpleNamespace(_object=obj, _original_vertex_coords={0: original}))
source_name = source._source_mesh.name
try:
    assert source._source_mesh.as_pointer() != pointer
    assert source.restore_source(obj)
    assert obj.data.as_pointer() == pointer and obj.mode == 'EDIT'
    restored = bmesh.from_edit_mesh(obj.data)
    restored.verts.ensure_lookup_table()
    assert restored.verts[0].co == original
finally:
    source.remove()
    source.remove()
assert source_name not in bpy.data.meshes
other, other_bm = edit_mesh('Second support fixture', vertices, faces)
assert obj.mode == 'OBJECT' and other.mode == 'EDIT'
assert bpy.context.selected_objects == [other] and bpy.context.active_object == other
assert len(other_bm.faces) == 1 and obj.data.as_pointer() == pointer

print('HOLEY_TOOLS_BLENDER_SUPPORT_OK')
