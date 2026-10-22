"""Native temporary-resource ownership, rollback, and preview handoff."""

import sys
from pathlib import Path

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.boolean_transaction import BooleanTransaction
from src.visible_geometry import detach_hidden_geometry


def resources():
    return tuple({item.as_pointer() for item in collection}
                 for collection in (bpy.data.objects, bpy.data.meshes, bpy.data.materials))


def signature(obj):
    obj.update_from_editmode()
    mesh = obj.data
    return (
        tuple(tuple(v.co) for v in mesh.vertices),
        tuple((tuple(f.vertices), f.hide, f.material_index) for f in mesh.polygons),
        tuple(mesh.materials),
        tuple(tuple(loop.uv) for loop in mesh.uv_layers['Transaction UV'].data),
        tuple(item.value for item in mesh.attributes['Transaction weight'].data),
    )


bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.object
mesh_pointer = obj.data.as_pointer()
obj.data.uv_layers.new(name='Transaction UV')
weights = obj.data.attributes.new('Transaction weight', 'FLOAT', 'POINT')
for index, item in enumerate(weights.data):
    item.value = index / 8
obj.data.polygons[0].hide = True
bpy.ops.object.mode_set(mode='EDIT')
source = signature(obj)


def allocate(transaction, stage, owner_is_source=False):
    """Stop at each allocation boundary, as if the following operation failed."""
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.data.vertices[0].co.x += 7
    if stage == 0:
        return
    transaction.hidden_geometry = detach_hidden_geometry(obj)
    if stage == 1:
        return
    transaction.cutter_mesh = bpy.data.meshes.new('Transaction cutter')
    if stage == 2:
        return
    transaction.cutter_object = bpy.data.objects.new('Transaction cutter', transaction.cutter_mesh)
    bpy.context.collection.objects.link(transaction.cutter_object)
    if stage == 3:
        return
    transaction.preview_object = bpy.data.objects.new('Transaction preview', obj.data.copy())
    bpy.context.collection.objects.link(transaction.preview_object)
    if stage == 4:
        return
    transaction.modifier_owner = obj if owner_is_source else transaction.preview_object
    modifier = transaction.modifier_owner.modifiers.new('Transaction modifier', 'BOOLEAN')
    transaction.modifier_name = modifier.name
    modifier.object = transaction.cutter_object
    material = bpy.data.materials.new('Transaction surface tag')
    transaction.temporary_materials.append(material)
    transaction.modifier_owner.data.materials.append(material)


for stage in range(6):
    for owner_is_source in (False, True) if stage == 5 else (False,):
        before = resources()
        transaction = BooleanTransaction(obj)
        try:
            allocate(transaction, stage, owner_is_source)
            obj.hide_set(True)
            transaction.rollback()
        finally:
            transaction.close()
            transaction.close()
        assert obj.mode == 'EDIT' and not obj.hide_get()
        assert obj.data.as_pointer() == mesh_pointer
        assert signature(obj) == source, (stage, owner_is_source)
        assert resources() == before, ('Leaked resource', stage, owner_is_source)
        assert not obj.modifiers

# A successful final transaction keeps the result, but not its backup/cutter.
before = resources()
transaction = BooleanTransaction(obj)
try:
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.data.vertices[0].co.x += 1
    transaction.cutter_mesh = bpy.data.meshes.new('Successful cutter')
finally:
    transaction.close()
assert obj.data.as_pointer() == mesh_pointer and resources() == before
assert obj.data.vertices[0].co.x == source[0][0][0] + 1

# Preview objects, their modifier, and in-use tags survive an explicit handoff.
bpy.ops.object.mode_set(mode='EDIT')
before = resources()
transaction = BooleanTransaction(obj)
try:
    allocate(transaction, 5)
    preview = transaction.preview_object
    cutter = transaction.cutter_object
    material = transaction.temporary_materials[0]
    transaction.rollback()  # Real edit object stays pristine during preview.
    transaction.release_preview()
finally:
    transaction.close()
assert preview.name in bpy.data.objects and cutter.name in bpy.data.objects
assert preview.modifiers[0].object == cutter
assert material.name in bpy.data.materials
preview_mesh, cutter_mesh = preview.data, cutter.data
bpy.data.objects.remove(preview, do_unlink=True)
bpy.data.objects.remove(cutter, do_unlink=True)
bpy.data.meshes.remove(preview_mesh)
bpy.data.meshes.remove(cutter_mesh)
bpy.data.materials.remove(material)
assert resources() == before

print('HOLEY_TOOLS_BOOLEAN_TRANSACTION_OK')
