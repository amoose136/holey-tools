"""Small, explicit fixtures shared by standalone Blender regression scripts.

These helpers do not register classes, reset scenes, install drawing handlers,
or silently populate test doubles with RNA defaults. Native registration and
undo integration remain the responsibility of the tests that exercise them.
"""

import inspect
from copy import deepcopy
from types import MethodType

import bmesh
import bpy

from src.operator import MESH_OT_holey_tools_resize_hole as Resize, _PersistentPreview


def bind_operator_methods(operator, operator_class=Resize):
    """Bind Python implementation methods without constructing an RNA owner."""
    for cls in reversed(operator_class.__mro__):
        for name, value in vars(cls).items():
            if isinstance(value, staticmethod):
                setattr(operator, name, value.__func__)
            elif inspect.isfunction(value) and not name.startswith('__'):
                setattr(operator, name, MethodType(value, operator))
    return operator


def rna_defaults(operator_class=Resize):
    """Copy explicit RNA defaults only when a fixture asks for them."""
    return {name: deepcopy(prop.keywords['default'])
            for name, prop in operator_class.__annotations__.items()
            if 'default' in getattr(prop, 'keywords', {})}


def activate_mesh(mesh):
    """Link one mesh and make it the sole selected, active object in Object Mode."""
    if bpy.context.object and bpy.context.object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for selected in bpy.context.selected_objects:
        selected.select_set(False)
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def edit_mesh(name, vertices, faces, edges=()):
    """Create exact supplied topology; leave normals/selection to the caller."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, edges, faces)
    obj = activate_mesh(mesh)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return obj, bm


def capture_redo_source(operator):
    """Capture the real production redo snapshot, without installing drawing.

    The returned owner must be removed by the caller after source restoration.
    It deliberately does not retain the completed operator's RNA wrapper.
    """
    owner = object.__new__(_PersistentPreview)
    owner._draw_handle = None
    owner._source_object_name = operator._object.name
    (owner._source_mesh_name, owner._source_mesh, owner._source_bmesh,
     owner._source_materials) = _PersistentPreview._capture_canonical_source(operator)
    return owner
