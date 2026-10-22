# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Restore geometry without invalidating Blender's edit-mesh undo identity."""

import bmesh
import bpy


def restore_mesh_contents(obj, source, materials=None):
    """Overwrite the existing mesh ID from a detached Mesh or Python BMesh.

    Edit-mode undo retains mesh identities independently of ``Mesh.users``.
    Swapping ``obj.data`` and removing its zero-user predecessor can therefore
    leave undo with a dangling native mesh pointer. Restore contents instead;
    the caller remains responsible for returning to Edit Mode when appropriate.
    """
    owned = not isinstance(source, bmesh.types.BMesh)
    snapshot = bmesh.new() if owned else source
    try:
        if owned:
            snapshot.from_mesh(source)
            materials = list(source.materials) if materials is None else materials
        if obj.mode == 'EDIT':
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.mode_set(mode='OBJECT')
        snapshot.to_mesh(obj.data)
        if materials is not None:
            obj.data.materials.clear()
            for material in materials:
                obj.data.materials.append(material)
        obj.data.update()
    finally:
        if owned:
            snapshot.free()
