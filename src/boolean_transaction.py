# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ownership and rollback for one Boolean reconstruction attempt."""

import bpy

from .mesh_state import restore_mesh_contents


class BooleanTransaction:
    """Own temporary native resources until cleanup or preview handoff.

    The caller synchronizes Edit Mode before construction, runs geometry work,
    calls rollback on failure, and always calls close. Restoring contents keeps
    the real mesh ID alive for Blender's native edit-mesh undo stack.
    """

    def __init__(self, obj):
        self.object = obj
        self.backup = obj.data.copy()
        self.source_hidden = obj.hide_get()
        self.cutter_object = None
        self.cutter_mesh = None
        self.preview_object = None
        self.modifier_owner = None
        self.modifier_name = None
        self.hidden_geometry = None
        self.temporary_materials = []
        self.closed = False

    def rollback(self):
        """Restore source geometry and Edit Mode, including material slots."""
        obj = self.object
        bpy.context.view_layer.objects.active = obj
        obj.hide_set(False)
        obj.select_set(True)
        restore_mesh_contents(obj, self.backup)
        bpy.ops.object.mode_set(mode='EDIT')
        obj.hide_set(self.source_hidden)

    def release_preview(self):
        """Relinquish objects/modifier after the modal owner accepts them."""
        self.preview_object = None
        self.cutter_object = None
        self.cutter_mesh = None
        self.modifier_owner = None
        self.modifier_name = None

    def close(self):
        """Release untransferred resources; safe to call more than once."""
        if self.closed:
            return
        self.closed = True
        if self.hidden_geometry is not None:
            self.hidden_geometry.free()
        if self.modifier_name is not None:
            modifier = self.modifier_owner.modifiers.get(self.modifier_name)
            if modifier is not None:
                self.modifier_owner.modifiers.remove(modifier)
        if self.preview_object is not None:
            preview_mesh = self.preview_object.data
            bpy.data.objects.remove(self.preview_object, do_unlink=True)
            if preview_mesh.users == 0:
                bpy.data.meshes.remove(preview_mesh)
        if self.cutter_object is not None:
            bpy.data.objects.remove(self.cutter_object, do_unlink=True)
        if self.cutter_mesh is not None and self.cutter_mesh.users == 0:
            bpy.data.meshes.remove(self.cutter_mesh)
        if self.backup.users == 0:
            bpy.data.meshes.remove(self.backup)
        for material in self.temporary_materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
