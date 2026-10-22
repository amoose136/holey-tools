"""Native RNA invocation can be safely cancelled by reload/unregister cleanup."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src
import src.operator as runtime
from src.modal_session import _sessions
from tests.blender_support import edit_mesh


n = 16
rings = ((3, 0), (1, 0), (1, -1), (3, -1))
vertices = [(r * math.cos(math.tau * i / n), r * math.sin(math.tau * i / n), z)
            for r, z in rings for i in range(n)]
faces = [(a * n + i, a * n + (i + 1) % n, b * n + (i + 1) % n, b * n + i)
         for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)) for i in range(n)]
obj, bm = edit_mesh('Modal session source', vertices, faces)
mesh = obj.data
bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
bm.faces.ensure_lookup_table()
for face in bm.faces:
    face.select = n <= face.index < 2 * n
bm.faces.active = bm.faces[n]
source = [vert.co.copy() for vert in bm.verts]
source_selection = {face.index for face in bm.faces if face.select}
mesh_pointer = mesh.as_pointer()
window = bpy.context.window
area = next(area for area in window.screen.areas if area.type == 'VIEW_3D')
region = next(region for region in area.regions if region.type == 'WINDOW')

src.register()


class MESH_OT_holey_modal_session_driver(runtime.MESH_OT_holey_tools_resize_hole):
    """Background Blender skips invoke; call it explicitly on a real RNA owner."""
    bl_idname = 'mesh.holey_modal_session_driver'
    bl_label = 'Modal session integration driver'
    __annotations__ = dict(runtime.MESH_OT_holey_tools_resize_hole.__annotations__)

    def execute(self, context):
        return self.invoke(context, SimpleNamespace(ctrl=False, oskey=False, mouse_x=0, mouse_y=0))


bpy.utils.register_class(MESH_OT_holey_modal_session_driver)


class MESH_OT_holey_fair_session_driver(src.MESH_OT_holey_tools_fair_hole):
    bl_idname = 'mesh.holey_fair_session_driver'
    bl_label = 'Fair modal session integration driver'
    __annotations__ = dict(src.MESH_OT_holey_tools_fair_hole.__annotations__)

    def execute(self, context):
        return self.invoke(context, SimpleNamespace(mouse_x=0, mouse_y=0))


bpy.utils.register_class(MESH_OT_holey_fair_session_driver)
try:
    with patch.object(bpy.types.SpaceView3D, 'draw_handler_add', side_effect=lambda *_args: object()), patch.object(
        bpy.types.SpaceView3D, 'draw_handler_remove', return_value=None
    ):
        for mode in ('BOOLEAN', 'SCALE'):
            with bpy.context.temp_override(window=window, area=area, region=region):
                assert bpy.ops.mesh.holey_modal_session_driver(resize_method=mode) == {'RUNNING_MODAL'}
                assert len(_sessions) == 1
                op = _sessions[0].operator
                assert op._lock_selection_and_fit(bpy.context, build_preview=False)
                op.diameter = 1.6
                op._begin_dimension_drag(bpy.context, SimpleNamespace(mouse_x=0, mouse_y=0), 'DIAMETER')
                op._apply_diameter_to_mesh(bpy.context)
                temporary = []
                if mode == 'BOOLEAN':
                    assert obj.hide_get()
                    temporary = [op._boolean_preview_display_object.name, op._boolean_preview_cutter_object.name]
                else:
                    assert any(vert.co != point for vert, point in zip(bmesh.from_edit_mesh(mesh).verts, source))
            # Invoke from View3D, clean up from the ordinary non-area context.
            runtime.cleanup_runtime()
            runtime.cleanup_runtime()
            assert not _sessions and not runtime._drag_timers and not runtime._draw_handles
            assert not runtime._modal_active and not obj.hide_get()
            assert obj.data.as_pointer() == mesh_pointer
            assert all(name not in bpy.data.objects for name in temporary)
            restored = bmesh.from_edit_mesh(mesh)
            assert [vert.co for vert in restored.verts] == source
            assert {face.index for face in restored.faces if face.select} == source_selection
            assert op.modal(bpy.context, SimpleNamespace(type='TIMER')) == {'CANCELLED'}
            op.cancel(bpy.context)

        with bpy.context.temp_override(window=window, area=area, region=region):
            assert bpy.ops.mesh.holey_fair_session_driver() == {'RUNNING_MODAL'}
            assert len(_sessions) == 1
        runtime.cleanup_runtime()
        assert not _sessions and not runtime._modal_active

finally:
    runtime.cleanup_runtime()
    bpy.utils.unregister_class(MESH_OT_holey_fair_session_driver)
    bpy.utils.unregister_class(MESH_OT_holey_modal_session_driver)
    src.unregister()

assert not _sessions
print('HOLEY_TOOLS_MODAL_SESSION_OK')
