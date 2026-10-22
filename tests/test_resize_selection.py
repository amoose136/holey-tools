"""Ctrl/Cmd invocation fits the exact current selection, without hover flooding."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.operator import MESH_OT_holey_tools_resize_hole as Resize
import src.operator as runtime
from tests.blender_support import bind_operator_methods, edit_mesh


def fixture(cone=False):
    count = 12
    vertices = [((2 - z * .5 if cone else 1) * math.cos(math.tau * i / count),
                 (2 - z * .5 if cone else 1) * math.sin(math.tau * i / count), z)
                for z in (0, 1, 2) for i in range(count)]
    faces = [(row * count + i, row * count + (i + 1) % count,
              (row + 1) * count + (i + 1) % count, (row + 1) * count + i)
             for row in range(2) for i in range(count)]
    obj, bm = edit_mesh('Selection shortcut', vertices, faces)
    bm.normal_update()
    for face in bm.faces:
        face.select = face.index < 6
    bm.faces.active = bm.faces[0]
    return obj, bm


def operator_and_context(obj):
    op = SimpleNamespace(diameter_estimator='AVERAGE', flood_angle=math.radians(60),
                         _draw_handle=None, _draw_error_reported=False)
    bind_operator_methods(op)
    op.poll = lambda _context: True
    warnings, handlers = [], []
    op.report = lambda _level, message: warnings.append(message)
    context = SimpleNamespace(
        active_object=obj, scene=bpy.context.scene,
        area=SimpleNamespace(type='VIEW_3D', regions=[SimpleNamespace(
            type='WINDOW', x=100, y=100, width=800, height=600)], tag_redraw=lambda: None),
        space_data=SimpleNamespace(region_3d=SimpleNamespace()),
        workspace=SimpleNamespace(status_text_set=lambda _text: None),
        window=SimpleNamespace(cursor_modal_set=lambda _cursor: None, cursor_modal_restore=lambda: None),
        window_manager=SimpleNamespace(modal_handler_add=lambda operator: handlers.append(operator)),
    )
    return op, context, warnings, handlers


def forbidden(*_args, **_kwargs):
    raise AssertionError('Selection shortcut must not hover or flood')


bpy.context.scene.tool_settings.use_proportional_edit = False
with patch.object(bpy.types.SpaceView3D, 'draw_handler_add', return_value=object()), patch.object(
    bpy.types.SpaceView3D, 'draw_handler_remove', return_value=None
):
    for cone in (False, True):
        for ctrl, cmd in ((True, False), (False, True), (True, True)):
            obj, bm = fixture(cone)
            initial_selection = {f.index for f in bm.faces if f.select}
            initial_points = [v.co.copy() for v in bm.verts]
            op, context, warnings, handlers = operator_and_context(obj)
            mode = 'BOOLEAN' if ctrl and cmd else 'SCALE' if cmd else 'AUTO'
            op.resize_method = mode
            op._flood_fill = op._set_hover_seed = op._update_hover = forbidden
            event = SimpleNamespace(ctrl=ctrl, oskey=cmd, mouse_x=950, mouse_y=250)
            assert Resize.invoke(op, context, event) == {'RUNNING_MODAL'}
            assert not warnings and handlers == [op]
            assert op._bvh is None and op._stage == 'CYLINDER'
            assert op._locked_indices == initial_selection
            assert op.resize_method == mode
            assert op._resize_requires_boolean == (mode == 'BOOLEAN')
            assert op._fit['is_cone'] == cone
            assert op._modal_initial_dimensions is not None
            assert {f.index for f in bm.faces if f.select} == initial_selection
            assert all(v.co == co for v, co in zip(bm.verts, initial_points))
            # The invocation must not start dragging or apply a dimension.
            assert not op._drag_active
            # Both Escape and accepting unchanged dimensions must be no-ops.
            finish = 'RET' if cmd else 'ESC'
            assert Resize.modal(op, context, SimpleNamespace(type=finish, value='PRESS')) == {'CANCELLED'}
            assert not runtime._modal_active
            assert {f.index for f in bm.faces if f.select} == initial_selection
            assert all(v.co == co for v, co in zip(bm.verts, initial_points))

    # No selection and an undersized selection cancel without entering hover
    # mode or installing modal/draw handlers.
    for selected in (set(), {0}):
        obj, bm = fixture()
        for face in bm.faces:
            face.select = face.index in selected
        op, context, warnings, handlers = operator_and_context(obj)
        op._flood_fill = op._set_hover_seed = op._update_hover = forbidden
        assert Resize.invoke(op, context, SimpleNamespace(ctrl=True, mouse_x=950, mouse_y=250)) == {'CANCELLED'}
        assert warnings and not handlers
        assert not runtime._modal_active
        assert {f.index for f in bm.faces if f.select} == selected

    # A plain click still uses the existing hover/flood-fill selection stage.
    for mode in ('AUTO', 'SCALE', 'BOOLEAN'):
        obj, bm = fixture()
        op, context, warnings, handlers = operator_and_context(obj)
        op.resize_method = mode
        hovered = []
        op._update_hover = lambda *_args: hovered.append(True)
        assert Resize.invoke(op, context, SimpleNamespace(mouse_x=950, mouse_y=250)) == {'RUNNING_MODAL'}
        assert op._stage == 'SELECT' and hovered == [True]
        assert op.resize_method == mode
        assert len(op._candidate_indices) > 6
        assert handlers == [op]
        assert Resize.modal(op, context, SimpleNamespace(type='ESC', value='PRESS', ctrl=False, ascii='')) == {'CANCELLED'}

assert not runtime._modal_active and not runtime._draw_handles
print('HOLEY_TOOLS_SELECTION_SHORTCUT_OK')
