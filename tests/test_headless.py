import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import DIAMETER_SYMBOL  # noqa: E402
from src.gpu_preview import _convex_hull_lines  # noqa: E402
from src.operator import (  # noqa: E402
    MESH_OT_holey_tools_resize_hole,
    _is_view_navigation_event,
)


assert ord(DIAMETER_SYMBOL) == 0x00D8, repr(DIAMETER_SYMBOL)

for navigation_event in (
    'MIDDLEMOUSE',
    'TRACKPADPAN',
    'TRACKPADZOOM',
    'MOUSEROTATE',
    'NDOF_MOTION',
    'NDOF_BUTTON_FIT',
):
    assert _is_view_navigation_event(navigation_event), navigation_event
assert not _is_view_navigation_event('LEFTMOUSE')
modal_context = SimpleNamespace(area=SimpleNamespace(type='VIEW_3D'))
for navigation_event in ('TRACKPADPAN', 'NDOF_MOTION'):
    assert MESH_OT_holey_tools_resize_hole.modal(
        SimpleNamespace(),
        modal_context,
        SimpleNamespace(type=navigation_event),
    ) == {'PASS_THROUGH'}
assert MESH_OT_holey_tools_resize_hole.modal(
    SimpleNamespace(),
    modal_context,
    SimpleNamespace(type='Z'),
) == {'PASS_THROUGH'}

angle_context = SimpleNamespace(
    scene=SimpleNamespace(holey_tools_flood_angle=math.radians(60.0))
)
angle_operator = SimpleNamespace(
    flood_angle=math.radians(60.0),
    _angle_numeric_input="60.0",
    _angle_numeric_replace=True,
    _show_flood_angle_hud=lambda _context: None,
    _rebuild_hover=lambda _context: None,
    _set_status_text=lambda _context: None,
)
angle_operator._update_flood_angle_numeric_preview = lambda context: (
    MESH_OT_holey_tools_resize_hole._update_flood_angle_numeric_preview(
        angle_operator, context
    )
)
assert not MESH_OT_holey_tools_resize_hole._handle_flood_angle_numeric_event(
    angle_operator,
    angle_context,
    SimpleNamespace(
        type='LEFTMOUSE',
        value='PRESS',
        ascii='',
        ctrl=False,
        alt=False,
    ),
)
for event_type, character in (('FOUR', '4'), ('FIVE', '5')):
    assert MESH_OT_holey_tools_resize_hole._handle_flood_angle_numeric_event(
        angle_operator,
        angle_context,
        SimpleNamespace(
            type=event_type,
            value='PRESS',
            ascii=character,
            ctrl=False,
            alt=False,
        ),
    )
assert abs(math.degrees(angle_operator.flood_angle) - 45.0) < 1.0e-6
assert (
    angle_context.scene.holey_tools_flood_angle == angle_operator.flood_angle
)

diameter_context = SimpleNamespace(
    scene=SimpleNamespace(
        unit_settings=SimpleNamespace(system='METRIC', scale_length=1.0)
    )
)
diameter_operator = SimpleNamespace(
    diameter=0.010,
    _format_diameter=lambda _context: "formatted",
    _build_cylinder_preview=lambda _context: None,
    _apply_diameter_to_mesh=lambda _context: None,
    _set_status_text=lambda _context: None,
)
diameter_operator._scene_unit_info = lambda context: (
    MESH_OT_holey_tools_resize_hole._scene_unit_info(context)
)
diameter_operator._step_diameter = lambda context, direction, fine: (
    MESH_OT_holey_tools_resize_hole._step_diameter(
        diameter_operator, context, direction, fine
    )
)
assert MESH_OT_holey_tools_resize_hole._handle_diameter_step_event(
    diameter_operator,
    diameter_context,
    SimpleNamespace(type='EQUAL', value='PRESS', shift=False),
)
assert abs(diameter_operator.diameter - 0.011) < 1.0e-9
assert MESH_OT_holey_tools_resize_hole._handle_diameter_step_event(
    diameter_operator,
    diameter_context,
    SimpleNamespace(type='MINUS', value='PRESS', shift=True),
)
assert abs(diameter_operator.diameter - 0.0109) < 1.0e-9

# Hit testing must be relative to the saved View3D WINDOW region. Modal
# operators invoked from the sidebar receive misleading mouse_region values.
gizmo_owner = SimpleNamespace(
    _gizmo_rect=(10.0, 20.0, 30.0, 40.0),
    _region=SimpleNamespace(x=100, y=200),
)
gizmo_event = SimpleNamespace(
    mouse_x=120,
    mouse_y=230,
    mouse_region_x=-500,
    mouse_region_y=-500,
)
assert MESH_OT_holey_tools_resize_hole._event_over_drag_gizmo(
    gizmo_owner, gizmo_event
)


hull_lines = _convex_hull_lines(
    [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0),
     (0.0, 1.0, 0.0), (1.0, 0.5, 0.0)]
)
assert len(hull_lines) == 8, hull_lines


bpy.ops.mesh.primitive_cylinder_add(
    vertices=16,
    radius=2.0,
    depth=4.0,
    end_fill_type='NOTHING',
)
obj = bpy.context.active_object
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()

operator = SimpleNamespace(
    _object=obj,
    flood_angle=math.radians(60.0),
    scale=1.5,
    _locked_indices={face.index for face in bm.faces},
)
operator._world_normal = lambda face: MESH_OT_holey_tools_resize_hole._world_normal(
    operator, face
)
operator._target_scale = lambda: MESH_OT_holey_tools_resize_hole._target_scale(operator)

triangles = MESH_OT_holey_tools_resize_hole._triangles_for_faces(
    operator, list(bm.faces)
)
assert len(triangles) == 16 * 2, len(triangles)
assert all(len(triangle) == 3 for triangle in triangles)

flood = MESH_OT_holey_tools_resize_hole._flood_fill(operator, bm.faces[0])
assert len(flood) == 16, len(flood)

operator.flood_angle = math.radians(10.0)
narrow_flood = MESH_OT_holey_tools_resize_hole._flood_fill(operator, bm.faces[0])
assert len(narrow_flood) == 1, len(narrow_flood)
operator.flood_angle = math.radians(60.0)

operator._fit = MESH_OT_holey_tools_resize_hole._fit_cylinder(operator, list(flood))
assert abs(operator._fit["radius"] - 2.0) < 1.0e-6, operator._fit
assert abs(abs(operator._fit["axis"].z) - 1.0) < 1.0e-6, operator._fit

# Exercise the extracted GPU-preview geometry builder without requiring a window.
operator.diameter = 0.0
operator._fill_batch = None
operator._line_batch = None
operator._tag_redraw = lambda _context: None
MESH_OT_holey_tools_resize_hole._build_cylinder_preview(operator, bpy.context)
assert len(operator._fill_coords) == 128, len(operator._fill_coords)
assert len(operator._line_coords) == 128, len(operator._line_coords)

# Give the upper boundary an oblique cut, fit both end surfaces, and verify that
# radial resizing projects the boundary back along the cylinder axis.
for vert in bm.verts:
    if vert.co.z > 0.0:
        vert.co.z = 2.0 + 0.25 * vert.co.x
operator._end_planes = MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
    operator, list(flood), operator._fit["axis"]
)
assert len(operator._end_planes) == 2, operator._end_planes

operator.scale = 1.0
operator.diameter = 6.0
assert abs(MESH_OT_holey_tools_resize_hole._target_scale(operator) - 1.5) < 1.0e-6

result = MESH_OT_holey_tools_resize_hole.execute(operator, bpy.context)
assert result == {'FINISHED'}, result

bm = bmesh.from_edit_mesh(obj.data)
for vert in bm.verts:
    assert abs(math.hypot(vert.co.x, vert.co.y) - 3.0) < 1.0e-5, vert.co
    if vert.co.z > 0.0:
        assert abs(vert.co.z - (2.0 + 0.25 * vert.co.x)) < 1.0e-5, vert.co
    else:
        assert abs(vert.co.z + 2.0) < 1.0e-5, vert.co

print("HOLEY_TOOLS_HEADLESS_TEST_OK")
