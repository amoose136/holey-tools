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


resize_properties = MESH_OT_holey_tools_resize_hole.__annotations__
for property_name in (
    "fairing_method",
    "fill_style",
    "circle_loops",
    "grid_match_curvature",
    "final_simplify",
):
    assert property_name in resize_properties, property_name


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

# Confirming immediately after fitting is a true no-op. It restores geometry
# and selection, tears down the modal interaction, and does not create a redo
# execution. Actual dimension changes continue through normal confirmation.
modal_cleanup = []
unchanged_modal = SimpleNamespace(
    _modal_initial_dimensions={
        "diameter": 1.0,
        "depth": 2.0,
        "truncated_depth": 0.75,
        "conic_angle": math.radians(60.0),
    },
    diameter=1.0,
    depth=2.0,
    truncated_depth=0.75,
    conic_angle=math.radians(60.0),
    depth_valid=True,
    angle_valid=True,
    _restore_original_geometry=lambda: modal_cleanup.append("geometry"),
    _restore_initial_selection=lambda: modal_cleanup.append("selection"),
    _finish_modal=lambda _context: modal_cleanup.append("finish"),
)
assert not MESH_OT_holey_tools_resize_hole._modal_dimensions_changed(
    unchanged_modal
)
assert MESH_OT_holey_tools_resize_hole._cancel_unchanged_modal(
    unchanged_modal, modal_context
)
assert modal_cleanup == ["geometry", "selection", "finish"]
unchanged_modal.truncated_depth += 0.001
assert MESH_OT_holey_tools_resize_hole._modal_dimensions_changed(
    unchanged_modal
)
assert not MESH_OT_holey_tools_resize_hole._cancel_unchanged_modal(
    unchanged_modal, modal_context
)
assert modal_cleanup == ["geometry", "selection", "finish"]

# A conic's hidden diameter is not a fourth dimension and must not trigger a
# commit when its visible Depth, Truncated Depth and Angle remain unchanged.
unchanged_modal.truncated_depth = 0.75
unchanged_modal.diameter = 1.0001
assert not MESH_OT_holey_tools_resize_hole._modal_dimensions_changed(unchanged_modal)
unchanged_modal._stage = 'CYLINDER'
unchanged_modal._drag_active = False
unchanged_modal._dimension_gizmo_at_event = lambda _event: None
unchanged_modal._cancel_unchanged_modal = lambda context: MESH_OT_holey_tools_resize_hole._cancel_unchanged_modal(unchanged_modal, context)
for event_type in ('LEFTMOUSE', 'RET'):
    assert MESH_OT_holey_tools_resize_hole.modal(
        unchanged_modal, modal_context,
        SimpleNamespace(type=event_type, value='PRESS'),
    ) == {'CANCELLED'}

angle_context = SimpleNamespace(
    scene=SimpleNamespace(holey_tools_flood_angle=math.radians(60.0))
)
angle_operator = SimpleNamespace(
    flood_angle=math.radians(25.0),
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
        unit_settings=SimpleNamespace(system='METRIC', scale_length=1.0),
        tool_settings=SimpleNamespace(use_snap=False),
    )
)
diameter_operator = SimpleNamespace(
    diameter=0.010,
    depth=0.020,
    truncated_depth=0.012,
    depth_valid=True,
    conic_angle=math.radians(10.0),
    angle_valid=False,
    _active_dimension='DIAMETER',
    _numeric_input="formatted",
    _format_diameter=lambda _context: "formatted",
    _format_depth=lambda _context: "formatted",
    _format_truncated_depth=lambda _context: "formatted",
    _format_conic_angle=lambda: "10°",
    _format_dimension=lambda _context, _value: "formatted",
    _build_cylinder_preview=lambda _context: None,
    _apply_diameter_to_mesh=lambda _context: None,
    _set_status_text=lambda _context: None,
    _tag_redraw=lambda _context: None,
)
diameter_operator._scene_unit_info = lambda context: (
    MESH_OT_holey_tools_resize_hole._scene_unit_info(context)
)
diameter_operator._parse_dimension = lambda context, text: (
    MESH_OT_holey_tools_resize_hole._parse_dimension(
        diameter_operator, context, text
    )
)
diameter_operator._active_dimension_value = lambda: (
    MESH_OT_holey_tools_resize_hole._active_dimension_value(diameter_operator)
)
diameter_operator._set_active_dimension_value = lambda value: (
    MESH_OT_holey_tools_resize_hole._set_active_dimension_value(
        diameter_operator, value
    )
)
diameter_operator._sync_dimension_text = lambda context: (
    MESH_OT_holey_tools_resize_hole._sync_dimension_text(
        diameter_operator, context
    )
)
diameter_operator._step_dimension = lambda context, direction, fine: (
    MESH_OT_holey_tools_resize_hole._step_dimension(
        diameter_operator, context, direction, fine
    )
)
diameter_operator._activate_dimension = lambda context, dimension: (
    MESH_OT_holey_tools_resize_hole._activate_dimension(
        diameter_operator, context, dimension
    )
)
# Snap-enabled selection clicks and one-pixel jitter must not round the fit or
# evaluate any geometry before a deliberate drag begins.
initial_apply = diameter_operator._apply_diameter_to_mesh
diameter_operator._apply_diameter_to_mesh = lambda _context: (_ for _ in ()).throw(AssertionError("Click jitter evaluated geometry"))
diameter_context.scene.tool_settings.use_snap = True
for dimension in ('DIAMETER', 'DEPTH', 'ANGLE'):
    diameter_operator.angle_valid = True
    MESH_OT_holey_tools_resize_hole._begin_dimension_drag(
        diameter_operator, diameter_context,
        SimpleNamespace(mouse_x=10, mouse_y=20), dimension,
    )
    before = diameter_operator._active_dimension_value()
    for x, y in ((10, 20), (11, 21)):
        MESH_OT_holey_tools_resize_hole._update_dimension_drag(
            diameter_operator, diameter_context,
            SimpleNamespace(mouse_x=x, mouse_y=y, shift=False, ctrl=False),
        )
        assert diameter_operator._active_dimension_value() == before
diameter_operator._apply_diameter_to_mesh = initial_apply
diameter_context.scene.tool_settings.use_snap = False
diameter_operator.angle_valid = False
diameter_operator._active_dimension = 'DIAMETER'
assert MESH_OT_holey_tools_resize_hole._handle_dimension_step_event(
    diameter_operator,
    diameter_context,
    SimpleNamespace(type='EQUAL', value='PRESS', shift=False),
)
assert abs(diameter_operator.diameter - 0.011) < 1.0e-9
assert MESH_OT_holey_tools_resize_hole._handle_dimension_step_event(
    diameter_operator,
    diameter_context,
    SimpleNamespace(type='MINUS', value='PRESS', shift=True),
)
assert abs(diameter_operator.diameter - 0.0109) < 1.0e-9
MESH_OT_holey_tools_resize_hole._begin_dimension_drag(
    diameter_operator,
    diameter_context,
    SimpleNamespace(mouse_x=10, mouse_y=20),
    'DEPTH',
)
assert diameter_operator._active_dimension == 'DEPTH'
MESH_OT_holey_tools_resize_hole._update_dimension_drag(
    diameter_operator,
    diameter_context,
    SimpleNamespace(mouse_x=1000, mouse_y=20, shift=False, ctrl=False),
)
assert abs(diameter_operator.depth - 0.020) < 1.0e-9
MESH_OT_holey_tools_resize_hole._update_dimension_drag(
    diameter_operator,
    diameter_context,
    SimpleNamespace(mouse_x=1000, mouse_y=220, shift=False, ctrl=False),
)
assert diameter_operator.depth < 0.020
MESH_OT_holey_tools_resize_hole._update_dimension_drag(
    diameter_operator,
    diameter_context,
    SimpleNamespace(mouse_x=1000, mouse_y=-180, shift=False, ctrl=False),
)
dragged_depth = diameter_operator.depth
assert dragged_depth > 0.020
assert MESH_OT_holey_tools_resize_hole._handle_dimension_step_event(
    diameter_operator,
    diameter_context,
    SimpleNamespace(type='EQUAL', value='PRESS', shift=False),
)
assert abs(diameter_operator.depth - (dragged_depth + 0.001)) < 1.0e-9
diameter_operator._numeric_reference = "41 mm"
diameter_operator._numeric_input = "30 mm"
MESH_OT_holey_tools_resize_hole._update_numeric_preview(
    diameter_operator, diameter_context
)
assert abs(diameter_operator.depth - 0.030) < 1.0e-8
diameter_operator._numeric_input = "0"
MESH_OT_holey_tools_resize_hole._update_numeric_preview(
    diameter_operator, diameter_context
)
assert diameter_operator.depth == 0.0
MESH_OT_holey_tools_resize_hole._step_dimension(
    diameter_operator, diameter_context, -1, False
)
assert diameter_operator.depth == 0.0
diameter_operator._active_dimension = 'DIAMETER'
diameter_operator._numeric_input = "0"
MESH_OT_holey_tools_resize_hole._update_numeric_preview(
    diameter_operator, diameter_context
)
assert diameter_operator.diameter == 0.0

# Cone included angle is an active dimension. It uses degrees for direct input,
# one-degree steps, and a horizontal drag whose rightward direction increases it.
diameter_operator.angle_valid = True
MESH_OT_holey_tools_resize_hole._activate_dimension(
    diameter_operator, diameter_context, 'ANGLE'
)
assert diameter_operator._active_dimension == 'ANGLE'
MESH_OT_holey_tools_resize_hole._step_dimension(
    diameter_operator, diameter_context, 1, False
)
assert abs(diameter_operator.conic_angle - math.radians(11.0)) < 1.0e-9
diameter_operator._numeric_input = "20 deg"
diameter_operator._parse_angle = MESH_OT_holey_tools_resize_hole._parse_angle
MESH_OT_holey_tools_resize_hole._update_numeric_preview(
    diameter_operator, diameter_context
)
assert abs(diameter_operator.conic_angle - math.radians(20.0)) < 1.0e-9
MESH_OT_holey_tools_resize_hole._begin_dimension_drag(
    diameter_operator,
    diameter_context,
    SimpleNamespace(mouse_x=10, mouse_y=20),
    'ANGLE',
)
MESH_OT_holey_tools_resize_hole._update_dimension_drag(
    diameter_operator,
    diameter_context,
    SimpleNamespace(mouse_x=50, mouse_y=20, shift=False, ctrl=False),
)
assert diameter_operator.conic_angle > math.radians(20.0)

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

depth_gizmo_owner = SimpleNamespace(
    _gizmo_rects={
        'DIAMETER': (10.0, 20.0, 30.0, 40.0),
        'DEPTH': (40.0, 10.0, 60.0, 50.0),
        'ANGLE': (70.0, 10.0, 100.0, 50.0),
    },
    _region=SimpleNamespace(x=100, y=200),
)
assert MESH_OT_holey_tools_resize_hole._dimension_gizmo_at_event(
    depth_gizmo_owner,
    SimpleNamespace(mouse_x=150, mouse_y=230),
) == 'DEPTH'
assert MESH_OT_holey_tools_resize_hole._dimension_gizmo_at_event(
    depth_gizmo_owner,
    SimpleNamespace(mouse_x=185, mouse_y=230),
) == 'ANGLE'


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

# A serialized redo resolution is authoritative even if the current topology
# would lead transient state reconstruction to infer a different value.
operator.fit_valid = True
operator.fit_angular_segments = 12
MESH_OT_holey_tools_resize_hole._build_resize_projection_state(
    operator, bm, list(flood)
)
assert operator._resize_angular_segments == 12
operator.fit_valid = False
operator.fit_angular_segments = 0
operator._resize_angular_segments = 0

# The same value survives the complete hidden-RNA store/restore path used by
# Blender's Adjust Last Operation panel.
operator._end_planes = []
operator._resize_requires_boolean = False
operator._resize_angular_segments = 16
MESH_OT_holey_tools_resize_hole._store_fit_for_redo(operator)
assert operator.fit_angular_segments == 16
assert operator.boolean_segments == 16
operator._fit = None
operator._locked_indices = set()
operator._resize_angular_segments = 0
assert MESH_OT_holey_tools_resize_hole._restore_fit_for_redo(operator)
assert operator._resize_angular_segments == 16

# Exercise the extracted GPU-preview geometry builder without requiring a window.
operator.diameter = 0.0
operator._tag_redraw = lambda _context: None
MESH_OT_holey_tools_resize_hole._build_cylinder_preview(operator, bpy.context)
assert len(operator._fill_coords) == 128 * 2, len(operator._fill_coords)
assert len(operator._line_coords) == 128 * 2, len(operator._line_coords)
smooth_lines = operator._line_coords
for source_count, cutter_count, live_count in ((3, 3, 3), (4, 9, 32), (512, 512, 32)):
    operator._resize_angular_segments = source_count
    operator.boolean_segments = cutter_count
    operator._boolean_preview_segments = live_count
    MESH_OT_holey_tools_resize_hole._build_cylinder_preview(operator, bpy.context)
    assert len(operator._fill_coords) == 128 * 2
    assert operator._line_coords == smooth_lines
# Conic geometry uses the same smooth overlay independently of its cutter.
operator._target_radius_at = lambda coordinate: 2.0 + .1 * coordinate
MESH_OT_holey_tools_resize_hole._build_cylinder_preview(operator, bpy.context)
assert len(operator._fill_coords) == len(operator._line_coords) == 128 * 2
del operator._target_radius_at
operator._resize_angular_segments = 16
operator._boolean_preview_segments = 0
operator.boolean_segments = 16

# Reconstruction depth fixes the preview at the fitted exterior-surface basis
# point and moves only the blind end, independently of the wall fit bounds.
operator._depth_fit = {
    "initial": 2.0,
    "outer_coordinate": 1.25,
    "inner_coordinate": -0.75,
    "surface_depth": True,
}
operator.depth = 1.0
operator._target_depth_scale = lambda: (
    MESH_OT_holey_tools_resize_hole._target_depth_scale(operator)
)
MESH_OT_holey_tools_resize_hole._build_cylinder_preview(operator, bpy.context)
preview_coordinates = {
    round((point - operator._fit["center"]).dot(operator._fit["axis"]), 6)
    for triangle in operator._fill_coords
    for point in triangle
}
assert preview_coordinates == {0.25, 1.25}, preview_coordinates
operator._depth_fit = None

# Boolean reconstruction simplifies a blind base only when its fitted depth is
# unchanged. A preserved, non-adjustable blind coordinate pair is unchanged by
# definition; through holes have no base to simplify.
unchanged_depth = SimpleNamespace(
    _depth_fit={"initial": 2.0},
    depth=2.0,
)
changed_depth = SimpleNamespace(
    _depth_fit={"initial": 2.0},
    depth=1.5,
)
assert MESH_OT_holey_tools_resize_hole._boolean_depth_unchanged(
    unchanged_depth, None
)
assert not MESH_OT_holey_tools_resize_hole._boolean_depth_unchanged(
    changed_depth, None
)
assert MESH_OT_holey_tools_resize_hole._boolean_depth_unchanged(
    SimpleNamespace(_depth_fit=None), (0.0, 1.0)
)
assert not MESH_OT_holey_tools_resize_hole._boolean_depth_unchanged(
    SimpleNamespace(_depth_fit=None), None
)

# Give the upper boundary an oblique cut, fit both end surfaces, and verify that
# radial resizing projects the boundary back along the cylinder axis.
for vert in bm.verts:
    if vert.co.z > 0.0:
        vert.co.z = 2.0 + 0.25 * vert.co.x
operator._end_planes = MESH_OT_holey_tools_resize_hole._fit_boundary_planes(
    operator, list(flood), operator._fit["axis"]
)
assert len(operator._end_planes) == 2, operator._end_planes
nonparallel_depth = MESH_OT_holey_tools_resize_hole._fit_adjustable_depth(
    operator, operator._end_planes, operator._fit
)
assert nonparallel_depth is not None
assert not nonparallel_depth["parallel"]

operator._resize_requires_boolean = False
operator._resize_angular_segments = 16
MESH_OT_holey_tools_resize_hole._store_fit_for_redo(operator)
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
