"""Live drag scheduling, native-float deduplication, and scoped wall caching."""

import math
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import GeometryMixin
import src.operator as runtime
from tests.blender_support import bind_operator_methods as bind_methods, edit_mesh


clock = [10.0]
applied, overlays, removed = [], [], []
timer = object()
context = SimpleNamespace(
    scene=bpy.context.scene, window=object(),
    area=SimpleNamespace(type='VIEW_3D', tag_redraw=lambda: None),
    window_manager=SimpleNamespace(
        event_timer_add=lambda *_args, **_kwargs: timer,
        event_timer_remove=removed.append,
    ),
)
context.scene.tool_settings.use_snap = False
op = bind_methods(SimpleNamespace(
    diameter=1.0, depth=1.0, truncated_depth=0.0, conic_angle=0.0,
    _stage='CYLINDER', _active_dimension='DIAMETER',
))
op._activate_dimension = lambda *_args: None
# Match native FloatProperty precision, rather than Python's float precision.
op._set_active_dimension_value = lambda value: setattr(op, 'diameter', struct.unpack('f', struct.pack('f', value))[0])
op._sync_dimension_text = op._set_status_text = lambda *_args: None
op._build_cylinder_preview = lambda *_args: overlays.append(op.diameter)
op._apply_diameter_to_mesh = lambda *_args: applied.append(op.diameter)


def move(x):
    op._update_dimension_drag(context, SimpleNamespace(mouse_x=x, mouse_y=0, ctrl=False, shift=False))


with patch.object(runtime.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(runtime, '_ui_scale', return_value=1.0):
    op._begin_dimension_drag(context, SimpleNamespace(mouse_x=0, mouse_y=0), 'DIAMETER')
    assert runtime._drag_timers and op._drag_preview_timer is timer
    move(10)
    assert len(applied) == len(overlays) == 1
    move(10)
    assert len(applied) == len(overlays) == 1, 'Same native-float value did work'
    for x in range(11, 51):
        move(x)
    assert len(applied) == 1 and op._drag_preview_pending
    latest = op.diameter
    assert latest != applied[-1] and overlays[-1] == latest
    clock[0] += .04
    # Unrelated timers do not flush our queue.
    op.modal(context, SimpleNamespace(type='TIMER', timer=object()))
    assert len(applied) == 1
    op.modal(context, SimpleNamespace(type='TIMER', timer=timer))
    assert applied[-1] == latest and len(applied) == 2
    op.modal(context, SimpleNamespace(type='TIMER', timer=timer))
    assert len(applied) == 2
    # A final release always uses the newest value, even within the interval.
    move(60)
    assert len(applied) == 2
    latest = op.diameter
    op.modal(context, SimpleNamespace(type='LEFTMOUSE', value='RELEASE'))
    assert len(applied) == 3 and applied[-1] == latest
    assert removed == [timer] and not runtime._drag_timers

    # Boolean throttling retains a pending solver update after the pointer
    # stops; our timer must deliver it when the half-second window expires.
    op._drag_active = True
    op._drag_preview_timer = timer
    op._boolean_preview_pending = True
    op._boolean_preview_last_update = clock[0]
    clock[0] += .1
    op.modal(context, SimpleNamespace(type='TIMER', timer=timer))
    assert len(applied) == 3
    clock[0] += .5
    op.modal(context, SimpleNamespace(type='TIMER', timer=timer))
    assert len(applied) == 4
    op._boolean_preview_pending = False
    op._remove_drag_timer(context)

    # Snap plateaus are also unchanged-value no-ops.
    context.scene.tool_settings.use_snap = True
    op.diameter = 0.001
    op._begin_dimension_drag(context, SimpleNamespace(mouse_x=0, mouse_y=0), 'DIAMETER')
    before = len(applied), len(overlays)
    move(4)
    move(5)
    assert (len(applied), len(overlays)) == before
    op._remove_drag_timer(context)
    context.scene.tool_settings.use_snap = False

    # Cancel must retire its timer as well, without applying deferred geometry.
    context.window = SimpleNamespace(cursor_modal_restore=lambda: None)
    context.workspace = SimpleNamespace(status_text_set=lambda _text: None)
    op._draw_handle = None
    op._restore_original_geometry = op._restore_initial_selection = lambda: None
    op._begin_dimension_drag(context, SimpleNamespace(mouse_x=0, mouse_y=0), 'DIAMETER')
    move(20)
    before = len(applied)
    op._drag_preview_pending = True
    assert op.modal(context, SimpleNamespace(type='ESC')) == {'CANCELLED'}
    assert len(applied) == before and not runtime._drag_timers
    assert op._resize_wall_cache is None

# A real edit mesh with a small wall and a large, unrelated region. Hashing
# the unrelated connectivity on every coordinate-only drag was the bottleneck.
segments = 16
vertices = [(math.cos(math.tau * i / segments), math.sin(math.tau * i / segments), z)
            for z in (0, 1) for i in range(segments)]
faces = [(i, (i + 1) % segments, (i + 1) % segments + segments, i + segments) for i in range(segments)]
obj, bm = edit_mesh('Live drag regression', vertices, faces)
mesh = obj.data
bmesh.ops.create_grid(bm, x_segments=128, y_segments=128, size=10)
bm.faces.ensure_lookup_table()
bm.normal_update()
wall = list(bm.faces[:segments])
op = bind_methods(SimpleNamespace(_object=obj, diameter_estimator='AVERAGE', resize_method='SCALE'))
op._fit = op._fit_cylinder(wall)
op._locked_indices = {face.index for face in wall}
op.fit_mesh_topology_signature = GeometryMixin._bmesh_topology_signature(bm)
op._drag_active = True

with patch.object(GeometryMixin, '_bmesh_topology_signature', wraps=GeometryMixin._bmesh_topology_signature) as signature:
    assert op._resolve_resize_wall_faces(bm) == wall
    assert signature.call_count == 1
    for _ in range(20):
        bm.verts[0].co.z += .001
        # Explicit Scaling calls this even without a previous Boolean source.
        op._replace_boolean_preview_with_source(consume=True)
        assert op._resolve_resize_wall_faces(bmesh.from_edit_mesh(mesh)) == wall
    assert signature.call_count == 1, 'Rehashed topology during coordinate-only drag'
    op._drag_active = False
    assert op._resolve_resize_wall_faces(bm) == wall
    assert signature.call_count == 2, 'Redo/release skipped connectivity validation'
    op._drag_active = True
    op._resolve_resize_wall_faces(bm)
    before = signature.call_count
    bm.verts.new((100, 100, 100))
    op._resolve_resize_wall_faces(bm)
    assert signature.call_count == before + 1, 'Topology change did not invalidate cache'
    op._locked_indices = set(range(4))
    assert len(op._resolve_resize_wall_faces(bm)) == 4
    assert signature.call_count == before + 2

# Report diagnostic timings, without a flaky machine-dependent speed assertion.
op._locked_indices = {face.index for face in wall}
op._resize_wall_cache = None
op._resolve_resize_wall_faces(bm)
start = time.perf_counter()
for _ in range(20):
    op._resolve_resize_wall_faces(bm)
cached = time.perf_counter() - start
start = time.perf_counter()
for _ in range(20):
    op._resolve_resize_wall_faces_uncached(bm)
uncached = time.perf_counter() - start
print(f'Wall validation, 20 updates / {len(bm.faces)} faces: uncached={uncached:.4f}s cached={cached:.4f}s')

# Exercise the real coordinate-deformation path, not just the resolver, and
# verify repeated live edits agree with a final source-based application.
bm.verts.ensure_lookup_table()
op._original_vertex_coords = {v.index: v.co.copy() for face in wall for v in face.verts}
op._proportional_weights = {index: 1.0 for index in op._original_vertex_coords}
op._resize_projection_ready = True
op._resize_base_requires_boolean = False
op._resize_surfaces_by_vertex = {}
op._depth_fit = None
op._end_planes = []
op.angle_valid = op.depth_valid = False
op._resize_wall_cache = None
unrelated = bm.verts[-1].co.copy()
times = {}
with patch.object(GeometryMixin, '_bmesh_topology_signature', wraps=GeometryMixin._bmesh_topology_signature) as signature:
    for dragging in (False, True):
        op._drag_active = dragging
        before = signature.call_count
        start = time.perf_counter()
        for i in range(10):
            op.diameter = 2.1 + i * .01
            op._apply_diameter_to_mesh(bpy.context)
        times[dragging] = time.perf_counter() - start
        assert signature.call_count - before == (1 if dragging else 10)
        if dragging:
            live_points = [v.co.copy() for face in wall for v in face.verts]
    op._drag_active = False
    op._apply_diameter_to_mesh(bpy.context, commit_boolean=True)
    final_points = [v.co.copy() for face in wall for v in face.verts]
    assert final_points == live_points
    assert bm.verts[-1].co == unrelated
print(f'Scaling, 10 updates: uncached={times[False]:.4f}s cached={times[True]:.4f}s')
op._resize_wall_cache = None
bpy.ops.object.mode_set(mode='OBJECT')

# Use a native WindowManager timer too, not just the event-test double.
native_op = bind_methods(SimpleNamespace(diameter=1.0, _active_dimension='DIAMETER'))
native_op._activate_dimension = lambda *_args: None
native_op._begin_dimension_drag(bpy.context, SimpleNamespace(mouse_x=0, mouse_y=0), 'DIAMETER')
assert isinstance(native_op._drag_preview_timer, bpy.types.Timer)
native_op._remove_drag_timer(bpy.context)
assert not runtime._drag_timers

# Reload/unregister cleanup also retires timers independently of operator RNA.
runtime._drag_timers.append((context.window_manager, timer))
runtime.cleanup_runtime()
assert not runtime._drag_timers
print('HOLEY_TOOLS_LIVE_DRAG_OK')
