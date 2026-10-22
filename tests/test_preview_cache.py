"""Headless drawing-cache regressions with a counted fake GPU backend.

Real Blender projection math is used; no graphics context is required and this
does not claim to visually validate a platform's native GPU backend.
"""

import sys
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.gpu_preview as preview
import src.preview_cache as cache_module
from src.fair_hole import MESH_OT_holey_tools_fair_hole as Fair
from src.extract_axis import MESH_OT_holey_tools_extract_axis as Extract
from src.operator import MESH_OT_holey_tools_resize_hole as Resize, _PersistentPreview
from src.preview_cache import PreviewDrawCache, clear_draw_cache, draw_cache


class Region:
    width, height, pointer = 800, 600, 1
    x = y = 0

    def as_pointer(self):
        return self.pointer


class Shader:
    def __init__(self, name):
        self.name = name
        self.bind = Mock()
        self.uniform_float = Mock()


def batch_factory(shader, primitive, attributes):
    return SimpleNamespace(shader=shader, primitive=primitive,
                           coords=list(attributes['pos']), draw=Mock())


shader_factory = Mock(side_effect=Shader)
batch_builder = Mock(side_effect=batch_factory)
backend = SimpleNamespace(
    shader=SimpleNamespace(from_builtin=shader_factory),
    state=SimpleNamespace(**{name: Mock() for name in (
        'blend_set', 'depth_test_set', 'depth_mask_set', 'face_culling_set')}),
    matrix=SimpleNamespace(push_pop=nullcontext, push_pop_projection=nullcontext,
                           load_identity=Mock(), load_projection_matrix=Mock()),
)
region = Region()
region_data = SimpleNamespace(perspective_matrix=Matrix.Identity(4))
context = SimpleNamespace(region=region, region_data=region_data, window=Region())
points = [Vector(co) for co in ((-.5, -.5, 0), (.5, -.5, 0), (.5, .5, 0), (-.5, .5, 0))]
triangles = [(points[0], points[1], points[2]), (points[0], points[2], points[3])]
lines = [(points[i], points[(i + 1) % 4]) for i in range(4)]
owner = SimpleNamespace(_fill_coords=triangles, _line_coords=lines, _stage='CYLINDER',
                        _region=region, _diameter_text='1 m', _draw_dimension_label=Mock())
real_project = cache_module.view3d_utils.location_3d_to_region_2d

with ExitStack() as stack:
    stack.enter_context(patch.object(cache_module, 'gpu', backend))
    stack.enter_context(patch.object(preview, 'gpu', backend))
    stack.enter_context(patch.object(cache_module, 'batch_for_shader', batch_builder))
    stack.enter_context(patch.object(cache_module, 'bpy', SimpleNamespace(context=context)))
    stack.enter_context(patch.object(preview, 'bpy', SimpleNamespace(context=context)))
    stack.enter_context(patch.object(preview, '_viewport_size', return_value=(1600, 1200)))
    stack.enter_context(patch.object(preview, '_line_width', return_value=4.0))
    colors = stack.enter_context(patch.object(preview, '_face_retopology_colors',
        return_value=((.1, .2, .3, .4), (.1, .2, .3, .9))))
    project = stack.enter_context(patch.object(cache_module.view3d_utils,
        'location_3d_to_region_2d', wraps=real_project))

    preview.GpuPreviewMixin._draw_preview_impl(owner)
    cache = draw_cache(owner)
    assert project.call_count == 4, 'Repeated triangle/line corners should project once'
    assert shader_factory.call_count == 2 and batch_builder.call_count == 2
    expected = [tuple(real_project(region, region_data, point)) + (0.0,)
                for triangle in triangles for point in triangle]
    projected = cache.geometry(triangles, lines, 'CYLINDER', region, region_data)
    assert projected.label_points == expected
    assert len(projected.line_batch.coords) == 16, 'Keep caps plus camera-dependent silhouette'

    # Label/theme changes stay live, while stationary redraws allocate nothing.
    colors.return_value = ((.6, .5, .4, .3), (.6, .5, .4, .9))
    owner._diameter_text = '2 m'
    for _ in range(100):
        preview.GpuPreviewMixin._draw_preview_impl(owner)
    assert project.call_count == 4 and batch_builder.call_count == 2
    assert shader_factory.call_count == 2
    assert owner._draw_dimension_label.call_count == 101
    assert cache.shader('UNIFORM_COLOR').uniform_float.call_args.args == ('color', (.6, .5, .4, .3))
    assert ('viewportSize', (1600, 1200)) in [call.args for call in cache.shader('POLYLINE_UNIFORM_COLOR').uniform_float.call_args_list]

    # Every geometry/view input invalidates, without recompiling the shaders.
    before = batch_builder.call_count
    owner._fill_coords = list(triangles)
    preview.GpuPreviewMixin._draw_preview_impl(owner)
    assert batch_builder.call_count == before + 2
    owner._line_coords = list(lines)
    preview.GpuPreviewMixin._draw_preview_impl(owner)
    assert batch_builder.call_count == before + 4
    region_data.perspective_matrix[0][3] = .25
    preview.GpuPreviewMixin._draw_preview_impl(owner)
    assert batch_builder.call_count == before + 6
    region.width += 20
    preview.GpuPreviewMixin._draw_preview_impl(owner)
    assert batch_builder.call_count == before + 8
    region.height += 10
    preview.GpuPreviewMixin._draw_preview_impl(owner)
    assert batch_builder.call_count == before + 10
    selected = cache.geometry(owner._fill_coords, owner._line_coords, 'SELECT', region, region_data)
    assert len(selected.line_batch.coords) == 8, 'Selection mode has no cylinder silhouette'
    assert batch_builder.call_count == before + 12
    assert shader_factory.call_count == 2

    # Split view handlers must not draw or build in an unrelated region.
    context.region = Region()
    context.region.pointer = 2
    before = batch_builder.call_count
    preview.GpuPreviewMixin._draw_preview_impl(owner)
    assert batch_builder.call_count == before
    context.region = region

    # Changing window context discards all old GPU resources on that owner.
    context.window.pointer += 10
    assert draw_cache(owner) is not cache
    preview.GpuPreviewMixin._draw_preview_impl(owner)
    assert shader_factory.call_count == 4
    assert batch_builder.call_count == before + 2

    # HUD roles keep one batch each; moving a handle doesn't accumulate history.
    hud = PreviewDrawCache()
    before = batch_builder.call_count
    for _ in range(2):
        preview.GpuPreviewMixin._draw_drag_gizmo(region, 100, 100, 68, (1, 1, 1, 1), cache=hud)
        preview.GpuPreviewMixin._draw_depth_drag_gizmo(region, 100, 80, 120, (1, 1, 1, 1), cache=hud)
        preview.GpuPreviewMixin._draw_flood_angle_wedge(region, (80, 80), 38, .5, (1, 1, 1, 1), cache=hud)
    assert batch_builder.call_count == before + 5
    for x in range(100):
        preview.GpuPreviewMixin._draw_drag_gizmo(region, x, 100, 68, (1, 1, 1, 1), cache=hud)
    assert len(hud._hud_batches) == 5 and len(hud._shaders) == 2

    # Real label layout still publishes three distinct, hittable conic handles.
    font = SimpleNamespace(SHADOW=1, size=Mock(), enable=Mock(), disable=Mock(),
                           shadow=Mock(), shadow_offset=Mock(), color=Mock(),
                           position=Mock(), draw=Mock(),
                           dimensions=lambda _font, text: (len(text) * 8, 16))
    label_owner = SimpleNamespace(_region=region, depth_valid=True, angle_valid=True,
        _diameter_text='2 m', _depth_text='1 m', _truncated_depth_text='.5 m',
        _conic_angle_text='90°', _active_dimension='DEPTH')
    with patch.object(preview, 'blf', font):
        before = batch_builder.call_count
        for _ in range(2):
            preview.GpuPreviewMixin._draw_dimension_label(
                label_owner, region, expected, (1, 1, 1, 1))
        assert batch_builder.call_count == before + 3
        labels = [call.args[1] for call in font.draw.call_args_list]
        assert 'Depth 1 m' in labels and 'Truncated Depth .5 m' in labels
        assert 'Included Angle 90°' in labels
        for dimension, (left, bottom, right, top) in label_owner._gizmo_rects.items():
            event = SimpleNamespace(mouse_x=(left + right) / 2, mouse_y=(bottom + top) / 2)
            assert preview.GpuPreviewMixin._dimension_gizmo_at_event(label_owner, event) == dimension
        label_owner._depth_text = '1.5 m'
        preview.GpuPreviewMixin._draw_dimension_label(label_owner, region, expected, (1, 1, 1, 1))
        assert 'Depth 1.5 m' in [call.args[1] for call in font.draw.call_args_list]

    # Cache failures never publish half-built batches or suppress the retry.
    retry = PreviewDrawCache()
    batch_builder.side_effect = RuntimeError('transient batch failure')
    try:
        retry.geometry(triangles, lines, 'CYLINDER', region, region_data)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Expected batch failure')
    assert retry._projected is None
    batch_builder.side_effect = batch_factory
    assert retry.geometry(triangles, lines, 'CYLINDER', region, region_data).fill_batch is not None

    # Behind-camera points omit complete elements; cached None still deduplicates.
    clipped = Vector((10, 0, 0))
    def clip_point(region, data, point):
        return None if point == clipped else real_project(region, data, point)
    project.side_effect = clip_point
    project.reset_mock()
    clipping = PreviewDrawCache()
    clipped_triangles = [triangles[0], (points[0], clipped, points[3])]
    clipped_lines = [(points[1], clipped), lines[0]]
    result = clipping.geometry(clipped_triangles, clipped_lines, 'SELECT', region, region_data)
    assert len(result.label_points) == 3 and len(result.line_batch.coords) == 2
    assert project.call_count == 5
    assert clipping.geometry(clipped_triangles, clipped_lines, 'SELECT', region, region_data) is result
    assert project.call_count == 5
    invisible = [(clipped, clipped, clipped)]
    empty = clipping.geometry(invisible, [], 'CYLINDER', region, region_data)
    assert empty.fill_batch is empty.line_batch is None and not empty.label_points
    project.side_effect = None

    # A draw failure still restores the shared GPU state in finally.
    failing = draw_cache(owner).geometry(owner._fill_coords, owner._line_coords,
                                        owner._stage, region, region_data)
    failing.fill_batch.draw.side_effect = RuntimeError('draw failure')
    try:
        preview.GpuPreviewMixin._draw_preview_impl(owner)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Expected draw failure')
    assert backend.state.blend_set.call_args.args == ('NONE',)
    assert backend.state.depth_mask_set.call_args.args == (True,)

    # Native resources have independent modal, fairing, and persistent owners.
    other = SimpleNamespace()
    assert draw_cache(other) is not draw_cache(owner)
    owner._tag_redraw = lambda _context: None
    Resize._clear_batches(owner, context)
    assert owner._gpu_preview_cache is None and owner._fill_coords is None
    other._tag_redraw = owner._tag_redraw
    Fair._clear_preview(other, context)
    assert other._gpu_preview_cache is None
    draw_cache(other)
    Extract._clear_batches(other, context)
    assert other._gpu_preview_cache is None
    persistent = object.__new__(_PersistentPreview)
    persistent._draw_handle = None
    draw_cache(persistent)
    persistent.remove()
    assert persistent._gpu_preview_cache is None
    clear_draw_cache(owner)

print('HOLEY_TOOLS_PREVIEW_CACHE_OK: 100 stationary redraws, 0 extra projections/batches')
