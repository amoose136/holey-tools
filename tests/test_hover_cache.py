"""Hover caching preserves flood/gap semantics and reuses preview geometry."""

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bmesh
import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fitting import FittingMixin
from tests.blender_support import bind_operator_methods, edit_mesh


def bind(op):
    bind_operator_methods(op)
    op._set_status_text = op._tag_redraw = lambda *_args: None
    return op


def fixture(cone=False, gap=False, dense=False):
    segments = 32
    vertices, faces = [], []
    # Two nearby but separate holes and a remote coaxial section. The original
    # axial/radial/normal filters must still reject both unrelated sections.
    for offset_x, offset_z in ((0, 0), (8, 0), (0, 20)):
        start = len(vertices)
        vertices.extend((offset_x + (1 + z * .2 if cone else 1) * math.cos(math.tau * i / segments),
                         (1 + z * .2 if cone else 1) * math.sin(math.tau * i / segments), z + offset_z)
                        for z in (0, 1) for i in range(segments))
        faces.extend((start + i, start + (i + 1) % segments,
                      start + segments + (i + 1) % segments, start + segments + i)
                     for i in range(segments) if not gap or i not in (8, 24))
    obj, bm = edit_mesh('Hover test', vertices, faces)
    if dense:
        bmesh.ops.create_grid(bm, x_segments=128, y_segments=128, size=10,
                             matrix=Matrix.Translation((100, 100, 100)))
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    bm.normal_update()
    op = bind(SimpleNamespace(_object=obj, _stage='SELECT', flood_angle=math.radians(25),
                              diameter_estimator='AVERAGE', _hover_face_index=-1, _fill_coords=None))
    return op, bm, segments - (2 if gap else 0)


def uncached(op, seed):
    op._stage = None
    try:
        return set(op._flood_fill(seed))
    finally:
        op._stage = 'SELECT'


context = SimpleNamespace()
for cone in (False, True):
    for gap in (False, True):
        op, bm, count = fixture(cone, gap)
        # Compare the accelerated spatial search with the original full scan
        # for full walls, both missing-face fragments, and transformed meshes.
        for transform in (Matrix.Identity(4), Matrix.Diagonal((2, .7, 1.3, 1)),
                          Matrix.Translation((10000, -20000, 30000))):
            op._object.matrix_world = transform
            for angle in (math.radians(5), math.radians(25), math.radians(60)):
                op.flood_angle = angle
                for index in (0, 15, count, count * 2):
                    expected = uncached(op, bm.faces[index])
                    assert set(op._flood_fill(bm.faces[index])) == expected, (cone, gap, angle, index)
        op._object.matrix_world = Matrix.Identity(4)
        op.flood_angle = math.radians(25)
        op._hover_cache = None
        with patch.object(FittingMixin, '_extend_flood_across_gaps', wraps=FittingMixin._extend_flood_across_gaps) as extend:
            op._set_hover_seed(context, bm.faces[0])
            preview = op._fill_coords
            region = op._hover_region
            assert len(op._candidate_indices) == count
            # Moving to another face of the connected seed component must not
            # rerun the fit/search or retessellate its highlight.
            op._set_hover_seed(context, bm.faces[1])
            assert op._hover_face_index == 1
            assert op._hover_region is region and op._fill_coords is preview
            assert extend.call_count == 1
            normal_count = len(op._hover_cache.normals)
            # Leave for a different hole and return: restore the saved triangles.
            op._set_hover_seed(context, bm.faces[count])
            op._set_hover_seed(context, bm.faces[0])
            assert op._fill_coords is preview and extend.call_count == 2
            assert len(op._hover_cache.normals) >= normal_count
            # Clearing the overlay on a ray miss must not suppress its return.
            op._fill_coords = None
            op._set_hover_seed(context, bm.faces[1])
            assert op._fill_coords is preview and extend.call_count == 2
            if gap:
                # Gap-extension faces are NOT aliased to the first component's
                # fit. A seed on the opposite fragment gets its own fit.
                op._set_hover_seed(context, bm.faces[15])
                assert extend.call_count == 3
                assert op._hover_region is not region

        # Angle changes discard fitted regions but retain expensive mesh data.
        cache = op._hover_cache
        tree = cache.spatial_index
        op.flood_angle = math.radians(5)
        op._rebuild_hover(context)
        assert op._hover_cache is cache and cache.spatial_index is tree
        assert len(op._candidate_indices) == 1
        op.diameter_estimator = 'MINIMUM'
        assert op._get_hover_cache() is cache
        assert not cache.regions and cache.spatial_index is tree
        assert op._hover_face_index == -1
        # Transform/count changes retire all world-space data.
        op._object.matrix_world = Matrix.Translation((0, 3, 0))
        assert op._get_hover_cache() is not cache
        cache = op._hover_cache
        bm.verts.new((0, 0, 10))
        assert op._get_hover_cache() is not cache

        # Visibility is frozen by the blocking selection stage; a fresh session
        # must honor hidden faces, and non-modal fitting retains no hover data.
        op._hover_cache = None
        bm.faces[3].hide = True
        assert bm.faces[3] not in op._flood_fill(bm.faces[0])
        assert set(op._flood_fill(bm.faces[0])) == uncached(op, bm.faces[0])
        op._stage = 'CYLINDER'
        assert op._get_hover_cache() is None

# Many tiny regions cannot retain an unbounded number of preview meshes.
op, bm, count = fixture()
op.flood_angle = math.radians(5)
for index in range(70):
    op._set_hover_seed(context, bm.faces[index])
assert op._hover_cache.region_count <= 32
assert len(op._hover_cache.regions) <= 32

op, bm, count = fixture(gap=True, dense=True)
with patch.object(FittingMixin, '_extend_flood_across_gaps', wraps=FittingMixin._extend_flood_across_gaps) as extend:
    start = time.perf_counter()
    op._set_hover_seed(context, bm.faces[0])
    cold = time.perf_counter() - start
    assert op._hover_cache.spatial_index is not None
    candidates = list(op._hover_cache.nearby_faces((0, 0, .5), 3))
    assert len(candidates) < 100, 'Spatial search included unrelated dense geometry'
    start = time.perf_counter()
    for _ in range(10):
        for index in range(8):
            op._set_hover_seed(context, bm.faces[index])
    cached = time.perf_counter() - start
    assert extend.call_count == 1
start = time.perf_counter()
for _ in range(10):
    uncached(op, bm.faces[0])
baseline = time.perf_counter() - start
print(f'Hover / {len(bm.faces)} faces: cold={cold:.4f}s; 80 cached moves={cached:.4f}s; 10 uncached fills={baseline:.4f}s')
op._hover_cache = op._hover_region = op._hover_preview_region = None
bpy.ops.object.mode_set(mode='OBJECT')
print('HOLEY_TOOLS_HOVER_CACHE_OK')
