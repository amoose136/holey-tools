import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gpu_preview import (  # noqa: E402
    DIAMETER_SYMBOL,
    _convex_hull_lines,
)
from src.operator import MESH_OT_holey_tools_resize_hole  # noqa: E402


assert ord(DIAMETER_SYMBOL) == 0x00D8, repr(DIAMETER_SYMBOL)


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
