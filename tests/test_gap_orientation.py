"""Gap bridging must preserve hole/stud winding, including on conical walls."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace

from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.blender_support import bind_operator_methods, edit_mesh


def fixture(slope, inward):
    count = 32
    vertices = [((1 + slope * z) * math.cos(math.tau * i / count),
                 (1 + slope * z) * math.sin(math.tau * i / count), z)
                for z in (0, 1) for i in range(count)]
    faces = []
    for i in range(count):
        if i in (8, 24):
            continue  # Two disconnected, compatible wall remnants.
        face = (i, (i + 1) % count, count + (i + 1) % count, count + i)
        faces.append(tuple(reversed(face)) if inward else face)
    source_count = len(faces)
    # An opposite-facing, disconnected face fits the same radius/axial span
    # perfectly. Distance tests and abs(dot) cannot distinguish it from a wall.
    offset = len(vertices)
    vertices.extend(vertices[index] for index in reversed(faces[12]))
    faces.append(tuple(range(offset, offset + 4)))
    obj, bm = edit_mesh('Gap orientation regression', vertices, faces)
    bm.normal_update()
    op = bind_operator_methods(SimpleNamespace(
        _object=obj, flood_angle=math.radians(12), diameter_estimator='AVERAGE',
    ))
    return obj, bm, op, source_count


for slope in (0, -.2, .2):
    for inward in (False, True):
        obj, bm, op, count = fixture(slope, inward)
        correct = set(bm.faces[:count])
        reversed_face = bm.faces[count]
        for transform in (
            Matrix.Identity(4),
            Matrix.Translation((4, 6, -2)) @ Matrix.Rotation(.7, 4, 'Y') @ Matrix.Scale(2, 4),
        ):
            obj.matrix_world = transform
            for stage in (None, 'SELECT'):
                op._stage = stage
                op._hover_cache = None
                for seed in (bm.faces[0], bm.faces[15]):
                    selected = set(op._flood_fill(seed))
                    assert reversed_face not in selected, (slope, inward, stage)
                    assert selected == correct, 'Keep both same-facing remnants across the gaps'
                    fit = op._fit_cylinder(list(selected))
                    assert fit['is_stud'] == (not inward)

print('HOLEY_TOOLS_GAP_ORIENTATION_OK')
