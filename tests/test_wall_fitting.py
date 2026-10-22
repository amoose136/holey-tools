"""A small stray rim face must not turn a cylindrical bore into a shallow cone."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace

import bmesh
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fitting import FittingMixin, _fit_wall_normal_plane
from tests.blender_support import bind_operator_methods, edit_mesh


def wall(slope=0.0, stray=False, missing=False, split=False, inward=True):
    count = 32
    points = []
    for row in range(2):
        for index in range(count):
            angle = math.tau * index / count
            height = row * (1 + .25 * math.cos(angle))
            radius = 1 + slope * height
            points.append((radius * math.cos(angle), radius * math.sin(angle), height))
    faces = []
    for index in range(count):
        if missing and index in (7, 8, 16, 24):
            continue
        following = (index + 1) % count
        corners = [index, following, count + following, count + index]
        if split and index != 1:
            # Collinear top-rim subdivisions have smaller radial distances,
            # despite exactly unchanged cylindrical face planes and normals.
            middle = (Vector(points[count + index]) + Vector(points[count + following])) * .5
            points.append(tuple(middle))
            corners.insert(3, len(points) - 1)
        faces.append(tuple(reversed(corners)) if inward else tuple(corners))
    if stray:
        first, second = count + 1, count + 2
        midpoint = (Vector(points[first]) + Vector(points[second])) * .5
        radial = Vector((midpoint.x, midpoint.y, 0)).normalized()
        extra = midpoint - radial * .12 + Vector((0, 0, .3))
        points.append(tuple(extra))
        corners = (second, first, len(points) - 1)
        faces.append(corners if inward else tuple(reversed(corners)))
    obj, bm = edit_mesh('Normal-plane fitting regression', points, faces)
    bm.normal_update()
    obj.matrix_world = Matrix.Translation((4, 6, -2)) @ Matrix.Rotation(.7, 4, 'Y')
    operator = bind_operator_methods(SimpleNamespace(
        _object=obj, diameter_estimator='AVERAGE', flood_angle=math.radians(25),
    ))
    return obj, bm, operator


for missing in (False, True):
    for split in (True, False):
        obj, bm, op = wall(stray=True, missing=missing, split=split)
        fit = FittingMixin._fit_cylinder(op, list(bm.faces))
        assert not fit['is_cone'], fit
        assert fit['fit_outlier_count'] == 1
        assert fit['slope'] == fit['half_angle'] == 0
        expected_axis = obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
        assert abs(fit['axis'].dot(expected_axis)) > .999999, fit
        assert abs(fit['inradius_estimates']['AVERAGE'] - math.cos(math.pi / 32)) < 1e-5
        assert not fit['is_stud']

# Shallow cones must stay cones: solve the contaminated normal plane rather
# than simply raising the cutoff above the user's false one-degree taper.
for half_angle in (.3, 1, 14):
    for sign in (-1, 1):
        for inward in (False, True):
            slope = sign * math.tan(math.radians(half_angle))
            obj, bm, op = wall(slope=slope, stray=True, missing=True, inward=inward)
            fit = FittingMixin._fit_cylinder(op, list(bm.faces))
            assert fit['is_cone'], (half_angle, sign, fit)
            # Unequal clipped generator lengths slightly bias the centroid-
            # based axis-line solve; classification must still keep the taper.
            assert abs(fit['half_angle'] - math.radians(half_angle)) < math.radians(.02), fit
            assert fit['fit_outlier_count'] == 1
            assert fit['is_stud'] == (not inward)

# Clean normals are not trimmed, even with redundant boundary vertices.
obj, bm, op = wall(split=True)
fit = FittingMixin._fit_cylinder(op, list(bm.faces))
assert not fit['is_cone'] and fit['fit_outlier_count'] == 0
obj, bm, op = wall(stray=True, split=True)
bm.faces[-1].hide = True
fit = FittingMixin._fit_cylinder(op, list(bm.faces))
assert not fit['is_cone'] and fit['fit_outlier_count'] == 0

# Require a dominant consensus by BOTH sample count and area. A deliberate
# cylinder/chamfer mixture or a major neighboring surface is not a stray face.
normals = np.asarray([(math.cos(math.tau * i / 32), math.sin(math.tau * i / 32), 0)
                      for i in range(32)] + [(math.cos(.4), 0, math.sin(.4))])
_axis, keep = _fit_wall_normal_plane(normals, np.ones(33))
assert keep.sum() == 32 and not keep[-1]
weights = np.ones(33)
weights[-1] = 100
_axis, keep = _fit_wall_normal_plane(normals, weights)
assert keep.all(), 'Do not discard a major-area support surface'
mixed = np.vstack((normals[:32], normals[:32] * math.cos(.2) + (0, 0, math.sin(.2))))
_axis, keep = _fit_wall_normal_plane(mixed, np.ones(64))
assert keep.all(), 'Do not discard a deliberately selected chamfer ring'

print('HOLEY_TOOLS_WALL_FITTING_OK')
