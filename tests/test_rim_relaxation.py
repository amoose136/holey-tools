"""Relax the refined rim, not the surrounding mesh or the captured fit."""

import json
from pathlib import Path
import sys

import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fairing import _organic_grid_frame
from src.grid_patcher import prepare_grid_steps
from src.ideal_surface import capture_organic_surface
from src.row_patcher import fill_refined_rows
from tests.blender_support import edit_mesh

data = json.loads(Path(__file__).with_name('fixtures').joinpath('suzanne_triangle_grid_hole.json').read_text())


def area_vector(face, matrix):
    p = np.asarray([matrix @ v.co for v in face.verts])
    return np.cross(p - p[0], np.roll(p, -1, axis=0) - p[0]).sum(axis=0)


for curvature in (False, True):
    outputs = []
    for rows, iterations, strength in ((0, 20, .5), (2, 20, .5), (0, 0, .5), (0, 20, 0.)):
        obj, bm = edit_mesh('Refined rim relaxation', data['vertices'], data['faces'])
        obj.matrix_world = Matrix(data['matrix'])
        loop = [bm.verts[i] for i in data['boundary']]
        edges = [bm.edges.get((v, loop[(i+1) % len(loop)])) for i, v in enumerate(loop)]
        prepared = prepare_grid_steps(bm, loop, edges, obj.matrix_world, 'REFINE', refinement_passes=1)
        loop, edges = prepared['boundary_verts'], prepared['boundary_edges']
        # A hidden face incident to one rim point must not move indirectly
        # through relaxation. Its remote geometry must not influence fitting.
        pinned = loop[0]
        hidden = bm.faces.new((pinned, bm.verts.new(pinned.co + Vector((3., 0., 0.))),
                               bm.verts.new(pinned.co + Vector((0., 0., 3.)))))
        hidden.hide = True
        bm.normal_update()
        center, axis, x, y, _points, _values = _organic_grid_frame(loop, edges, obj.matrix_world)
        surface = capture_organic_surface(loop, obj.matrix_world, center, x, y, axis,
                                          match_curvature=curvature)
        assert surface is not None
        original = {v: v.co.copy() for v in bm.verts}
        face_normals = {f: area_vector(f, obj.matrix_world) for f in bm.faces}
        uv = bm.loops.layers.uv.new('Preserved UV')
        uv_values = {}
        for face in bm.faces:
            for corner in face.loops:
                corner[uv].uv = corner.vert.co.xy
                uv_values[corner] = corner[uv].uv.copy()
        result = fill_refined_rows(bm, loop, edges, obj.matrix_world, fill_rows=rows,
                                   relaxation_iterations=iterations, relaxation_strength=strength,
                                   match_curvature=curvature)
        assert all(v.co == co for v, co in original.items() if v not in loop)
        assert pinned.co == original[pinned] and hidden.hide
        assert all((corner[uv].uv - value).length == 0. for corner, value in uv_values.items())
        for face, normal in face_normals.items():
            assert np.dot(area_vector(face, obj.matrix_world), normal) > .09 * np.dot(normal, normal)
        moved = [v for v in loop if v.co != original[v]]
        assert len(moved) == result['relaxed_rim_vertices']
        if iterations and strength:
            assert moved
            projected = moved + result['verts']
            world = np.asarray([obj.matrix_world @ v.co for v in projected])
            assert np.max(np.linalg.norm(surface.project(world) - world, axis=1)) < 2e-6
            outputs.append(([v.co.copy() for v in loop], [v.co.copy() for v in result['verts']]))
        else:
            assert not moved and all(v.co == co for v, co in original.items())
    assert outputs[0][0] == outputs[1][0]
    assert all(min((p - q).length for q in outputs[0][1]) < 1e-6 for p in outputs[1][1])

print('HOLEY_TOOLS_RIM_RELAXATION_OK')
