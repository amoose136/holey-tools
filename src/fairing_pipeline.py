# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared, transactional fill policy for Fair Hole and Resize Hole."""

import numpy as np

from .fairing import (
    PATCH_RINGS,
    FairingError,
    fair_boundary_loop,
    finalize_fair_patch,
    ordered_boundary_loop,
    prepare_fair_boundary,
    resolved_fill_style,
)
from .grid_patcher import prepare_grid_steps
from .visible_geometry import visible_bmesh_copy, hidden_boundary_vertices


def _probe_grid_patch(obj, bm, loop_edges, grid_options):
    """Return a failure reason, or None, without changing the original mesh."""
    bm.edges.index_update()
    indices = [edge.index for edge in loop_edges]
    trial = None
    try:
        options = dict(grid_options)
        if any(face.hide for face in bm.faces):
            trial, vertices, edges, _faces = visible_bmesh_copy(bm)
            trial_edges = [edges[edge] for edge in loop_edges]
            options['protected_vertices'] = {
                vertices[vert] for vert in hidden_boundary_vertices(bm) if vert in vertices
            }
        else:
            trial = bm.copy()
            trial.edges.ensure_lookup_table()
            trial_edges = [trial.edges[index] for index in indices]
        trial_loop, trial_edges = ordered_boundary_loop(
            edges=trial_edges
        )
        prepare_grid_steps(
            trial, trial_loop, trial_edges, obj.matrix_world, 'ROWS', **options
        )
    except (FairingError, ValueError, RuntimeError, IndexError, KeyError, np.linalg.LinAlgError) as error:
        return str(error)
    finally:
        if trial is not None:
            trial.free()
    return None


def fair_hole_patch(
    obj, bm, loop_verts, loop_edges, hypothesis, fill_style='AUTO', *,
    circle_loops=PATCH_RINGS, final_simplify=True, refit=None,
    grid_step='ROWS', fill_rows=0, match_curvature=True,
    relaxation_iterations=20, relaxation_strength=.5,
):
    """Try the same Manhattan grid everywhere; Auto alone may fall back.

    Probe on a standalone BMesh so failed preparation cannot move, dissolve,
    or reclassify the original rim. Successful probes replay deterministically
    on the live mesh, retaining element identities outside the repaired ring.
    A verified numeric plan reuses fitting/relaxation within this invocation.
    Explicit Grid and development inspection stages never silently fall back.
    """
    topology = resolved_fill_style(hypothesis, fill_style)
    fallback_error = ''
    grid_options = dict(
        hypothesis=hypothesis,
        refinement_passes=1,
        fill_rows=fill_rows,
        match_curvature=match_curvature,
        relaxation_iterations=relaxation_iterations,
        relaxation_strength=relaxation_strength,
    )
    if topology == 'GRID':
        if fill_style == 'AUTO' and grid_step == 'ROWS':
            # This dictionary lives only through the private probe/live replay.
            # Reuse verified numeric plans/fits, never trial BMesh identities.
            grid_options['_row_plan_cache'] = {}
            probe_error = _probe_grid_patch(obj, bm, loop_edges, grid_options)
            if probe_error is not None:
                fallback_error = probe_error
                topology = 'CIRCLE'
        if topology == 'GRID':
            try:
                result = prepare_grid_steps(
                    bm, loop_verts, loop_edges, obj.matrix_world,
                    grid_step, **grid_options,
                )
            except np.linalg.LinAlgError as error:
                raise FairingError("The grid surface fit was singular") from error
            return bm, result

    prepared = prepare_fair_boundary(
        obj, bm, loop_verts, loop_edges, hypothesis, topology
    )
    bm = prepared['bm']
    loop_verts, loop_edges = prepared['loop_verts'], prepared['loop_edges']
    result = prepared.get('result')
    if result is None:
        if prepared['forced_method'] is not None and refit is not None:
            hypothesis = refit(loop_verts, loop_edges, prepared['forced_method'])
        result = fair_boundary_loop(
            bm, loop_verts, loop_edges, obj.matrix_world, hypothesis,
            topology, circle_loops=circle_loops, match_curvature=match_curvature,
        )
    bm, result = finalize_fair_patch(
        obj, bm, result,
        result.get('boundary_verts', loop_verts),
        result.get('boundary_edges', loop_edges),
        final_simplify=final_simplify,
    )
    result['grid_fallback_reason'] = fallback_error
    return bm, result
