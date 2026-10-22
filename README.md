# Holey Tools

Holey Tools is a Blender 5.2+ extension for editing mesh holes to precise physical dimensions and repairing open boundaries. It contains the interactive **Resize Hole** and **Fair Hole** tools.

## Documentation

- [Version 1.0 release notes](RELEASE_NOTES.md)
- [User guide](src/README.md)
- [Architecture](src/docs/architecture.md)
- [Resize Hole workflow](src/docs/resize-hole-workflow.md)
- [Fair Hole workflow](src/docs/fair-hole-workflow.md)

The uploadable extension is built from `src/`; tests and development scripts remain at the repository root.

## Build

Build with the newest reachable Git tag matching `#.#.#`:

```sh
./build
```

Pass a version to override the tag, for example `./build 1.0.0`. Use `./build --development-controls 1.0.0` to compile in the staged Fair Hole inspection controls; normal builds omit them. On Windows, use `./build.ps1`, optionally with a version and `-DevelopmentControls`. The helpers package a temporary source copy, so version and development overrides do not modify the working tree. Set `BLENDER` to the Blender executable when it is not available on `PATH` or in the standard installation location.

Builds run Blender with factory startup, avoiding enabled user add-ons and their registration/cleanup conflicts without changing saved preferences.

Fair Hole and Resize Hole share the self-contained Manhattan patcher. Auto tries it first for spherical/organic surfaces, falling back to five circular rings if a complete grid cannot be built. No native build is required; the retired QuadWild/Bi-MDF and cross-field backends have been removed. See the [fairing workflow](src/docs/fair-hole-workflow.md).

## Tests

Run the active suite from ordinary Python:

```sh
python3 tests/run.py --jobs 4
```

The runner gives every script its own process, including a factory-startup Blender for each mesh/operator test. Native undo tests remain isolated even when running concurrently. Set `BLENDER` or pass `--blender PATH` when Blender is not on `PATH` or in the standard macOS app location. Use `--list` to inspect the suite, select individual tests with `python3 tests/run.py modes undo_lifecycle`, and add `--verbose` to show successful test output. Failures, native crashes, launch errors, and per-script timeouts all fail the suite. The default timeout is 180 seconds and can be changed with `--timeout`. Parked Extract Axis tests are excluded unless named explicitly or enabled with `--include-experimental`.

Dimension and conic-control tests run without Blender:

```sh
python3 tests/test_resize_dimensions.py
```

Mesh and operator regressions run with Blender, for example:

```sh
blender --background --factory-startup --python-exit-code 1 --python tests/test_modes.py
```

`python3 tests/test_resize_state.py` also runs without Blender and checks scoped cache invalidation, isolation between operators, and preservation of user choices and resource owners. Mesh/operator scripts use the Blender invocation above. Factory startup isolates tests from installed add-ons, and `--python-exit-code 1` makes Python failures fail the command.

`tests/test_undo_lifecycle.py` exercises native undo/redo across conic Scaling/Boolean transitions and verifies that snapshots and rollback preserve the real mesh datablock's identity. Run it in a separate Blender process: the former mesh-swapping implementation crashes inside native undo rather than raising a Python exception.

`tests/test_boolean_transaction.py` checks rollback at each temporary-resource allocation stage, source attributes/material restoration, idempotent cleanup, and explicit live-preview ownership handoff. The undo lifecycle regression also injects a failure after native subtraction to verify full rollback and resource release.

`tests/test_inscribed_diameter.py` checks blind- and through-hole clearance for low-sided polygons, segment-aware Boolean cutters, blue-overlay gauge sizing, direct scaling, and fresh fits after repolygonization.

`tests/test_resize_selection.py` verifies Ctrl/Cmd-click uses the exact selected wall faces without flood filling, preserves geometry on invocation/cancellation, handles invalid selections, and retains the normal hover-selection workflow.

`tests/test_live_drag.py` checks unchanged-value suppression, latest-value drag coalescing, pause/release delivery, timer cleanup, and topology-cache invalidation. It also compares cached scaling with final execution on a dense mesh and reports diagnostic timings without machine-dependent speed assertions.

`tests/test_hover_cache.py` compares cached hover floods and indexed missing-face searches with the full-scan path on cylinders, cones, fragments, and transformed meshes. It checks preview reuse, cache limits, invalidation, hidden faces, and diagnostic dense-mesh timings.

`tests/test_gap_orientation.py` verifies that gap bridging retains same-facing disconnected wall remnants while rejecting geometrically matching reversed faces. It covers cylindrical/conical holes and studs, transforms, and cached/uncached selection.

`tests/test_wall_fitting.py` checks that an isolated rim-face normal and uneven boundary subdivisions do not turn a cylinder into a shallow cone. It also covers genuine shallow/steep cones, holes/studs, missing faces, hidden outliers, and safeguards against discarding major surfaces or deliberate cylinder/chamfer mixtures.

`tests/test_dissolve_completion.py` uses native limited dissolve to close planar openings, verifies a completed N-plane result without duplicate patches/refits, preserves other openings, and checks that deleting the local support surface remains a failure. It also exercises successful native operator reporting/selection.

`tests/test_hidden_geometry.py` checks visible-only picking and fairing probes, shared-boundary vertex behavior, and preservation of hidden topology, hide flags, UVs, and custom face data across native Boolean subtraction and live-preview cancellation.

`tests/test_modal_session.py` exercises registered RNA operators through live Scaling/Boolean previews and Fair Hole selection, then cancels them through extension cleanup from another area. It checks source restoration, mesh identity, timer/object release, and repeated cleanup. `tests/test_grid_patcher.py` also checks probe/replay equivalence at every preparation stage with an attached hidden face.

`tests/test_grid_quality.py` checks scale-consistent sliver improvement, projected 3D cell distortion, fixed source/parity-cut positions, convergence, and disabled smoothing. The Suzanne sweep regression additionally compares distortion and worst edge ratios against the old first-valid layout.

`tests/test_rim_relaxation.py` verifies that the refined rim participates in the final surface-aware solve while exterior vertices, hidden geometry, source UVs, and parity-cut endpoints stay fixed. It checks source-face orientation, reprojection onto the frozen fit, disabled relaxation, and partial/full consistency with C2 both on and off. The interior-only spike regression remains isolated in `tests/test_curvature_overshoot.py`.

`tests/test_organic_surface.py` isolates the fitted surface from topology relaxation. It reproduces the folded Suzanne brow with 37 Circle rings, checks outward and inward error against the intact reference at three scales, and verifies frozen evaluation, unchanged source geometry, and concave/collinear parameter-domain coverage. A negative control reproduces the former height-fit bulge with the same output topology.

`tests/test_fairing_performance.py` compares cached and uncached Suzanne repairs for identical coordinates/topology, verifies that Auto fits and relaxes only once, and counts eliminated duplicate candidate solves. It checks numeric replay after freeing the probe mesh, partial-row reuse, and invalidation for source, transform, protected-rim, smoothing, and analytic projection changes. Timings are reported without hardware-dependent speed assertions.

`tests/test_sphere_fairing.py` reproduces the saved sphere opening's 354-quad repair at 20 and 100 passes. It checks near-identical distortion with multicolor batching, exact ordered-wavefront equivalence on a general fit, local-only boundary tracking, independent/bounded batches, singular stacked-solve fallback, and frozen vectorized sphere/ray intersections. Timings are diagnostic; topology, spherical membership, and geometry-quality bounds are enforced.

`tests/test_grid_sweep.py` verifies perpendicular fallback on a local Suzanne opening, quad/corner transitions across walk seeds, the single required odd-rim parity cut, Auto probe/replay, explicit Grid, and rejection when both sweep directions are invalid. `tests/test_manhattan.py` also checks unequal counts in both directions, three-to-one mutations without boundary cuts, one-cut parity repair for odd rims, and partial/full projection with source faces and UVs preserved.

`tests/test_grid_surface_fit.py` verifies that Grid honors analytic/organic fit choices in the operator and shared reconstruction pipeline, projects sphere interiors exactly in transformed meshes, preserves topology/support, and limits the C2 toggle to Organic.

`tests/test_small_grid_hole.py` uses a small extracted spherical-grid fixture to verify that triangle/n-gon boundaries are not collapsed by ambiguous slides. It checks Auto and explicit Grid from different walk seeds/directions, a 2×1 quad fill without cuts, and preservation of the outer source boundary.

`tests/test_grid_notches.py` verifies existing-vertex U-shaped notch caps, winding/UV preservation, protected boundaries, and bounded local filling. A local fixture from the current larger Suzanne opening checks collapsed-spoke triangle recovery, independent triangle completion, Auto/Grid replay, three quad mutations, and no boundary cuts or changes to other openings.

`tests/test_curvature_overshoot.py` reproduces the Suzanne corner's C2 spike at three scales and verifies bounded continuation, reference-only local/full-opening surface checks, exact rim heights, unchanged source vertices, and matching partial/full projections. It also checks actual brow-quad warpage at 100 relaxation passes/full strength with C2 both on and off; proximity to a reference surface alone did not detect that twisted cell. `tests/test_ideal_surface.py` checks density-independent derivative estimation and rejection of narrow, noise-amplifying stencils without losing exact quadratic curvature, as well as concave-domain guide sampling.

`tests/test_face_curvature.py` checks refined-face tangent constraints, curvature from normal variation, transformed planes, area-centroid invariance under redundant edge splits, hidden-face exclusion, and frozen evaluation. The Suzanne profile regression also requires improved agreement with border-face normals without reintroducing the bulge.

`tests/test_preview_cache.py` uses real Blender projection math with a counted fake GPU backend to verify stationary redraw reuse, camera/viewport/geometry invalidation, clipped-point handling, bounded handle caches, live colors/labels, context isolation, failure recovery, and overlay cleanup. It does not replace native GPU visual testing.

`tests/blender_support.py` shares method binding, optional RNA defaults, exact-topology mesh creation, and canonical redo snapshots across regressions. It does not register classes or populate test-double defaults implicitly. Native RNA/modal/undo integration tests remain separate from method-level tests. `tests/test_blender_support.py` verifies those fixture contracts, and `python3 tests/test_runner.py` checks runner discovery, isolation commands, and failure reporting without requiring Blender.
