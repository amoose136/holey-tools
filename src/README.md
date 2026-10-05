# Holey Tools

Holey Tools provides precision mesh-editing tools for holes. Version 0.9 contains **Resize Hole**, **Fair Hole**, and **Extract Axis**.

## Requirements

- Blender 5.2 or newer
- A mesh object in Edit Mode.

## Resize Hole

1. Open **3D Viewport > Sidebar > Holey Tools**.

2. Choose a diameter estimate:
    - **Minimum** uses the narrowest selected radius and is useful for chamfered holes.
    - **Average** uses the mean selected radius.
    - **Maximum** uses the widest selected radius.

3. Click **Resize Hole** and hover a cylindrical or conical wall.

4. Adjust the flood-fill angle with `[` and `]`, then click the wall.

5. Type a diameter or drag the horizontal on-screen handle. Blind holes with either planar ends or a recoverable reconstruction surface expose the vertical depth handle: drag down for deeper or up for shallower. Detected cones also expose a horizontal **Angle** handle for their half-angle. Typing and +/- edit the most recently touched handle, defaulting to diameter. Hold Shift for fine adjustment.

    Ctrl temporarily inverts Blender's snapping toggle; snapped changes use 0.1 mm increments.

    Viewport navigation remains available from the middle mouse button, trackpad, mouse wheel, or 3D mouse while the tool is active.

6. Click away from the handle or press Enter to finish. Press Esc to cancel.

The tool respects Blender's proportional-editing toggle, radius, falloff, connected mode, projected mode, and global snapping toggle. The final diameter remains editable from Blender's Adjust Last Operation panel, where the minimum, average, or maximum diameter estimate can also be changed. Boolean reconstructions additionally expose cutter segments, the surface hypothesis, Grid/Circle fill, circle loops, grid angle, and final simplification. Cutter segments initially match the detected source resolution and can be overridden from 3 to 512. For direct planar resizing, **Depth** is the axis distance between the two end-plane intersections. For Boolean reconstruction, its fixed basis is the hole-axis intersection with the accepted exterior reconstruction surface and its endpoint is the blind-cap intersection. A detected cone also exposes **Conic Angle**, expressed as the half-angle between its wall and axis.

Direct rim deformation is used only when every exterior end surface resolves to one plane. In that case enlargement projects the resized rim along the hole axis onto a frozen base-mesh snapshot, while each interior blind-cap rim is projected against its own fit. Shrinkage likewise intersects each rim's fitted plane mathematically. During enlargement, target-radius probes recheck that the frozen surface is still coplanar with the original collar. Crossing from that collar onto a curved or sloped cap switches the operation to Exact Boolean reconstruction after a tolerant multi-probe check filters isolated BVH seams and small mesh noise.

Any exterior surface classified as two or three planes, spherical, conic/cylindrical, or organic switches to reconstruction. Live dragging leaves the source mesh in a pristine snapshot and shows a reversible 16-sided Exact Boolean result. Drag-time Boolean rebuilds run at most once every 0.5 seconds; the GPU cutter remains continuous and release applies the latest pending value. On confirmation the old wall and cavity geometry are removed at the source mesh's angular resolution, exterior openings are faired back to their inferred surfaces, and the requested circular-cone cutter is subtracted with Blender's Exact Boolean solver. The operation keeps a mesh backup and restores it if reconstruction or the Boolean fails. The cutter begins with explicit axial clearance outside the reconstructed mesh so its cap cannot remain coplanar with the exterior support.

## Current limitations

- Resize Hole operates on one connected hole wall at a time.
- The selected wall must contain enough geometry to define a stable circular-cone axis and radial regression.
- Minimum and maximum estimates are sensitive to genuinely irregular wall vertices.
- Depth adjustment requires either two usable planar end arcs or distinct exterior and blind-cap surfaces that both intersect the fitted axis.
- Cylindrical through-holes expose diameter only; conical through-holes retain top-to-bottom depth and angle controls.
- Counterbore holes are not (yet) supported.
- Boolean reconstruction requires at least one recoverable closed exterior rim; branched or irrecoverably fragmented non-planar boundaries are rejected rather than deformed directly.

## Fair Hole

1. Open **3D Viewport > Sidebar > Holey Tools** and click **Fair Hole**.
2. Hover any edge of an open boundary. The complete connected non-manifold edge loop is highlighted by connectivity alone; no face-angle threshold is used.
3. Click the highlighted loop or press Enter. Fair Hole limited-dissolves the ring, creates the patch, merges it by distance, expands the patch faces by one ring, then limited-dissolves the expanded region. Press Esc to cancel without changing the mesh.

The Adjust Last Operation panel separates **Surface** from **Fill**. Surface defaults to **Auto**, or can be forced to N-Plane, Cylinder, Sphere, or Organic when the automatic residual comparison chooses the wrong model. Fill defaults to the method-specific topology, or can be forced to a source-aligned **Grid** or concentric **Circle** rings. Forced analytic surfaces still require a stable fit, but are not rejected merely because their residual loses to another model. Disable **Final Simplify** to preserve the generated patch topology. Required seam welding still runs, but the final selection growth and Limited Dissolve do not. When Circle fill is active, **Circle Loops** controls the number of concentric quad rings before the center fan. When Grid fill is active, **Grid Angle Offset** rotates the grid relative to its automatically inferred source alignment.

Fair Hole tries increasingly general surface hypotheses. It first groups the faces touching the rim into at most three planar runs and caps their fitted intersection. If that fails, it evaluates the shared circular cone/cylinder fit and sphere fit together and uses the lower residual. Fits whose scale-relative residual exceeds 3% are rejected. A cylinder first runs boundary-inclusive Limited Dissolve, scales the rim to zero along the fitted axis, and merges at 0.0001. It then grows the affected face selection once and runs Limited Dissolve again. Sphere fits also run the boundary-inclusive pre-dissolve and refit before building the patch; cones retain surface-projected rings. The final fallback snapshots the non-boundary spoke edges incident to the rim. It preserves every spoke-connected boundary vertex and dissolves only unsupported rim vertices whose adjacent surface normals differ by at most 10°; exterior source-grid edges are never included in the dissolve. The grid is built only after that pass. A four-way folded SVD of the saved spokes determines grid orientation even when its two orthogonal direction families are equally populated; their boundary anchors also align the grid-line positions to the source mesh. Interval count follows the simplified boundary vertex count and aspect ratio, and constrained triangulation clips the grid to the opening. A thin-plate height field fitted to the rim and exterior one-ring then continues the surrounding slope smoothly, moving points only in the fitted-axis direction. Grid vertices are projected directly onto forced spherical or cylindrical surfaces; circular topology uses the same selected surface model.

Current Fair Hole limitations:

- The boundary must be one simple closed component of edges with exactly one linked face; branched, open, and already-filled selections are rejected.
- Analytic hypotheses use only faces directly touching the boundary. Very sparse or noisy rims may therefore fall through to organic fairing.
- The organic fallback is a thin-plate curvature-minimizing patch, not a continuity-constrained CAD surface or a topology remesher.

## Extract Axis

1. Open **3D Viewport > Sidebar > Holey Tools** and find **Geometry Nodes**.
2. Choose the face-group attribute name, then click **Extract Axis**.
3. Hover the desired wall or surface region. The tint, `[` / `]` flood-angle controls, direct angle entry, and viewport navigation work like Resize Hole.
4. Click the region or press Enter to store it.

The tool replaces a face-domain integer attribute (by default `holey_axis_region`) with the flooded selection and adds an **Extract Axis** Geometry Nodes modifier. Its reusable **Holey Tools Extract Axis** group uses the bundled **Fit Circular Cone** asset, so the fit remains useful for conical surfaces, missing faces, and irregular topology. Its outputs are:

- **Geometry**: input geometry joined with the aligned axis indicator.
- **Indicator**: the aligned indicator geometry by itself.
- **Axis** and **Center**: a normalized direction and point describing the fitted axis. Its sign is arbitrary, so either direction represents the same line.
- **Radius**, **Slope**, and **Half Angle**: the fitted circular-cone parameters.
- **Valid**: true when the named attribute produces a usable fit.

The **Face Group** string input defaults to the most recently extracted attribute name and can be changed per modifier. **Indicator Size** uniformly scales the marker. Leave **Indicator Object** blank to use the generated mesh arrow, or choose a mesh object to place its origin at `Center` and align its local +Z direction with `Axis`.

## Source layout

- `constants.py` contains version metadata, fallback colors, and shared UI options.
- `geometry.py` contains circular-cone fitting, gap-aware flood fill, proportional weights, and mesh deformation.
- `gpu_preview.py` contains screen-space tinting, cone/cylinder drawing, labels, and drag gizmos.
- `operator.py` contains modal interaction, numeric input, redo state, and runtime cleanup.
- `extract_axis.py` contains the face-group extraction modal operator.
- `fair_hole.py` contains connected-boundary picking and the Fair Hole operator.
- `fairing.py` contains boundary validation, analytic hypotheses, patch topology, and the organic thin-plate solve with a Laplacian fallback.
- `axis_nodes.py` loads the bundled fitting assets and builds the Extract Axis modifier.
- `assets/holeyToolsAssets.blend` contains reusable Geometry Nodes math and fitting assets.
- `ui.py` contains the sidebar panel and registered scene setting.
- `__init__.py` is Blender's extension entry point and registers the operator and UI.

More detail is available in the developer documentation:

- [Architecture](docs/architecture.md)
- [Resize Hole workflow](docs/resize-hole-workflow.md)
- [Fair Hole workflow](docs/fair-hole-workflow.md)
- [Extract Axis workflow](docs/extract-axis-workflow.md)

## License

Copyright 2026 Amos Manneschmidt. Holey Tools is licensed under the GNU General Public License, version 3 or later. See `LICENSE`.
