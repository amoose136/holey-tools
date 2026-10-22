# Holey Tools 🧀

Holey Tools provides precision mesh-editing tools for holes with **Resize Hole** and **Fair Hole**.

## Resize Hole

1. Open **3D Viewport > Sidebar > Holey Tools**.

2. Choose a diameter estimate:
    - **Minimum** uses the narrowest selected radius and is useful for chamfered holes.
    - **Average** uses the mean selected radius.
    - **Maximum** uses the widest selected radius.

    The **Mode** dropdown below Diameter Estimate preselects **Auto**, **Scaling**, or **Boolean** for the next resize, including Ctrl/Cmd-click selection-based resizing. Auto chooses from the fitted geometry; Scaling and Boolean force that method. You can still change the method in Adjust Last Operation.

3. Click **Resize Hole** and hover a cylindrical or conical wall.

    Alternatively, Ctrl-click (Windows/Linux) or Cmd-click (macOS) **Resize Hole** to fit the currently selected visible wall faces directly and skip step 4. This uses the exact face selection without flood filling; select only the cylindrical or conical wall region you want to resize.

4. Adjust the flood-fill angle with `[` and `]`, then click the wall.

5. Cylinders expose diameter and, when recoverable, physical hole depth. Detected conic regions instead expose theoretical apex **Depth**, physical **Truncated Depth**, and the full included **Conic Angle** at the tip. The apex may lie beyond the modeled frustum, and the truncation circle may open into a continuing cylindrical bore as in a countersink. Changing Conic Angle pivots the cone around that bottom/truncation circle, preserving both its axial position and radius while updating the theoretical apex depth. Changing either Depth or Truncated Depth moves the other by the same distance, preserving the bottom-circle diameter while moving its axial position. Type a value or drag its on-screen handle; downward depth drags make the corresponding depth greater. Typing and +/- edit the most recently touched handle. Hold Shift for fine adjustment.

    Setting Conic Angle to **0°** removes the countersink taper and keeps the connected through-hole at its original diameter. It does not close the hole; setting Depth to zero is the separate close/fair operation.

    Ctrl temporarily inverts Blender's snapping toggle; snapped changes use 0.1 mm increments.

    Viewport navigation remains available from the middle mouse button, trackpad, mouse wheel, or 3D mouse while the tool is active.

6. After changing a dimension, click away from the handle or press Enter to finish. Clicking away or pressing Enter without changing any value is a no-op that restores the original selection and geometry. Press Esc to cancel.

Use **Adjust Last Operation** to change dimensions or switch between **Auto**, **Scaling**, and **Boolean**. Boolean mode also exposes cutter resolution and fairing options. Scaling's **Final Simplify** is off by default.

- Set Diameter or an available Depth to **0** to close and fair the hole.
- Cylinder diameter measures the usable inside circle shown by the blue overlay; changing Boolean segments preserves that clearance.
- Proportional editing and snapping are respected.

See the [Resize Hole workflow](docs/resize-hole-workflow.md) for technical details.

## Current limitations

Hidden faces are excluded from picking, fitting, surface projection, fairing probes, and Boolean operands. They remain preserved through confirmation, cancellation, and undo; visible vertices shared with hidden faces can still move with the edit, as in Blender's normal transforms. Full-mesh backups and visibility bookkeeping remain necessary to preserve that geometry.

- Resize Hole operates on one connected hole wall at a time.
- The selected wall must contain enough geometry to define a stable circular-cone axis and radial regression.
- Minimum and maximum estimates are sensitive to genuinely irregular wall vertices. IE they must be on smoothly varying surface without discontinuities.
- Depth adjustment requires either two usable planar end arcs or distinct exterior and blind-cap surfaces that both intersect the fitted axis.
- Cylindrical through-holes expose diameter only; conical regions with two recoverable end circles expose theoretical apex depth, physical truncation depth, and included tip angle.
- Counterbore holes are not (yet) supported as a single operation but one can handle them in two steps.
- Boolean reconstruction requires at least one recoverable closed exterior rim; branched or irrecoverably fragmented non-planar boundaries are rejected rather than deformed directly.

## Fair Hole

Fill an open hole by continuing the surrounding surface. It tries to fit an ideal surface using the faces surrounding a non-manifold edge and will attempt to grid fill the region and stretch it to the intersection of up to 3 planes, a cylinder side wall, a sphere, or an organic surface.

1. In **Edit Mode**, open **3D Viewport > Sidebar > Holey Tools** and click **Fair Hole**.
2. Hover an edge of the hole to highlight its boundary loop.
3. Click or press Enter to fill it. Press Esc to cancel.

Start with **Auto Fit** and **Auto Topology**, the defaults beside the button. For spherical and organic surfaces, Auto tries a quad grid first and falls back to circular rings if needed.

Use **Adjust Last Operation** to override the surface fit or choose **Grid** or **Circle** topology. Circle exposes a ring count; Organic Grid offers **Match Curvature (C2)** to blend with the surrounding shape.

The boundary must be one simple closed loop. If a forced Grid fill fails, try Auto Topology or Circle instead.

See the [Fair Hole workflow](docs/fair-hole-workflow.md) for algorithm details and development-only inspection controls.

## Source layout

- `constants.py` contains version metadata, fallback colors, and shared UI options.
- `geometry.py` composes the geometry interface and owns proportional deformation, transactional reconstruction, and preview cleanup.
- `fitting.py` contains circular-cone fitting, gap-aware flood fill, boundary planes, and depth classification.
- `resize_dimensions.py` contains Blender-independent dimension math and coupled conic-control callbacks.
- `resize_surfaces.py` contains support-surface classification, frozen projections, and resize-mode policy.
- `boolean_bore.py` contains topology-preserving cone/bore junctions and local cutter construction.
- `gpu_preview.py` contains screen-space tinting, cone/cylinder drawing, labels, and drag gizmos.
- `operator.py` contains modal interaction, numeric input, redo state, and runtime cleanup.
- `fair_hole.py` contains connected-boundary picking and the Fair Hole operator.
- `grid_patcher.py` contains the inspectable boundary dissolve, pentagon repair, spoke-slide, and triangle-completion stages.
- `face_curvature.py` snapshots refined border-face tangents and estimates curvature from neighboring face normals for the Organic/C2 ideal surface.
- `ideal_surface.py` provides a shared frozen Organic evaluator for Circle and Grid; `intrinsic_surface.py` uses a 3D bending reference when folded support cannot safely be represented by a height chart. The panel labels this fallback separately rather than claiming exact C2 matching.
- `fairing.py` contains boundary validation, analytic hypotheses, patch topology, and the organic thin-plate solve with a Laplacian fallback.
- `fairing_pipeline.py` shares the Grid/Circle policy and isolated grid trial between both operators.
- `ui.py` contains the sidebar panel and registered scene setting.
- `__init__.py` is Blender's extension entry point and registers the operator and UI.

More detail is available in the developer documentation:

- [Architecture](docs/architecture.md)
- [Resize Hole workflow](docs/resize-hole-workflow.md)
- [Fair Hole workflow](docs/fair-hole-workflow.md)

## License

Copyright 2026 Amos Manneschmidt. Holey Tools is licensed under the GNU General Public License, version 3 or later. See `LICENSE`.
