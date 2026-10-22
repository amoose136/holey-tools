# Holey Tools

Holey Tools provides precision mesh-editing tools for holes. Version 0.9 contains **Resize Hole** and **Fair Hole**.

## Requirements

- Blender 5.2 or newer
- A mesh object in Edit Mode.

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

Set Diameter or an available Depth to **0** to close the hole using shared fairing, including both openings of a through-hole. A through-cone shortened with Depth or Truncated Depth keeps a cylindrical connection from its new small circle to the original exit surface, preserving the small-circle radius. Auto reconstructs this connection with Exact Boolean; explicit Scaling inserts the connecting strip directly. Zero Truncated Depth alone retains that cylinder rather than closing the hole.

For cylindrical holes, blind or through, **Diameter** measures the circle fitting between the polygon's flat walls, not across its corner vertices. The smooth blue overlay shows that inside circle. An N-sided Boolean cutter uses vertex radius `Diameter / (2 × cos(π/N))`, so changing Boolean Segments preserves the usable opening instead of changing which circular gauge fits. The overlay's 128 segments never affect this correction. Direct scaling keeps the original side count. Cone and stud sizing conventions are unchanged.

The tool respects Blender's proportional-editing toggle, radius, falloff, connected mode, projected mode, and global snapping toggle. Adjust Last Operation exposes **Resize Method** with **Auto**, **Scaling**, and **Exact Boolean** choices. Auto uses direct deformation only for a one-plane exterior support and otherwise reconstructs the surface before cutting; the explicit choices override that decision. Scaling exposes its own **Final Simplify** option, off by default, which expands the resized-wall selection and runs Limited Dissolve after a shrink. Cylinders retain the diameter estimator and Diameter control. For conic regions with two recoverable end circles, the outer and truncation radii are derived from apex Depth, Truncated Depth, and the full included Conic Angle, so Diameter is not an independent fourth constraint. A countersink is anchored at its fitted small circle at the geometry level, ensuring Depth, Truncated Depth, and Conic Angle never resize the connected through-hole even during redo reconstruction. Geometry recognizes whether Depth, Truncated Depth, or Angle changed without relying on redo callback order: Angle keeps the truncation plane fixed, Truncated Depth places it directly, and a lone Depth edit derives its position from the fixed bore radius. Boolean reconstructions additionally expose cutter segments, the reconstruction fit followed by topology, circle loops, C2 curvature matching, and their separately default-enabled final simplification. Boolean simplification covers every face of the reconstructed fair surface plus one exterior connected topological face ring before running Limited Dissolve in edge-selection context; cutter-derived wall and cap faces are excluded so cleanup cannot bridge the new opening. Cutter segments initially match the detected source resolution and can be overridden from 3 to 512.

Auto uses direct deformation for planar exterior supports and recognized coaxial countersink connections. A conic between a planar exterior and a cylindrical bore scales directly while preserving the bore diameter. When resizing the cylinder instead, the shared cone rim slides along the hole axis to preserve the cone angle; intermediate cone rows adjust along their generators and its larger rim stays fixed. For other surfaces, enlargement projects the resized rim along the hole axis onto a frozen base-mesh snapshot, while each interior blind-cap rim is projected against its own fit. Shrinkage likewise intersects each rim's fitted surface mathematically. During enlargement, target-radius probes recheck that the frozen surface is still coplanar with the original collar. Crossing from that collar onto a curved or sloped cap switches the operation to Exact Boolean reconstruction after a tolerant multi-probe check filters isolated BVH seams and small mesh noise.

Any exterior surface classified as two or three planes, spherical, conic/cylindrical, or organic switches to reconstruction. Live dragging keeps the real source object's mesh pristine and shows a reversible Exact Boolean result. Source meshes up to 32 angular segments use their final cutter resolution; higher-resolution sources use a 32-sided live cutter. The exterior opening is faired once on a separate mesh, and the modal path hides the untouched edit cage while a display object holds that reconstruction and one unapplied Boolean modifier. Subsequent changes mutate only its small cutter mesh, avoiding destructive source-mesh swaps, polluted redo input, and the overlapping edit-cage artifacts of an unapplied modifier on the edited object. Preview updates skip final simplification, run at most once every 0.5 seconds during dragging, and apply the latest pending value on release while the GPU cutter remains continuous. On confirmation the temporary display is removed, the source object is shown again, and the old wall and cavity geometry are rebuilt at the source mesh's angular resolution before Blender's Exact Boolean solver subtracts the requested circular-cone cutter. For a countersink feeding a compatible coaxial cylinder, Exact Boolean preserves the existing cylindrical wall and moves only its junction ring for depth edits. The local cone cutter follows the source ring's angular alignment and stops just inside the bore; it no longer re-cuts the cylinder along its full length. Different cutter segment counts use one short transition strip at the junction. If the edit would consume the next source cylinder row, the general compound cutter remains the safe fallback. A canonical pre-preview mesh snapshot is restored before every Adjust Last Operation execution, so redo never compounds a direct preview or treats a previous Boolean result and old blind cap as its source. The operation keeps a mesh backup and restores it if reconstruction or the Boolean fails. For a conic cutter, the wider ring is always the model-side end and extends by an explicit axial clearance beyond the reconstructed exterior surface so its cap cannot remain coplanar with the mesh.

## Current limitations

Hidden faces are excluded from picking, fitting, surface projection, fairing probes, and Boolean operands. They remain preserved through confirmation, cancellation, and undo; visible vertices shared with hidden faces can still move with the edit, as in Blender's normal transforms. Full-mesh backups and visibility bookkeeping remain necessary to preserve that geometry.

- Resize Hole operates on one connected hole wall at a time.
- The selected wall must contain enough geometry to define a stable circular-cone axis and radial regression.
- Minimum and maximum estimates are sensitive to genuinely irregular wall vertices.
- Depth adjustment requires either two usable planar end arcs or distinct exterior and blind-cap surfaces that both intersect the fitted axis.
- Cylindrical through-holes expose diameter only; conical regions with two recoverable end circles expose theoretical apex depth, physical truncation depth, and included tip angle.
- Counterbore holes are not (yet) supported.
- Boolean reconstruction requires at least one recoverable closed exterior rim; branched or irrecoverably fragmented non-planar boundaries are rejected rather than deformed directly.

## Fair Hole

1. Open **3D Viewport > Sidebar > Holey Tools** and click **Fair Hole**.
2. Hover an edge of an open boundary. The complete loop is highlighted by connectivity alone; no face-angle threshold is used.
3. Click or press Enter. **Fit** and **Topology** default to **Auto Fit** and **Auto Topology** on each invocation; the dropdowns to the right of Fair Hole present them in that order and can preselect another choice. Auto Topology tries a complete Manhattan grid for spherical and organic fits, then falls back to Circle with five rings if the grid cannot be built.
4. In Adjust Last Operation, choose the fit first and topology second, or toggle **Match Curvature (C2)**. Normal builds run Grid to completion and expose the default-enabled C2 toggle for Organic fits; analytic fits use the selected ideal surface directly. Builds made with `--development-controls` additionally expose **Stop After** stages, row limits, and smoothing diagnostics.

Step 1 dissolves border vertices with exactly two connected mesh edges, preserving vertices with additional source edges. It does not use an angle threshold and never destroys the last triangular loop or a triangular support face. Step 2 first gathers the faces incident to the surviving ring edges and merges all vertices belonging to those faces by distance at **0.001 mesh units**, then inspects five-sided faces with one edge exposed to the selected opening. It extends each exposed endpoint's neighboring pentagon edge to their intersection, then welds the endpoints there, restoring a roughly rectangular quad. The merge includes outer support vertices of those ring faces, but excludes vertices outside the face region. Pentagon repair leaves the remaining source vertices fixed. Ambiguous or unsafe intersections are skipped and counted in the redo panel.

Step 3 first aligns quad border vertices toward the nearest corrected corner, then slides triangle-touching vertices, and finally runs **three T-only refinement passes before any triangle completion**. Every three-edge boundary vertex with one inward spoke participates, including triangle-touching vertices. Ordinary T points with no adjacent triangle slide along their spoke toward a straight border (180° between its branches) and right angles to the inward spoke. A T point diagonally opposite a repaired pentagon corner across a quad can instead move in both planar directions to make that quad rectangular, without moving the repaired corner. Beside a triangle, these points move in both planar directions to bring their own corner and the two neighboring corners of the completed-side quad toward 90° together, ignoring the unfinished triangle-side branch. Repaired pentagon corners remain locked. Step 3 leaves the triangles intact for inspection.

Step 4 completes the triangle patterns using the refined boundary: bowties become two quads with a shared new edge and a middle boundary vertex positioned to minimize the turn toward its two neighboring rim vertices, while staying in both source triangle planes; triangles with one exposed boundary edge are completed by a 180° rotation of their opposite corner about the face normal through the edge midpoint (A + B − C). This makes a parallelogram, rectangular for a right triangle, and also works for edge-adjacent triangles without requiring quad neighbors. This is the **Refined Edge** state. Exactly **one post-mirror refinement pass** runs; this is fixed rather than a redo option. The three pre-mirror passes in Step 3 are still run. Moves preserve valid faces and a non-crossing boundary, and the panel separately reports pre-mirror T and post-mirror move counts. After refinement, a three-edge U-shaped run with two inside corners is capped by one quad using its four existing vertices, without splitting source edges. Convexity, source connectivity, and opening-containment checks gate the cap; the remaining Refined Edge stays selected.

Steps 1–4 leave the remaining border selected for inspection. **Step 5: Fill Quad Rows** always uses **Manhattan Grid**; the retired native, cross-field, and older local backends have been removed. No native executable or Grid Density control is needed. At Step 4, inspect the interlaced **U1 / V1 / U2 / V2** counts and template size. The template trims encroached cells and uses single-row all-quad mutations for opposing count differences. The saved Suzanne example gives **8 / 5 / 2 / 5**, an **8 × 5** template, and three mutations with no boundary cuts. Unequal side counts use quad mutations and held-corner transitions instead of subdividing the rim. Even-length refined perimeters receive zero cuts; an odd perimeter needs one reported parity cut to retain all-quads. **Rows to Fill** limits inspection layers, with **0** filling the complete patch. **Interior Relaxation** defaults to 20 passes with **Relaxation Strength** 0.5; zero passes disables smoothing. Interior and unpinned refined-rim vertices relax on the frozen ideal surface, with fold prevention and quad angle/aspect optimization; exterior geometry, hidden-face-adjacent rim points, and parity-cut endpoints remain fixed. Held-station layouts are ranked by shape quality rather than accepting the first valid patch. For well-conditioned Organic height charts, **Match Curvature (C2)** defaults on and estimates rim slopes/curvature from exterior faces; disabling it or insufficient samples uses thin plate. Folded or grazing support instead uses a **3D bending continuation (folded chart)** for either setting. Matching a discrete source is approximate, not exact C2 continuity of polygon faces. No final dissolve runs. The planner tries the longest-side sweep first, then its perpendicular orientation on failure; each sweep requires one connected interval per row. See the [Fair Hole workflow](docs/fair-hole-workflow.md) for fitting and inspection details.

Fit can be Auto Fit, N-Plane, Cylinder, Sphere, or Organic. Auto Fit tries at most three adjacent planes first, then compares cylinder/cone and sphere fits, and finally uses organic continuation. Explicit Grid requires the Manhattan patcher and reports an error if it cannot fill; only Auto Topology can fall back to Circle. Grid respects the selected fit: Sphere, Cylinder, and N-Plane project points analytically. Circle retains adjustable **Circle Loops**, defaulting to five. Organic Circle and Grid share the same frozen fitting policy, including the 3D bending fallback; newly generated topology never determines the ideal surface. **Final Simplify** remains available for completed non-Grid repairs. Resize Hole uses this same pipeline for Boolean reconstruction, retaining its subdivisions until the final subtraction cleanup.

Grid preparation also welds safe collapsed corner edges after pentagon repair and before triangle completion, so nearly triangular quads are classified correctly. Repaired corners remain fixed, and the panel reports these targeted welds separately. Curved triangle pairs that can only produce a tiny bowtie spoke complete individually instead. These checks use live connectivity rather than saved face or vertex IDs. Collapsed quad spokes also use a conservative local length ratio, and neighbor repairs are followed by another degree-two-rim cleanup. Triangle pairs with distinct safe completion directions mirror separately instead of creating a coupled bowtie edge.

Resize Hole uses the same complete Manhattan/Circle pipeline; development inspection stages never produce a partial Boolean operand.

Current Fair Hole limitations:

- The boundary must be one simple closed component of edges with exactly one linked face.
- Grid Steps 1–4 intentionally leave the opening; Step 5 validates a boundary-locked Manhattan patch. Infeasible, crossing, or folded patches are rejected and the operation restores its input.
- Pentagon repair requires one exposed edge and a safe, nearby intersection of the neighboring edge directions; not every five-sided face is a clipped rectangular corner.

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
