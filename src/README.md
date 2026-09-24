# Holey Tools

Holey Tools provides precision mesh-editing tools for holes. Version 0.9 contains
**Resize Hole** and **Extract Axis**.

## Requirements

- Blender 5.2 or newer
- A mesh object in Edit Mode with a cylindrical or conical hole/stud wall.

## Resize Hole

1. Open **3D Viewport > Sidebar > Holey Tools**.

2. Choose a diameter estimate:
    - **Minimum** uses the narrowest selected radius and is useful for chamfered holes.
    - **Average** uses the mean selected radius.
    - **Maximum** uses the widest selected radius.

3. Click **Resize Hole** and hover a cylindrical or conical wall.

4. Adjust the flood-fill angle with `[` and `]`, then click the wall.

5. Type a diameter or drag the horizontal on-screen handle. When two planar end arcs are recoverable, use the vertical handle to adjust depth: drag down for deeper or up for shallower. Detected cones also expose a horizontal **Angle** handle for their half-angle. Typing and +/- edit the most recently touched handle, defaulting to diameter. Hold Shift for fine adjustment.

    Ctrl temporarily inverts Blender's snapping toggle; snapped changes use 0.1 mm increments.

    Viewport navigation remains available from the middle mouse button, trackpad, mouse wheel, or 3D mouse while the tool is active.

6. Click away from the handle or press Enter to finish. Press Esc to cancel.

The tool respects Blender's proportional-editing toggle, radius, falloff, connected mode, projected mode, and global snapping toggle. The final diameter remains editable from Blender's Adjust Last Operation panel, where the minimum, average, or maximum diameter estimate can also be changed. **Depth** appears when two planar end arcs can be recovered, even when missing wall faces break the arcs into fragments. It is the axis distance from the top/model-side circle center to the bottom circle center. The top circle stays fixed while the other end moves. A detected cone also exposes **Conic Angle**, expressed as the half-angle between its wall and axis; changing it pivots the taper about that top circle.

## Current limitations

- Resize Hole operates on one connected hole wall at a time.
- The selected wall must contain enough geometry to define a stable circular-cone axis and radial regression.
- Minimum and maximum estimates are sensitive to genuinely irregular wall vertices.
- Depth adjustment is unavailable for non-planar ends or walls from which two distinct end arcs cannot be recovered.
- Cylindrical through-holes expose diameter only; conical through-holes retain top-to-bottom depth and angle controls.
- Counterbore holes are not (yet) supported.
- Holes capped by non-planar ends will not be handled well yet. For that, an upcoming non-destructive reconstruction mode is planned that will rebuild the surface without the hole and then build the hole with a boolean and a node tree.

## Extract Axis

1. Open **3D Viewport > Sidebar > Holey Tools** and find **Geometry Nodes**.
2. Choose the face-group attribute name, then click **Extract Axis**.
3. Hover the desired wall or surface region. The tint, `[` / `]` flood-angle
   controls, direct angle entry, and viewport navigation work like Resize Hole.
4. Click the region or press Enter to store it.

The tool replaces a face-domain integer attribute (by default
`holey_axis_region`) with the flooded selection and adds an **Extract Axis**
Geometry Nodes modifier. Its reusable **Holey Tools Extract Axis** group uses the
bundled **Fit Circular Cone** asset, so the fit remains useful for conical surfaces,
missing faces, and irregular topology. Its outputs are:

- **Geometry**: input geometry joined with the aligned axis indicator.
- **Indicator**: the aligned indicator geometry by itself.
- **Axis** and **Center**: a normalized direction and point describing the fitted
  axis. Its sign is arbitrary, so either direction represents the same line.
- **Radius**, **Slope**, and **Half Angle**: the fitted circular-cone parameters.
- **Valid**: true when the named attribute produces a usable fit.

The **Face Group** string input defaults to the most recently extracted attribute
name and can be changed per modifier. **Indicator Size** uniformly scales the
marker. Leave **Indicator Object** blank to use the generated mesh arrow, or choose
a mesh object to place its origin at `Center` and align its local +Z direction with
`Axis`.

## Source layout

- `constants.py` contains version metadata, fallback colors, and shared UI options.
- `geometry.py` contains circular-cone fitting, gap-aware flood fill, proportional weights, and mesh deformation.
- `gpu_preview.py` contains screen-space tinting, cone/cylinder drawing, labels, and drag gizmos.
- `operator.py` contains modal interaction, numeric input, redo state, and runtime cleanup.
- `extract_axis.py` contains the face-group extraction modal operator.
- `axis_nodes.py` loads the bundled fitting assets and builds the Extract Axis modifier.
- `assets/holeyToolsAssets.blend` contains reusable Geometry Nodes math and fitting assets.
- `ui.py` contains the sidebar panel and registered scene setting.
- `__init__.py` is Blender's extension entry point and registers the operator and UI.

More detail is available in the developer documentation:

- [Architecture](docs/architecture.md)
- [Resize Hole workflow](docs/resize-hole-workflow.md)
- [Extract Axis workflow](docs/extract-axis-workflow.md)

## License

Copyright 2026 Amos Manneschmidt. Holey Tools is licensed under the GNU General Public License, version 3 or later. See `LICENSE`.
