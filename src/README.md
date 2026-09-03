# Holey Tools

Holey Tools provides precision mesh-editing tools for holes. Version 0.9 currently contains one tool: **Resize Hole**.

## Requirements

- Blender 5.2 or newer
- A mesh object in Edit Mode that has some cylindrical indentation or through hole.

## Resize Hole

1. Open **3D Viewport > Sidebar > Holey Tools**.

2. Choose a diameter estimate:
    - **Minimum** uses the narrowest selected radius and is useful for chamfered holes.
    - **Average** uses the mean selected radius.
    - **Maximum** uses the widest selected radius.

3. Click **Resize Hole** and hover a cylindrical wall.

4. Adjust the flood-fill angle with `[` and `]`, then click the wall.

5. Type a diameter or drag the horizontal on-screen handle. When planar end loops are available, use the vertical handle below it to adjust depth. Typing and +/- edit the most recently touched handle, defaulting to diameter. Hold Shift for fine adjustment.

    Ctrl temporarily inverts Blender's snapping toggle; snapped changes use 0.1 mm increments.

    Viewport navigation remains available from the middle mouse button, trackpad, mouse wheel, or 3D mouse while the tool is active.

6. Click away from the handle or press Enter to finish. Press Esc to cancel.

The tool respects Blender's proportional-editing toggle, radius, falloff, connected mode, projected mode, and global snapping toggle. The final diameter remains editable from Blender's Adjust Last Operation panel, where the minimum, average, or maximum diameter estimate can also be changed. A **Depth** field appears there when the selected wall has exactly two planar boundary loops. Parallelism is not required: depth is measured between the loop centers on the fitted cylinder axis. The end attached to the surrounding model stays fixed while the other end moves.

## Current limitations

- Resize Hole operates on one connected hole wall at a time.
- The selected wall must contain enough geometry to define a stable cylinder.
- Minimum and maximum estimates are sensitive to genuinely irregular wall vertices.
- Depth adjustment is unavailable for non-planar ends and wall selections that do not produce exactly two boundary loops.
- Through-holes expose diameter adjustment only; depth is reserved for blind holes and studs with a fixed model-side boundary.
- Conics are not supported.
- Countersunk holes do not preserves angles on adjustment (yet).
- Counterbore holes are not (yet) supported.
- Holes capped by non-planar ends will not be handled well yet. For that, an upcoming non-destructive reconstruction mode is planned that will rebuild the surface without the hole and then build the hole with a boolean and a node tree.

## Source layout

- `constants.py` contains version metadata, fallback colors, and shared UI options.
- `geometry.py` contains cylinder fitting, flood fill, proportional weights, and mesh deformation.
- `gpu_preview.py` contains screen-space tinting, cylinder drawing, labels, and the drag gizmos.
- `operator.py` contains modal interaction, numeric input, redo state, and runtime cleanup.
- `ui.py` contains the sidebar panel and registered scene setting.
- `__init__.py` is Blender's extension entry point and registers the operator and UI.

More detail is available in the developer documentation:

- [Architecture](docs/architecture.md)
- [Resize Hole workflow](docs/resize-hole-workflow.md)

## License

Copyright 2026 Amos Manneschmidt. Holey Tools is licensed under the GNU General Public License, version 3 or later. See `LICENSE`.
