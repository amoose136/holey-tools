# Holey Tools 1.0

Changes since 0.9.4:

- Added **Fair Hole**, with automatic surface fitting, curvature-aware repair, and boundary-aligned quad Grid or Circle fills.
- Added conic and countersink editing with **Depth**, **Truncated Depth**, and full tip-angle controls, preserving compatible connected bores.
- Added Boolean reconstruction for non-planar holes, **Auto / Scaling / Boolean** mode overrides, and adjustable cutter resolution.
- Expanded blind-hole depth support beyond planar ends. Setting Diameter or Depth to zero now closes and fairs the hole.
- Added **Ctrl/Cmd-click Resize Hole** to operate directly on selected faces instead of flood filling.
- Corrected low-sided cylindrical hole sizing so changing Boolean segments preserves usable circular clearance.
- Sped up hover selection, live previews, and final fairing, particularly spherical repairs.
- Improved undo/redo, cancellation, local geometry cleanup, and hidden-face preservation.
