# Architecture

This document describes the high-level structure of Holey Tools and the boundaries
between its modules. The extension contains two Blender operators: Resize Hole owns
interactive deformation, while Extract Axis publishes a flooded region to a
generated Geometry Nodes modifier backed by reusable fitting assets.

## Package structure

```text
src/
├── __init__.py              Blender extension entry point and legacy metadata
├── axis_nodes.py            Cone-fit wrapper, modifier, and axis indicator
├── assets/                  Bundled Geometry Nodes fitting assets
├── blender_manifest.toml    Extension Platform metadata
├── constants.py             Version, colors, and shared option definitions
├── extract_axis.py          Extract Axis modal operator and face-group storage
├── geometry.py              Fitting, flood fill, and mesh deformation
├── gpu_preview.py           Screen-space rendering and drag gizmo
├── operator.py              Modal state machine, input, redo, and cleanup
├── ui.py                    Sidebar panel and registered scene properties
└── docs/                    Developer documentation
```

## Dependency direction

```mermaid
flowchart TD
    Entry[__init__.py] --> Operator[operator.py]
    Entry --> Extract[extract_axis.py]
    Entry --> UI[ui.py]
    Operator --> Geometry[geometry.py]
    Operator --> Preview[gpu_preview.py]
    Operator --> Constants[constants.py]
    Extract --> AxisNodes[axis_nodes.py]
    Extract --> Geometry
    Extract --> Preview
    AxisNodes --> Constants
    UI --> Constants
    Preview --> Constants
```

Dependencies should continue to point downward in this diagram. Geometry and GPU code must not import the Blender operator class; this avoids circular imports and keeps those responsibilities independently testable.

## Module responsibilities

### `constants.py`

Contains values shared across modules:

- `ADDON_VERSION`, which drives the viewport and sidebar version display.
- `DIAMETER_ESTIMATOR_ITEMS`, shared by the scene setting and operator property.
- The fallback Face Retopology color used when theme access is unavailable.

It should not import Blender or any other Holey Tools module.

### `geometry.py`

Defines `GeometryMixin`, which is inherited by the operator. It owns operations that read or change mesh geometry:

- Normal-angle flood fill with fitted-surface bridging across missing faces.
- Circular-cone fitting using normal PCA, least-squares axis-plane intersection,
  line coordinates, and radius-on-height regression.
- Minimum, average, and maximum radius estimation.
- Boundary-plane fitting, planarity validation, and adjacent-face classification.
- Proportional-edit weight calculation.
- Absolute radial/axial/angle deformation and end-plane reprojection along cone generators.

Geometry methods operate through attributes on the operator, but do not know about modal events, sidebar layout, or GPU drawing.

### `gpu_preview.py`

Defines `GpuPreviewMixin` and small drawing helpers. It owns:

- Triangulating selected faces for the hover tint.
- Building ideal circular-cone/cylinder preview geometry.
- Projecting world coordinates into viewport pixels.
- Drawing fill and outlines in `POST_PIXEL`, independently of depth/X-Ray state.
- Drawing diameter/depth/conic-angle labels and drag gizmos.
- Hit-testing the drag gizmo.

The preview color comes from `ThemeView3D.face_retopology`. The diameter symbol is defined by a Unicode escape so source-file encoding cannot silently change it.

### `operator.py`

Defines `MESH_OT_holey_tools_resize_hole`, the modal Blender operator. It combines `GeometryMixin` and `GpuPreviewMixin` with `bpy.types.Operator` and owns:

- Operator dimension properties and transient modal state.
- Viewport ray casting and hover updates.
- Mouse, keyboard, numeric, snapping, and navigation events.
- Original-selection and original-coordinate restoration.
- Serialization of fitted data into hidden redo properties.
- Creation and cleanup of draw handlers.

`_PersistentPreview` is a plain Python object in this module. It retains only drawing data after the Blender operator's RNA wrapper is destroyed, preventing stale-RNA `ReferenceError` failures while the Adjust Last Operation panel remains active.

### `extract_axis.py`

Defines `MESH_OT_holey_tools_extract_axis`. It uses the same normal-angle flood,
screen-space tint, bracket/direct-entry controls, ray casting, viewport-navigation
pass-through, and cancellation semantics as Resize Hole. Confirmation writes a
zero/one integer attribute on the face domain and selects the same faces in Edit
Mode. Integer storage is used because BMesh can update it directly; the generated
Named Attribute node reads it as a boolean selection.

### `axis_nodes.py`

Creates and reuses the **Holey Tools Extract Axis** wrapper and appends the bundled
**Fit Circular Cone** asset and its dependencies when needed. The wrapper isolates
the named face group and samples the fit's axis point, direction, radius, slope,
half angle, and rank. This least-squares fit supports circular cones and is more
tolerant of missing faces and irregular topology than selecting a PCA outlier.
The tool installs the wrapper as an **Extract Axis** modifier. It joins either a
generated mesh arrow or an optional object's geometry at the fitted point, aligned
from local +Z to the axis. Generated wrappers carry a private version marker so an
unrelated user node group with the preferred name is never overwritten.

### `ui.py`

Contains the Holey Tools sidebar panel and the scene-level diameter-estimator setting. It refers to the operator by its stable ID, `mesh.holey_tools_resize_hole`, rather than importing the operator class.

### `__init__.py`

This is the extension entry point Blender loads. It exposes `bl_info` for legacy development installs and directly registers the operator, panel, and scene properties. Its unregister path calls the operator module's runtime cleanup before removing Blender classes.

## State ownership

Operator-instance attributes hold all state specific to an invocation, including the active object, selected face indices, fitted circular cone, original coordinates, preview geometry, numeric input, active dimension, and gizmo rectangles.

`operator.py` has three module-level runtime values shared by both modal tools:

- `_modal_active` records whether a Holey Tools modal session is active.
- `_draw_handles` tracks every installed viewport draw handler for cleanup.
- `_persistent_preview` owns the post-operation cone/cylinder preview.

No runtime state is written to disk or sent over the network.

## Adding another tool

New Holey Tools should normally receive their own operator module. Reuse focused geometry or preview helpers only when their behavior is genuinely shared. Register the new operator in `__init__.py`, expose its UI from `ui.py`, and add headless tests for its core behavior.
