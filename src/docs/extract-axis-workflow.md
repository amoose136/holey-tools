# Extract Axis workflow

This document follows the data from an Edit Mode flood fill into the reusable Geometry Nodes fitted-axis modifier.

## 1. Preview a region

`MESH_OT_holey_tools_extract_axis` starts with the same non-destructive selection workflow as Resize Hole. A View3D ray cast finds the face under the pointer and `GeometryMixin._flood_fill()` crosses visible shared edges whose adjacent world-space normals differ by no more than the configured flood angle. The GPU overlay tints the candidate faces without changing Blender's selection.

Bracket keys adjust the threshold, typing replaces it in degrees, and viewport navigation events pass through to Blender. Escape restores the selection captured at invoke time.

## 2. Store the face group

Click or Enter replaces the named face-domain integer attribute with zero for every face outside the flood and one for every face inside it. The default name is `holey_axis_region`, but the sidebar and Adjust Last Operation panel expose it as a string. The real Edit Mode face selection is updated to match.

The integer representation is deliberate: BMesh exposes integer custom-data layers in Edit Mode, and Geometry Nodes converts their zero/one values through a Boolean Named Attribute socket.

## 3. Fit the circular cone

Confirmation appends **Fit Circular Cone** and its dependencies from the bundled `assets/holeyToolsAssets.blend` library when needed. The generated **Holey Tools Extract Axis** group separates the named faces, evaluates their position and normal fields through the fit asset, and samples the group-constant result.

```text
named face geometry -> Fit Circular Cone -> sample fit result
                                       -> Axis, Center, Radius, Slope, Half Angle
```

The fit combines principal components, least-squares plane intersection, projection to a line, line coordinates, and linear regression. It operates on the surface samples rather than requiring complete boundary loops, which makes it tolerant of missing faces and unusual topology. A rank of at least two, a nonzero direction, and a nonempty named face group are required for `Valid`.

Axis sign is not defined: `Axis` and `-Axis` describe the same line. `Slope` changes sign with that choice, while its magnitude and `Half Angle` describe the same cone.

## 4. Install and configure the modifier

The tool creates or updates one **Extract Axis** modifier on the active mesh and sets its `Face Group` input to the stored attribute. Running the tool again updates that modifier instead of stacking duplicates.

When `Indicator Object` is blank, the group creates a small mesh arrow pointing down local +Z, places its base at `Center`, aligns it to `Axis`, and joins it with the input geometry. Choosing a mesh object replaces that arrow with the object's geometry; its origin and local +Z direction define how it is placed. `Indicator Size` uniformly scales either form. The separate `Indicator` output remains available when the group is reused inside another node tree.
