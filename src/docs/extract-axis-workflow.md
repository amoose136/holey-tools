# Extract Axis workflow

This document follows the data from an Edit Mode flood fill into the reusable
Geometry Nodes axis result.

## 1. Preview a region

`MESH_OT_holey_tools_extract_axis` starts with the same non-destructive selection
workflow as Resize Hole. A View3D ray cast finds the face under the pointer and
`GeometryMixin._flood_fill()` crosses visible shared edges whose adjacent world-space
normals differ by no more than the configured flood angle. The GPU overlay tints the
candidate faces without changing Blender's selection.

Bracket keys adjust the threshold, typing replaces it in degrees, and viewport
navigation events pass through to Blender. Escape restores the selection captured at
invoke time.

## 2. Store the face group

Click or Enter replaces the named face-domain integer attribute with zero for every
face outside the flood and one for every face inside it. The default name is
`holey_axis_region`, but the sidebar and Adjust Last Operation panel expose it as a
string. The real Edit Mode face selection is updated to match.

The integer representation is deliberate: BMesh exposes integer custom-data layers
in Edit Mode, and Geometry Nodes converts their zero/one values through a Boolean
Named Attribute socket.

## 3. Generate the Geometry Nodes wrapper

Confirmation ensures that **Holey Tools Extract Axis** exists. The tool creates only
a reusable node group datablock; it does not add or reorder object modifiers.

The wrapper accepts the source `Geometry`, a `Face Group` attribute-name string, and
an `Indicator Size`. It separates the selected faces and passes their geometry and
position field to Blender's bundled **Geometry Principal Components** asset.

```text
selected geometry -> Geometry Principal Components
axis = outlier(Longest Axis, Shortest Axis)
indicator location = Group Center
```

For a cylindrical wall, two position components describe the similar radial spread
and the remaining component describes its axial spread. Depending on wall depth,
that axial component can be the longest or shortest. The wrapper compares adjacent
component gaps and chooses the outlier. Axis sign is not defined: `axis` and `-axis`
describe the same line.

The wrapper creates a subdivision-one icosphere at `Group Center`. `Geometry` joins
it to the input geometry, while `Indicator` exposes it separately for downstream
styling or replacement. `Valid` combines the Named Attribute node's `Exists` result
with a nonzero selected-face count.

## 4. Consume the result

Insert the generated group into a Geometry Nodes tree, connect geometry that contains
the stored face group, and use `Axis` and `Center` to orient or position downstream
geometry. Change the group-node's `Face Group` string when several extracted regions
live on the same mesh. Use `Indicator` separately when the center marker should not
be joined to the model.
