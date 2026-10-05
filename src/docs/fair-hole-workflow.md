# Fair Hole workflow

Fair Hole is a blocking modal operator that discovers one open boundary and fills it using the simplest surface model that passes its validation gates.

## Boundary selection

The mouse is tested against projected edges with exactly one linked face. A hit seeds a connectivity flood over all neighboring one-face edges; face-normal angle does not participate. The component is valid only if it contains at least three edges, every vertex has degree two within the component, and one ordered walk visits every edge before returning to the seed. This rejects open chains, branches, and multiple loops connected at a vertex.

The complete component is drawn as a screen-space line overlay. Clicking or pressing Enter confirms it. Esc restores the pre-invocation mesh selection.

On confirmation, the chosen ring first receives a 0.1° limited-dissolve pass. Collinear boundary vertices whose adjoining support normals remain compatible are removed before fitting, reducing triangulation noise without crossing sharp surface seams.

## Fit cascade

Only the faces directly touching the chosen rim are analytic fit samples. Each hypothesis either returns a complete result or yields to the next one:

1. **N-plane intersection.** Consecutive support samples are progressively added while their maximum plane-fit error remains below 0.5% of the rim scale. The disconnected runs describing the same geometric plane are merged before the limit is checked. More than three distinct planes rejects the hypothesis. One plane projects the rim center, two planes define an intersection line and choose its nearest point, and three planes define an intersection point. Plane-normal angles below 0.4° are treated as parallel and do not produce an unstable distant line or point intersection.
2. **Circular cone/cylinder.** The shared normal-PCA and radius-on-axis regression fit is applied to the touching faces. Insufficient rank or a relative radial residual above 3% rejects it.
3. **Sphere.** Adjacent face-normal lines are intersected in least squares. Their average distance to the center defines the radius. The larger of normalized radial error and normal-alignment error must be at most 3%.

    The cone/cylinder and sphere fits both run whenever the plane hypothesis fails. If both pass, the lower normalized residual determines the chosen model. Before a selected sphere patch is constructed, its support region receives Blender's boundary-inclusive Limited Dissolve and the sphere is refitted against the surviving boundary.

4. **Organic.** This unconditional final hypothesis snapshots the non-loop spokes attached to the rim, then applies a targeted 10° boundary dissolve. Every spoke-connected vertex and exterior edge is protected; only redundant boundary-only vertices with compatible support normals are removed. It only then fits/projects the simplified loop. The spoke directions are folded modulo 90° before SVD so balanced orthogonal source-grid families do not collapse to a diagonal axis. Their boundary anchor positions also establish the grid-line phase. Boundary count and aspect ratio determine the interval budget; constrained Delaunay clips the grid to the loop. A thin-plate interpolation constrained by the rim and its exterior one-ring carries both height and surrounding slope into the patch. It assigns only fitted-axis depth; a graph Laplacian remains the degeneracy fallback.

The plane case uses a center triangle fan. A cylinder runs Blender's Limited Dissolve on the initial support selection with boundary dissolve enabled, scales the surviving rim vertices to zero only along the fitted axis, and merges by distance at 0.0001. The affected faces are selected, grown by one adjacency step, and passed through Limited Dissolve again. This closes missing cylindrical wall patches without pulling vertices radially off the cylinder. Conic and sphere cases default to two concentric quad rings and a center fan, with ring vertices projected back onto their accepted analytic surface. Organic uses the clipped, SVD-aligned grid described above. If no stable clipped grid can be produced, it falls back to one fitted-plane center seed.

After the accepted patch is built, its vertices and former rim seam are merged by a scale-aware distance. The patch-face selection then grows across one adjacent face ring before a second 0.1° limited-dissolve pass. Normal delimiters preserve intentional plane breaks while redundant coplanar or nearly tangent topology is removed.

## Redo overrides

The Adjust Last Operation panel exposes the surface hypothesis and fill topology as separate controls. **Surface: Auto** follows the cascade above. N-Plane, Cylinder, Sphere, and Organic force that hypothesis; forced cylinder and sphere fits retain their numerical rank requirements but bypass the automatic 3% residual gate. Cylinder also fixes the fitted slope to zero.

**Fill: Auto** keeps the hypothesis-specific topology: a plane center fan, cylinder axial collapse, radial rings for cones and spheres, or the organic grid. **Grid** instead clips the SVD-aligned source grid to the boundary, then projects its new vertices onto the selected analytic surface or evaluates the thin-plate continuation. **Circle** creates concentric rings and a center fan, using that same selected surface. Blender restores the operator's original input before each redo, so switching either control does not accumulate edits.

For Circle fill, **Circle Loops** sets the number of concentric quad rings before the center fan; zero produces a fan directly from the boundary. For Grid fill, **Grid Angle Offset** rotates the inferred SVD/source-edge alignment without changing the fitted surface or projection axis.

**Final Simplify** is enabled by default. It performs the one-ring selection growth and final Limited Dissolve described above. Disabling it preserves the generated grid, rings, or fan; required seam welding and the cylinder's fixed 0.0001 merge still run.

## Commit and failure behavior

The fit and patch are created only after confirmation. The original mesh is snapshotted before the first dissolve; a singular solve, invalid boundary, or zero-area result restores that snapshot rather than leaving a partially simplified ring. On success the surviving patch faces and vertices are selected, mesh normals are updated, and Blender records the operation for Undo. The Adjust Last Operation panel reports the selected hypothesis and its residual.

The current implementation deliberately does not remesh the surrounding surface, infer trimmed CAD continuity, or bridge branching and open boundary graphs.
