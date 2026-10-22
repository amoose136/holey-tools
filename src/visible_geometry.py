# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Visible-only working geometry with lossless hidden-geometry preservation."""

import bmesh
import bpy
from mathutils.bvhtree import BVHTree


def visible_bvh(bm):
    """Return a picking BVH and its polygon-to-source-face mapping."""
    faces = tuple(face for face in bm.faces if not face.hide)
    if not faces:
        return None, ()
    vertices = list({vert for face in faces for vert in face.verts})
    indices = {vert: index for index, vert in enumerate(vertices)}
    polygons = [[indices[vert] for vert in face.verts] for face in faces]
    return BVHTree.FromPolygons([vert.co for vert in vertices], polygons), faces


def touches_hidden(vert):
    """Identify shared boundaries that topology cleanup must not dissolve."""
    return vert.hide or any(face.hide for face in vert.link_faces)


def is_visible_vertex(vert):
    """Keep standard Blender behavior for visible mask-boundary vertices."""
    return not vert.hide and (
        not vert.link_faces or any(not face.hide for face in vert.link_faces)
    )


def visible_valence(vert):
    return sum(not edge.hide and (
        not edge.link_faces or any(not face.hide for face in edge.link_faces)
    ) for edge in vert.link_edges)


def hidden_boundary_vertices(bm):
    """Visible vertices whose topology is also owned by hidden source faces."""
    if not any(face.hide for face in bm.faces):
        return set()
    return {vert for face in bm.faces if not face.hide for vert in face.verts
            if touches_hidden(vert)}


def update_visible_normals(bm):
    """Do not recompute hidden-face geometry during coordinate-only previews."""
    visible = [face for face in bm.faces if not face.hide]
    if len(visible) == len(bm.faces):
        bm.normal_update()
        return
    for face in visible:
        face.normal_update()
    for vert in {vert for face in visible for vert in face.verts}:
        vert.normal_update()


def visible_bmesh_copy(bm):
    """Geometry-only visible copy for disposable fitting/probe meshes."""
    copied = bmesh.new()
    try:
        vertices, faces = {}, {}
        for face in bm.faces:
            if face.hide:
                continue
            for vert in face.verts:
                if vert not in vertices:
                    vertices[vert] = copied.verts.new(vert.co)
            target = copied.faces.new([vertices[vert] for vert in face.verts])
            target.material_index = face.material_index
            target.smooth = face.smooth
            faces[face] = target
        edges = {edge: copied.edges.get((vertices[edge.verts[0]], vertices[edge.verts[1]]))
                 for face in faces for edge in face.edges}
        copied.normal_update()
        return copied, vertices, edges, faces
    except Exception:
        copied.free()
        raise


def detach_hidden_geometry(obj):
    """Remove hidden geometry from a native Boolean operand, retaining a backup.

    The real mesh ID is preserved. Join IDs are temporary vertex attributes;
    matching original seam vertices are welded by ID AND position on restore,
    not by proximity to unrelated geometry or interpolated Boolean attributes.
    Call in Object Mode; the caller owns rollback and must free the returned
    state even if reconstruction or the solver raises.
    """
    if not any(face.hide for face in obj.data.polygons):
        return None
    bm = bmesh.new()
    hidden = None
    try:
        bm.from_mesh(obj.data)
        name = '_holey_hidden_join'
        while obj.data.attributes.get(name) is not None:
            name += '_'
        layer = bm.verts.layers.int.new(name)
        for index, vert in enumerate(bm.verts, 1):
            vert[layer] = index
        hidden = bm.copy()
        hidden_faces = [face for face in bm.faces if face.hide]
        hidden_edges = {edge for face in hidden_faces for edge in face.edges}
        hidden_verts = {vert for edge in hidden_edges for vert in edge.verts}
        bmesh.ops.delete(bm, geom=hidden_faces, context='FACES_ONLY')
        bmesh.ops.delete(bm, geom=[edge for edge in bm.edges if not edge.link_faces
                                 and (edge.hide or edge in hidden_edges)], context='EDGES')
        bmesh.ops.delete(bm, geom=[vert for vert in bm.verts if not vert.link_edges
                                 and (vert.hide or vert in hidden_verts)], context='VERTS')
        bmesh.ops.delete(hidden, geom=[face for face in hidden.faces if not face.hide], context='FACES_ONLY')
        bmesh.ops.delete(hidden, geom=[edge for edge in hidden.edges if not edge.link_faces and not edge.hide], context='EDGES')
        bmesh.ops.delete(hidden, geom=[vert for vert in hidden.verts if not vert.link_edges and not vert.hide], context='VERTS')
        bm.verts.index_update()
        vertex_map = {vert[layer] - 1: vert.index for vert in bm.verts}
        bm.to_mesh(obj.data)
        obj.data.update()
        state = HiddenGeometry(hidden, name, vertex_map)
        hidden = None
        return state
    finally:
        bm.free()
        if hidden is not None:
            hidden.free()


class HiddenGeometry:
    def __init__(self, hidden, layer_name, vertex_map):
        self.hidden = hidden
        self.layer_name = layer_name
        self.vertex_map = vertex_map

    def restore(self, obj):
        """Append untouched hidden faces and reconnect surviving original seams."""
        joined = bmesh.new()
        snapshot = bpy.data.meshes.new('Holey Tools Hidden Geometry')
        try:
            joined.from_mesh(obj.data)
            visible_vertex_count = len(joined.verts)
            self.hidden.to_mesh(snapshot)
            joined.from_mesh(snapshot)
            joined.verts.ensure_lookup_table()
            layer = joined.verts.layers.int.get(self.layer_name)
            if layer is not None:
                original = {}
                for vert in joined.verts[:visible_vertex_count]:
                    original.setdefault(vert[layer], []).append(vert)
                weld = {}
                for vert in joined.verts[visible_vertex_count:]:
                    for target in original.get(vert[layer], ()):
                        if target.co == vert.co:
                            weld[vert] = target
                            break
                if weld:
                    bmesh.ops.weld_verts(joined, targetmap=weld)
                joined.verts.layers.int.remove(layer)
            joined.to_mesh(obj.data)
            obj.data.update()
        finally:
            joined.free()
            bpy.data.meshes.remove(snapshot)

    def free(self):
        self.hidden.free()
