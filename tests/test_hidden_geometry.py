"""Hidden faces stay out of geometric work and survive native Boolean unchanged."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy
from mathutils import Matrix, Vector
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import GeometryMixin
from src.resize_surfaces import ResizeSurfaceMixin
from src.visible_geometry import detach_hidden_geometry, visible_bvh, visible_bmesh_copy, is_visible_vertex
import src.fairing_pipeline as pipeline
import src.operator as runtime
from src.modal_session import begin_modal_session, _sessions
from tests.blender_support import activate_mesh as activate, bind_operator_methods, rna_defaults


def hidden_signature(mesh):
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        tag = bm.faces.layers.int.get('hidden_test_tag')
        uv = bm.loops.layers.uv.get('hidden_test_uv')
        return sorted((face[tag] if tag else 0, face.material_index,
                       tuple(sorted(tuple(v.co) for v in face.verts)),
                       tuple(sorted(tuple(loop[uv].uv) for loop in face.loops)) if uv else ())
                      for face in bm.faces if face.hide)
    finally:
        bm.free()


# Shared boundary: detaching must not duplicate the original seam or lose UVs.
mesh = bpy.data.meshes.new('Hidden seam')
mesh.from_pydata([(0,0,0),(1,0,0),(2,0,0),(0,1,0),(1,1,0),(2,1,0)], [],
                 [(0,1,4,3),(1,2,5,4)])
obj = activate(mesh)
bm = bmesh.new()
bm.from_mesh(mesh)
bm.faces.ensure_lookup_table()
bm.faces[1].hide = True
tag = bm.faces.layers.int.new('hidden_test_tag')
uv = bm.loops.layers.uv.new('hidden_test_uv')
for face in bm.faces:
    face[tag] = face.index + 17
    for loop in face.loops:
        loop[uv].uv = loop.vert.co.xy
bm.to_mesh(mesh)
bm.free()
before = hidden_signature(mesh)
state = detach_hidden_geometry(obj)
assert len(mesh.polygons) == 1
state.restore(obj)
state.free()
assert len(mesh.vertices) == 6 and len(mesh.polygons) == 2
assert hidden_signature(mesh) == before
assert not any(a.name.startswith('_holey_hidden_join') for a in mesh.attributes)

# Visible mask-boundary vertices retain Blender's usual transform behavior.
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(mesh)
bm.faces.ensure_lookup_table()
probe = SimpleNamespace(_object=obj)
GeometryMixin._capture_deformation_state(probe, bpy.context, bm, set(bm.faces[0].verts))
assert len(probe._original_vertex_coords) == 4
bm.verts.ensure_lookup_table()
assert all(is_visible_vertex(bm.verts[i]) for i in probe._original_vertex_coords)
bpy.ops.object.mode_set(mode='OBJECT')


# Hidden geometry in front of the visible face must not occlude hover picking.
mesh = bpy.data.meshes.new('Hidden picking')
bm = bmesh.new()
for z, hide in ((0, False), (1, True)):
    face = bm.faces.new([bm.verts.new(co) for co in ((-1,-1,z),(1,-1,z),(1,1,z),(-1,1,z))])
    face.hide = hide
bm.normal_update()
tree, mapping = visible_bvh(bm)
assert len(mapping) == 1 and not mapping[0].hide
hit = tree.ray_cast(Vector((0,0,2)), Vector((0,0,-1)))
assert abs(hit[0].z) < 1e-7
copy, vertices, edges, faces = visible_bmesh_copy(bm)
assert len(copy.faces) == 1 and len(copy.verts) == 4
assert len(faces) == 1 and all(not face.hide for face in faces)
copy.free()

# The grid trial must not copy hidden topology into its probe.
visible_face = next(face for face in bm.faces if not face.hide)
def check_probe(trial, *_args, **_kwargs):
    assert len(trial.faces) == 1 and all(not face.hide for face in trial.faces)
    raise ValueError('visible-only probe reached')
with patch.object(pipeline, 'prepare_grid_steps', side_effect=check_probe):
    assert pipeline._probe_grid_patch(SimpleNamespace(matrix_world=Matrix.Identity(4)), bm,
                                      list(visible_face.edges), {}) == 'visible-only probe reached'
bm.free()


def bore_fixture():
    mesh = bpy.data.meshes.new('Visible blind bore')
    n = 16
    rings = ((3,0),(1,0),(1,-1),(3,-2))
    vertices = [(r*math.cos(math.tau*i/n), r*math.sin(math.tau*i/n), z)
                for r,z in rings for i in range(n)]
    faces = [(a*n+i, a*n+(i+1)%n, b*n+(i+1)%n, b*n+i)
             for a,b in ((0,1),(1,2),(3,0)) for i in range(n)]
    faces += [tuple(range(2*n,3*n)), tuple(reversed(range(3*n,4*n)))]
    mesh.from_pydata(vertices, [], faces)
    obj = activate(mesh)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    # Deliberately intersects the proposed cutter. Native Boolean must never
    # see or alter this hidden cube, even when it is inside the same mesh ID.
    tag = bm.faces.layers.int.new('hidden_test_tag')
    uv = bm.loops.layers.uv.new('hidden_test_uv')
    cube = bmesh.ops.create_cube(bm, size=.8, matrix=Matrix.Translation((0,0,-1)))
    hidden_faces = {face for vert in cube['verts'] for face in vert.link_faces}
    for face in hidden_faces:
        face.hide_set(True)
        face[tag] = 123
        for loop in face.loops:
            loop[uv].uv = (.2,.7)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.faces.ensure_lookup_table()
    defaults = rna_defaults()
    defaults.update(_object=obj, _stage='SELECT', diameter_estimator='AVERAGE', resize_method='BOOLEAN',
                    _candidate_indices=set(range(n,2*n)), _initial_selection=set(range(n,2*n)),
                    _draw_handle=None)
    op = SimpleNamespace(**defaults)
    bind_operator_methods(op)
    op._tag_redraw = op._set_status_text = lambda *_args: None
    op.report = lambda *_args: print('Hidden fixture:', *_args)
    assert op._lock_selection_and_fit(bpy.context, build_preview=False)
    return obj, op


obj, op = bore_fixture()
obj.update_from_editmode()
before = hidden_signature(obj.data)
signature = GeometryMixin._object_topology_signature(obj)
bm = bmesh.from_edit_mesh(obj.data)
hidden_points = {v: v.co.copy() for f in bm.faces if f.hide for v in f.verts}
for vert in hidden_points:
    vert.co.x += 5
assert GeometryMixin._object_topology_signature(obj) == signature, 'Hidden geometry entered fingerprint'
# Return the hidden cube to the cutter's path before the real test.
for vert, point in hidden_points.items():
    vert.co = point
op.diameter = 1.6
op._apply_diameter_to_mesh(bpy.context, commit_boolean=True)
obj.update_from_editmode()
assert hidden_signature(obj.data) == before
assert not any(a.name.startswith('_holey_hidden_join') for a in obj.data.attributes)

obj, op = bore_fixture()
obj.update_from_editmode()
before = hidden_signature(obj.data)
op.diameter = 1.6
begin_modal_session(op, bpy.context)
op._apply_diameter_to_mesh(bpy.context)
display = op._boolean_preview_display_object
display_name = display.name
cutter_name = op._boolean_preview_cutter_object.name
assert display is not None and not any(f.hide for f in display.data.polygons)
assert len(display.data.polygons) < 100
assert hidden_signature(obj.data) == before
runtime.cleanup_runtime()
runtime.cleanup_runtime()  # Repeated reload cleanup is harmless.
assert not _sessions and not runtime._modal_active and not obj.hide_get()
assert display_name not in bpy.data.objects and cutter_name not in bpy.data.objects
assert op._boolean_preview_source is None and op._boolean_reconstruction_source is None
assert op.modal(bpy.context, SimpleNamespace(type='MOUSEMOVE')) == {'CANCELLED'}
obj.update_from_editmode()
assert hidden_signature(obj.data) == before

# Dense hidden geometry is not inserted into any picking/search/probe mesh.
obj, op = bore_fixture()
bm = bmesh.from_edit_mesh(obj.data)
grid = bmesh.ops.create_grid(bm, x_segments=128, y_segments=128, size=10,
                            matrix=Matrix.Translation((100,100,100)))
for face in {face for vert in grid['verts'] for face in vert.link_faces}:
    face.hide = True
visible = [face for face in bm.faces if not face.hide]
op._stage = 'SELECT'
cache = op._get_hover_cache()
assert len(cache.visible_faces) == len(visible) == 50
list(cache.nearby_faces((0,0,0), 10))
assert len(cache.centers) == 50
tree, mapping = visible_bvh(bm)
assert len(mapping) == 50
copy, _verts, _edges, _faces = visible_bmesh_copy(bm)
assert len(copy.faces) == 50
copy.free()
bvh, span = ResizeSurfaceMixin._surface_bvh(list(bm.faces), obj.matrix_world)
assert span < 25, 'Hidden grid affected projection bounds'
print(f'Visible working sets: {len(visible)} faces out of {len(bm.faces)} total')
op._hover_cache = None
bpy.ops.object.mode_set(mode='OBJECT')
print('HOLEY_TOOLS_HIDDEN_GEOMETRY_OK')
