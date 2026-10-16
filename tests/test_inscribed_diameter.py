"""Blind polygonal bores retain their usable diameter across cutter resolutions."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.operator import MESH_OT_holey_tools_resize_hole as Resize, _PersistentPreview
from src.resize_dimensions import _update_diameter_estimator


def fixture(count):
    if bpy.context.object and bpy.context.object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    vertices = [(r * math.cos(math.tau * i / count), r * math.sin(math.tau * i / count), z)
                for r, z in ((4, 1), (1, 1), (1, -1), (4, -2)) for i in range(count)]
    faces = []
    for a, b in ((0, 1), (1, 2), (3, 0)):
        for i in range(count):
            j = (i + 1) % count
            faces.append((a * count + i, a * count + j, b * count + j, b * count + i))
    faces.extend((tuple(range(2 * count, 3 * count)), tuple(reversed(range(3 * count, 4 * count)))))
    mesh = bpy.data.meshes.new('Blind polygonal bore')
    mesh.from_pydata(vertices, (), faces)
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    wall = list(bm.faces[count:2 * count])
    op = SimpleNamespace(_object=obj, _locked_indices={f.index for f in wall},
                         diameter_estimator='AVERAGE', resize_method='AUTO',
                         fit_valid=False, scale=1.0, scaling_final_simplify=False)
    op._world_normal = lambda face: Resize._world_normal(op, face)
    op._target_scale = lambda: Resize._target_scale(op)
    op._target_depth_scale = lambda: Resize._target_depth_scale(op)
    op._preview_radius_at = lambda z: Resize._preview_radius_at(op, z)
    op._tag_redraw = lambda _context: None
    op.report = lambda _level, message: (_ for _ in ()).throw(AssertionError(message))
    op._fit = Resize._fit_cylinder(op, wall)
    op._end_planes = Resize._fit_boundary_planes(op, wall, op._fit['axis'], op._fit)
    Resize._build_resize_projection_state(op, bm, wall)
    Resize._capture_deformation_state(op, bpy.context, bm, {v for f in wall for v in f.verts})
    Resize._store_fit_for_redo(op)
    owner = object.__new__(_PersistentPreview)
    owner._draw_handle = None
    owner._source_object_name = obj.name
    (owner._source_mesh_name, owner._source_mesh, owner._source_bmesh,
     owner._source_materials) = _PersistentPreview._capture_canonical_source(op)
    return obj, op, owner


for source_count in (3, 4, 8, 32):
    obj, op, owner = fixture(source_count)
    inside = math.cos(math.pi / source_count)
    assert op.fit_angular_segments == source_count
    assert op.depth_valid and not op.fit_is_stud and not op.fit_is_cone
    assert abs(op.diameter - 2 * inside) < 1.0e-6
    assert abs(Resize._target_scale(op) - 1) < 1.0e-6
    original = [v.co.copy() for v in bmesh.from_edit_mesh(obj.data).verts]
    assert Resize.execute(op, bpy.context) == {'FINISHED'}
    assert all(v.co == co for v, co in zip(bmesh.from_edit_mesh(obj.data).verts, original))
    for estimator in ('MINIMUM', 'AVERAGE', 'MAXIMUM'):
        op.diameter_estimator = estimator
        _update_diameter_estimator(op, None)
        assert abs(op.diameter - 2 * inside) < 1.0e-6
    op.diameter = 2.4 * inside
    assert Resize.execute(op, bpy.context) == {'FINISHED'}
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    assert len(bm.verts) == 4 * source_count
    assert max(abs(math.hypot(v.co.x, v.co.y) - 1.2)
               for v in bm.verts[source_count:3 * source_count]) < 1.0e-6
    for target_count in (3, 4, 8, 32, 64):
        assert owner.restore_source(obj)
        op.resize_method = 'BOOLEAN'
        op.boolean_segments = target_count
        op.diameter = 1.2 * 2 * inside
        assert Resize.execute(op, bpy.context) == {'FINISHED'}
        assert op.fit_angular_segments == source_count
        bm = bmesh.from_edit_mesh(obj.data)
        bm.normal_update()
        assert all(edge.is_manifold for edge in bm.edges), (source_count, target_count)
        # Measure the resulting flat walls, not the distances to corner verts.
        wall = [f for f in bm.faces if abs(f.normal.z) < 1.0e-5
                and f.normal.dot(Vector((f.calc_center_median().x, f.calc_center_median().y, 0))) < 0]
        assert len(wall) == target_count, (source_count, target_count, len(wall))
        distances = [abs(f.normal.dot(f.verts[0].co - Vector((0, 0, f.verts[0].co.z)))) for f in wall]
        assert max(abs(d - op.diameter / 2) for d in distances) < 1.0e-5
        assert abs(Resize._preview_radius_at(op, 0) - op.diameter / 2) < 1.0e-6
        Resize._build_cylinder_preview(op, bpy.context)
        assert len(op._line_coords) == 256
        for point in (p for line in op._line_coords for p in line):
            offset = point - op._fit['center']
            radial = offset - op._fit['axis'] * offset.dot(op._fit['axis'])
            assert abs(radial.length - op.diameter / 2) < 1.0e-5
        # A live-preview resolution override needs its own correction, not the
        # final segment count currently displayed in the redo panel.
        cutter, _ = Resize._boolean_cutter_geometry(op, obj, segment_override=16)
        expected_radius = op.diameter / (2 * math.cos(math.pi / 16))
        assert max(abs(math.hypot(v.x, v.y) - expected_radius) for v in cutter) < 1.0e-5
        # Refit as a genuinely new operation: the same inside diameter must be
        # recovered from its new polygonization, without applying it twice.
        probe = SimpleNamespace(_object=obj, diameter_estimator='AVERAGE')
        probe._world_normal = lambda face: Resize._world_normal(probe, face)
        probe._fit = Resize._fit_cylinder(probe, wall)
        recovered_count = Resize._estimate_angular_segments(probe, wall)
        assert recovered_count == target_count
        assert not probe._fit['is_cone']
        assert abs(probe._fit['inradius_estimates']['AVERAGE'] - op.diameter / 2) < 1.0e-5
        probe._locked_indices = {f.index for f in wall}
        probe._end_planes = Resize._fit_boundary_planes(probe, wall, probe._fit['axis'], probe._fit)
        Resize._build_resize_projection_state(probe, bm, wall)
        Resize._store_fit_for_redo(probe)
        assert abs(probe.diameter - op.diameter) < 2.0e-5
        assert not probe._resize_requires_boolean
    owner.remove()

print('HOLEY_TOOLS_INSCRIBED_DIAMETER_OK')
