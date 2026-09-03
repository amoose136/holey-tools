import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.axis_nodes import (  # noqa: E402
    ensure_axis_node_group,
    ensure_principal_components_group,
)
from src.constants import AXIS_FACE_ATTRIBUTE  # noqa: E402
from src.extract_axis import assign_axis_face_group  # noqa: E402


bpy.ops.mesh.primitive_cylinder_add(
    vertices=24,
    radius=2.0,
    depth=4.0,
    end_fill_type='NOTHING',
)
obj = bpy.context.active_object
obj.data.transform(Matrix.Translation((3.0, -2.0, 1.0)))
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()
all_faces = {face.index for face in bm.faces}
assign_axis_face_group(obj, all_faces, AXIS_FACE_ATTRIBUTE)

layer = bm.faces.layers.int.get(AXIS_FACE_ATTRIBUTE)
assert layer is not None
assert all(face[layer] == 1 for face in bm.faces)
bpy.ops.object.mode_set(mode='OBJECT')

attribute = obj.data.attributes.get(AXIS_FACE_ATTRIBUTE)
assert attribute is not None
assert attribute.domain == 'FACE'
assert attribute.data_type == 'INT'

axis_group = ensure_axis_node_group()
assert axis_group is ensure_axis_node_group()
principal_group = ensure_principal_components_group()
principal_nodes = [
    node
    for node in axis_group.nodes
    if node.bl_idname == 'GeometryNodeGroup'
    and node.node_tree is principal_group
]
assert len(principal_nodes) == 1
assert not any(node.bl_idname == 'FunctionNodeMatrixSVD' for node in axis_group.nodes)
interface = {
    (item.name, item.in_out)
    for item in axis_group.interface.items_tree
    if item.item_type == 'SOCKET'
}
assert {
    ('Geometry', 'INPUT'),
    ('Face Group', 'INPUT'),
    ('Indicator Size', 'INPUT'),
    ('Geometry', 'OUTPUT'),
    ('Indicator', 'OUTPUT'),
    ('Axis', 'OUTPUT'),
    ('Center', 'OUTPUT'),
    ('Rotation', 'OUTPUT'),
    ('Principal Components', 'OUTPUT'),
    ('Valid', 'OUTPUT'),
} <= interface

# Evaluate the generated group through a tiny modifier wrapper. Storing its value
# outputs as named point attributes makes them inspectable from the evaluated mesh
# in a headless Blender process.
wrapper = bpy.data.node_groups.new("Extract Axis Test Wrapper", 'GeometryNodeTree')
wrapper.is_modifier = True
wrapper.interface.new_socket(
    name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry'
)
wrapper.interface.new_socket(
    name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry'
)
group_input = wrapper.nodes.new('NodeGroupInput')
group_output = wrapper.nodes.new('NodeGroupOutput')
extract = wrapper.nodes.new('GeometryNodeGroup')
extract.node_tree = axis_group
extract.inputs["Face Group"].default_value = AXIS_FACE_ATTRIBUTE
extract.inputs["Indicator Size"].default_value = 0.25
store = wrapper.nodes.new('GeometryNodeStoreNamedAttribute')
store.domain = 'POINT'
store.data_type = 'FLOAT_VECTOR'
store.inputs["Selection"].default_value = True
store.inputs["Name"].default_value = "test_extracted_axis"
store_valid = wrapper.nodes.new('GeometryNodeStoreNamedAttribute')
store_valid.domain = 'POINT'
store_valid.data_type = 'BOOLEAN'
store_valid.inputs["Selection"].default_value = True
store_valid.inputs["Name"].default_value = "test_axis_valid"
store_center = wrapper.nodes.new('GeometryNodeStoreNamedAttribute')
store_center.domain = 'POINT'
store_center.data_type = 'FLOAT_VECTOR'
store_center.inputs["Selection"].default_value = True
store_center.inputs["Name"].default_value = "test_axis_center"
wrapper.links.new(group_input.outputs["Geometry"], extract.inputs["Geometry"])
wrapper.links.new(extract.outputs["Geometry"], store.inputs["Geometry"])
wrapper.links.new(extract.outputs["Axis"], store.inputs["Value"])
wrapper.links.new(store.outputs["Geometry"], store_valid.inputs["Geometry"])
wrapper.links.new(extract.outputs["Valid"], store_valid.inputs["Value"])
wrapper.links.new(store_valid.outputs["Geometry"], store_center.inputs["Geometry"])
wrapper.links.new(extract.outputs["Center"], store_center.inputs["Value"])
wrapper.links.new(store_center.outputs["Geometry"], group_output.inputs["Geometry"])

modifier = obj.modifiers.new(name="Extract Axis Test", type='NODES')
modifier.node_group = wrapper
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = obj.evaluated_get(depsgraph)
evaluated_mesh = bpy.data.meshes.new_from_object(evaluated)
valid_attribute = evaluated_mesh.attributes.get("test_axis_valid")
assert valid_attribute is not None
assert valid_attribute.data[0].value
center_attribute = evaluated_mesh.attributes.get("test_axis_center")
assert center_attribute is not None
center = center_attribute.data[0].vector
center_error = max(
    abs(center[index] - value)
    for index, value in enumerate((3.0, -2.0, 1.0))
)
assert center_error < 1.0e-5, tuple(center)
# The joined center indicator adds its icosphere vertices to the evaluated mesh.
indicator_vertex_count = len(evaluated_mesh.vertices) - len(obj.data.vertices)
axis_attribute = evaluated_mesh.attributes.get("test_extracted_axis")
assert axis_attribute is not None
axis = axis_attribute.data[0].vector
assert abs(axis.x) < 1.0e-5, tuple(axis)
assert abs(axis.y) < 1.0e-5, tuple(axis)
assert abs(abs(axis.z) - 1.0) < 1.0e-5, tuple(axis)
assert indicator_vertex_count > 0, indicator_vertex_count

# A shallow cylinder makes the axial direction the shortest rather than longest
# component. The wrapper's outlier selection must still return the Z axis.
bpy.ops.mesh.primitive_cylinder_add(
    vertices=24,
    radius=2.0,
    depth=0.5,
    end_fill_type='NOTHING',
)
shallow = bpy.context.active_object
assign_axis_face_group(
    shallow,
    {polygon.index for polygon in shallow.data.polygons},
    AXIS_FACE_ATTRIBUTE,
)
shallow_modifier = shallow.modifiers.new(name="Extract Axis Test", type='NODES')
shallow_modifier.node_group = wrapper
shallow_evaluated = shallow.evaluated_get(bpy.context.evaluated_depsgraph_get())
shallow_mesh = bpy.data.meshes.new_from_object(shallow_evaluated)
shallow_axis = shallow_mesh.attributes["test_extracted_axis"].data[0].vector
assert abs(shallow_axis.x) < 1.0e-5, tuple(shallow_axis)
assert abs(shallow_axis.y) < 1.0e-5, tuple(shallow_axis)
assert abs(abs(shallow_axis.z) - 1.0) < 1.0e-5, tuple(shallow_axis)

print("extract axis tests passed", tuple(axis))
