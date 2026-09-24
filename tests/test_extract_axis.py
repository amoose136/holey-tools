import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.axis_nodes import (  # noqa: E402
    ensure_axis_modifier,
    ensure_axis_node_group,
    ensure_fit_circular_cone_group,
)
from src.constants import AXIS_FACE_ATTRIBUTE  # noqa: E402
from src.extract_axis import assign_axis_face_group  # noqa: E402


def interface_socket(group, name, in_out):
    return next(
        item
        for item in group.interface.items_tree
        if item.item_type == 'SOCKET'
        and item.name == name
        and item.in_out == in_out
    )


def add_store(wrapper, geometry, name, data_type, value):
    store = wrapper.nodes.new('GeometryNodeStoreNamedAttribute')
    store.domain = 'POINT'
    store.data_type = data_type
    store.inputs["Selection"].default_value = True
    store.inputs["Name"].default_value = name
    wrapper.links.new(geometry, store.inputs["Geometry"])
    wrapper.links.new(value, store.inputs["Value"])
    return store.outputs["Geometry"]


def evaluate_axis_group(obj, axis_group, indicator_object=None):
    """Evaluate value outputs by storing them on the modifier result."""
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
    extract.inputs["Indicator Object"].default_value = indicator_object
    wrapper.links.new(group_input.outputs["Geometry"], extract.inputs["Geometry"])

    geometry = extract.outputs["Geometry"]
    geometry = add_store(
        wrapper,
        geometry,
        "test_extracted_axis",
        'FLOAT_VECTOR',
        extract.outputs["Axis"],
    )
    geometry = add_store(
        wrapper,
        geometry,
        "test_axis_valid",
        'BOOLEAN',
        extract.outputs["Valid"],
    )
    geometry = add_store(
        wrapper,
        geometry,
        "test_axis_center",
        'FLOAT_VECTOR',
        extract.outputs["Center"],
    )
    geometry = add_store(
        wrapper,
        geometry,
        "test_axis_radius",
        'FLOAT',
        extract.outputs["Radius"],
    )
    geometry = add_store(
        wrapper,
        geometry,
        "test_axis_slope",
        'FLOAT',
        extract.outputs["Slope"],
    )
    geometry = add_store(
        wrapper,
        geometry,
        "test_axis_half_angle",
        'FLOAT',
        extract.outputs["Half Angle"],
    )
    wrapper.links.new(geometry, group_output.inputs["Geometry"])

    modifier = obj.modifiers.new(name="Extract Axis Probe", type='NODES')
    modifier.node_group = wrapper
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = bpy.data.meshes.new_from_object(evaluated)
    obj.modifiers.remove(modifier)
    return mesh


# Use a cone with several omitted face-group members. This exercises the robust
# authored fit rather than relying on complete loops or regular topology.
bpy.ops.mesh.primitive_cone_add(
    vertices=32,
    radius1=2.0,
    radius2=1.0,
    depth=4.0,
    end_fill_type='NOTHING',
)
obj = bpy.context.active_object
obj.data.transform(Matrix.Translation((3.0, -2.0, 1.0)))
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()
selected_faces = {
    face.index
    for face in bm.faces
    if face.index not in {1, 7, 13, 22}
}
assign_axis_face_group(obj, selected_faces, AXIS_FACE_ATTRIBUTE)
layer = bm.faces.layers.int.get(AXIS_FACE_ATTRIBUTE)
assert layer is not None
assert sum(face[layer] for face in bm.faces) == len(selected_faces)
bpy.ops.object.mode_set(mode='OBJECT')

attribute = obj.data.attributes.get(AXIS_FACE_ATTRIBUTE)
assert attribute is not None
assert attribute.domain == 'FACE'
assert attribute.data_type == 'INT'
assert sum(value.value for value in attribute.data) == len(selected_faces)

fit_group = ensure_fit_circular_cone_group()
fit_dependencies = {
    node.node_tree.name
    for node in fit_group.nodes
    if node.bl_idname == 'GeometryNodeGroup' and node.node_tree is not None
}
assert {
    'Principal Components',
    'Least Squares Plane Intersection',
    'Project To Line',
    'Line Coordinates',
    'Linear Regression',
} <= fit_dependencies

axis_group = ensure_axis_node_group()
assert axis_group is ensure_axis_node_group()
fit_nodes = [
    node
    for node in axis_group.nodes
    if node.bl_idname == 'GeometryNodeGroup' and node.node_tree is fit_group
]
assert len(fit_nodes) == 1
interface = {
    (item.name, item.in_out)
    for item in axis_group.interface.items_tree
    if item.item_type == 'SOCKET'
}
assert {
    ('Geometry', 'INPUT'),
    ('Face Group', 'INPUT'),
    ('Indicator Object', 'INPUT'),
    ('Indicator Size', 'INPUT'),
    ('Geometry', 'OUTPUT'),
    ('Indicator', 'OUTPUT'),
    ('Axis', 'OUTPUT'),
    ('Center', 'OUTPUT'),
    ('Radius', 'OUTPUT'),
    ('Slope', 'OUTPUT'),
    ('Half Angle', 'OUTPUT'),
    ('Valid', 'OUTPUT'),
} <= interface

evaluated_mesh = evaluate_axis_group(obj, axis_group)
valid_attribute = evaluated_mesh.attributes.get("test_axis_valid")
assert valid_attribute is not None
assert valid_attribute.data[0].value
axis = evaluated_mesh.attributes["test_extracted_axis"].data[0].vector
assert abs(axis.x) < 5.0e-4, tuple(axis)
assert abs(axis.y) < 5.0e-4, tuple(axis)
assert abs(abs(axis.z) - 1.0) < 5.0e-4, tuple(axis)
center = evaluated_mesh.attributes["test_axis_center"].data[0].vector
assert abs(center.x - 3.0) < 5.0e-3, tuple(center)
assert abs(center.y + 2.0) < 5.0e-3, tuple(center)
slope = evaluated_mesh.attributes["test_axis_slope"].data[0].value
assert abs(abs(slope) - 0.25) < 1.0e-3, slope
half_angle = evaluated_mesh.attributes["test_axis_half_angle"].data[0].value
assert half_angle > 0.0, half_angle
default_indicator_points = len(evaluated_mesh.vertices) - len(obj.data.vertices)
assert default_indicator_points > 3, default_indicator_points

# The operator-facing helper creates one reusable modifier and writes the
# per-modifier face-group string rather than adding duplicates on every run.
bpy.ops.object.mode_set(mode='EDIT')
modifier = ensure_axis_modifier(obj, "custom_axis_region")
assert obj.mode == 'EDIT'
assert modifier == ensure_axis_modifier(obj, "custom_axis_region")
assert len([candidate for candidate in obj.modifiers if candidate.type == 'NODES']) == 1
face_group_socket = interface_socket(axis_group, "Face Group", 'INPUT')
face_group_input = getattr(modifier.properties.inputs, face_group_socket.identifier)
assert face_group_input.value == "custom_axis_region"
modifier = ensure_axis_modifier(obj, AXIS_FACE_ATTRIBUTE)
bpy.ops.object.mode_set(mode='OBJECT')
face_group_input = getattr(modifier.properties.inputs, face_group_socket.identifier)
assert face_group_input.value == AXIS_FACE_ATTRIBUTE

# Supplying a mesh object replaces the generated arrow. Its local +Z direction
# is aligned to the fitted axis and its geometry is joined to the modifier input.
custom_mesh = bpy.data.meshes.new("Custom Axis Indicator Mesh")
custom_mesh.from_pydata(
    [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.0, 1.0)],
    [],
    [(0, 1, 2)],
)
custom_indicator = bpy.data.objects.new("Custom Axis Indicator", custom_mesh)
bpy.context.collection.objects.link(custom_indicator)
indicator_socket = interface_socket(axis_group, "Indicator Object", 'INPUT')
indicator_input = getattr(modifier.properties.inputs, indicator_socket.identifier)
indicator_input.value = custom_indicator
assert indicator_input.value == custom_indicator, indicator_input.type
assert indicator_input.type == 'VALUE', indicator_input.type
obj.update_tag()
bpy.context.view_layer.update()
assert (
    getattr(modifier.properties.inputs, indicator_socket.identifier).value
    == custom_indicator
)
custom_evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
custom_result = bpy.data.meshes.new_from_object(custom_evaluated)
assert len(custom_result.vertices) == len(obj.data.vertices) + 3, (
    len(custom_result.vertices),
    len(obj.data.vertices),
)

print("extract axis tests passed", tuple(axis), slope)
