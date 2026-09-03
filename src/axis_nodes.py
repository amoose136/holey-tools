# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generated wrapper around Blender's bundled principal-components asset."""

from pathlib import Path

import bpy

from .constants import AXIS_FACE_ATTRIBUTE, AXIS_NODE_GROUP_NAME


_GROUP_MARKER = "holey_tools_extract_axis"
_GROUP_VERSION = 3
_PRINCIPAL_GROUP_NAME = "Geometry Principal Components"


def _new_node(nodes, node_type, name, location):
    """Create and position one clearly named node."""
    node = nodes.new(node_type)
    node.name = name
    node.label = name
    node.location = location
    return node


def _generated_group():
    """Find the generated wrapper independently of Blender name suffixes."""
    return next(
        (
            group
            for group in bpy.data.node_groups
            if group.bl_idname == 'GeometryNodeTree'
            and group.get(_GROUP_MARKER) is not None
        ),
        None,
    )


def _has_principal_interface(group):
    """Return whether a node group has the bundled asset's required sockets."""
    sockets = {
        (item.name, item.in_out)
        for item in group.interface.items_tree
        if item.item_type == 'SOCKET'
    }
    return {
        ("Geometry", 'INPUT'),
        ("Position", 'INPUT'),
        ("Group Center", 'OUTPUT'),
        ("Principal Components", 'OUTPUT'),
        ("Longest Axis", 'OUTPUT'),
        ("Shortest Axis", 'OUTPUT'),
    } <= sockets


def ensure_principal_components_group():
    """Append and return Blender's bundled Geometry Principal Components asset."""
    for existing in bpy.data.node_groups:
        if (
            existing.bl_idname == 'GeometryNodeTree'
            and existing.name.startswith(_PRINCIPAL_GROUP_NAME)
            and _has_principal_interface(existing)
        ):
            return existing

    resource_root = Path(bpy.utils.resource_path('LOCAL'))
    asset_library = (
        resource_root
        / "datafiles"
        / "assets"
        / "nodes"
        / "principal_components.blend"
    )
    if not asset_library.is_file():
        raise RuntimeError(
            "Blender's bundled Principal Components node asset could not be found"
        )

    with bpy.data.libraries.load(
        str(asset_library), link=False, assets_only=True
    ) as (available, loaded):
        if _PRINCIPAL_GROUP_NAME not in available.node_groups:
            raise RuntimeError(
                "The bundled Geometry Principal Components node group is unavailable"
            )
        loaded.node_groups = [_PRINCIPAL_GROUP_NAME]

    principal = loaded.node_groups[0]
    if principal is None or not _has_principal_interface(principal):
        raise RuntimeError("The bundled Principal Components node group is incompatible")
    return principal


def _set_face_group_default(group, attribute_name):
    """Set the generated group's string input without rebuilding its graph."""
    for item in group.interface.items_tree:
        if (
            item.item_type == 'SOCKET'
            and item.in_out == 'INPUT'
            and item.name == "Face Group"
        ):
            item.default_value = attribute_name
            return


def ensure_axis_node_group(attribute_name=AXIS_FACE_ATTRIBUTE):
    """Return the reusable principal-axis group, creating it when needed.

    The wrapper isolates the named face group and delegates its analysis to
    Blender's bundled Geometry Principal Components asset. It chooses the PCA
    axis whose component differs most from the other two, which selects the
    cylinder direction for both shallow and deep cylindrical regions. A small
    icosphere is joined at the group center as a visible center indicator.
    """
    principal_group = ensure_principal_components_group()
    group = _generated_group()
    if group is not None and group.get(_GROUP_MARKER) == _GROUP_VERSION:
        _set_face_group_default(group, attribute_name)
        return group

    if group is None:
        name = AXIS_NODE_GROUP_NAME
        if bpy.data.node_groups.get(name) is not None:
            name = f"{name} (Generated)"
        group = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    else:
        group.nodes.clear()
        group.interface.clear()

    group[_GROUP_MARKER] = _GROUP_VERSION
    group.description = (
        "Extract a principal axis and center from a named face group, with a "
        "visible center indicator"
    )
    group.is_modifier = True

    geometry_in = group.interface.new_socket(
        name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry'
    )
    geometry_in.description = "Geometry containing the named face group"
    face_group_in = group.interface.new_socket(
        name="Face Group", in_out='INPUT', socket_type='NodeSocketString'
    )
    face_group_in.default_value = attribute_name
    face_group_in.description = "Name of an integer or boolean face-domain attribute"
    indicator_size_in = group.interface.new_socket(
        name="Indicator Size", in_out='INPUT', socket_type='NodeSocketFloat'
    )
    indicator_size_in.default_value = 0.1
    indicator_size_in.min_value = 0.0
    indicator_size_in.subtype = 'DISTANCE'
    indicator_size_in.description = "Radius of the center-point icosphere"

    group.interface.new_socket(
        name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry'
    )
    indicator_out = group.interface.new_socket(
        name="Indicator", in_out='OUTPUT', socket_type='NodeSocketGeometry'
    )
    indicator_out.description = "Icosphere placed at the analyzed group center"
    axis_out = group.interface.new_socket(
        name="Axis", in_out='OUTPUT', socket_type='NodeSocketVector'
    )
    axis_out.description = "Principal cylinder axis; its sign is arbitrary"
    center_out = group.interface.new_socket(
        name="Center", in_out='OUTPUT', socket_type='NodeSocketVector'
    )
    center_out.description = "Center reported by Geometry Principal Components"
    rotation_out = group.interface.new_socket(
        name="Rotation", in_out='OUTPUT', socket_type='NodeSocketRotation'
    )
    rotation_out.description = "Complete principal-components orientation"
    components_out = group.interface.new_socket(
        name="Principal Components", in_out='OUTPUT', socket_type='NodeSocketVector'
    )
    components_out.description = "Variance along the three principal axes"
    valid_out = group.interface.new_socket(
        name="Valid", in_out='OUTPUT', socket_type='NodeSocketBool'
    )
    valid_out.description = "The named face group exists and selects at least one face"

    nodes = group.nodes
    links = group.links
    group_input = _new_node(nodes, 'NodeGroupInput', "Group Input", (-940, 80))
    group_output = _new_node(nodes, 'NodeGroupOutput', "Group Output", (940, 80))

    named = _new_node(
        nodes, 'GeometryNodeInputNamedAttribute', "Read Face Group", (-940, -180)
    )
    named.data_type = 'BOOLEAN'
    separate = _new_node(
        nodes, 'GeometryNodeSeparateGeometry', "Selected Face Geometry", (-700, 180)
    )
    separate.domain = 'FACE'
    links.new(group_input.outputs["Face Group"], named.inputs["Name"])
    links.new(group_input.outputs["Geometry"], separate.inputs["Geometry"])
    links.new(named.outputs["Attribute"], separate.inputs["Selection"])

    position = _new_node(nodes, 'GeometryNodeInputPosition', "Position", (-700, 420))
    principal = _new_node(
        nodes, 'GeometryNodeGroup', "Geometry Principal Components", (-440, 260)
    )
    principal.node_tree = principal_group
    links.new(separate.outputs["Selection"], principal.inputs["Geometry"])
    links.new(position.outputs["Position"], principal.inputs["Position"])

    # A cylindrical wall has two similar radial components and one outlier.
    # Choose Longest for a deep wall and Shortest for a shallow wall according
    # to which adjacent pair of ordered component variances is less similar.
    split_components = _new_node(
        nodes, 'ShaderNodeSeparateXYZ', "Separate Components", (-180, 80)
    )
    longest_gap = _new_node(nodes, 'ShaderNodeMath', "Longest Gap", (20, 120))
    longest_gap.operation = 'SUBTRACT'
    shortest_gap = _new_node(nodes, 'ShaderNodeMath', "Shortest Gap", (20, 20))
    shortest_gap.operation = 'SUBTRACT'
    use_shortest = _new_node(nodes, 'ShaderNodeMath', "Use Shortest Axis", (220, 80))
    use_shortest.operation = 'GREATER_THAN'
    choose_axis = _new_node(nodes, 'GeometryNodeSwitch', "Cylinder Axis", (420, 180))
    choose_axis.input_type = 'VECTOR'
    links.new(
        principal.outputs["Principal Components"], split_components.inputs["Vector"]
    )
    links.new(split_components.outputs["X"], longest_gap.inputs[0])
    links.new(split_components.outputs["Y"], longest_gap.inputs[1])
    links.new(split_components.outputs["Y"], shortest_gap.inputs[0])
    links.new(split_components.outputs["Z"], shortest_gap.inputs[1])
    links.new(shortest_gap.outputs["Value"], use_shortest.inputs[0])
    links.new(longest_gap.outputs["Value"], use_shortest.inputs[1])
    links.new(use_shortest.outputs["Value"], choose_axis.inputs["Switch"])
    links.new(principal.outputs["Longest Axis"], choose_axis.inputs["False"])
    links.new(principal.outputs["Shortest Axis"], choose_axis.inputs["True"])

    domain_size = _new_node(
        nodes, 'GeometryNodeAttributeDomainSize', "Selected Face Count", (-420, -160)
    )
    has_faces = _new_node(nodes, 'ShaderNodeMath', "Has Selected Faces", (-180, -180))
    has_faces.operation = 'GREATER_THAN'
    has_faces.inputs[1].default_value = 0.0
    valid = _new_node(nodes, 'FunctionNodeBooleanMath', "Valid Face Group", (20, -180))
    valid.operation = 'AND'
    links.new(separate.outputs["Selection"], domain_size.inputs["Geometry"])
    links.new(domain_size.outputs["Face Count"], has_faces.inputs[0])
    links.new(named.outputs["Exists"], valid.inputs[0])
    links.new(has_faces.outputs["Value"], valid.inputs[1])

    visible_size = _new_node(
        nodes, 'ShaderNodeMath', "Visible Indicator Size", (180, -480)
    )
    visible_size.operation = 'MULTIPLY'
    links.new(group_input.outputs["Indicator Size"], visible_size.inputs[0])
    links.new(valid.outputs["Boolean"], visible_size.inputs[1])
    sphere = _new_node(nodes, 'GeometryNodeMeshIcoSphere', "Center Indicator", (380, -420))
    sphere.inputs["Subdivisions"].default_value = 1
    links.new(visible_size.outputs["Value"], sphere.inputs["Radius"])
    move_sphere = _new_node(
        nodes, 'GeometryNodeTransform', "Place at Center", (580, -360)
    )
    links.new(sphere.outputs["Mesh"], move_sphere.inputs["Geometry"])
    links.new(principal.outputs["Group Center"], move_sphere.inputs["Translation"])

    join = _new_node(nodes, 'GeometryNodeJoinGeometry', "Geometry + Indicator", (760, 260))
    links.new(group_input.outputs["Geometry"], join.inputs["Geometry"])
    links.new(move_sphere.outputs["Geometry"], join.inputs["Geometry"])

    links.new(join.outputs["Geometry"], group_output.inputs["Geometry"])
    links.new(move_sphere.outputs["Geometry"], group_output.inputs["Indicator"])
    links.new(choose_axis.outputs["Output"], group_output.inputs["Axis"])
    links.new(principal.outputs["Group Center"], group_output.inputs["Center"])
    links.new(principal.outputs["Rotation"], group_output.inputs["Rotation"])
    links.new(
        principal.outputs["Principal Components"],
        group_output.inputs["Principal Components"],
    )
    links.new(valid.outputs["Boolean"], group_output.inputs["Valid"])

    return group
