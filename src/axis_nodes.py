# SPDX-FileCopyrightText: 2026 Amos Manneschmidt
# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometry Nodes modifier for robust circular-cone axis extraction."""

from pathlib import Path

import bpy

from .constants import AXIS_FACE_ATTRIBUTE, AXIS_NODE_GROUP_NAME


_ASSET_LIBRARY = Path(__file__).with_name("assets") / "holeyToolsAssets.blend"
_FIT_GROUP_NAME = "Fit Circular Cone"
_GROUP_MARKER = "holey_tools_extract_axis"
_GROUP_VERSION = 4
_MODIFIER_NAME = "Extract Axis"


def _new_node(nodes, node_type, name, location):
    """Create and position one clearly named node."""
    node = nodes.new(node_type)
    node.name = name
    node.label = name
    node.location = location
    return node


def _socket(sockets, name):
    """Return a named socket with a useful compatibility error."""
    socket = sockets.get(name)
    if socket is None:
        raise RuntimeError(
            f"Missing socket {name!r}; available sockets are "
            f"{[candidate.name for candidate in sockets]}"
        )
    return socket


def _interface_socket(group, name, in_out):
    """Return a named group-interface socket."""
    return next(
        (
            item
            for item in group.interface.items_tree
            if item.item_type == 'SOCKET'
            and item.name == name
            and item.in_out == in_out
        ),
        None,
    )


def _has_interface(group, required):
    """Return whether a node group exposes every required socket."""
    sockets = {
        (item.name, item.in_out)
        for item in group.interface.items_tree
        if item.item_type == 'SOCKET'
    }
    return required <= sockets


def _has_fit_interface(group):
    return _has_interface(
        group,
        {
            ("Position", 'INPUT'),
            ("Normal", 'INPUT'),
            ("Group ID", 'INPUT'),
            ("Axis Point", 'OUTPUT'),
            ("Axis Direction", 'OUTPUT'),
            ("Radius", 'OUTPUT'),
            ("Slope", 'OUTPUT'),
            ("Half Angle", 'OUTPUT'),
            ("Rank", 'OUTPUT'),
        },
    )


def ensure_fit_circular_cone_group():
    """Append and return the bundled robust circular-cone fitting asset."""
    for existing in bpy.data.node_groups:
        if (
            existing.bl_idname == 'GeometryNodeTree'
            and existing.name.startswith(_FIT_GROUP_NAME)
            and _has_fit_interface(existing)
        ):
            return existing

    if not _ASSET_LIBRARY.is_file():
        raise RuntimeError(
            "The bundled Holey Tools Geometry Nodes asset library could not be found"
        )

    with bpy.data.libraries.load(
        str(_ASSET_LIBRARY), link=False, assets_only=True
    ) as (available, loaded):
        if _FIT_GROUP_NAME not in available.node_groups:
            raise RuntimeError(
                f"The bundled {_FIT_GROUP_NAME!r} node group is unavailable"
            )
        loaded.node_groups = [_FIT_GROUP_NAME]

    fit_group = loaded.node_groups[0]
    if fit_group is None or not _has_fit_interface(fit_group):
        raise RuntimeError(
            f"The bundled {_FIT_GROUP_NAME!r} node group is incompatible"
        )
    return fit_group


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


def _set_face_group_default(group, attribute_name):
    """Set the generated group's string input without rebuilding its graph."""
    socket = _interface_socket(group, "Face Group", 'INPUT')
    if socket is not None:
        socket.default_value = attribute_name


def _sample_field(nodes, links, geometry, value, data_type, name, location):
    """Sample a group-constant field from the first selected mesh point."""
    sample = _new_node(nodes, 'GeometryNodeSampleIndex', name, location)
    sample.data_type = data_type
    sample.domain = 'POINT'
    _socket(sample.inputs, "Index").default_value = 0
    links.new(geometry, _socket(sample.inputs, "Geometry"))
    links.new(value, _socket(sample.inputs, "Value"))
    return _socket(sample.outputs, "Value")


def _build_default_arrow(nodes, links):
    """Build a unit mesh arrow pointing along local positive Z."""
    shaft = _new_node(nodes, 'GeometryNodeMeshCylinder', "Arrow Shaft", (120, -560))
    _socket(shaft.inputs, "Vertices").default_value = 12
    _socket(shaft.inputs, "Radius").default_value = 0.045
    _socket(shaft.inputs, "Depth").default_value = 0.7

    move_shaft = _new_node(
        nodes, 'GeometryNodeTransform', "Place Arrow Shaft", (340, -560)
    )
    _socket(move_shaft.inputs, "Translation").default_value = (0.0, 0.0, 0.35)
    links.new(
        _socket(shaft.outputs, "Mesh"),
        _socket(move_shaft.inputs, "Geometry"),
    )

    head = _new_node(nodes, 'GeometryNodeMeshCone', "Arrow Head", (120, -760))
    _socket(head.inputs, "Vertices").default_value = 12
    _socket(head.inputs, "Radius Top").default_value = 0.0
    _socket(head.inputs, "Radius Bottom").default_value = 0.11
    _socket(head.inputs, "Depth").default_value = 0.3

    move_head = _new_node(
        nodes, 'GeometryNodeTransform', "Place Arrow Head", (340, -760)
    )
    _socket(move_head.inputs, "Translation").default_value = (0.0, 0.0, 0.85)
    links.new(
        _socket(head.outputs, "Mesh"),
        _socket(move_head.inputs, "Geometry"),
    )

    join = _new_node(nodes, 'GeometryNodeJoinGeometry', "Default Arrow", (560, -640))
    links.new(
        _socket(move_shaft.outputs, "Geometry"),
        _socket(join.inputs, "Geometry"),
    )
    links.new(
        _socket(move_head.outputs, "Geometry"),
        _socket(join.inputs, "Geometry"),
    )
    return _socket(join.outputs, "Geometry")


def ensure_axis_node_group(attribute_name=AXIS_FACE_ATTRIBUTE):
    """Return the reusable cone-axis modifier group, creating it when needed.

    The wrapper evaluates the bundled ``Fit Circular Cone`` field group over the
    named face selection. It joins an aligned unit arrow to the modifier input,
    or aligns geometry from an optional object when one is supplied.
    """
    fit_group = ensure_fit_circular_cone_group()
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
    group.description = "Fit a circular cone to a named face group and display its axis"
    group.color_tag = 'GEOMETRY'
    group.is_modifier = True

    geometry_in = group.interface.new_socket(
        name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry'
    )
    geometry_in.description = "Modifier geometry containing the named face group"
    face_group_in = group.interface.new_socket(
        name="Face Group", in_out='INPUT', socket_type='NodeSocketString'
    )
    face_group_in.default_value = attribute_name
    face_group_in.description = "Name of an integer or boolean face-domain attribute"
    indicator_object_in = group.interface.new_socket(
        name="Indicator Object", in_out='INPUT', socket_type='NodeSocketObject'
    )
    indicator_object_in.description = (
        "Optional mesh whose local +Z axis is aligned to the fitted axis"
    )
    indicator_size_in = group.interface.new_socket(
        name="Indicator Size", in_out='INPUT', socket_type='NodeSocketFloat'
    )
    indicator_size_in.default_value = 0.25
    indicator_size_in.min_value = 0.0
    indicator_size_in.subtype = 'DISTANCE'
    indicator_size_in.description = "Uniform scale of the axis indicator"

    group.interface.new_socket(
        name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry'
    )
    indicator_out = group.interface.new_socket(
        name="Indicator", in_out='OUTPUT', socket_type='NodeSocketGeometry'
    )
    indicator_out.description = "Aligned custom geometry or the generated mesh arrow"
    axis_out = group.interface.new_socket(
        name="Axis", in_out='OUTPUT', socket_type='NodeSocketVector'
    )
    axis_out.description = "Normalized fitted cone axis; its sign is arbitrary"
    center_out = group.interface.new_socket(
        name="Center", in_out='OUTPUT', socket_type='NodeSocketVector'
    )
    center_out.description = "A point on the fitted cone axis"
    radius_out = group.interface.new_socket(
        name="Radius", in_out='OUTPUT', socket_type='NodeSocketFloat'
    )
    radius_out.description = "Fitted radius at the axis point"
    slope_out = group.interface.new_socket(
        name="Slope", in_out='OUTPUT', socket_type='NodeSocketFloat'
    )
    slope_out.description = "Change in radius per unit distance along the axis"
    angle_out = group.interface.new_socket(
        name="Half Angle", in_out='OUTPUT', socket_type='NodeSocketFloat'
    )
    angle_out.subtype = 'ANGLE'
    angle_out.description = "Fitted cone half angle"
    valid_out = group.interface.new_socket(
        name="Valid", in_out='OUTPUT', socket_type='NodeSocketBool'
    )
    valid_out.description = "The face group exists and produced a usable cone fit"

    nodes = group.nodes
    links = group.links
    group_input = _new_node(nodes, 'NodeGroupInput', "Group Input", (-1280, 120))
    group_output = _new_node(nodes, 'NodeGroupOutput', "Group Output", (1320, 120))

    named = _new_node(
        nodes, 'GeometryNodeInputNamedAttribute', "Read Face Group", (-1280, -180)
    )
    named.data_type = 'BOOLEAN'
    separate = _new_node(
        nodes, 'GeometryNodeSeparateGeometry', "Selected Face Geometry", (-1040, 220)
    )
    separate.domain = 'FACE'
    links.new(_socket(group_input.outputs, "Face Group"), _socket(named.inputs, "Name"))
    links.new(_socket(group_input.outputs, "Geometry"), _socket(separate.inputs, "Geometry"))
    links.new(_socket(named.outputs, "Attribute"), _socket(separate.inputs, "Selection"))
    selected_geometry = _socket(separate.outputs, "Selection")
    named_exists = _sample_field(
        nodes,
        links,
        _socket(group_input.outputs, "Geometry"),
        _socket(named.outputs, "Exists"),
        'BOOLEAN',
        "Sample Face Group Exists",
        (-1040, -40),
    )

    position = _new_node(nodes, 'GeometryNodeInputPosition', "Position", (-1040, 500))
    normal = _new_node(nodes, 'GeometryNodeInputNormal', "Normal", (-1040, 620))
    fit = _new_node(nodes, 'GeometryNodeGroup', _FIT_GROUP_NAME, (-800, 440))
    fit.node_tree = fit_group
    _socket(fit.inputs, "Group ID").default_value = 0
    links.new(_socket(position.outputs, "Position"), _socket(fit.inputs, "Position"))
    links.new(_socket(normal.outputs, "Normal"), _socket(fit.inputs, "Normal"))

    axis_point = _sample_field(
        nodes, links, selected_geometry, _socket(fit.outputs, "Axis Point"),
        'FLOAT_VECTOR', "Sample Axis Point", (-520, 620),
    )
    sampled_axis = _sample_field(
        nodes, links, selected_geometry, _socket(fit.outputs, "Axis Direction"),
        'FLOAT_VECTOR', "Sample Axis Direction", (-520, 440),
    )
    radius = _sample_field(
        nodes, links, selected_geometry, _socket(fit.outputs, "Radius"),
        'FLOAT', "Sample Radius", (-520, 260),
    )
    slope = _sample_field(
        nodes, links, selected_geometry, _socket(fit.outputs, "Slope"),
        'FLOAT', "Sample Slope", (-520, 80),
    )
    half_angle = _sample_field(
        nodes, links, selected_geometry, _socket(fit.outputs, "Half Angle"),
        'FLOAT', "Sample Half Angle", (-520, -100),
    )
    rank = _sample_field(
        nodes, links, selected_geometry, _socket(fit.outputs, "Rank"),
        'INT', "Sample Fit Rank", (-520, -280),
    )

    normalize_axis = _new_node(
        nodes, 'ShaderNodeVectorMath', "Normalize Axis", (-240, 440)
    )
    normalize_axis.operation = 'NORMALIZE'
    links.new(sampled_axis, _socket(normalize_axis.inputs, "Vector"))
    axis = _socket(normalize_axis.outputs, "Vector")

    domain_size = _new_node(
        nodes, 'GeometryNodeAttributeDomainSize', "Selected Face Count", (-800, -220)
    )
    links.new(selected_geometry, _socket(domain_size.inputs, "Geometry"))
    has_faces = _new_node(nodes, 'FunctionNodeCompare', "Has Selected Faces", (-560, -460))
    has_faces.data_type = 'INT'
    has_faces.operation = 'GREATER_THAN'
    _socket(has_faces.inputs, "B").default_value = 0
    links.new(_socket(domain_size.outputs, "Face Count"), _socket(has_faces.inputs, "A"))
    has_rank = _new_node(nodes, 'FunctionNodeCompare', "Usable Fit Rank", (-280, -420))
    has_rank.data_type = 'INT'
    has_rank.operation = 'GREATER_EQUAL'
    _socket(has_rank.inputs, "B").default_value = 2
    links.new(rank, _socket(has_rank.inputs, "A"))
    axis_length = _new_node(nodes, 'ShaderNodeVectorMath', "Axis Length", (-280, -560))
    axis_length.operation = 'LENGTH'
    links.new(axis, _socket(axis_length.inputs, "Vector"))
    has_axis = _new_node(nodes, 'FunctionNodeCompare', "Has Axis", (-40, -560))
    has_axis.data_type = 'FLOAT'
    has_axis.operation = 'GREATER_THAN'
    _socket(has_axis.inputs, "B").default_value = 1.0e-6
    links.new(_socket(axis_length.outputs, "Value"), _socket(has_axis.inputs, "A"))
    named_and_faces = _new_node(
        nodes, 'FunctionNodeBooleanMath', "Existing Face Group", (-280, -700)
    )
    named_and_faces.operation = 'AND'
    links.new(named_exists, named_and_faces.inputs[0])
    links.new(_socket(has_faces.outputs, "Result"), named_and_faces.inputs[1])
    rank_and_axis = _new_node(
        nodes, 'FunctionNodeBooleanMath', "Usable Cone Fit", (-40, -420)
    )
    rank_and_axis.operation = 'AND'
    links.new(_socket(has_rank.outputs, "Result"), rank_and_axis.inputs[0])
    links.new(_socket(has_axis.outputs, "Result"), rank_and_axis.inputs[1])
    valid = _new_node(nodes, 'FunctionNodeBooleanMath', "Valid Axis", (200, -480))
    valid.operation = 'AND'
    links.new(_socket(named_and_faces.outputs, "Boolean"), valid.inputs[0])
    links.new(_socket(rank_and_axis.outputs, "Boolean"), valid.inputs[1])
    valid_output = _socket(valid.outputs, "Boolean")

    default_arrow = _build_default_arrow(nodes, links)
    object_info = _new_node(nodes, 'GeometryNodeObjectInfo', "Custom Indicator Object", (120, -920))
    object_info.transform_space = 'ORIGINAL'
    _socket(object_info.inputs, "As Instance").default_value = False
    links.new(_socket(group_input.outputs, "Indicator Object"), _socket(object_info.inputs, "Object"))
    custom_size = _new_node(
        nodes, 'GeometryNodeAttributeDomainSize', "Custom Indicator Size", (360, -920)
    )
    links.new(_socket(object_info.outputs, "Geometry"), _socket(custom_size.inputs, "Geometry"))
    has_custom = _new_node(
        nodes, 'FunctionNodeCompare', "Has Custom Indicator", (580, -920)
    )
    has_custom.data_type = 'INT'
    has_custom.operation = 'GREATER_THAN'
    _socket(has_custom.inputs, "B").default_value = 0
    links.new(_socket(custom_size.outputs, "Point Count"), _socket(has_custom.inputs, "A"))
    choose_indicator = _new_node(nodes, 'GeometryNodeSwitch', "Choose Indicator", (780, -720))
    choose_indicator.input_type = 'GEOMETRY'
    links.new(_socket(has_custom.outputs, "Result"), _socket(choose_indicator.inputs, "Switch"))
    links.new(default_arrow, _socket(choose_indicator.inputs, "False"))
    links.new(_socket(object_info.outputs, "Geometry"), _socket(choose_indicator.inputs, "True"))

    indicator_scale = _new_node(nodes, 'ShaderNodeCombineXYZ', "Indicator Scale", (560, -560))
    for component in ("X", "Y", "Z"):
        links.new(
            _socket(group_input.outputs, "Indicator Size"),
            _socket(indicator_scale.inputs, component),
        )
    align = _new_node(
        nodes, 'FunctionNodeAlignRotationToVector', "Align Indicator", (560, -360)
    )
    align.axis = 'Z'
    _socket(align.inputs, "Factor").default_value = 1.0
    links.new(axis, _socket(align.inputs, "Vector"))
    transform_indicator = _new_node(
        nodes, 'GeometryNodeTransform', "Place Axis Indicator", (1000, -540)
    )
    links.new(_socket(choose_indicator.outputs, "Output"), _socket(transform_indicator.inputs, "Geometry"))
    links.new(axis_point, _socket(transform_indicator.inputs, "Translation"))
    links.new(_socket(align.outputs, "Rotation"), _socket(transform_indicator.inputs, "Rotation"))
    links.new(_socket(indicator_scale.outputs, "Vector"), _socket(transform_indicator.inputs, "Scale"))
    valid_indicator = _new_node(
        nodes, 'GeometryNodeSwitch', "Hide Invalid Indicator", (1000, -260)
    )
    valid_indicator.input_type = 'GEOMETRY'
    links.new(valid_output, _socket(valid_indicator.inputs, "Switch"))
    links.new(_socket(transform_indicator.outputs, "Geometry"), _socket(valid_indicator.inputs, "True"))
    indicator = _socket(valid_indicator.outputs, "Output")

    join = _new_node(nodes, 'GeometryNodeJoinGeometry', "Geometry + Axis", (1120, 180))
    links.new(_socket(group_input.outputs, "Geometry"), _socket(join.inputs, "Geometry"))
    links.new(indicator, _socket(join.inputs, "Geometry"))

    links.new(_socket(join.outputs, "Geometry"), _socket(group_output.inputs, "Geometry"))
    links.new(indicator, _socket(group_output.inputs, "Indicator"))
    links.new(axis, _socket(group_output.inputs, "Axis"))
    links.new(axis_point, _socket(group_output.inputs, "Center"))
    links.new(radius, _socket(group_output.inputs, "Radius"))
    links.new(slope, _socket(group_output.inputs, "Slope"))
    links.new(half_angle, _socket(group_output.inputs, "Half Angle"))
    links.new(valid_output, _socket(group_output.inputs, "Valid"))

    return group


def ensure_axis_modifier(obj, attribute_name=AXIS_FACE_ATTRIBUTE):
    """Create or update the generated Extract Axis modifier on ``obj``."""
    restore_edit_mode = obj.mode == 'EDIT'
    if restore_edit_mode:
        if bpy.context.active_object != obj:
            raise RuntimeError("Extract Axis needs the edited mesh to be active")
        bpy.ops.object.mode_set(mode='OBJECT')

    try:
        node_group = ensure_axis_node_group(attribute_name)
        modifier = next(
            (
                candidate
                for candidate in obj.modifiers
                if candidate.type == 'NODES'
                and candidate.node_group is not None
                and candidate.node_group.get(_GROUP_MARKER) is not None
            ),
            None,
        )
        if modifier is None:
            modifier = obj.modifiers.new(name=_MODIFIER_NAME, type='NODES')
        modifier.node_group = node_group

        face_group_socket = _interface_socket(node_group, "Face Group", 'INPUT')
        if face_group_socket is not None:
            modifier_input = getattr(
                modifier.properties.inputs,
                face_group_socket.identifier,
            )
            modifier_input.value = attribute_name
        obj.update_tag()
        return modifier
    finally:
        if restore_edit_mode:
            bpy.ops.object.mode_set(mode='EDIT')
