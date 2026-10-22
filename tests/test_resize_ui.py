"""Sidebar resize defaults use the same modes as the operator and survive registration."""

import sys
from pathlib import Path
from types import SimpleNamespace

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src
from src.ui import VIEW3D_PT_holey_tools as Panel


class Layout:
    def __init__(self, parent=None):
        self.properties = parent.properties if parent is not None else []
        self.operators = parent.operators if parent is not None else {}
        self.labels = parent.labels if parent is not None else []
        self.alignment = 'EXPAND'

    def split(self, **_kwargs):
        return Layout(self)

    row = column = split

    def label(self, text="", **_kwargs):
        self.labels.append((text, self.alignment))

    def separator(self):
        pass

    def prop(self, data, name, **_kwargs):
        self.properties.append(name)

    def operator(self, name, **_kwargs):
        operator = SimpleNamespace()
        self.operators[name] = operator
        return operator


src.register()
try:
    scene = bpy.context.scene
    assert scene.holey_tools_resize_method == 'AUTO'
    scene_items = scene.bl_rna.properties['holey_tools_resize_method'].enum_items
    operator_items = bpy.ops.mesh.holey_tools_resize_hole.get_rna_type().properties['resize_method'].enum_items
    assert [(item.identifier, item.name) for item in scene_items] == [
        ('AUTO', 'Auto'), ('SCALE', 'Scaling'), ('BOOLEAN', 'Boolean')]
    assert [(item.identifier, item.name) for item in scene_items] == [
        (item.identifier, item.name) for item in operator_items]
    for mode in ('AUTO', 'SCALE', 'BOOLEAN'):
        scene.holey_tools_resize_method = mode
        layout = Layout()
        Panel.draw(SimpleNamespace(layout=layout), bpy.context)
        assert layout.properties[:2] == ['holey_tools_diameter_estimator', 'holey_tools_resize_method']
        assert layout.labels[:2] == [('Diameter Estimate:', 'RIGHT'), ('Mode:', 'RIGHT')]
        operator = layout.operators['mesh.holey_tools_resize_hole']
        assert operator.resize_method == mode
        assert operator.diameter_estimator == scene.holey_tools_diameter_estimator
        assert operator.flood_angle == scene.holey_tools_flood_angle
finally:
    src.unregister()

assert not hasattr(bpy.types.Scene, 'holey_tools_resize_method')
print('HOLEY_TOOLS_RESIZE_UI_OK')
