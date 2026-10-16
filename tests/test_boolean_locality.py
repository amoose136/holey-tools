import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bmesh
from mathutils import Matrix, Vector


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import GeometryMixin  # noqa: E402


fit = {
    "axis": Vector((0.0, 0.0, 1.0)),
    "basis_x": Vector((1.0, 0.0, 0.0)),
    "basis_y": Vector((0.0, 1.0, 0.0)),
    "center": Vector((0.0, 0.0, 0.0)),
    "radius": 1.0,
    "slope": 0.0,
    "z_min": -1.0,
    "z_max": 1.0,
    "is_cone": False,
}
operator = SimpleNamespace(
    _fit=fit,
    _depth_fit=None,
    _resize_surfaces_by_vertex={},
    diameter=2.0,
    angle_valid=False,
    boolean_segments=8,
    _resize_angular_segments=0,
    _locked_indices=set(),
)
remote_object = SimpleNamespace(
    matrix_world=Matrix.Identity(4),
    data=SimpleNamespace(vertices=(
        SimpleNamespace(co=Vector((20.0, 0.0, -100.0))),
        SimpleNamespace(co=Vector((20.0, 0.0, 100.0))),
    )),
)
cutter_vertices, _cutter_faces = GeometryMixin._boolean_cutter_geometry(
    operator, remote_object
)
cutter_coordinates = [vertex.z for vertex in cutter_vertices]
assert min(cutter_coordinates) > -2.0, min(cutter_coordinates)
assert max(cutter_coordinates) < 2.0, max(cutter_coordinates)


bm = bmesh.new()
ring = [
    bm.verts.new((
        math.cos(math.tau * index / 8),
        math.sin(math.tau * index / 8),
        0.0,
    ))
    for index in range(8)
]
outer = [
    bm.verts.new((
        10.0 * math.cos(math.tau * (index + 0.5) / 8),
        10.0 * math.sin(math.tau * (index + 0.5) / 8),
        0.0,
    ))
    for index in range(8)
]
support_faces = []
for index in range(8):
    following = (index + 1) % 8
    support_faces.append(bm.faces.new((
        ring[index], outer[index], ring[following]
    )))
bm.normal_update()
loop_edges = [
    bm.edges.get((ring[index], ring[(index + 1) % 8]))
    for index in range(8)
]
surface = {
    "exterior": True,
    "closed": True,
    "hypothesis": {
        "method": "PLANES",
        "planes": ({
            "point": Vector(),
            "normal": Vector((0.0, 0.0, 1.0)),
        },),
    },
    "loop_edges": loop_edges,
    "loop_verts": ring,
    "loop_world_points": [vert.co.copy() for vert in ring],
    "support_faces": support_faces,
}
operator._object = SimpleNamespace(matrix_world=Matrix.Identity(4))
operator._resize_surfaces_by_vertex = {
    vert.index: surface for vert in ring
}
operator.diameter = 10.0
assert GeometryMixin._enlargement_stretches_sparse_planar_support(operator)
operator.diameter = 2.5
assert not GeometryMixin._enlargement_stretches_sparse_planar_support(operator)
bm.free()

print("HOLEY_TOOLS_BOOLEAN_LOCALITY_OK")
