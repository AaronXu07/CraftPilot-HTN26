import numpy as np
import pytest

from copilot.engine.shapes import SHAPE_TYPES, shape_local_bbox
from copilot.engine.solids import make_solid, polygon_sdf
from copilot.engine.transform import rot_y

SHAPES = {
    "box": {"type": "box", "size": [4, 6, 2]},
    "cylinder": {"type": "cylinder", "radius": 3, "height": 10},
    "cylinder_x": {"type": "cylinder", "radius": 3, "height": 10, "axis": "x"},
    "cylinder_z": {"type": "cylinder", "radius": 2, "height": 6, "axis": "z"},
    "frustum": {"type": "cylinder", "radius": 3, "height": 10, "radius_top": 1},
    "sphere": {"type": "sphere", "radius": 5},
    "half_sphere": {"type": "sphere", "radius": 5, "half": True},
    "ellipsoid": {"type": "ellipsoid", "radii": [4, 2, 3]},
    "cone": {"type": "cone", "radius": 6, "height": 8},
    "cone_frustum": {"type": "cone", "radius": 6, "height": 8, "radius_top": 2},
    "pyramid": {"type": "pyramid", "base": [10, 8], "height": 6},
    "pyramid_frustum": {"type": "pyramid", "base": [10, 8], "height": 6, "top": [4, 2]},
    "wedge": {"type": "wedge", "size": [10, 5, 6], "slope_axis": "x"},
    "wedge_neg": {"type": "wedge", "size": [10, 5, 6], "slope_axis": "-z"},
    "prism": {"type": "prism", "sides": 6, "radius": 5, "height": 4},
    "torus": {"type": "torus", "major": 6, "minor": 1.5},
    "torus_x": {"type": "torus", "major": 6, "minor": 1.5, "axis": "x"},
    "capsule": {"type": "capsule", "radius": 2, "height": 10},
    "extrude": {"type": "extrude", "profile": [[-5, -5], [5, -5], [5, 0], [0, 0], [0, 5], [-5, 5]], "height": 7},
    "revolve": {"type": "revolve", "profile": [[0, 0], [4, 0], [4, 3], [1, 6], [0, 6]]},
    "sweep": {"type": "sweep", "radius": 1, "path": [[0, 0, 0], [10, 0, 0], [10, 0, 10]]},
    "plane_cut": {"type": "plane_cut", "normal": [0, 1, 0], "offset": 3},
    "block": {"type": "block", "state": "minecraft:lantern"},
    "line": {"type": "line", "from": [0, 0, 0], "to": [10, 10, 0], "thickness": 2},
}

INSIDE = {
    "box": [[0, 3, 0], [1.9, 0.1, 0.9]],
    "cylinder": [[0, 5, 0], [2.9, 9.9, 0]],
    "cylinder_x": [[0, 3, 0], [4.9, 0.2, 0]],
    "cylinder_z": [[0, 2, 2.9], [1.9, 2, 0]],
    "frustum": [[0, 5, 0], [0.9, 9.9, 0], [2.9, 0.1, 0]],
    "sphere": [[0, 0, 0], [4.9, 0, 0]],
    "half_sphere": [[0, 2, 0], [0, 4.9, 0]],
    "ellipsoid": [[0, 0, 0], [3.9, 0, 0], [0, 1.9, 0]],
    "cone": [[0, 1, 0], [5.5, 0.1, 0], [0, 7.9, 0]],
    "cone_frustum": [[0, 7.9, 0], [1.9, 7.9, 0]],
    "pyramid": [[0, 1, 0], [4.9, 0.05, 0], [0, 5.9, 0]],
    "pyramid_frustum": [[0, 5.9, 0], [1.9, 5.9, 0.9]],
    "wedge": [[4.9, 4.9, 0], [-4.9, 0.02, 0], [0, 2.4, 2.9]],
    "wedge_neg": [[0, 4.9, -2.9], [0, 0.02, 2.9]],
    "prism": [[0, 2, 0], [4.3, 2, 0], [0, 2, 4.3]],
    "torus": [[6, 0, 0], [0, 1.4, 6]],
    "torus_x": [[0, 6, 0], [1.4, 0, 6]],
    "capsule": [[0, 0, 0], [0, 4.9, 0], [1.9, 0, 0]],
    "extrude": [[-2, 3, 2], [2, 3, -2], [4.9, 6.9, -4.9]],
    "revolve": [[0, 3, 0], [3.9, 1, 0], [0, 1, 3.9]],
    "sweep": [[5, 0, 0], [10, 0, 5], [10, 0.9, 10]],
    "plane_cut": [[0, 0, 0], [100, 2.9, -50]],
    "block": [[0.5, 0.5, 0.5]],
    "line": [[5, 5, 0], [5, 5, 0.9]],
}

OUTSIDE = {
    "box": [[2.1, 3, 0], [0, 6.1, 0], [0, -0.1, 0]],
    "cylinder": [[3.1, 5, 0], [0, 10.1, 0], [2.5, 5, 2.5]],
    "cylinder_x": [[5.1, 3, 0], [0, 6.1, 0], [0, -0.1, 0]],
    "cylinder_z": [[0, 2, 3.1], [2.1, 2, 0]],
    "frustum": [[2, 9.9, 0], [3.1, 0.1, 0]],
    "sphere": [[5.1, 0, 0]],
    "half_sphere": [[0, -0.1, 0], [0, 5.1, 0]],
    "ellipsoid": [[4.1, 0, 0], [0, 2.1, 0]],
    "cone": [[5.9, 7, 0], [0, 8.1, 0], [3.5, 4.5, 0]],
    "cone_frustum": [[2.5, 7.9, 0], [6.1, 0, 0]],
    "pyramid": [[4.9, 5.9, 0], [0, 6.1, 0], [5.1, 0, 0]],
    "pyramid_frustum": [[2.5, 5.9, 0], [0, 6.1, 0]],
    "wedge": [[-4.9, 4.9, 0], [0, 5.1, 0], [5.1, 1, 0]],
    "wedge_neg": [[0, 4.9, 2.9], [0, 5.1, 0]],
    "prism": [[5.1, 2, 0], [4.5, 2, 0.1], [0, 4.1, 0]],
    "torus": [[0, 0, 0], [6, 1.6, 0], [7.6, 0, 0]],
    "torus_x": [[0, 0, 0], [1.6, 6, 0]],
    "capsule": [[2.1, 0, 0], [0, 5.1, 0]],
    "extrude": [[2, 3, 2], [0, 7.1, -2], [-5.1, 3, 0]],
    "revolve": [[4.1, 1, 0], [3, 5, 0], [0, 6.1, 0]],
    "sweep": [[5, 1.1, 0], [0, 0, 5]],
    "plane_cut": [[0, 3.1, 0]],
    "block": [[1.5, 0.5, 0.5], [0.5, -0.5, 0.5]],
    "line": [[5, 5, 1.1], [0, 10, 0], [11, 11, 0]],
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_inside_outside(name):
    s = make_solid(SHAPES[name])
    ins = np.array(INSIDE[name], dtype=float)
    out = np.array(OUTSIDE[name], dtype=float)
    d_in = s.sdf(ins)
    d_out = s.sdf(out)
    assert np.all(d_in <= 0.0), f"{name}: inside points got {d_in}"
    assert np.all(d_out > 0.0), f"{name}: outside points got {d_out}"


@pytest.mark.parametrize("name", list(SHAPES))
def test_bbox_matches_shapes(name):
    s = make_solid(SHAPES[name])
    lo, hi = s.local_bbox()
    lo2, hi2 = shape_local_bbox(s.shape)
    assert np.allclose(lo, lo2) and np.allclose(hi, hi2)


def test_all_shape_types_covered():
    assert set(s["type"] for s in SHAPES.values()) == set(SHAPE_TYPES)


def test_bbox_contains_shape():
    """Points well outside the bbox are outside; the bbox is never violated by interior samples."""
    rng = np.random.default_rng(0)
    for name, sh in SHAPES.items():
        if name == "plane_cut":
            continue
        s = make_solid(sh)
        lo, hi = s.local_bbox()
        pts = rng.uniform(lo - 3, hi + 3, size=(3000, 3))
        d = s.sdf(pts)
        inside = pts[d <= 0]
        assert np.all(inside >= lo - 1e-6) and np.all(inside <= hi + 1e-6), name


def test_rotation_invariance_sphere_and_cylinder():
    """Rotating query points about y leaves y-symmetric solids unchanged."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-8, 8, size=(2000, 3))
    for sh in ({"type": "sphere", "radius": 5}, {"type": "cylinder", "radius": 3, "height": 6}, {"type": "cone", "radius": 4, "height": 5}, {"type": "torus", "major": 5, "minor": 1}):
        s = make_solid(sh)
        d0 = s.sdf(pts)
        d1 = s.sdf(pts @ rot_y(37.0).T)
        assert np.allclose(d0, d1, atol=1e-6), sh


def test_distance_accuracy():
    """Box and sphere distances are exact; cone/pyramid within 0.1 block on the axis."""
    b = make_solid({"type": "box", "size": [4, 4, 4]})
    assert np.isclose(b.sdf(np.array([[5, 2, 0.0]]))[0], 3.0)
    assert np.isclose(b.sdf(np.array([[0, 2, 0.0]]))[0], -2.0)
    s = make_solid({"type": "sphere", "radius": 3})
    assert np.isclose(s.sdf(np.array([[0, 5, 0.0]]))[0], 2.0)
    c = make_solid({"type": "cone", "radius": 6, "height": 8})
    assert abs(c.sdf(np.array([[0, 9, 0.0]]))[0] - 1.0) < 0.1
    p = make_solid({"type": "pyramid", "base": [8, 8], "height": 4})
    assert abs(p.sdf(np.array([[0, 5, 0.0]]))[0] - 1.0) < 0.1


def test_polygon_sdf_sign_and_distance():
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    d = polygon_sdf(np.array([[0, 0], [2, 0], [0, 3.0], [1.5, 1.5]]), square)
    assert d[0] < 0 and np.isclose(d[1], 1.0) and np.isclose(d[2], 2.0) and np.isclose(d[3], np.hypot(0.5, 0.5))


def test_shrunk_rounding():
    b = make_solid({"type": "box", "size": [6, 6, 6]})
    r = b.shrunk(1.0)
    assert r is not None
    lo, hi = r.local_bbox()
    assert np.allclose(lo, [-2, 1, -2]) and np.allclose(hi, [2, 5, 2])
    corner = np.array([[3, 6, 3.0]])
    assert b.sdf(corner)[0] <= 1e-9
    assert r.sdf(corner)[0] - 1.0 > 0.3  # rounded box excludes the sharp corner
    assert make_solid({"type": "block", "state": "minecraft:stone"}).shrunk(0.5) is None
