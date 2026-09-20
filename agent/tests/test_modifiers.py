import numpy as np

from copilot.engine.modifiers import build_object_sdf
from copilot.engine.noise import noise3
from copilot.engine.scene import Scene, SceneEditor


def obj_sdf(mods, shape=None, pos=(0, 0, 0), rot=(0, 0, 0)):
    e = SceneEditor(Scene("m"))
    e.add("a", shape or {"type": "box", "size": [10, 10, 10]}, pos=list(pos), rot=list(rot), material="m", modifiers=mods)
    return e.scene, build_object_sdf(e.scene, e.scene.get("a"))


def test_plain_transform():
    _, f = obj_sdf([], pos=(20, 5, 0))
    assert f(np.array([[20, 10, 0.0]]))[0] < 0
    assert f(np.array([[20, 4.9, 0.0]]))[0] > 0
    assert f(np.array([[25.5, 10, 0.0]]))[0] > 0


def test_rotation_about_y():
    _, f = obj_sdf([], shape={"type": "box", "size": [20, 4, 2]}, rot=(0, 90, 0))
    # after a 90° turn the long axis lies along z
    assert f(np.array([[0, 2, 9.0]]))[0] < 0
    assert f(np.array([[9, 2, 0.0]]))[0] > 0


def test_shell_hollow_interior():
    _, f = obj_sdf([{"type": "shell", "thickness": 2}])
    assert f(np.array([[0, 5, 0.0]]))[0] > 0  # hollow centre
    assert f(np.array([[4.5, 5, 0.0]]))[0] < 0  # inside the 2-thick wall
    assert f(np.array([[2.5, 5, 0.0]]))[0] > 0  # just inside the inner surface
    assert f(np.array([[5.5, 5, 0.0]]))[0] > 0  # outside


def test_round_removes_corners():
    _, f0 = obj_sdf([])
    _, f1 = obj_sdf([{"type": "round", "radius": 1.5}])
    corner = np.array([[4.9, 9.9, 4.9]])
    assert f0(corner)[0] < 0 and f1(corner)[0] > 0
    assert f1(np.array([[0, 5, 0.0]]))[0] < 0
    assert f1(np.array([[4.9, 5, 0.0]]))[0] < 0  # face centre still inside


def test_array_count():
    sc, f = obj_sdf([{"type": "array", "count": 4, "offset": [15, 0, 0]}], shape={"type": "box", "size": [4, 4, 4]})
    xs = np.array([[15 * i, 2, 0.0] for i in range(5)])
    d = f(xs)
    assert np.all(d[:4] < 0) and d[4] > 0
    bb = sc.object_bbox("a")
    assert bb.hi[0] == 2 + 45


def test_mirror_symmetry():
    sc, f = obj_sdf([{"type": "mirror", "axis": "x", "plane": 20}], shape={"type": "box", "size": [4, 4, 4]}, pos=(10, 0, 0))
    assert f(np.array([[10, 2, 0.0]]))[0] < 0
    assert f(np.array([[30, 2, 0.0]]))[0] < 0
    assert f(np.array([[20, 2, 0.0]]))[0] > 0
    bb = sc.object_bbox("a")
    assert bb.lo[0] == 8 and bb.hi[0] == 32
    _, g = obj_sdf([{"type": "mirror", "axis": "x", "plane": 20, "keep_original": False}], shape={"type": "box", "size": [4, 4, 4]}, pos=(10, 0, 0))
    assert g(np.array([[10, 2, 0.0]]))[0] > 0 and g(np.array([[30, 2, 0.0]]))[0] < 0


def test_taper_narrows_top():
    _, f = obj_sdf([{"type": "taper", "top_scale": 0.2}])
    assert f(np.array([[4.5, 0.5, 0.0]]))[0] < 0  # wide at the bottom
    assert f(np.array([[4.5, 9.5, 0.0]]))[0] > 0  # narrow at the top
    assert f(np.array([[0.5, 9.5, 0.0]]))[0] < 0


def test_twist_rotates_top():
    _, f = obj_sdf([{"type": "twist", "deg_per_block": 9}], shape={"type": "box", "size": [12, 10, 2]})
    # at y=10 the box has turned 90°: the long axis is along z
    assert f(np.array([[5.5, 9.9, 0.0]]))[0] > 0
    assert f(np.array([[0, 9.9, 5.5]]))[0] < 0
    assert f(np.array([[5.5, 0.1, 0.0]]))[0] < 0


def test_boolean_modifier():
    e = SceneEditor(Scene("b"))
    e.add("cut", {"type": "sphere", "radius": 3}, pos=[0, 0, 0], material="x", visible=False)
    e.add("a", {"type": "box", "size": [10, 10, 10]}, pos=[0, 0, 0], material="m", modifiers=[{"type": "boolean", "target": "cut", "op": "subtract"}])
    f = build_object_sdf(e.scene, e.scene.get("a"))
    assert f(np.array([[0, 1, 0.0]]))[0] > 0  # carved by the sphere (sphere bottom sits at y=0, centre y=3)
    assert f(np.array([[4, 8, 4.0]]))[0] < 0
    e2 = SceneEditor(Scene("b2"))
    e2.add("t", {"type": "box", "size": [4, 4, 4]}, pos=[20, 0, 0], material="x", visible=False)
    e2.add("a", {"type": "box", "size": [4, 4, 4]}, pos=[0, 0, 0], material="m", modifiers=[{"type": "boolean", "target": "t", "op": "union"}])
    g = build_object_sdf(e2.scene, e2.scene.get("a"))
    assert g(np.array([[20, 2, 0.0]]))[0] < 0


def test_noise_displace_and_noise_function():
    pts = np.random.default_rng(0).uniform(-50, 50, size=(500, 3))
    n = noise3(pts, 4.0, 7)
    assert n.shape == (500,) and n.min() >= -1 and n.max() <= 1 and n.std() > 0.05
    assert np.allclose(n, noise3(pts, 4.0, 7))  # deterministic
    assert not np.allclose(n, noise3(pts, 4.0, 8))
    _, f = obj_sdf([{"type": "noise_displace", "amplitude": 1.0, "scale": 3, "seed": 1}])
    _, f0 = obj_sdf([])
    d = f(pts)
    assert np.all(np.abs(d - f0(pts)) <= 1.0 + 1e-9)
