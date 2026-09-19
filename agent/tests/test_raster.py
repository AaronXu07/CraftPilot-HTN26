import numpy as np
import pytest

from copilot.engine.raster import RasterError, rasterize
from copilot.engine.scene import Scene, SceneEditor
from tests.engine_helpers import occupied_world


def test_box_fills_exact_voxels():
    e = SceneEditor(Scene("r"))
    e.add("keep", {"type": "box", "size": [14, 18, 14]}, pos=[17, 0, 13], material="stone")
    r = rasterize(e.scene)
    w = occupied_world(r)
    assert w[:, 0].min() == 10 and w[:, 0].max() == 23
    assert w[:, 1].min() == 0 and w[:, 1].max() == 17
    assert w[:, 2].min() == 6 and w[:, 2].max() == 19
    assert len(w) == 14 * 18 * 14
    assert r.material_names == ["", "stone"]
    assert r.block_count() == 14 * 18 * 14


def test_odd_cylinder_equator():
    e = SceneEditor(Scene("r"))
    e.add("tower", {"type": "cylinder", "radius": 4.5, "height": 22}, pos=[34, 0, 4], material="stone")
    r = rasterize(e.scene)
    w = occupied_world(r)
    row = w[(w[:, 1] == 0) & (w[:, 2] == 4)]
    assert row[:, 0].min() == 30 and row[:, 0].max() == 38 and len(row) == 9
    assert w[:, 1].max() == 21


def test_subtract_intersect_paint():
    e = SceneEditor(Scene("r"))
    e.add("a", {"type": "box", "size": [10, 10, 10]}, pos=[0, 0, 0], material="stone")
    e.add("cut", {"type": "box", "size": [4, 4, 4]}, pos=[0, 0, 0], op="subtract")
    r = rasterize(e.scene)
    assert r.material[r.world_to_index(0, 1, 0)] == 0
    assert r.sdf[r.world_to_index(0, 1, 0)] > 0  # carved surface keeps a valid sdf
    assert r.material[r.world_to_index(4, 1, 4)] > 0
    e.add("clip", {"type": "box", "size": [10, 5, 10]}, pos=[0, 0, 0], op="intersect")
    r = rasterize(e.scene)
    w = occupied_world(r)
    assert w[:, 1].max() == 4
    e.paint({"type": "box", "size": [10, 2, 10]}, pos=[0, 0, 0], material="moss")
    r = rasterize(e.scene)
    assert r.material_names == ["", "stone", "moss"]
    assert r.material_name_at(4, 0, 4) == "moss"
    assert r.material_name_at(4, 3, 4) == "stone"
    assert r.material_name_at(0, 0, 0) == ""  # paint never adds geometry


def test_later_add_owns_overlap():
    e = SceneEditor(Scene("r"))
    e.add("a", {"type": "box", "size": [6, 6, 6]}, pos=[0, 0, 0], material="stone")
    e.add("b", {"type": "box", "size": [6, 6, 6]}, pos=[3, 0, 0], material="wood")
    r = rasterize(e.scene)
    assert r.material_name_at(1, 1, 0) == "wood"
    assert r.owner[r.world_to_index(1, 1, 0)] == 1
    assert r.material_name_at(-2, 1, 0) == "stone"


def test_props_and_hidden_objects():
    e = SceneEditor(Scene("r"))
    e.add("a", {"type": "box", "size": [4, 4, 4]}, pos=[0, 0, 0], material="stone")
    e.add("lantern", {"type": "block", "state": "minecraft:lantern"}, pos=[0, 4, 0])
    e.add("ghost", {"type": "box", "size": [40, 4, 4]}, pos=[100, 0, 0], material="stone", visible=False)
    r = rasterize(e.scene)
    assert r.props == {(0, 4, 0): "minecraft:lantern"}
    assert r.shape[0] < 20


def test_scene_sdf_agrees_with_grid():
    e = SceneEditor(Scene("r"))
    e.add("a", {"type": "cylinder", "radius": 5, "height": 8}, pos=[0, 0, 0], material="stone")
    e.add("cut", {"type": "sphere", "radius": 3}, pos=[0, 4, 0], op="subtract")
    r = rasterize(e.scene)
    idx = np.argwhere(np.ones(r.shape, dtype=bool))
    pts = idx + np.asarray(r.origin) + 0.5
    f, owner = r.scene_sdf(pts)
    grid_occ = (r.material > 0)[idx[:, 0], idx[:, 1], idx[:, 2]]
    assert np.array_equal(f <= 0, grid_occ)
    assert np.all(owner[grid_occ] == 0) and np.all(owner[~grid_occ] == -1)


def test_size_cap():
    e = SceneEditor(Scene("r"))
    e.add("huge", {"type": "box", "size": [300, 10, 10]}, pos=[0, 0, 0], material="stone")
    with pytest.raises(RasterError):
        rasterize(e.scene)


def test_cache_reuses_unchanged_objects():
    e = SceneEditor(Scene("r"))
    e.add("a", {"type": "box", "size": [10, 10, 10]}, pos=[0, 0, 0], material="stone")
    e.add("b", {"type": "box", "size": [4, 4, 4]}, pos=[20, 0, 0], material="stone")
    e.add("c", {"type": "sphere", "radius": 3}, pos=[0, 20, 0], material="stone")
    cache = {}
    r1 = rasterize(e.scene, cache=cache)
    assert r1.timings["objects_evaluated"] == 3
    e.move("b", [0, 1, 0])
    r2 = rasterize(e.scene, cache=cache)
    assert r2.timings["objects_evaluated"] == 1 and r2.timings["cache_hits"] == 2
    assert r2.material_name_at(20, 1, 0) == "stone"
    r3 = rasterize(e.scene, cache=cache)
    assert r3.timings["objects_evaluated"] == 0


def test_empty_scene():
    r = rasterize(Scene("empty"))
    assert r.block_count() == 0
    f, o = r.scene_sdf(np.zeros((3, 3)))
    assert np.all(np.isinf(f))
