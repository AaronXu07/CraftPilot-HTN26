import json

import numpy as np
import pytest

from copilot.engine.scene import Scene, SceneEditor, SceneError, apply_op


def make():
    e = SceneEditor(Scene(name="t"))
    e.define_material("stone_wall", {"base": "stone_bricks", "palette": [["stone_bricks", 0.7], ["cracked_stone_bricks", 0.3]]})
    e.add("keep", {"type": "box", "size": [14, 18, 14]}, pos=[17, 0, 13], material="stone_wall", group="outer_wall", modifiers=[{"type": "shell", "thickness": 1}])
    e.add("tower_ne", {"type": "cylinder", "radius": 4.5, "height": 22}, pos=[34, 0, 4], material="stone_wall", group="outer_wall", tags=["tower"])
    e.add("gate_cut", {"type": "box", "size": [4, 6, 3]}, pos=[24, 0, -1], op="subtract")
    return e


def test_add_and_bbox():
    e = make()
    bb = e.scene.object_bbox("keep")
    assert bb.lo.tolist() == [10, 0, 6] and bb.hi.tolist() == [24, 18, 20]
    bb = e.scene.object_bbox("tower_ne")
    assert bb.lo.tolist() == [30, 0, 0] and bb.hi.tolist() == [39, 22, 9]
    assert "bbox" in e.bbox()


def test_add_errors():
    e = make()
    with pytest.raises(SceneError):
        e.add("keep", {"type": "box", "size": [1, 1, 1]})
    with pytest.raises(SceneError):
        e.add("Bad-Id", {"type": "box", "size": [1, 1, 1]})
    with pytest.raises(SceneError):
        e.add("x", {"type": "nope"})
    with pytest.raises(SceneError):
        e.add("x", {"type": "box", "size": [1, -1, 1]})
    with pytest.raises(SceneError):
        e.move("nope", [1, 0, 0])
    assert "WARNING" in e.add("huge", {"type": "box", "size": [300, 1, 1]})


def test_delete_duplicate_rename():
    e = make()
    e.duplicate("tower_ne", "tower_nw", offset=[-34, 0, 0])
    assert e.scene.get("tower_nw").transform.pos.tolist() == [0, 0, 4]
    assert e.scene.index_of("tower_nw") == e.scene.index_of("tower_ne") + 1
    e.rename("tower_nw", "tower_w")
    assert e.scene.has("tower_w") and not e.scene.has("tower_nw")
    e.delete(["tower_w", "gate_cut"])
    assert [o.id for o in e.scene.objects] == ["keep", "tower_ne"]


def test_move_rotate_scale():
    e = make()
    e.move("keep", [1, 2, 3])
    assert e.scene.get("keep").transform.pos.tolist() == [18, 2, 16]
    e.move_to("keep", [0, 0, 0])
    assert e.scene.get("keep").transform.pos.tolist() == [0, 0, 0]
    e.rotate("keep", 45)
    assert e.scene.get("keep").transform.rot[1] == 45
    e.rotate("keep", 45, pivot=[10, 0, 0])
    assert e.scene.get("keep").transform.rot[1] == 90
    pos = e.scene.get("keep").transform.pos
    assert np.allclose(pos, [10 - 10 * np.cos(np.deg2rad(45)), 0, 10 * np.sin(np.deg2rad(45))], atol=1e-3)  # positions are stored to 4 decimals
    e.scale("tower_ne", 2)
    assert e.scene.get("tower_ne").shape["radius"] == 9 and e.scene.get("tower_ne").shape["height"] == 44
    e.scale("gate_cut", [1, 2, 1])
    assert e.scene.get("gate_cut").shape["size"] == [4, 12, 3]


def test_align_stack():
    e = make()
    e.add("roof", {"type": "cone", "radius": 5.5, "height": 7}, pos=[0, 0, 0], material="stone_wall")
    e.stack("roof", on="tower_ne")
    bb = e.scene.object_bbox("roof")
    tb = e.scene.object_bbox("tower_ne")
    assert bb.lo[1] == tb.hi[1]
    assert np.isclose(bb.center[0], tb.center[0]) and np.isclose(bb.center[2], tb.center[2])
    e.align("gate_cut", "x", "min", to="keep")
    assert e.scene.object_bbox("gate_cut").lo[0] == e.scene.object_bbox("keep").lo[0]
    e.align("gate_cut", "y", "max", to=30)
    assert e.scene.object_bbox("gate_cut").hi[1] == 30


def test_mirror_copy_exact_footprint():
    e = make()
    e.mirror_copy("tower_ne", axis="x", plane=17, new_suffix="_w")
    a = e.scene.object_bbox("tower_ne")
    b = e.scene.object_bbox("tower_ne_w")
    assert b.lo[0] == 2 * 17 - a.hi[0] and b.hi[0] == 2 * 17 - a.lo[0]
    assert b.lo[2] == a.lo[2] and b.hi[2] == a.hi[2]
    # wedge slope flips
    e.add("roof_e", {"type": "wedge", "size": [6, 3, 8], "slope_axis": "x"}, pos=[10, 0, 0], material="stone_wall")
    e.mirror_copy("roof_e", axis="x", plane=0, new_suffix="_m")
    assert e.scene.get("roof_e_m").shape["slope_axis"] == "-x"


def test_set_shape_and_type_change():
    e = make()
    msg = e.set_shape("keep", type="prism", sides=6)
    assert "prism n6" in msg
    s = e.scene.get("keep").shape
    assert s["sides"] == 6 and s["height"] == 18 and s["radius"] == 7
    e.set_shape("keep", height=20)
    assert e.scene.get("keep").shape["height"] == 20
    e.set_op("gate_cut", "add")
    assert e.scene.get("gate_cut").op == "add"
    with pytest.raises(SceneError):
        e.set_op("gate_cut", "weird")


def test_set_anchor_keeps_world_position():
    e = make()
    before = e.scene.object_bbox("keep")
    e.set_anchor("keep", "center")
    after = e.scene.object_bbox("keep")
    assert np.allclose(before.lo, after.lo) and np.allclose(before.hi, after.hi)


def test_reorder_and_modifiers():
    e = make()
    e.reorder("gate_cut", before="keep")
    assert [o.id for o in e.scene.objects][0] == "gate_cut"
    e.reorder("gate_cut", after="tower_ne")
    assert [o.id for o in e.scene.objects][-1] == "gate_cut"
    e.add_modifier("tower_ne", {"type": "round", "radius": 0.5})
    e.add_modifier("tower_ne", {"type": "array", "count": 3, "offset": [0, 0, 10]}, index=0)
    mods = e.scene.get("tower_ne").modifiers
    assert mods[0]["type"] == "array" and mods[1]["type"] == "round"
    e.set_modifier("tower_ne", 0, count=4)
    assert e.scene.get("tower_ne").modifiers[0]["count"] == 4
    bb = e.scene.object_bbox("tower_ne")
    assert bb.hi[2] == 9 + 30
    e.remove_modifier("tower_ne", 0)
    assert len(e.scene.get("tower_ne").modifiers) == 1
    with pytest.raises(SceneError):
        e.add_modifier("tower_ne", {"type": "boolean", "target": "missing"})
    with pytest.raises(SceneError):
        e.remove_modifier("tower_ne", 5)


def test_group_ungroup_select():
    e = make()
    assert e.select("group:outer_wall") == "keep, tower_ne"
    assert e.select("tag:tower") == "tower_ne"
    assert e.select("op:subtract") == "gate_cut"
    assert e.select("tower_*") == "tower_ne"
    assert e.select("material:stone_wall -tag:tower") == "keep"
    assert e.select("above_y:20") == "tower_ne"
    e.group(["tower_ne", "gate_cut"], "towers")
    assert e.scene.get("gate_cut").group == "towers"
    e.group(["towers"], "castle")
    assert e.scene.groups["towers"]["parent"] == "castle"
    assert set(e.scene.group_members("castle")) == {"tower_ne", "gate_cut"}
    e.ungroup("towers")
    assert e.scene.get("tower_ne").group == "castle"
    assert "towers" not in e.scene.groups
    e.move("castle", [0, 1, 0])
    assert e.scene.get("gate_cut").transform.pos[1] == 1


def test_describe_and_measure():
    e = make()
    out = e.describe()
    assert out.startswith("scene t  bbox 10..39 x 0..22 y 0..20 z")
    assert "[outer_wall]" in out and "gate_cut" in out and "SUBTRACT" in out and "shell(1)" in out
    full = e.describe(detail="full")
    assert "materials:" in full and "bbox (" in full
    assert "gap 6" in e.measure("keep", "tower_ne")
    assert "y=22" in e.top_of("tower_ne")
    assert "z=6" in e.side_of("keep", "north")
    assert "stone_wall" in e.list_materials()


def test_serialisation_round_trip_and_purity():
    e = make()
    sc = e.scene
    js = sc.to_json()
    back = Scene.from_json(js)
    assert back.hash() == sc.hash()
    assert json.loads(back.to_json()) == json.loads(js)
    new_scene, msg = apply_op(sc, "move", ids="keep", delta=[5, 0, 0])
    assert sc.get("keep").transform.pos[0] == 17  # input untouched
    assert new_scene.get("keep").transform.pos[0] == 22


def test_paint_and_inline_material():
    e = make()
    msg = e.paint({"type": "box", "size": [14, 2, 14]}, pos=[17, 0, 13], material="mossy")
    assert "paint" in msg and e.scene.objects[-1].op == "paint"
    e.add("trim", {"type": "box", "size": [2, 2, 2]}, pos=[0, 0, 0], material={"name": "gold", "base": "gold_block"})
    assert "gold" in e.scene.materials and e.scene.get("trim").material == "gold"
    e.set_material("keep", "gold")
    assert e.scene.get("keep").material == "gold"


def test_block_shape_anchor():
    e = make()
    e.add("lantern", {"type": "block", "state": "lantern[hanging=true]"}, pos=[3, 5, 3], anchor="center")
    o = e.scene.get("lantern")
    assert o.transform.anchor == "bottom_min" and o.shape["state"] == "minecraft:lantern[hanging=true]"
    bb = e.scene.object_bbox(o)
    assert bb.lo.tolist() == [3, 5, 3] and bb.hi.tolist() == [4, 6, 4]
