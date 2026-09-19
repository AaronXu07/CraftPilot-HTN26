import numpy as np

from copilot.engine.coretypes import FitResult, RasterResult
from copilot.engine.lint import findings_text, lint, lint_scene_only
from copilot.engine.registry import default_registry
from copilot.engine.scene import Scene, apply_op


def box_raster(scene, obj_index=0, size=(12, 6, 12), origin=(0, 0, 0), hollow=False, names=("", "wall")):
    X, Y, Z = size
    r = RasterResult(origin=origin, material=np.zeros(size, np.int16), material_names=list(names), owner=np.full(size, -1, np.int32), sdf=np.full(size, np.inf, np.float32), props={})
    r.material[:, :, :] = 1
    r.owner[:, :, :] = obj_index
    if hollow:
        r.material[1:-1, 1:-1, 1:-1] = 0
        r.owner[1:-1, 1:-1, 1:-1] = -1
    return r


def scene_box(size=(12, 6, 12), material="medieval_stone"):
    sc = Scene()
    sc, _ = apply_op(sc, "add", id="hall", shape={"type": "box", "size": list(size)}, material=material)
    return sc


def rules(findings):
    return {f.rule for f in findings}


def test_blank_facade_r1():
    sc = scene_box()
    r = box_raster(sc)
    f = FitResult.full_blocks(r)
    bm = {(x, y, z): "minecraft:stone_bricks" for x in range(12) for y in range(6) for z in range(12)}
    fs = lint(sc, r, f, bm, default_registry())
    r1 = [x for x in fs if x.rule == "R1"]
    assert r1 and "hall" in r1[0].objects
    assert "façade" in r1[0].message
    # punching windows (holes) in every face removes the R1 finding for those faces
    r.material[3, 2:4, 3:9] = 0
    r.material[8, 2:4, 3:9] = 0
    r2 = box_raster(sc)
    for x in (2, 5, 9):
        for z in (0, 11):
            r2.material[x, 2:4, z] = 0
        for z in (2, 5, 9):
            for xx in (0, 11):
                r2.material[xx, 2:4, z] = 0
    f2 = FitResult.full_blocks(r2)
    fs2 = lint(sc, r2, f2, bm, default_registry())
    assert "R1" not in rules(fs2)


def test_scale_r7_and_unknown_material_r8():
    sc = Scene()
    sc, _ = apply_op(sc, "add", id="huge", shape={"type": "box", "size": [500, 4, 4]}, material="medieval_stone")
    sc, _ = apply_op(sc, "add", id="odd", shape={"type": "box", "size": [4, 4, 4]}, pos=[0, 0, 20], material="unobtainium")
    fs = lint_scene_only(sc)
    r7 = [x for x in fs if x.rule == "R7"]
    r8 = [x for x in fs if x.rule == "R8"]
    assert r7 and r7[0].severity == "error" and "huge" in r7[0].objects
    assert r8 and "odd" in r8[0].objects
    # a bare block id counts as a material
    sc, _ = apply_op(sc, "set_material", ids="odd", material="stone_bricks")
    assert "R8" not in rules(lint_scene_only(sc))


def test_roof_r2():
    sc = Scene()
    sc, _ = apply_op(sc, "add", id="tower", shape={"type": "cylinder", "radius": 4, "height": 10}, material="medieval_stone")
    sc, _ = apply_op(sc, "add", id="tower_roof", shape={"type": "cone", "radius": 3, "height": 6}, pos=[0, 10, 0], material="slate_roof")
    fs = lint_scene_only(sc)
    assert "R2" not in rules(fs)
    r = box_raster(sc, size=(10, 16, 10))
    f = FitResult.full_blocks(r)
    fs = lint(sc, r, f, None, default_registry())
    r2 = [x for x in fs if x.rule == "R2"]
    assert r2 and any("overhang" in x.message for x in r2)
    # widen the roof: overhang ok; flatten it: slope warning
    sc, _ = apply_op(sc, "set_shape", id="tower_roof", radius=5.5, height=1)
    fs = lint(sc, r, f, None, default_registry())
    r2 = [x for x in fs if x.rule == "R2"]
    assert r2 and any("slope" in x.message for x in r2) and not any("overhang" in x.message for x in r2)


def test_entrance_r3_and_variety_r4():
    sc = scene_box((12, 6, 12))
    r = box_raster(sc, hollow=True)
    f = FitResult.full_blocks(r)
    bm = {(x, y, z): "minecraft:stone" for x in range(12) for y in range(6) for z in range(12)}
    fs = lint(sc, r, f, bm, default_registry())
    assert "R3" in rules(fs)
    assert "R4" not in rules(fs)  # under 1000 blocks
    big = {(x, y, z): "minecraft:stone" for x in range(20) for y in range(6) for z in range(20)}
    fs = lint(sc, r, f, big, default_registry())
    assert "R4" in rules(fs)
    # carve a 2-tall doorway through the south wall: entrance satisfied
    r.material[5, 0:2, 11] = 0
    r.material[5, 0:2, 10] = 0
    f = FitResult.full_blocks(r)
    fs = lint(sc, r, f, bm, default_registry())
    assert "R3" not in rules(fs)
    # a door prop also counts
    r2 = box_raster(sc, hollow=True)
    r2.props = {(5, 0, 11): "minecraft:oak_door[facing=south]"}
    assert "R3" not in rules(lint(sc, r2, FitResult.full_blocks(r2), bm, default_registry()))


def test_floating_and_isolated_r5_and_props_r6():
    sc = scene_box()
    r = box_raster(sc, size=(8, 8, 8))
    r.material[:, :, :] = 0
    r.material[0:4, 0, 0:4] = 1  # a slab on the ground
    r.material[7, 5, 7] = 1  # isolated single
    r.material[1:3, 5, 1:3] = 1  # floating 2x2
    r.props = {(6, 3, 6): "minecraft:torch", (6, 3, 2): "minecraft:lantern[hanging=true]"}
    f = FitResult.full_blocks(r)
    fs = lint(sc, r, f, None, default_registry())
    r5 = [x for x in fs if x.rule == "R5"]
    assert any("isolated" in x.message for x in r5) and any("floating" in x.message for x in r5)
    r6 = [x for x in fs if x.rule == "R6"]
    assert r6 and "torch" in r6[0].message and "lantern" not in r6[0].message
    assert "R5" in findings_text(fs)


def test_lint_without_raster_only_scene_rules():
    sc = scene_box()
    assert lint(sc, None, None, None) == lint_scene_only(sc)
