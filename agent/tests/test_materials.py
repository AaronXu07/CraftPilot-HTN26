import numpy as np
import pytest

from copilot.engine.materials import (
    PRESETS,
    MaterialError,
    MaterialSpec,
    choose_blocks,
    material_fit_modes,
    noise3,
    pick_palette_entry,
    presets_text,
    resolve_material,
    validate_material_spec,
)
from copilot.engine.registry import default_registry


def test_all_presets_validate():
    reg = default_registry()
    assert len(PRESETS) >= 25
    for name, spec in PRESETS.items():
        n = validate_material_spec(spec, reg)
        assert n["base"].startswith("minecraft:")
        assert abs(sum(w for _, w in n["palette"]) - 1.0) < 1e-6, name
        assert n["fit"] in ("none", "slab", "stairs", "stairs+slab", "walls")
    assert "medieval_stone" in presets_text()


def test_validate_fills_defaults_and_normalises():
    n = validate_material_spec({"base": "stone_bricks"})
    assert n == {"base": "minecraft:stone_bricks", "palette": [["minecraft:stone_bricks", 1.0]], "fit": "stairs+slab", "noise": {"scale": 3.0, "seed": 0}}
    n = validate_material_spec({"palette": [["oak_planks", 3], ["spruce_planks", 1]], "fit": "slab+stairs"})
    assert n["base"] == "minecraft:oak_planks"
    assert [w for _, w in n["palette"]] == [0.75, 0.25]
    assert n["fit"] == "stairs+slab"
    n = validate_material_spec({"preset": "copper_roof", "fit": "none"})
    assert n["fit"] == "none" and n["base"] == "minecraft:cut_copper"
    n = validate_material_spec({"base": "stone_bricks", "faces": {"top": "stone_brick_slab", "bottom": None}, "gradient": {"axis": "y", "from": 0, "to": 3, "palette": ["cobblestone"]}})
    assert n["faces"] == {"top": "minecraft:stone_brick_slab", "bottom": None}
    assert n["gradient"]["palette"] == [["minecraft:cobblestone", 1.0]]


@pytest.mark.parametrize("bad,msg", [
    ({"base": "stone_brick"}, "unknown base"),
    ({"palette": [["stone", 0]]}, "weight"),
    ({"base": "stone", "fit": "curvy"}, "fit must be"),
    ({"base": "stone", "gradient": {"axis": "w", "from": 0, "to": 1, "palette": ["stone"]}}, "axis"),
    ({"base": "stone", "gradient": {"axis": "y", "from": 5, "to": 1, "palette": ["stone"]}}, "to > from"),
    ({"base": "stone", "faces": {"roof": "stone"}}, "faces key"),
    ({"preset": "nope"}, "unknown preset"),
    ({"base": "stone", "noise": {"scale": 0}}, "noise.scale"),
    ("stone", "must be an object"),
])
def test_validate_errors(bad, msg):
    with pytest.raises(MaterialError) as e:
        validate_material_spec(bad)
    assert msg in str(e.value)


def test_noise_is_coherent_and_deterministic():
    pts = np.array([[x + 0.5, y + 0.5, 0.5] for x in range(40) for y in range(40)], dtype=float)
    a = noise3(pts, 3.0, 1)
    b = noise3(pts, 3.0, 1)
    assert np.array_equal(a, b)
    assert 0.0 <= a.min() and a.max() < 1.0
    grid = a.reshape(40, 40)
    neighbour_diff = np.abs(np.diff(grid, axis=0)).mean()
    random_diff = np.abs(grid.ravel() - np.random.default_rng(0).permutation(grid.ravel())).mean()
    assert neighbour_diff < random_diff * 0.7


def test_palette_picks_respect_weights_and_clump():
    spec = MaterialSpec.from_dict({"base": "stone_bricks", "palette": [["stone_bricks", 0.7], ["cracked_stone_bricks", 0.2], ["mossy_stone_bricks", 0.1]]})
    pts = np.array([[x + 0.5, y + 0.5, z + 0.5] for x in range(30) for y in range(20) for z in range(3)], dtype=float)
    idx = pick_palette_entry(spec, pts, 0)
    frac = np.bincount(idx, minlength=3) / len(idx)
    assert abs(frac[0] - 0.7) < 0.03 and abs(frac[1] - 0.2) < 0.03 and abs(frac[2] - 0.1) < 0.03
    grid = idx.reshape(30, 20, 3)
    same_neighbour = (grid[1:, :, :] == grid[:-1, :, :]).mean()
    chance = (frac ** 2).sum()
    assert same_neighbour > chance + 0.15
    assert np.array_equal(idx, pick_palette_entry(spec, pts, 0))
    assert not np.array_equal(idx, pick_palette_entry(spec, pts, 5))


def test_gradient_and_faces():
    spec = MaterialSpec.from_dict({"base": "stone_bricks", "gradient": {"axis": "y", "from": 0, "to": 3, "palette": ["cobblestone"]}, "faces": {"top": "stone_brick_slab"}})
    pts = np.array([[x + 0.5, y + 0.5, z + 0.5] for x in range(40) for y in range(10) for z in range(2)], dtype=float)
    top = np.array([(int(p[1]) == 9) for p in pts])
    ch = choose_blocks(spec, pts, 0, top_exposed=top)
    ys = pts[:, 1].astype(int)
    low = ch[ys <= 2]
    high = ch[(ys >= 5) & (ys < 9)]
    assert (low == "minecraft:cobblestone").all()  # the band itself is never eroded
    assert (high == "minecraft:stone_bricks").all()
    assert (ch[ys == 9] == "minecraft:stone_brick_slab").all()
    boundary = ch[ys == 3]  # noisy fringe just above the band
    assert 0 < (boundary == "minecraft:cobblestone").mean() < 1


def test_resolve_material_lookup_order():
    scene_mats = {"wall": {"base": "spruce_planks"}}
    assert resolve_material("wall", scene_mats).base == "minecraft:spruce_planks"
    assert resolve_material("copper_roof", scene_mats).base == "minecraft:cut_copper"
    assert resolve_material("stone_bricks", scene_mats).base == "minecraft:stone_bricks"
    assert resolve_material({"base": "bricks"}, scene_mats).base == "minecraft:bricks"
    w = []
    assert resolve_material("nope", scene_mats, w).base == "minecraft:stone"
    assert w and "nope" in w[0]
    w = []
    assert resolve_material("bad", {"bad": {"base": "zzz"}}, w).base == "minecraft:stone"
    assert w


def test_material_fit_modes_with_scene():
    from copilot.engine.scene import Scene, apply_op

    sc = Scene()
    sc, _ = apply_op(sc, "define_material", name="wall", spec={"base": "stone_bricks", "fit": "slab"})
    sc, _ = apply_op(sc, "add", id="a", shape={"type": "box", "size": [4, 4, 4]}, material="wall")
    sc, _ = apply_op(sc, "add", id="b", shape={"type": "box", "size": [4, 4, 4]}, pos=[10, 0, 0], material="glass_clear")
    modes = material_fit_modes(sc)
    assert modes["wall"] == "slab" and modes["glass_clear"] == "none"
