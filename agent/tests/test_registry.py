import os

import pytest

from copilot.engine.registry import WOODS, Registry, default_registry


@pytest.fixture(scope="module")
def reg():
    return default_registry()


def test_fallback_loads(reg):
    assert reg.source in ("fallback", "cache")
    assert len(reg) >= 900
    assert reg.has("minecraft:stone") and reg.has("stone") and reg.has("oak_stairs")
    assert not reg.has("minecraft:not_a_block")


def test_wood_families(reg):
    for w in WOODS:
        fam = reg.family(f"{w}_planks")
        for form in ("full", "stairs", "slab", "fence", "fence_gate", "door", "trapdoor", "button", "pressure_plate", "sign", "log", "stripped_log"):
            assert form in fam, (w, form, fam)
        assert fam["stairs"] == f"minecraft:{w}_stairs"
    assert reg.family("crimson_planks")["log"] == "minecraft:crimson_stem"
    assert reg.family("bamboo_planks")["log"] == "minecraft:bamboo_block"
    assert reg.family("oak_log")["stairs"] == "minecraft:oak_stairs"


def test_stone_families(reg):
    bases = ["stone", "cobblestone", "mossy_cobblestone", "stone_bricks", "mossy_stone_bricks", "andesite", "polished_andesite",
             "diorite", "granite", "cobbled_deepslate", "polished_deepslate", "deepslate_bricks", "deepslate_tiles", "tuff",
             "polished_tuff", "tuff_bricks", "blackstone", "polished_blackstone", "polished_blackstone_bricks", "sandstone",
             "smooth_sandstone", "red_sandstone", "bricks", "mud_bricks", "nether_bricks", "red_nether_bricks", "end_stone_bricks",
             "prismarine", "prismarine_bricks", "dark_prismarine", "purpur_block", "quartz_block", "smooth_quartz", "cut_copper"]
    assert len(bases) >= 20
    for b in bases:
        fam = reg.family(b)
        assert "stairs" in fam and "slab" in fam, (b, fam)
        assert reg.has(fam["stairs"]) and reg.has(fam["slab"])
    assert reg.family("oak_planks")["stairs"] == "minecraft:oak_stairs"
    assert reg.family("stone_bricks")["wall"] == "minecraft:stone_brick_wall"
    assert reg.family("red_concrete")["powder"] == "minecraft:red_concrete_powder"
    assert reg.family("deepslate_tiles")["stairs"] == "minecraft:deepslate_tile_stairs"
    assert reg.family("copper_block")["stairs"] == "minecraft:cut_copper_stairs"
    assert reg.family("white_wool")["carpet"] == "minecraft:white_carpet"
    assert reg.family("glass")["pane"] == "minecraft:glass_pane"
    assert "stairs" not in reg.family("furnace")


def test_families_report_is_short(reg):
    rep = reg.families_report()
    assert len(rep) < 80
    assert "stone_bricks" not in rep and "oak_planks" not in rep


def test_validate_state(reg):
    ok, st = reg.validate_state("minecraft:oak_stairs[facing=north,half=top]")
    assert ok and st == "minecraft:oak_stairs[facing=north,half=top,shape=straight,waterlogged=false]"
    ok, st = reg.validate_state("oak_stairs[half=top]", fill_defaults=False)
    assert ok and st == "minecraft:oak_stairs[half=top]"
    ok, msg = reg.validate_state("oak_stairs[facing=up]")
    assert not ok and "allowed" in msg
    ok, msg = reg.validate_state("oak_stairs[color=red]")
    assert not ok and "no property" in msg
    ok, msg = reg.validate_state("stone_bricks_wall")
    assert not ok and "stone_brick_wall" in msg
    assert reg.normalize("stone") == "minecraft:stone"
    with pytest.raises(ValueError):
        reg.normalize("bogus_block")
    assert reg.default_state("lantern") == "minecraft:lantern[hanging=false,waterlogged=false]"


def test_colors(reg):
    assert reg.color("oak_planks") == reg.color("oak_stairs") == reg.color("minecraft:oak_slab[type=top]")
    assert reg.color("white_concrete") != reg.color("black_concrete")
    assert reg.color("lime_stained_glass_pane") == reg.color("lime_stained_glass")
    assert reg.color("mud_brick_wall") == reg.color("mud_bricks")
    assert reg.color("totally_unknown") == (128, 128, 128)
    assert all(0 <= c <= 255 for c in reg.color("stripped_oak_log"))


def test_nearest_and_search(reg):
    assert reg.nearest((233, 236, 236), "wool") == "minecraft:white_wool"
    assert reg.nearest((8, 10, 15), "concrete") == "minecraft:black_concrete"
    assert "glass" in reg.nearest((40, 120, 220), "glass")
    assert reg.is_full_block(reg.nearest((150, 100, 80)))
    res = reg.search("mossy stone")
    assert "minecraft:mossy_stone_bricks" in res
    assert reg.search("cobblston")[0] == "minecraft:cobblestone"
    assert reg.search("zzzz") == []
    assert reg.search("lantern", limit=2) == ["minecraft:lantern", "minecraft:soul_lantern"] or "minecraft:lantern" in reg.search("lantern")


def test_categories_and_full(reg):
    assert "wood" in reg.categories("spruce_planks")
    assert "color:red" in reg.categories("red_wool") and "wool" in reg.categories("red_wool")
    assert "light" in reg.categories("lantern") and not reg.is_full_block("lantern")
    assert reg.is_full_block("stone_bricks") and not reg.is_full_block("stone_brick_stairs")
    assert not reg.is_full_block("poppy") and not reg.is_full_block("oak_door")
    assert reg.is_full_block("oak_leaves")


def test_catalog_text(reg):
    cat = reg.catalog_text()
    assert len(cat) < 12000
    assert "stone_bricks(stairs/slab/wall)" in cat
    assert "spruce" in cat and "light_blue" in cat and "lantern" in cat
    assert "1060" in cat or str(len(reg)) in cat


def test_load_from_bridge_like_object(tmp_path):
    class B:
        def blocks(self):
            return [{"id": "minecraft:stone", "properties": {}, "default": "minecraft:stone"},
                    {"id": "minecraft:oak_stairs", "properties": {"facing": ["north", "south"], "half": ["top", "bottom"]},
                     "default": "minecraft:oak_stairs[facing=north,half=bottom]"}]

    r = Registry.load(bridge=B(), cache_dir=str(tmp_path))
    assert r.source == "mod" and len(r) == 2
    assert os.path.exists(tmp_path / "blocks_live.json")
    assert r.validate_state("oak_stairs")[1] == "minecraft:oak_stairs[facing=north,half=bottom]"
    r2 = Registry.load(cache_dir=str(tmp_path))
    assert r2.source == "cache" and len(r2) == 2
