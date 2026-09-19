import os

import pytest

from copilot.engine.schematic import export_litematic, materials_list, materials_list_text


def test_export_and_reload(tmp_path):
    bm = {(10, 64, -5): "minecraft:stone", (11, 64, -5): "minecraft:oak_stairs[facing=east,half=bottom]", (10, 66, -4): "minecraft:lantern[hanging=true]"}
    path = export_litematic(bm, "test_build", str(tmp_path / "test_build"))
    assert path.endswith(".litematic") and os.path.exists(path)
    from litemapy import Schematic

    s = Schematic.load(path)
    assert s.name == "test_build"
    reg = list(s.regions.values())[0]
    assert abs(reg.width) == 2 and abs(reg.height) == 3 and abs(reg.length) == 2
    assert reg[0, 0, 0].id == "minecraft:stone"
    st = reg[1, 0, 0]
    assert st.id == "minecraft:oak_stairs" and st["facing"] == "east" and st["half"] == "bottom"
    assert reg[0, 2, 1].id == "minecraft:lantern"
    assert reg.count_blocks() == 3


def test_materials_list():
    bm = {(0, 0, 0): "minecraft:stone", (1, 0, 0): "minecraft:stone", (2, 0, 0): "minecraft:oak_stairs[facing=east]", (3, 0, 0): "minecraft:oak_stairs[facing=west]"}
    assert materials_list(bm) == [("minecraft:oak_stairs", 2), ("minecraft:stone", 2)]
    txt = materials_list_text(bm)
    assert "4 blocks" in txt and "oak_stairs" in txt and "stacks" in txt
    assert materials_list_text({}) == "no blocks"


def test_export_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        export_litematic({}, "x", str(tmp_path / "x"))
