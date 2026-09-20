import os

import pytest

from copilot.engine.schematic import export_litematic, materials_list, materials_list_text, stacks_text


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
    assert txt.startswith("4 blocks (1 stack") and "oak_stairs" in txt and "stone" in txt
    assert materials_list_text({}) == "no blocks"


def test_materials_list_survival_stacks():
    # survival players count in stacks of 64 and shulker boxes of 27 stacks
    assert stacks_text(0) == "0"
    assert stacks_text(63) == "63"
    assert stacks_text(64) == "1 stack"
    assert stacks_text(100) == "1 stack + 36"
    assert stacks_text(64 * 27) == "27 stacks = 1 shulker"
    assert stacks_text(64 * 28 + 1) == "28 stacks + 1 = 1 shulker + 1 stack"
    assert stacks_text(64 * 60) == "60 stacks = 2 shulkers + 6 stacks"
    bm = {(i, 0, 0): "minecraft:stone_bricks" for i in range(1800)}
    bm.update({(i, 1, 0): "minecraft:oak_stairs[facing=east]" for i in range(100)})
    bm.update({(i, 2, 0): "minecraft:lantern" for i in range(63)})
    lines = materials_list_text(bm).splitlines()
    assert lines[0] == "1963 blocks (31 stacks, 2 shulker boxes), 3 block types:"
    assert lines[1].split() == ["stone_bricks", "1800", "(28", "stacks", "+", "8", "=", "1", "shulker", "+", "1", "stack)"]
    assert lines[2].split() == ["oak_stairs", "100", "(1", "stack", "+", "36)"]
    assert lines[3].split() == ["lantern", "63", "(63)"]
    # long lists are capped with a remainder line
    many = {(i, 0, 0): f"minecraft:block_{i}" for i in range(45)}
    tail = materials_list_text(many, limit=40).splitlines()[-1]
    assert tail.strip() == "... 5 more types (5 blocks)"


def test_export_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        export_litematic({}, "x", str(tmp_path / "x"))
