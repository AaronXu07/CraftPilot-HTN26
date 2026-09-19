"""In-game placement: state strings, anchoring, layer chunking, and the fake bridge round trip."""

from __future__ import annotations

import pytest
from test_engine import SAFETY, simple_program

from craftpilot.blocks.state import BlockRef
from craftpilot.engine.pipeline import generate
from craftpilot.grid.ops import quarter_turns_for_facing, rotate_cw
from craftpilot.place.anchor import (
    facing_from_yaw,
    look_from_yaw,
    plan_origin,
    trimmed_bbox,
)
from craftpilot.place.bridge import FORCE_FLAGS, FakeBridge
from craftpilot.place.placer import (
    AIR,
    PlaceContext,
    PlaceOptions,
    grid_to_blocks,
    is_attachable,
    layer_chunks,
    place_grid,
)
from craftpilot.program.model import Bounds
from craftpilot.program.validate import repair


def _grid(facing: str = "south"):
    program, _ = repair(simple_program(), SAFETY)
    grid = generate(program, Bounds(width=13, height=16, depth=11), 3)
    if facing != "south":
        rotate_cw(grid, quarter_turns_for_facing(facing))
    return grid


def test_state_string():
    assert BlockRef.make("minecraft:oak_stairs", half="bottom", facing="north").state_string() == \
        "minecraft:oak_stairs[facing=north,half=bottom]"
    assert BlockRef.make("minecraft:stone").state_string() == "minecraft:stone"


def test_grid_to_blocks_counts_and_offsets():
    grid = _grid()
    x0, y0, z0, x1, y1, z1 = trimmed_bbox(grid)
    origin = (100, 64, -20)
    solid = grid_to_blocks(grid, origin, clear=False)
    assert len(solid) == grid.block_count()
    everything = grid_to_blocks(grid, origin, clear=True)
    assert len(everything) == (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
    assert sum(1 for b in everything if b[3] == AIR) == len(everything) - len(solid)
    xs = [b[0] for b in everything]
    ys = [b[1] for b in everything]
    zs = [b[2] for b in everything]
    assert (min(xs), min(ys), min(zs)) == (x0 + 100, y0 + 64, z0 - 20)
    assert (max(xs), max(ys), max(zs)) == (x1 + 100, y1 + 64, z1 - 20)
    assert all("[" in b[3] or ":" in b[3] for b in everything)


def test_is_attachable():
    for bid in ("minecraft:oak_door", "minecraft:lantern", "minecraft:spruce_trapdoor", "minecraft:red_wall_banner",
                "minecraft:vine", "minecraft:glow_lichen", "minecraft:ladder", "minecraft:stone_button"):
        assert is_attachable(bid), bid
    for bid in ("minecraft:sea_lantern", "minecraft:oak_stairs", "minecraft:iron_bars", "minecraft:oak_fence_gate",
                "minecraft:stone", "minecraft:air"):
        assert not is_attachable(bid), bid


def test_layer_chunks_align_to_layers_and_put_attachables_last():
    sizes = [400, 900, 300, 2000, 10]
    blocks = []
    for y, n in enumerate(sizes):
        for i in range(n):
            state = "minecraft:oak_door[half=lower]" if i % 7 == 0 else "minecraft:stone"
            blocks.append((i % 50, y, i // 50, state))
    chunks = layer_chunks(blocks, max_blocks=1500)
    assert sum(len(c) for c in chunks) == len(blocks)
    assert sorted(b for c in chunks for b in c) == sorted(blocks)
    # Bottom up across chunks.
    last_y = -1
    for c in chunks:
        assert min(b[1] for b in c) >= last_y
        last_y = max(b[1] for b in c)
    # Layers 0+1 fit together (1300); layer 2 (300) would overflow so it starts a new chunk;
    # layer 3 (2000) is split in two; layer 4 rides alone after it.
    assert [sorted({b[1] for b in c}) for c in chunks] == [[0, 1], [2], [3], [3], [4]]
    assert [len(c) for c in chunks] == [1300, 300, 1500, 500, 10]
    # Within each layer of each chunk, attachables come after solids.
    for c in chunks:
        for y in {b[1] for b in c}:
            kinds = [is_attachable(b[3]) for b in c if b[1] == y]
            assert kinds == sorted(kinds)


def test_yaw_to_directions():
    assert [look_from_yaw(y) for y in (0, 90, 180, 270, -90, 359)] == \
        ["south", "west", "north", "east", "east", "south"]
    assert facing_from_yaw(180) == "south"


@pytest.mark.parametrize("look,yaw", [("south", 0), ("west", 90), ("north", 180), ("east", 270)])
def test_plan_origin_puts_building_gap_blocks_ahead_and_centred(look, yaw):
    player = {"pos": [10.5, 64.0, 10.5], "yaw": yaw}
    bbox = (0, 0, 0, 4, 9, 6)  # 5 wide, 10 tall, 7 deep, inclusive
    ox, oy, oz = plan_origin(player, bbox, gap=2, sink=0)
    assert oy == 64
    assert plan_origin(player, bbox, gap=2, sink=1)[1] == 63
    wx0, wx1 = ox + bbox[0], ox + bbox[3]
    wz0, wz1 = oz + bbox[2], oz + bbox[5]
    if look == "south":
        assert wz0 == 10 + 1 + 2 and wx0 <= 10 <= wx1 and (wx0 + wx1) // 2 == 10
    elif look == "north":
        assert wz1 == 10 - 1 - 2 and (wx0 + wx1) // 2 == 10
    elif look == "east":
        assert wx0 == 10 + 1 + 2 and (wz0 + wz1) // 2 == 10
    else:
        assert wx1 == 10 - 1 - 2 and (wz0 + wz1) // 2 == 10


def test_place_grid_with_fake_bridge():
    bridge = FakeBridge(pos=(0.5, 64.0, 0.5), yaw=180.0)  # looking north -> building faces south
    player = bridge.player()
    grid = _grid(facing_from_yaw(player["yaw"]))
    ctx = PlaceContext(bridge=bridge, player=player, opts=PlaceOptions(chunk_blocks=500, delay_ms=60))
    res = place_grid(grid, ctx, "test")
    assert len(bridge.calls) == 1
    call = bridge.calls[0]
    assert call["flags"] == FORCE_FLAGS and call["postprocess"] is False
    assert all(d == 60 for _, d in call["chunks"])
    assert res["chunks"] == len(call["chunks"]) and res["queued"] == call["count"]
    assert res["blocks"] == grid.block_count()
    first_blocks, _ = call["chunks"][0]
    assert min(b[1] for b in first_blocks) == res["origin"][1] == 64
    # Nothing landed in or behind the player's block: the building is north of z=0.
    assert res["world_bbox"][1][2] <= 0 - 1 - 2
    # The door faces the player: it sits on the south wall (the foundation outset and steps stick out one more).
    assert grid.door is not None
    door_world_z = grid.door[2] + res["origin"][2]
    z_min, z_max = res["world_bbox"][0][2], res["world_bbox"][1][2]
    assert z_max - 2 <= door_world_z <= z_max and door_world_z > (z_min + z_max) / 2


def test_serve_build_without_placing_returns_summary():
    from fastapi.testclient import TestClient

    from craftpilot.serve import app

    client = TestClient(app)
    r = client.post("/build", json={"text": "small stone cottage", "use_llm": False, "place": False,
                                    "bounds": {"width": 13, "height": 16, "depth": 11}, "seed": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schematic"].endswith(".litematic")
    assert "summary" in body and "#1" in body["summary"]
    assert "placement" not in body
