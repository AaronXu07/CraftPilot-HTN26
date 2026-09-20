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
    chunk_size_for,
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


def test_place_grid_shows_the_placing_outline_before_queueing():
    bridge = FakeBridge(pos=(0.5, 64.0, 0.5), yaw=180.0)
    player = bridge.player()
    grid = _grid(facing_from_yaw(player["yaw"]))
    ctx = PlaceContext(bridge=bridge, player=player, opts=PlaceOptions())
    res = place_grid(grid, ctx, "test")
    assert bridge.outlines == [{"min": res["world_bbox"][0], "max": res["world_bbox"][1], "phase": "placing"}]
    assert res["warnings"] == []


def test_place_grid_survives_an_outline_failure():
    class NoOutline(FakeBridge):
        def outline(self, lo, hi, phase):
            from craftpilot.place.bridge import BridgeError
            raise BridgeError("mod said no")

    bridge = NoOutline()
    grid = _grid("south")
    res = place_grid(grid, PlaceContext(bridge=bridge, player=bridge.player(), opts=PlaceOptions()), "test")
    assert len(bridge.calls) == 1 and any("outline not shown" in w for w in res["warnings"])


@pytest.mark.parametrize("yaw,facing,swap", [(180.0, "south", False), (270.0, "west", True)])
def test_run_build_outlines_bounds_then_trimmed_box(tmp_path, yaw, facing, swap):
    from craftpilot.cli import run_build

    bridge = FakeBridge(pos=(0.5, 64.0, 0.5), yaw=yaw)
    player = bridge.player()
    assert facing_from_yaw(yaw) == facing
    ctx = PlaceContext(bridge=bridge, player=player, opts=PlaceOptions())
    program = simple_program()
    bounds = Bounds(width=13, height=16, depth=11)
    res = run_build(program, bounds, 1, tmp_path / "b.litematic", False, "t", "text", [], facing, ctx)
    assert [o["phase"] for o in bridge.outlines] == ["generating", "placing"]
    gen = bridge.outlines[0]
    w = gen["max"][0] - gen["min"][0] + 1
    h = gen["max"][1] - gen["min"][1] + 1
    d = gen["max"][2] - gen["min"][2] + 1
    assert (w, h, d) == ((11, 16, 13) if swap else (13, 16, 11))
    placing = bridge.outlines[1]
    assert [placing["min"], placing["max"]] == res["placement"]["world_bbox"]
    # The trimmed box sits inside the bounds box.
    assert all(placing["min"][i] >= gen["min"][i] and placing["max"][i] <= gen["max"][i] for i in range(3))


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


def test_serve_anchors_to_the_pose_sent_with_the_command(monkeypatch):
    """The player may walk off while the LLM composes; the build must land where the outline was drawn."""
    from fastapi.testclient import TestClient

    from craftpilot import serve
    from craftpilot.place import bridge as bridge_mod

    live = FakeBridge(pos=(500.5, 70.0, 500.5), yaw=0.0)   # where the player is *now*
    monkeypatch.setattr(bridge_mod, "HttpBridge", lambda *a, **k: live)
    client = TestClient(serve.app)
    r = client.post("/build", json={"text": "small stone cottage", "use_llm": False, "place": True, "seed": 1,
                                    "bounds": {"width": 13, "height": 16, "depth": 11},
                                    "pos": [0.5, 64.0, 0.5], "yaw": 180.0})
    assert r.status_code == 200, r.text
    placement = r.json()["placement"]
    assert placement["origin"][1] == 64                      # feet height when the command was typed
    assert placement["world_bbox"][1][2] <= -3               # north of z=0, not next to (500, 500)
    assert r.json()["facing"] == "south"
    # Without a pose the service falls back to the live position, as before.
    r2 = client.post("/build", json={"text": "small stone cottage", "use_llm": False, "place": True, "seed": 1,
                                     "bounds": {"width": 13, "height": 16, "depth": 11}})
    assert r2.json()["placement"]["origin"][1] == 70


def test_build_lands_exactly_where_the_bounds_outline_was(tmp_path):
    """The mod anchors its hologram with the full bounds box; the placed blocks must share that origin."""
    from craftpilot.cli import run_build

    bridge = FakeBridge(pos=(0.5, 64.0, 0.5), yaw=180.0)
    ctx = PlaceContext(bridge=bridge, player=bridge.player(), opts=PlaceOptions())
    res = run_build(simple_program(), Bounds(width=13, height=16, depth=11), 1, tmp_path / "b.litematic", False,
                    "t", "text", [], "south", ctx)
    generating, placing = bridge.outlines
    assert generating["phase"] == "generating" and placing["phase"] == "placing"
    assert res["placement"]["origin"] == generating["min"]            # ghost origin == build origin
    # The trimmed (visual) box sits inside the bounds box, offset by the engine's margins.
    assert all(placing["min"][i] >= generating["min"][i] and placing["max"][i] <= generating["max"][i]
               for i in range(3))
    assert (placing["min"], placing["max"]) != (generating["min"], generating["max"])   # the margins are real


def test_chunk_size_scales_with_the_build():
    opts = PlaceOptions(chunk_blocks=1500, chunk_max=6000, target_seconds=6.0, delay_ms=60)
    assert chunk_size_for(2_000, opts) == 1500                        # small: keep the layer pace
    assert chunk_size_for(70_000, opts) == 1750                       # 40 chunks x 3 ticks (1 + 60 ms) = 6 s
    assert chunk_size_for(150_000, opts) == 3750
    assert chunk_size_for(1_000_000, opts) == 6000                    # capped per tick
    fast = PlaceOptions(chunk_blocks=1500, chunk_max=6000, target_seconds=6.0, delay_ms=0)
    assert chunk_size_for(150_000, fast) == 1500                      # 120 chunks x 1 tick fits already
    blocks = [(i % 40, i // 1600, (i // 40) % 40, "minecraft:stone") for i in range(150_000)]
    chunks = layer_chunks(blocks, chunk_size_for(len(blocks), opts))
    assert 30 <= len(chunks) <= 60
    last_y = -1
    for c in chunks:
        assert min(b[1] for b in c) >= last_y
        last_y = max(b[1] for b in c)
