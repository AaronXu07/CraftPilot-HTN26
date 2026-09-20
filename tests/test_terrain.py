"""Siting a build on real terrain: origin choice, ground level, cut / fill / grading / steps."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from fastapi.testclient import TestClient

from craftpilot import terrain
from craftpilot.blocks.catalog import known_blocks
from craftpilot.blocks.state import BlockRef
from craftpilot.config import SETTINGS
from craftpilot.engine.pipeline import generate
from craftpilot.grid.ops import quarter_turns_for_facing, rotate_cw
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.place.bridge import FakeBridge
from craftpilot.place.placer import PlaceContext, PlaceOptions, place_grid
from craftpilot.program.exemplars import load_one
from craftpilot.terrain import PlacementRequest, TerrainSample, place

AIR = "minecraft:air"
GRASS = "minecraft:grass_block"
PLAYER = (0.5, 64.0, 0.5)


@pytest.fixture(scope="module")
def cottage():
    """The cottage exemplar rendered and rotated to face north (player looking south, yaw 0)."""
    ex = load_one(SETTINGS.exemplars_dir, "cottage")
    grid = generate(ex.program, ex.program.bounds, 7)
    rotate_cw(grid, quarter_turns_for_facing("north"))
    return grid


def sample(fn, r: int = 40, top: str = GRASS) -> TerrainSample:
    return TerrainSample.from_heights({(x, z): (fn(x, z), top) for x in range(-r, r) for z in range(-r, r)})


def flat(x, z):
    return 63


def slope(x, z):
    """63 in front of the player rising one per row from z=5 to 68 (the cottage stands at z >= 3)."""
    return 63 + max(0, min(5, z - 4))


def world_blocks(grid, origin) -> dict[tuple[int, int, int], str]:
    ox, oy, oz = origin
    return {(ox + int(x), oy + int(y), oz + int(z)): grid.palette[int(grid.block[x, y, z])].block_id
            for x, y, z in zip(*np.nonzero(grid.block >= 0))}


# ------------------------------------------------------------------------------------ unit level


def test_terrain_sample_roundtrip_keeps_nulls():
    hs = {(0, 0): (63, GRASS), (1, 0): (70, "minecraft:stone"), (0, 1): (62, "minecraft:water[level=0]")}
    s = TerrainSample.from_heights(hs)
    assert s.heights == [[63, 70], [62, None]] and s.tops[1][1] is None
    assert s.to_heights() == hs
    assert TerrainSample(x0=0, z0=0, heights=[[1]]).to_heights() == {(0, 0): (1, GRASS)}  # tops default to grass


def test_facing_from_yaw_faces_the_player():
    assert terrain.facing_from_yaw(0) == "north"  # looking south -> door faces north, toward the player
    assert terrain.facing_from_yaw(90) == "east"
    assert terrain.facing_from_yaw(180) == "south"
    assert terrain.facing_from_yaw(-90) == "west"


@pytest.mark.parametrize("facing,yaw", [("north", 0.0), ("south", 180.0), ("east", 90.0), ("west", 270.0)])
def test_place_uses_the_placer_anchor_rule(facing, yaw):
    ex = load_one(SETTINGS.exemplars_dir, "cottage")
    grid = generate(ex.program, ex.program.bounds, 7)
    rotate_cw(grid, quarter_turns_for_facing(facing))
    p = place(grid, PlacementRequest(pos=PLAYER, yaw=yaw))
    assert p["facing"] == facing
    x0, z0, x1, z1 = p["footprint"]
    near = {"north": z0 == 3, "south": z1 == -3, "east": x1 == -3, "west": x0 == 3}[facing]  # 2 air blocks from the player
    assert near and (x0 <= 0 <= x1 if facing in ("north", "south") else z0 <= 0 <= z1)


def test_choose_ground_is_a_little_below_the_median_and_sinks_tall_builds():
    base = {(x, 0) for x in range(10)}
    heights = {(x, 0): (60 + x, GRASS) for x in range(10)}
    assert terrain.choose_ground(heights, base, feet_y=99, build_height=5) == (64, "")  # 40th percentile of 60..69 is 63
    assert terrain.choose_ground({}, base, feet_y=99, build_height=5)[0] == 99
    high = {c: (300, "minecraft:stone") for c in base}
    ground, note = terrain.choose_ground(high, base, feet_y=301, build_height=30)
    assert ground == terrain.MAX_BUILD_Y - 30 + 1 and "sunk" in note
    with pytest.raises(ValueError):
        terrain.choose_ground(high, base, feet_y=301, build_height=60)


def test_materials_come_from_the_building_and_exist_in_26_2(cottage):
    known = known_blocks()
    assert known is not None and "minecraft:cobblestone" in known
    f = terrain.foundation_block(cottage, known)
    assert f in known and f == "minecraft:cobblestone"  # the cottage's foundation ring
    s = terrain.stairs_block(cottage, f, known)
    assert s in known and s.endswith("_stairs")
    assert terrain.stairs_for("minecraft:stone_bricks") == "minecraft:stone_brick_stairs"
    assert terrain.stairs_for("minecraft:deepslate_tiles") == "minecraft:deepslate_tile_stairs"
    assert terrain.filler_for(GRASS) == "minecraft:dirt" and terrain.filler_for("minecraft:sand") == "minecraft:sand"


# --------------------------------------------------------------------------------- placement


def test_no_terrain_places_at_the_feet_with_no_edits(cottage):
    p = place(cottage, PlacementRequest(pos=PLAYER, yaw=0.0))
    assert p["origin"][1] == 64 and p["site"] is None and p["edits"] == [] and p["facing"] == "north"
    assert p["schematic_size"][0] == cottage.W and p["schematic_size"][2] == cottage.D


def test_flat_world_needs_no_edits(cottage):
    p = place(cottage, PlacementRequest(pos=PLAYER, yaw=0.0, terrain=sample(flat)))
    assert p["origin"][1] == 64 and p["site"]["strategy"] == "flat" and p["edits"] == []


def test_slope_cuts_the_hill_fills_the_low_side_and_grades_the_apron(cottage):
    p = place(cottage, PlacementRequest(pos=PLAYER, yaw=0.0, terrain=sample(slope)), known=known_blocks())
    site = p["site"]
    assert site["strategy"] == "fill" and site["range"] == 5 and site["lowest"] == 63 and site["highest"] == 68
    ground = p["origin"][1]
    assert 66 <= ground <= 69  # most of the base stands on the 68 shelf; the front rows get a plinth
    blocks = world_blocks(cottage, p["origin"])
    edits = {(x, y, z): st for x, y, z, st in p["edits"]}
    assert not set(edits) & set(blocks), "edits never overlap schematic blocks"
    z0 = p["footprint"][1]
    base = {(x, z) for (x, y, z) in blocks if y == ground}
    for (x, z) in base:
        h = slope(x, z)
        # solid right under the base everywhere: terrain or foundation fill
        assert h >= ground - 1 or edits.get((x, ground - 1, z)) == "minecraft:cobblestone", (x, z)
        # nothing of the hill left inside the building above ground
        for y in range(ground, h + 1):
            assert (x, y, z) in blocks or edits.get((x, y, z)) == AIR, (x, y, z)
    fills = [p_ for p_, st in edits.items() if st == "minecraft:cobblestone"]
    graded = [p_ for p_, st in edits.items() if st in (GRASS, "minecraft:dirt")]
    assert fills and graded and site["stats"]["fill"] == len(fills)
    # the apron in front (toward the player, downhill) is re-topped with grass, never more than one block per column
    tops = {}
    for (x, y, z), st in edits.items():
        if st == GRASS:
            tops[(x, z)] = max(tops.get((x, z), -999), y)
    dz_col = p["origin"][0] + cottage.door[0]
    front = sorted((z, y) for (x, z), y in tops.items() if x == dz_col and z < z0)
    assert front and all(b[1] - a[1] <= 1 for a, b in pairwise(front))


def test_door_gets_steps_down_to_the_graded_ground(cottage):
    p = place(cottage, PlacementRequest(pos=PLAYER, yaw=0.0, terrain=sample(slope)), known=known_blocks())
    ground = p["origin"][1]
    stairs = [(x, y, z, st) for x, y, z, st in p["edits"] if "_stairs[" in st]
    assert stairs and p["site"]["stats"]["steps"] >= len(stairs)
    dx = p["origin"][0] + cottage.door[0]
    mine = sorted((z, y) for x, y, z, st in stairs if x == dx)
    assert mine and mine[-1][1] == ground - 1  # first step right below the platform, just outside the ring
    assert all(b[1] - a[1] == 1 for a, b in pairwise(mine))  # one step per column
    assert all("facing=south" in st for _, _, _, st in stairs)  # ascending back toward the door (north face)


def test_cliff_is_cut_to_a_terrace_and_a_tall_build_is_refused(cottage):
    cliff = sample(lambda x, z: 63 + (10 if z > 10 else 0))
    p = place(cottage, PlacementRequest(pos=PLAYER, yaw=0.0, terrain=cliff))
    assert p["site"]["strategy"] == "stilts" and p["site"]["range"] == 10 and p["origin"][1] == 64
    assert p["site"]["stats"]["cut"] > 0 and not any(st == "minecraft:cobblestone" for *_, st in p["edits"])
    high = sample(lambda x, z: 305)  # the cottage is ~20 tall: sunk to fit, not refused
    p2 = place(cottage, PlacementRequest(pos=(0.5, 306.0, 0.5), yaw=0.0, terrain=high))
    assert p2["origin"][1] + p2["schematic_size"][1] - 1 <= terrain.MAX_BUILD_Y and "sunk" in p2["site"]["note"]
    tower = SemanticGrid(3, 60, 3, np.random.default_rng(0))  # a 60-block pillar cannot be sunk far enough
    tower.block[1, :, 1] = tower.intern(BlockRef.make("minecraft:stone_bricks"))
    with pytest.raises(ValueError):
        place(tower, PlacementRequest(pos=(0.5, 306.0, 0.5), yaw=0.0, terrain=high))


def test_water_and_unknown_columns_are_left_alone(cottage):
    # grass to z=17 (the cottage's last row), a lake behind it at 18..19, nothing sampled beyond
    lake = TerrainSample.from_heights({(x, z): (60, "minecraft:water[level=0]") if z > 17 else (63 + (2 if z > 8 else 0), GRASS)
                                       for x in range(-40, 40) for z in range(-40, 20)})
    p = place(cottage, PlacementRequest(pos=PLAYER, yaw=0.0, terrain=lake))
    assert p["site"]["unknown"] == 0 and p["edits"]
    assert not any(z > 17 for _, _, z, _ in p["edits"])  # the lake is not graded, the unknown is not invented
    short = TerrainSample.from_heights({(x, z): (63, GRASS) for x in range(-40, 40) for z in range(-40, 10)})
    p2 = place(cottage, PlacementRequest(pos=PLAYER, yaw=0.0, terrain=short))
    assert p2["site"]["unknown"] > 0 and "outside the terrain sample" in p2["site"]["summary"]
    assert not any(z > 9 for _, _, z, _ in p2["edits"])


def _grid(facing: str):
    ex = load_one(SETTINGS.exemplars_dir, "cottage")
    grid = generate(ex.program, ex.program.bounds, 7)
    rotate_cw(grid, quarter_turns_for_facing(facing))
    return grid


def test_placer_seats_the_build_on_the_surveyed_ground():
    bridge = FakeBridge(pos=PLAYER, yaw=0.0, terrain=slope)  # looking south, the ground rises away
    ctx = PlaceContext(bridge=bridge, player=bridge.player(), opts=PlaceOptions())
    res = place_grid(_grid("north"), ctx, "t")
    assert res["site"] is not None and res["site"]["strategy"] == "fill"
    ground = res["origin"][1]
    assert ground == res["site"]["ground"] and 66 <= ground <= 69
    assert any(w.startswith("Site:") for w in res["warnings"])
    sent = [b for blocks, _ in bridge.calls[0]["chunks"] for b in blocks]
    ys = [b[1] for b in sent]
    assert min(ys) < ground  # the foundation goes below the plinth row
    assert sent == sorted(sent, key=lambda b: b[1]) or all(a[1] <= c[1] for a, c in pairwise(sent))  # bottom up
    fills = [b for b in sent if b[3] == "minecraft:cobblestone" and b[1] < ground]
    stairs = [b for b in sent if "_stairs[" in b[3] and b[1] < ground]
    grass = [b for b in sent if b[3] == GRASS]
    assert fills and stairs and grass
    # air inside the base only: the apron is graded, not flattened into a crater
    x0, z0, x1, z1 = terrain.footprint_of(_grid("north"), (res["origin"][0], res["origin"][2]))
    air_cols = {(b[0], b[2]) for b in sent if b[3] == AIR}
    assert air_cols and all(x0 <= x <= x1 and z0 <= z <= z1 for x, z in air_cols)


def test_placer_without_a_survey_places_at_the_feet():
    bridge = FakeBridge(pos=PLAYER, yaw=0.0)  # empty fake world: nothing to survey
    ctx = PlaceContext(bridge=bridge, player=bridge.player(), opts=PlaceOptions())
    res = place_grid(_grid("north"), ctx, "t")
    assert res["site"] is None and res["origin"][1] == 64
    off = FakeBridge(pos=PLAYER, yaw=0.0, terrain=slope)
    ctx = PlaceContext(bridge=off, player=off.player(), opts=PlaceOptions(terrain=False))
    res = place_grid(_grid("north"), ctx, "t")
    assert res["site"] is None and res["origin"][1] == 64


def test_placer_sink_applies_on_top_of_the_site_ground():
    bridge = FakeBridge(pos=PLAYER, yaw=0.0, terrain=slope)
    ctx = PlaceContext(bridge=bridge, player=bridge.player(), opts=PlaceOptions(sink=1))
    res = place_grid(_grid("north"), ctx, "t")
    ref = FakeBridge(pos=PLAYER, yaw=0.0, terrain=slope)
    res0 = place_grid(_grid("north"), PlaceContext(bridge=ref, player=ref.player(), opts=PlaceOptions()), "t")
    assert res["origin"][1] == res0["origin"][1] - 1


def test_serve_build_places_on_terrain(monkeypatch):

    from craftpilot import serve
    from craftpilot.place import bridge as bridge_mod

    live = FakeBridge(pos=PLAYER, yaw=0.0, terrain=slope)
    monkeypatch.setattr(bridge_mod, "HttpBridge", lambda *a, **k: live)
    client = TestClient(serve.app)
    r = client.post("/build", json={"text": "a cozy cottage", "use_llm": False, "place": True, "seed": 7})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["placement"]["site"]["strategy"] == "fill" and data["placement"]["origin"][1] > 64
    assert any(n.startswith("Site:") for n in data["notes"])
    assert "placing at" in data["summary"]


def test_edits_use_only_known_blocks(cottage):
    known = known_blocks()
    p = place(cottage, PlacementRequest(pos=PLAYER, yaw=0.0, terrain=sample(slope)), known=known)
    ids = {st.split("[", 1)[0] for *_, st in p["edits"]}
    assert ids and ids <= known | {AIR}
