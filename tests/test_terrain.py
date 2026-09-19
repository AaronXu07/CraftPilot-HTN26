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
from craftpilot.grid.enums import Role
from craftpilot.grid.ops import quarter_turns_for_facing, rotate_cw
from craftpilot.grid.semantic import SemanticGrid
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


@pytest.mark.parametrize("facing", ["north", "south", "east", "west"])
def test_origin_puts_the_front_two_blocks_from_the_player(cottage, facing):
    ex = load_one(SETTINGS.exemplars_dir, "cottage")
    grid = generate(ex.program, ex.program.bounds, 7)
    rotate_cw(grid, quarter_turns_for_facing(facing))
    ox, oz = terrain.choose_origin_xz(PLAYER, facing, grid)
    cols = terrain.column_mask(grid)
    xs, zs = np.nonzero(cols)
    wx0, wx1, wz0, wz1 = ox + xs.min(), ox + xs.max(), oz + zs.min(), oz + zs.max()
    px, pz = 0, 0
    if facing == "north":
        assert wz0 == pz + 3 and wx0 <= px <= wx1
    elif facing == "south":
        assert wz1 == pz - 3 and wx0 <= px <= wx1
    elif facing == "east":
        assert wx1 == px - 3 and wz0 <= pz <= wz1
    else:
        assert wx0 == px + 3 and wz0 <= pz <= wz1
    # the door is on the side nearest the player (behind the foundation ring and the step column)
    dx, _, dz = grid.door
    assert {"north": 0 < dz + oz - wz0 <= 3, "south": 0 < wz1 - (dz + oz) <= 3,
            "east": 0 < wx1 - (dx + ox) <= 3, "west": 0 < dx + ox - wx0 <= 3}[facing]


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


def test_build_endpoint_returns_placement(monkeypatch, tmp_path):
    from craftpilot import serve

    monkeypatch.setattr(SETTINGS, "schematics_dir", tmp_path)
    client = TestClient(serve.app)
    body = {"text": "a cozy cottage", "player": "t", "origin": [0.5, 64.0, 0.5], "yaw": 0.0, "use_llm": False, "seed": 7,
            "terrain": sample(slope).model_dump()}
    r = client.post("/build", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    pl = data["placement"]
    assert pl["facing"] == "north" and pl["site"]["strategy"] == "fill" and pl["edits"]
    assert any(n.startswith("Site:") for n in data["notes"])
    # edits and regenerate reuse the same site
    r2 = client.post("/regenerate", json={"player": "t", "seed": 8})
    assert r2.status_code == 200 and r2.json()["placement"]["origin"][:1] == pl["origin"][:1]
    # without terrain: feet level, no edits
    r3 = client.post("/build", json={"text": "a cozy cottage", "player": "u", "origin": [0, 64, 0], "yaw": 180.0, "use_llm": False, "seed": 7})
    assert r3.status_code == 200 and r3.json()["placement"]["edits"] == [] and r3.json()["placement"]["facing"] == "south"


def test_edits_use_only_known_blocks(cottage):
    known = known_blocks()
    p = place(cottage, PlacementRequest(pos=PLAYER, yaw=0.0, terrain=sample(slope)), known=known)
    ids = {st.split("[", 1)[0] for *_, st in p["edits"]}
    assert ids and ids <= known | {AIR}
    assert Role.STEP is not None  # the engine's own steps are skipped, ours continue past the ring
