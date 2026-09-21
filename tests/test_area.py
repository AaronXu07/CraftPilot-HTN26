"""A base area marked with two points: facing, footprint, origin, and the ghost -> commit round trip."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_engine import SAFETY, simple_program

from craftpilot.config import SETTINGS
from craftpilot.engine.pipeline import generate
from craftpilot.grid.ops import quarter_turns_for_facing, rotate_cw
from craftpilot.place.anchor import (
    area_footprint,
    area_origin,
    facing_from_area,
    normalise_area,
)
from craftpilot.place.bridge import FakeBridge
from craftpilot.place.placer import AIR, PlaceContext, PlaceOptions, place_grid
from craftpilot.program.model import Bounds
from craftpilot.program.validate import repair


def test_normalise_area_orders_corners():
    assert normalise_area([[10, 64, -3], [2, 70, 5]]) == (2, 64, -3, 10, 70, 5)
    with pytest.raises(ValueError):
        normalise_area([[1, 2, 3]])


def test_facing_is_the_side_the_player_stands_on():
    area = (0, 64, 0, 9, 64, 19)
    assert facing_from_area(area, [5.0, 65.0, 30.0]) == "south"
    assert facing_from_area(area, [5.0, 65.0, -4.0]) == "north"
    assert facing_from_area(area, [25.0, 65.0, 10.0]) == "east"
    assert facing_from_area(area, [-9.0, 65.0, 10.0]) == "west"
    # Standing inside the area (or no position): fall back to the look direction, as when aiming.
    assert facing_from_area(area, [4.5, 65.0, 10.5], yaw=180.0) == "south"
    assert facing_from_area(area, None, yaw=0.0) == "north"


def test_footprint_swaps_for_an_east_or_west_front():
    area = (0, 64, 0, 9, 64, 19)   # 10 wide (x), 20 deep (z) in the world
    assert area_footprint(area, "south") == (10, 20)
    assert area_footprint(area, "north") == (10, 20)
    assert area_footprint(area, "east") == (20, 10)
    assert area_footprint(area, "west") == (20, 10)


def test_origin_sits_on_the_marked_blocks_and_centres_a_smaller_grid():
    area = (100, 63, -20, 109, 65, -1)
    assert area_origin(area, 10, 20) == (100, 64, -20)
    assert area_origin(area, 10, 20, sink=1) == (100, 63, -20)
    assert area_origin(area, 8, 16) == (101, 64, -18)


@pytest.mark.parametrize("facing", ["south", "east"])
def test_place_grid_lands_inside_the_area(facing):
    program, _ = repair(simple_program(), SAFETY)
    area = normalise_area([[200, 70, 300], [212, 70, 310]])     # 13 x 11 in the world
    w, d = area_footprint(area, facing)
    grid = generate(program, Bounds(width=w, height=16, depth=d), 3)
    if facing != "south":
        rotate_cw(grid, quarter_turns_for_facing(facing))
    assert (grid.W, grid.D) == (13, 11)
    live = FakeBridge(pos=(0.5, 64.0, 0.5), yaw=0.0)
    ctx = PlaceContext(bridge=live, player=live.player(), opts=PlaceOptions(terrain=False), area=area)
    res = place_grid(grid, ctx, "t")
    assert res["origin"] == [200, 71, 300]
    xs = [x for (x, y, z), s in live.world.items() if s != AIR]
    zs = [z for (x, y, z), s in live.world.items() if s != AIR]
    ys = [y for (x, y, z), s in live.world.items() if s != AIR]
    assert 200 <= min(xs) and max(xs) <= 212 and 300 <= min(zs) and max(zs) <= 310
    assert min(ys) == 71


def test_area_ghost_then_commit_is_the_same_build(monkeypatch, tmp_path):
    """/build with an area answers a ghost anchored on the area; the G commit places exactly that."""
    from craftpilot import serve
    from craftpilot.place import bridge as bridge_mod

    live = FakeBridge(pos=(106.5, 65.0, 130.5), yaw=0.0)        # the player stands south of the area
    monkeypatch.setattr(bridge_mod, "HttpBridge", lambda *a, **k: live)
    monkeypatch.setattr(SETTINGS, "schematics_dir", tmp_path)
    client = TestClient(serve.app)
    area = [[100, 64, 100], [114, 64, 112]]                     # 15 wide, 13 deep
    r = client.post("/build", json={"text": "small stone cottage", "player": "a", "use_llm": False, "seed": 9,
                                    "ghost": True, "place": True, "area": area,
                                    "pos": [106.5, 65.0, 130.5], "yaw": 0.0})
    assert r.status_code == 200, r.text
    body = r.json()
    ghost = body["ghost"]
    assert body["bounds"]["width"] == 15 and body["bounds"]["depth"] == 13
    assert ghost["anchor"]["facing"] == "south" and ghost["anchor"]["turns"] == 0
    assert ghost["anchor"]["origin"] == [100, 65, 100]
    assert live.calls == []

    r = client.post("/regenerate", json={"player": "a", "place": True, "seed": 9})
    assert r.status_code == 200, r.text
    placement = r.json()["placement"]
    assert placement["origin"] == [100, 65, 100]
    placed = {pos for pos, state in live.world.items() if state != AIR}
    flat = ghost["blocks"]
    ox, oy, oz = placement["origin"]
    ghost_cells = {(flat[i] + ox, flat[i + 1] + oy, flat[i + 2] + oz) for i in range(0, len(flat), 4)}
    assert ghost_cells == placed
    assert all(100 <= x <= 114 and 100 <= z <= 112 for x, _, z in placed)


def test_area_from_the_east_rotates_the_footprint(monkeypatch, tmp_path):
    from craftpilot import serve
    from craftpilot.place import bridge as bridge_mod

    live = FakeBridge(pos=(140.5, 65.0, 106.5), yaw=90.0)       # east of the area
    monkeypatch.setattr(bridge_mod, "HttpBridge", lambda *a, **k: live)
    monkeypatch.setattr(SETTINGS, "schematics_dir", tmp_path)
    client = TestClient(serve.app)
    area = [[100, 64, 100], [114, 64, 112]]                     # 15 (x) by 13 (z)
    r = client.post("/build", json={"text": "small stone cottage", "player": "b", "use_llm": False, "seed": 9,
                                    "ghost": True, "place": True, "area": area,
                                    "pos": [140.5, 65.0, 106.5], "yaw": 90.0})
    assert r.status_code == 200, r.text
    body = r.json()
    # Generated facing south with the front along z, then turned to face east: the cloud is 13 wide, 15 deep.
    assert (body["ghost"]["width"], body["ghost"]["depth"]) == (13, 15)
    assert body["ghost"]["anchor"]["facing"] == "east" and body["ghost"]["anchor"]["turns"] == 3   # south -> east clockwise
    r = client.post("/regenerate", json={"player": "b", "place": True, "seed": 9})
    assert r.status_code == 200, r.text
    placed = {pos for pos, state in live.world.items() if state != AIR}
    assert placed and all(100 <= x <= 114 and 100 <= z <= 112 for x, _, z in placed)
    assert r.json()["facing"] == "east"
