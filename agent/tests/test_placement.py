import pytest

from copilot.placement import (
    estimate_seconds,
    map_bbox,
    place_scene,
    plan_anchor,
    rotate_state,
    rotate_xz,
    transform_block_map,
    undo_world,
)
from copilot.session import Session
from mock_mod import MockBridge

STAIR = "minecraft:oak_stairs[facing={f},half=bottom,shape=inner_left,waterlogged=false]"


@pytest.mark.parametrize("k,facing", [(0, "north"), (1, "east"), (2, "south"), (3, "west"), (4, "north")])
def test_rotate_state_facing(k, facing):
    assert rotate_state(STAIR.format(f="north"), k) == STAIR.format(f=facing)


def test_rotate_state_other_properties():
    assert rotate_state("minecraft:oak_log[axis=x]", 1) == "minecraft:oak_log[axis=z]"
    assert rotate_state("minecraft:oak_log[axis=x]", 2) == "minecraft:oak_log[axis=x]"
    assert rotate_state("minecraft:oak_log[axis=y]", 3) == "minecraft:oak_log[axis=y]"
    assert rotate_state("minecraft:white_banner[rotation=14]", 1) == "minecraft:white_banner[rotation=2]"
    assert rotate_state("minecraft:white_banner[rotation=0]", 3) == "minecraft:white_banner[rotation=12]"
    assert rotate_state("minecraft:oak_fence[east=false,north=true,south=false,waterlogged=false,west=false]", 1) == "minecraft:oak_fence[east=true,north=false,south=false,waterlogged=false,west=false]"
    assert rotate_state("minecraft:wall_torch[facing=up]", 1) == "minecraft:wall_torch[facing=up]"
    assert rotate_state("minecraft:oak_door[facing=north,half=lower,hinge=left,open=false,powered=false]", 1) == "minecraft:oak_door[facing=east,half=lower,hinge=left,open=false,powered=false]"
    assert rotate_state("minecraft:rail[shape=north_south,waterlogged=false]", 1) == "minecraft:rail[shape=east_west,waterlogged=false]"
    assert rotate_state("minecraft:stone", 1) == "minecraft:stone"


def test_rotate_xz_voxel_math():
    assert rotate_xz(3, 5, 0) == (3, 5)
    assert rotate_xz(3, 5, 1) == (-6, 3)
    assert rotate_xz(3, 5, 4) == (3, 5)
    # 4 quarter turns of a footprint come back to itself; 1 turn maps a 2x3 footprint to 3x2
    fp = {(x, z) for x in range(2) for z in range(3)}
    r1 = {rotate_xz(x, z, 1) for x, z in fp}
    xs = {p[0] for p in r1}
    zs = {p[1] for p in r1}
    assert len(xs) == 3 and len(zs) == 2


@pytest.mark.parametrize("facing,expect_lo,expect_hi", [
    ("north", (-4, 64, -12), (5, 68, -3)),
    ("east", (3, 64, -4), (12, 68, 5)),
    ("south", (-4, 64, 3), (5, 68, 12)),
    ("west", (-12, 64, -4), (-3, 68, 5)),
])
def test_plan_anchor_places_build_in_front(facing, expect_lo, expect_hi):
    player = {"pos": [0.5, 64.0, 0.5], "facing": facing}
    scene_map = {(x, y, z): "minecraft:stone" for x in range(10) for y in range(5) for z in range(10)}
    anchor, k = plan_anchor(player, ((0, 0, 0), (10, 5, 10)))
    world = transform_block_map(scene_map, anchor, k)
    lo, hi = map_bbox(world)
    assert (lo, hi) == (expect_lo, expect_hi)
    assert len(world) == len(scene_map)


def test_front_faces_player():
    # front marker at the scene's +z edge; after placement facing east it must be on the west (player) side
    scene_map = {(x, 0, z): "minecraft:stone" for x in range(4) for z in range(4)}
    scene_map[(1, 1, 3)] = "minecraft:torch"  # front (+z) marker
    anchor, k = plan_anchor({"pos": [0.5, 64.0, 0.5], "facing": "east"}, ((0, 0, 0), (4, 2, 4)))
    world = transform_block_map(scene_map, anchor, k)
    torch = next(p for p, s in world.items() if s == "minecraft:torch")
    lo, hi = map_bbox(world)
    assert torch[0] == lo[0]  # on the west edge, nearest the player


def test_diff_placement_sends_only_changes():
    b = MockBridge()
    s = Session("p")
    bm = {(x, 0, z): "minecraft:stone" for x in range(3) for z in range(3)}
    msg = place_scene(s, b, bm, mode="diff", animate=False)
    assert "9 set" in msg and b.calls[-1]["count"] == 9
    assert s.world.anchor is not None and s.world.pre_scan is not None and s.world.placements == 1
    assert len(s.world.placed) == 9
    # change one block, add one, remove one
    bm2 = dict(bm)
    bm2[(0, 0, 0)] = "minecraft:oak_planks"
    bm2[(0, 1, 0)] = "minecraft:glass"
    del bm2[(2, 0, 2)]
    msg2 = place_scene(s, b, bm2, mode="diff", animate=False)
    assert "2 set, 1 cleared" in msg2 and b.calls[-1]["count"] == 3
    assert s.world.placements == 2 and len(s.world.placed) == 9
    # unchanged -> nothing sent
    msg3 = place_scene(s, b, bm2, mode="diff", animate=False)
    assert "up to date" in msg3 and len(b.calls) == 2
    # full resends everything
    msg4 = place_scene(s, b, bm2, mode="full", animate=False)
    assert b.calls[-1]["count"] == 9
    assert "9 set" in msg4


def test_animated_chunks_bottom_up():
    b = MockBridge()
    s = Session("p")
    bm = {(x, y, 0): "minecraft:stone" for x in range(4) for y in range(3)}
    place_scene(s, b, bm, mode="diff", animate=True, chunk_size=4, delay_ms=60)
    call = b.calls[-1]
    assert len(call["chunks"]) == 3 and all(c["delay_ms"] == 60 for c in call["chunks"])
    assert estimate_seconds(12, 4, 60) > 0


def test_undo_world_restores_including_air_and_ground():
    b = MockBridge(ground_y=63)
    s = Session("p")
    b.set_block(0, 64, -5, "minecraft:oak_planks")  # pre-existing block the build will overwrite
    bm = {(x, 0, z): "minecraft:stone" for x in range(3) for z in range(3)}
    bm[(1, 1, 1)] = "minecraft:glass"
    place_scene(s, b, bm, mode="diff", animate=False)
    placed = dict(s.world.placed)
    assert any(b.get_block(*p) == st for p, st in placed.items())
    # edit: remove a block (cleared to air) then undo everything
    bm2 = {k: v for k, v in bm.items() if k != (1, 1, 1)}
    place_scene(s, b, bm2, mode="diff", animate=False)
    msg = undo_world(s, b)
    assert "restored" in msg
    for p in placed:
        assert b.get_block(*p) == ("minecraft:oak_planks" if p == (0, 64, -5) else ("minecraft:air" if p[1] > 63 else b.get_block(*p)))
    assert b.get_block(0, 64, -5) == "minecraft:oak_planks"
    assert s.world.placed == {} and s.world.pre_scan is None and s.world.anchor is None
    assert undo_world(s, b).startswith("nothing to undo")
