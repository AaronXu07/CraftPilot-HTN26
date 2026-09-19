"""Scene -> world placement (plan.md §9).

Conventions
-----------
* Scene coords: x east, y up, z south, origin = ground level at the build anchor. The scene's **+z
  (south) side is the front** of a build (that is where a gatehouse / entrance goes).
* At placement the scene is rotated by `quarter_turns` clockwise (viewed from above: north -> east ->
  south -> west) so that the front faces the player, then translated by `anchor` (world coordinate of
  the scene origin). Player facing north -> 0 turns, east -> 1, south -> 2, west -> 3.
* The build is centred laterally on the player with 2 air blocks between the player and its nearest
  block; ground level (scene y=0) maps to the block the player stands in (floor(player.y)).
* A voxel [x, x+1) x [z, z+1) rotated one quarter turn CW about the origin becomes voxel (-z-1, x).
* Block-state properties are remapped per quarter turn with `PROPERTY_REMAP` (facing, axis,
  rotation, north/east/south/west connectivity, rail shapes). Stairs `shape` (inner_left, ...) and
  door `hinge` are invariant under rotation (only mirroring would swap them) and are left alone.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .engine.coretypes import BlockMap, IVec3, format_state, parse_state

FACING_TURNS = {"north": 0, "east": 1, "south": 2, "west": 3}
FACING_VEC = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
CW = {"north": "east", "east": "south", "south": "west", "west": "north"}

# value remaps applied once per clockwise quarter turn, keyed by property name
_RAIL_CW = {
    "north_south": "east_west", "east_west": "north_south",
    "ascending_north": "ascending_east", "ascending_east": "ascending_south",
    "ascending_south": "ascending_west", "ascending_west": "ascending_north",
    "north_east": "south_east", "south_east": "south_west", "south_west": "north_west", "north_west": "north_east",
}
PROPERTY_REMAP: Dict[str, Dict[str, str]] = {
    "facing": {**CW, "up": "up", "down": "down"},
    "axis": {"x": "z", "z": "x", "y": "y"},
    "rotation": {str(i): str((i + 4) % 16) for i in range(16)},
    "shape": _RAIL_CW,  # only rail shapes are listed; stair shapes are absent and pass through untouched
    "orientation": {},  # jigsaw; left alone
}
# connectivity props (fences, walls, panes, vines, redstone, mushroom blocks): the value that was on
# `north` moves to `east`, etc.
_CONNECTIVITY = ("north", "east", "south", "west")


def rotate_state(state: str, quarter_turns: int) -> str:
    """Rotate a block state string by k clockwise quarter turns (k in 0..3)."""
    k = int(quarter_turns) % 4
    if k == 0 or "[" not in state:
        return state
    block_id, props = parse_state(state)
    for _ in range(k):
        new_props = dict(props)
        for key, val in props.items():
            table = PROPERTY_REMAP.get(key)
            if table and val in table:
                new_props[key] = table[val]
        if all(c in props for c in _CONNECTIVITY):
            new_props["east"] = props["north"]
            new_props["south"] = props["east"]
            new_props["west"] = props["south"]
            new_props["north"] = props["west"]
        props = new_props
    return format_state(block_id, props)


def rotate_xz(x: int, z: int, quarter_turns: int) -> Tuple[int, int]:
    """Rotate voxel (x, z) about the scene origin by k CW quarter turns: (x, z) -> (-z-1, x)."""
    k = int(quarter_turns) % 4
    for _ in range(k):
        x, z = -z - 1, x
    return x, z


def rotate_bbox_xz(lo: Sequence[float], hi: Sequence[float], quarter_turns: int) -> Tuple[List[float], List[float]]:
    """Rotate a continuous bbox (lo, hi) about the origin (x/z only)."""
    x0, z0, x1, z1 = float(lo[0]), float(lo[2]), float(hi[0]), float(hi[2])
    for _ in range(int(quarter_turns) % 4):
        x0, z0, x1, z1 = -z1, x0, -z0, x1
    return [x0, float(lo[1]), z0], [x1, float(hi[1]), z1]


def transform_block_map(block_map: BlockMap, anchor: Sequence[int], quarter_turns: int) -> BlockMap:
    """Scene-space BlockMap -> world-space BlockMap (rotate about origin, remap states, translate)."""
    ax, ay, az = (int(v) for v in anchor)
    k = int(quarter_turns) % 4
    out: BlockMap = {}
    cache: Dict[str, str] = {}
    for (x, y, z), state in block_map.items():
        rx, rz = rotate_xz(int(x), int(z), k)
        if k:
            st = cache.get(state)
            if st is None:
                st = rotate_state(state, k)
                cache[state] = st
        else:
            st = state
        out[(rx + ax, int(y) + ay, rz + az)] = st
    return out


def map_bbox(block_map: BlockMap) -> Optional[Tuple[IVec3, IVec3]]:
    """Inclusive integer bbox of a BlockMap, or None if empty."""
    if not block_map:
        return None
    xs = [p[0] for p in block_map]
    ys = [p[1] for p in block_map]
    zs = [p[2] for p in block_map]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _bbox_pair(scene_bbox: Any) -> Tuple[List[float], List[float]]:
    if hasattr(scene_bbox, "lo") and hasattr(scene_bbox, "hi"):
        return [float(v) for v in scene_bbox.lo], [float(v) for v in scene_bbox.hi]
    lo, hi = scene_bbox
    return [float(v) for v in lo], [float(v) for v in hi]


def plan_anchor(player: Dict[str, Any], scene_bbox: Any, gap: int = 2) -> Tuple[IVec3, int]:
    """Choose (anchor, quarter_turns) so the build stands `gap` air blocks in front of the player,
    centred laterally, front (scene +z) facing the player, ground at the player's feet.

    `scene_bbox` is a Bbox or (lo, hi) pair in scene coordinates (hi exclusive, as voxel ranges).
    """
    facing = str(player.get("facing") or facing_from_yaw(float(player.get("yaw", 180.0)))).lower()
    k = FACING_TURNS.get(facing, 0)
    px, py, pz = (float(v) for v in player["pos"])
    fx, fz = FACING_VEC[facing]
    lo, hi = _bbox_pair(scene_bbox)
    rlo, rhi = rotate_bbox_xz(lo, hi, k)
    bx, bz = math.floor(px), math.floor(pz)
    if fz != 0:  # player looks along z: near side is rlo.z (looking south) or rhi.z (looking north)
        if fz > 0:
            az = bz + 1 + gap - int(round(rlo[2]))
        else:
            az = bz - gap - int(round(rhi[2]))
        ax = int(round(px - (rlo[0] + rhi[0]) / 2.0))
    else:
        if fx > 0:
            ax = bx + 1 + gap - int(round(rlo[0]))
        else:
            ax = bx - gap - int(round(rhi[0]))
        az = int(round(pz - (rlo[2] + rhi[2]) / 2.0))
    ay = math.floor(py) - int(round(lo[1]))
    return (int(ax), int(ay), int(az)), k


def facing_from_yaw(yaw: float) -> str:
    """Minecraft yaw -> compass facing (0 = south, 90 = west, 180 = north, 270 = east)."""
    y = (float(yaw) % 360 + 360) % 360
    if y < 45 or y >= 315:
        return "south"
    if y < 135:
        return "west"
    if y < 225:
        return "north"
    return "east"


def estimate_seconds(n_blocks: int, chunk_size: int = 1500, delay_ms: int = 60) -> float:
    """Rough wall-clock for an animated placement (chunks spaced by delay_ms, plus overhead)."""
    if n_blocks <= 0:
        return 0.0
    chunks = math.ceil(n_blocks / max(1, chunk_size))
    return round(chunks * delay_ms / 1000.0 + 0.2 + n_blocks / 200000.0, 1)


def _diff(old: BlockMap, new: BlockMap) -> Tuple[Dict[IVec3, str], List[IVec3]]:
    """(added_or_changed, removed) between two world maps. Uses engine.diff when available."""
    try:
        from .engine.diff import diff_block_maps  # type: ignore

        d = diff_block_maps(old, new)
        added: Dict[IVec3, str] = {}
        added.update(getattr(d, "added", {}) or {})
        changed = getattr(d, "changed", {}) or {}
        for k, v in changed.items():
            added[k] = v[1] if isinstance(v, (tuple, list)) else v
        removed = list(getattr(d, "removed", []) or [])
        return added, removed
    except ImportError:
        pass
    added = {k: v for k, v in new.items() if old.get(k) != v}
    removed = [k for k in old if k not in new]
    return added, removed


def _layer_chunks(blocks: List[Tuple[int, int, int, str]], chunk_size: int) -> List[List[Tuple[int, int, int, str]]]:
    """Bottom-up chunks of at most chunk_size blocks. Uses engine.diff.layer_chunks when available."""
    try:
        from .engine.diff import layer_chunks  # type: ignore

        return [list(c) for c in layer_chunks(blocks, chunk_size=chunk_size)]
    except ImportError:
        pass
    ordered = sorted(blocks, key=lambda b: (b[1], b[2], b[0]))
    return [ordered[i : i + chunk_size] for i in range(0, len(ordered), chunk_size)]


def _touched(session) -> set:
    t = getattr(session.world, "touched", None)
    if t is None:
        t = set()
        session.world.touched = t
    return t


def place_scene(session, bridge, block_map: BlockMap, mode: str = "diff", animate: bool = True, chunk_size: int = 1500, delay_ms: int = 60) -> str:
    """Place a scene-space BlockMap into the world through `bridge`, recording state on `session`.

    mode="diff": send only blocks that differ from the last placement (removed -> air).
    mode="full": resend every block (plus air for removed ones).
    """
    if not block_map:
        return "nothing to place: the scene rasterises to 0 blocks"
    world = session.world
    if world.anchor is None:
        player = bridge.player()
        lo, hi = map_bbox(block_map)
        scene_bbox = (lo, (hi[0] + 1, hi[1] + 1, hi[2] + 1))
        world.anchor, world.quarter_turns = plan_anchor(player, scene_bbox)
    new_map = transform_block_map(block_map, world.anchor, world.quarter_turns)
    lo, hi = map_bbox(new_map)
    plo = (lo[0] - 1, lo[1] - 1, lo[2] - 1)
    phi = (hi[0] + 1, hi[1] + 1, hi[2] + 1)
    if world.pre_scan is None:
        world.pre_scan = bridge.scan(plo, phi)
        world.pre_scan_bbox = (plo, phi)
    else:
        # extend the snapshot for any newly covered region (only positions we have not seen yet)
        missing = [p for p in new_map if p not in world.pre_scan]
        if missing:
            mlo, mhi = map_bbox({p: "" for p in missing})
            extra = bridge.scan(mlo, mhi)
            for p, s in extra.items():
                world.pre_scan.setdefault(p, s)
            world.pre_scan_bbox = (tuple(min(a, b) for a, b in zip(world.pre_scan_bbox[0], mlo)), tuple(max(a, b) for a, b in zip(world.pre_scan_bbox[1], mhi)))  # type: ignore[assignment]
    old_map = world.placed if mode == "diff" else {}
    added, removed = _diff(old_map, new_map)
    if mode != "diff":
        removed = [k for k in world.placed if k not in new_map]
    blocks: List[Tuple[int, int, int, str]] = [(x, y, z, s) for (x, y, z), s in added.items()]
    blocks += [(x, y, z, "minecraft:air") for (x, y, z) in removed]
    touched = _touched(session)
    touched.update(added.keys())
    touched.update(removed)
    if not blocks:
        return f"world already up to date ({len(new_map)} blocks placed, anchor {world.anchor}, {world.quarter_turns * 90}° rotation)"
    chunks = _layer_chunks(blocks, chunk_size)
    delay = int(delay_ms) if animate else 0
    queued = bridge.setblocks([(c, delay) for c in chunks], flags=3)
    world.placed = dict(new_map)
    world.placements += 1
    secs = estimate_seconds(len(blocks), chunk_size, delay) if animate else 0.0
    return (
        f"placed {mode}: {len(added)} set, {len(removed)} cleared ({queued} sent in {len(chunks)} chunks, ~{secs:g}s); "
        f"total {len(new_map)} blocks at anchor {world.anchor} rotated {world.quarter_turns * 90}°"
    )


def undo_world(session, bridge, chunk_size: int = 4000) -> str:
    """Restore every position touched by this session to its pre-placement state (incl. air)."""
    world = session.world
    if world.pre_scan is None:
        return "nothing to undo in the world (no placement yet)"
    touched = _touched(session) | set(world.placed.keys())
    blocks = [(x, y, z, world.pre_scan.get((x, y, z), "minecraft:air")) for (x, y, z) in touched]
    blocks.sort(key=lambda b: (-b[1], b[2], b[0]))  # top-down so nothing floats mid-restore
    chunks = [blocks[i : i + chunk_size] for i in range(0, len(blocks), chunk_size)]
    n = bridge.setblocks([(c, 0) for c in chunks], flags=3) if chunks else 0
    world.placed = {}
    world.pre_scan = None
    world.pre_scan_bbox = None
    world.anchor = None
    world.quarter_turns = 0
    world.placements = 0
    touched.clear()
    return f"world restored: {n} blocks reset to their pre-build state"
