"""Object placement geometry, kept for reference and tests.

Placement itself goes through the shared terrain placer (`craftpilot.place.placer.place_grid`): an object
grid is turned to face south once (`objects.orient.face_south`, using PRESENTATION_TURNS below) and from
then on is placed, previewed and undone exactly like an engine-built building. The helpers here are the
original centred-anchor arithmetic; `objects.flow` is the live path.
"""
from __future__ import annotations

import math

import numpy as np

from craftpilot.grid.semantic import SemanticGrid
from craftpilot.objects.orient import DEFAULT_FRONT, PRESENTATION_TURNS

FACING_TURNS = {"north": 0, "east": 1, "south": 2, "west": 3}
FACING_VEC = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}

__all__ = ["DEFAULT_FRONT", "FACING_TURNS", "FACING_VEC", "PRESENTATION_TURNS", "facing_from_yaw", "grid_blocks",
           "layer_chunks", "plan_anchor", "rotate_state", "rotate_xz", "to_world"]


def facing_from_yaw(yaw: float) -> str:
    y = (float(yaw) % 360 + 360) % 360
    if y < 45 or y >= 315:
        return "south"
    if y < 135:
        return "west"
    if y < 225:
        return "north"
    return "east"


def rotate_xz(x: int, z: int, quarter_turns: int) -> tuple[int, int]:
    """CW quarter turns about the origin on voxel coordinates: (x, z) -> (-z-1, x)."""
    for _ in range(int(quarter_turns) % 4):
        x, z = -z - 1, x
    return x, z


_CW = {"north": "east", "east": "south", "south": "west", "west": "north"}


def rotate_state(state: str, quarter_turns: int) -> str:
    """Rotate a block state's `facing` (stairs) by k CW quarter turns; other states pass through. Objects
    only carry full blocks, slabs (no facing) and straight stairs, so this is all the remapping they need."""
    k = int(quarter_turns) % 4
    if not k or "facing=" not in state:
        return state
    head, _, props = state.partition("[")
    props = props.rstrip("]")
    out = []
    for kv in props.split(","):
        key, _, val = kv.partition("=")
        if key == "facing" and val in _CW:
            for _ in range(k):
                val = _CW[val]
        out.append(f"{key}={val}")
    return f"{head}[{','.join(out)}]"


def grid_blocks(grid: SemanticGrid) -> list[tuple[int, int, int, str]]:
    """Grid -> scene-space blocks (centred on x/z, y from 0), already turned so the presented side faces +z."""
    xs, ys, zs = np.nonzero(grid.block >= 0)
    if xs.size == 0:
        return []
    cx = (int(xs.min()) + int(xs.max()) + 1) // 2
    cz = (int(zs.min()) + int(zs.max()) + 1) // 2
    states = [f"{ref.block_id}" + ("[" + ",".join(f"{k}={v}" for k, v in ref.props) + "]" if ref.props else "") for ref in grid.palette]
    front = str((getattr(grid, "report", None) or {}).get("object_front", DEFAULT_FRONT))
    turns = PRESENTATION_TURNS.get(front, PRESENTATION_TURNS[DEFAULT_FRONT])
    states = [rotate_state(st, turns) for st in states]
    out = []
    for x, y, z in zip(xs.tolist(), ys.tolist(), zs.tolist()):
        rx, rz = rotate_xz(x - cx, z - cz, turns)
        out.append((rx, int(y), rz, states[int(grid.block[x, y, z])]))
    return out


def plan_anchor(player: dict, blocks: list[tuple[int, int, int, str]], gap: int = 2) -> tuple[tuple[int, int, int], int]:
    """(anchor, quarter_turns): the object stands `gap` air blocks in front of the player, centred, facing them."""
    facing = str(player.get("facing") or facing_from_yaw(float(player.get("yaw", 0.0)))).lower()
    k = FACING_TURNS.get(facing, 0)
    px, py, pz = (float(v) for v in player["pos"])
    fx, fz = FACING_VEC[facing]
    rot = [rotate_xz(x, z, k) for x, _, z, _ in blocks]
    rxs = [r[0] for r in rot]
    rzs = [r[1] for r in rot]
    lo_x, hi_x, lo_z, hi_z = min(rxs), max(rxs) + 1, min(rzs), max(rzs) + 1
    lo_y = min(b[1] for b in blocks)
    bx, bz = math.floor(px), math.floor(pz)
    if fz != 0:
        az = (bz + 1 + gap - lo_z) if fz > 0 else (bz - gap - hi_z)
        ax = round(px - (lo_x + hi_x) / 2.0)
    else:
        ax = (bx + 1 + gap - lo_x) if fx > 0 else (bx - gap - hi_x)
        az = round(pz - (lo_z + hi_z) / 2.0)
    ay = math.floor(py) - lo_y
    return (ax, ay, az), k


def to_world(blocks: list[tuple[int, int, int, str]], anchor: tuple[int, int, int], k: int) -> list[tuple[int, int, int, str]]:
    ax, ay, az = anchor
    cache: dict[str, str] = {}
    out = []
    for x, y, z, s in blocks:
        rx, rz = rotate_xz(x, z, k)
        st = cache.get(s)
        if st is None:
            st = cache[s] = rotate_state(s, k)
        out.append((rx + ax, y + ay, rz + az, st))
    return out


def layer_chunks(blocks: list[tuple[int, int, int, str]], chunk_size: int) -> list[list[tuple[int, int, int, str]]]:
    """Bottom-up chunks so the object rises out of the ground."""
    ordered = sorted(blocks, key=lambda b: (b[1], b[2], b[0]))
    return [ordered[i:i + chunk_size] for i in range(0, len(ordered), chunk_size)]

