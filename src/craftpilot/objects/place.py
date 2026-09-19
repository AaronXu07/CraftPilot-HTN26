"""Place an object grid in the running game through the Fabric mod's HTTP bridge (mod/, port 7777).

The bridge API (see agent/copilot/bridge.py): GET /health, GET /player -> {pos, yaw, facing}, POST /setblocks
{"chunks": [{"blocks": [[x, y, z, state], …], "delay_ms": n}], "flags": 3}, GET /setblocks/status.
Objects contain only full blocks, so rotation is a coordinate rotation with no block-state remapping.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from craftpilot.grid.semantic import SemanticGrid

DEFAULT_MOD_URL = "http://127.0.0.1:7777"
FACING_TURNS = {"north": 0, "east": 1, "south": 2, "west": 3}
FACING_VEC = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
# TripoSR: "x back, y right, z up" with the input camera on +x, so the side seen in the reference image
# faces +x after the z-up -> y-up rotation. Three CW quarter turns take +x to +z, the agent's convention
# for "the front faces the player standing south of the build".
PRESENTATION_TURNS = 3


class ModUnavailable(RuntimeError):
    pass


@dataclass
class Placement:
    anchor: tuple[int, int, int]
    quarter_turns: int
    blocks: int
    chunks: int
    estimated_seconds: float
    undo_file: Path | None = None


def mod_url() -> str:
    return os.environ.get("COPILOT_MOD_URL", DEFAULT_MOD_URL).rstrip("/")


def _get(path: str, timeout: float = 5.0) -> dict:
    try:
        with urllib.request.urlopen(mod_url() + path, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise ModUnavailable(f"mod not reachable at {mod_url()}{path}: {exc}") from exc


def _post(path: str, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(mod_url() + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        raise ModUnavailable(f"mod rejected {path}: HTTP {exc.code} {exc.read()[:200]!r}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ModUnavailable(f"mod not reachable at {mod_url()}{path}: {exc}") from exc


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


def grid_blocks(grid: SemanticGrid) -> list[tuple[int, int, int, str]]:
    """Grid -> scene-space blocks (centred on x/z, y from 0), already turned so the presented side faces +z."""
    xs, ys, zs = np.nonzero(grid.block >= 0)
    if xs.size == 0:
        return []
    cx = (int(xs.min()) + int(xs.max()) + 1) // 2
    cz = (int(zs.min()) + int(zs.max()) + 1) // 2
    states = [f"{ref.block_id}" + ("[" + ",".join(f"{k}={v}" for k, v in ref.props) + "]" if ref.props else "") for ref in grid.palette]
    out = []
    for x, y, z in zip(xs.tolist(), ys.tolist(), zs.tolist()):
        rx, rz = rotate_xz(x - cx, z - cz, PRESENTATION_TURNS)
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
    out = []
    for x, y, z, s in blocks:
        rx, rz = rotate_xz(x, z, k)
        out.append((rx + ax, y + ay, rz + az, s))
    return out


def layer_chunks(blocks: list[tuple[int, int, int, str]], chunk_size: int) -> list[list[tuple[int, int, int, str]]]:
    """Bottom-up chunks so the object rises out of the ground."""
    ordered = sorted(blocks, key=lambda b: (b[1], b[2], b[0]))
    return [ordered[i:i + chunk_size] for i in range(0, len(ordered), chunk_size)]


def estimate_seconds(n_blocks: int, chunk_size: int, delay_ms: int) -> float:
    return math.ceil(n_blocks / max(chunk_size, 1)) * delay_ms / 1000.0 + 1.0


def health() -> dict:
    return _get("/health")


def place(grid: SemanticGrid, gap: int = 2, chunk_size: int = 1500, delay_ms: int = 60, undo_dir: Path | None = None,
          wait: bool = True, max_wait_s: float = 120.0) -> Placement:
    """Place the object in front of the player. Writes an undo record (world positions) when `undo_dir` is given."""
    h = health()
    if not h.get("world_loaded", True):
        raise ModUnavailable("no world loaded (open a single-player world first)")
    player = _get("/player")
    scene = grid_blocks(grid)
    if not scene:
        raise ValueError("empty object")
    anchor, k = plan_anchor(player, scene, gap=gap)
    world = to_world(scene, anchor, k)
    chunks = layer_chunks(world, chunk_size)
    body = {"chunks": [{"blocks": [[x, y, z, s] for x, y, z, s in c], "delay_ms": delay_ms} for c in chunks], "flags": 3}
    _post("/setblocks", body)
    est = estimate_seconds(len(world), chunk_size, delay_ms)
    undo_file = None
    if undo_dir is not None:
        undo_dir.mkdir(parents=True, exist_ok=True)
        undo_file = undo_dir / "placed.json"
        undo_file.write_text(json.dumps({"anchor": anchor, "turns": k, "positions": [[x, y, z] for x, y, z, _ in world]}))
    if wait:
        deadline = time.time() + min(max_wait_s, est * 3 + 5)
        while time.time() < deadline:
            try:
                st = _get("/setblocks/status")
            except ModUnavailable:
                break
            if int(st.get("pending_blocks", 0) or 0) == 0 and int(st.get("pending_chunks", 0) or 0) == 0:
                break
            time.sleep(0.5)
    return Placement(anchor=anchor, quarter_turns=k, blocks=len(world), chunks=len(chunks), estimated_seconds=est,
                     undo_file=undo_file)


def undo(undo_file: Path, chunk_size: int = 4000) -> int:
    """Replace every recorded position with air. Returns the block count."""
    data = json.loads(Path(undo_file).read_text())
    positions = data.get("positions", [])
    if not positions:
        return 0
    blocks = [(int(x), int(y), int(z), "minecraft:air") for x, y, z in positions]
    chunks = layer_chunks(blocks, chunk_size)
    _post("/setblocks", {"chunks": [{"blocks": [[x, y, z, s] for x, y, z, s in c], "delay_ms": 0} for c in chunks], "flags": 3})
    return len(blocks)
