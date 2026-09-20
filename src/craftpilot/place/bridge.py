"""HTTP client for the Fabric bridge mod (mod/README.md).

The mod hosts a small HTTP server inside the game. All coordinates are absolute world coordinates and
block states are command-syntax strings (``minecraft:oak_stairs[facing=north]``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from craftpilot.config import SETTINGS

Block = tuple[int, int, int, str]
Chunk = tuple[list[Block], int]  # (blocks, delay_ms before the next chunk)
Heightmap = dict[tuple[int, int], tuple[int, str]]  # (x, z) -> (y of the top solid block, its state); void columns absent
HEIGHTMAP_CHUNK = 128  # /heightmap requests are split into <= 128x128 rectangles

# Block.NOTIFY_LISTENERS | Block.FORCE_STATE: write the state verbatim, no neighbour updates.
# The engine already bakes stair shapes, door hinges, connections and support into every state.
FORCE_FLAGS = 2 | 16
NOTIFY_ALL = 3


class BridgeError(RuntimeError):
    """The mod is unreachable or answered with an error."""


class HttpBridge:
    def __init__(self, base_url: str | None = None, timeout: float = 10.0):
        self.base_url = (base_url or SETTINGS.mod_url).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _get(self, path: str, timeout: float) -> Any:
        try:
            r = self._client.get(path, timeout=timeout)
        except httpx.HTTPError as e:
            raise BridgeError(f"mod unreachable at {self.base_url}{path}: {e}") from e
        if r.status_code >= 400:
            raise BridgeError(f"GET {path} -> {r.status_code}: {r.text[:200]}")
        return r.json()

    def _post(self, path: str, body: dict[str, Any], timeout: float) -> Any:
        try:
            r = self._client.post(path, json=body, timeout=timeout)
        except httpx.HTTPError as e:
            raise BridgeError(f"mod unreachable at {self.base_url}{path}: {e}") from e
        if r.status_code >= 400:
            raise BridgeError(f"POST {path} -> {r.status_code}: {r.text[:200]}")
        return r.json()

    def health(self) -> dict:
        """{ok, mod_version, mc_version, world_loaded, player, pending_chunks}"""
        return self._get("/health", timeout=3.0)

    def player(self) -> dict:
        """{name, pos: [x, y, z], yaw, pitch, facing, looking_at, dimension}"""
        return self._get("/player", timeout=5.0)

    def setblocks(self, chunks: list[Chunk], flags: int = FORCE_FLAGS, postprocess: bool = False) -> dict:
        """Queue blocks; the mod places one chunk per game tick. Returns {queued, chunks, invalid, invalid_samples}."""
        if not any(blocks for blocks, _ in chunks):
            return {"queued": 0, "chunks": 0, "invalid": 0}
        body = {
            "chunks": [{"blocks": [[int(x), int(y), int(z), str(s)] for x, y, z, s in blocks],
                        "delay_ms": int(delay)} for blocks, delay in chunks],
            "flags": int(flags),
            "postprocess": bool(postprocess),
        }
        return self._post("/setblocks", body, timeout=60.0)

    def status(self) -> dict:
        """{pending_chunks, pending_blocks, placed_total, postprocessed}"""
        return self._get("/setblocks/status", timeout=5.0)

    def cancel(self) -> dict:
        return self._post("/setblocks/cancel", {}, timeout=5.0)

    def scan(self, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> dict[tuple[int, int, int], str]:
        """Inclusive box (keep it small; the mod caps the volume) -> {pos: state}, air included."""
        data = self._post("/scan", {"min": list(lo), "max": list(hi)}, timeout=60.0)
        palette = data["palette"]
        return {(int(x), int(y), int(z)): palette[int(i)] for x, y, z, i in data["blocks"]}

    def heightmap(self, lo: tuple[int, int], hi: tuple[int, int]) -> Heightmap:
        """Inclusive (x, z) rectangle -> surface heightmap: the y of the top motion-blocking non-leaf
        block per column (water surfaces count as ground) and that block's state. Void columns absent."""
        x0, z0 = min(lo[0], hi[0]), min(lo[1], hi[1])
        x1, z1 = max(lo[0], hi[0]), max(lo[1], hi[1])
        out: Heightmap = {}
        for bx in range(x0, x1 + 1, HEIGHTMAP_CHUNK):
            for bz in range(z0, z1 + 1, HEIGHTMAP_CHUNK):
                box_hi = [min(bx + HEIGHTMAP_CHUNK - 1, x1), min(bz + HEIGHTMAP_CHUNK - 1, z1)]
                out.update(decode_heightmap(self._post("/heightmap", {"min": [bx, bz], "max": box_hi}, timeout=60.0)))
        return out

    def say(self, text: str) -> None:
        self._post("/say", {"text": text}, timeout=5.0)

    def outline(self, lo: tuple[int, int, int], hi: tuple[int, int, int], phase: str) -> None:
        """Show the in-progress box (inclusive world coords); phase is 'generating' or 'placing'.

        The mod animates it and clears it itself once the placement queue drains."""
        self._post("/outline", {"min": list(lo), "max": list(hi), "phase": phase}, timeout=5.0)

    def clear_outline(self) -> None:
        self._post("/outline", {"clear": True}, timeout=5.0)

    def wait_idle(self, poll_s: float = 0.25, timeout_s: float = 600.0) -> dict:
        deadline = time.time() + timeout_s
        while True:
            st = self.status()
            if st.get("pending_chunks", 0) == 0 or time.time() > deadline:
                return st
            time.sleep(poll_s)


def decode_heightmap(data: dict[str, Any]) -> Heightmap:
    """{min:[x0,z0], max:[x1,z1], palette, heights, tops} (row-major, null for void) -> {(x, z): (y, state)}."""
    palette = data.get("palette", [])
    x0, z0 = (int(v) for v in data["min"])
    x1 = int(data["max"][0])
    width = x1 - x0 + 1
    tops = data.get("tops", [])
    out: Heightmap = {}
    for i, h in enumerate(data.get("heights", [])):
        if h is None:
            continue
        idx = tops[i] if i < len(tops) else None
        state = palette[int(idx)] if idx is not None and int(idx) < len(palette) else "minecraft:stone"
        out[(x0 + i % width, z0 + i // width)] = (int(h), state)
    return out


class FakeBridge:
    """In-memory stand-in for tests: records every setblocks call and applies it to a dict world.

    ``terrain(x, z) -> y`` gives the fake a ground surface (grass at that y, dirt below) so the
    heightmap survey has something to see; without it the world is empty and the survey finds nothing.
    """

    def __init__(self, pos: tuple[float, float, float] = (0.5, 64.0, 0.5), yaw: float = 180.0,
                 name: str = "tester", terrain: Callable[[int, int], int] | None = None):
        self._player = {"name": name, "pos": list(pos), "yaw": yaw, "pitch": 0.0, "facing": "north",
                        "looking_at": None, "dimension": "minecraft:overworld"}
        self.terrain = terrain
        self.world: dict[tuple[int, int, int], str] = {}
        self.calls: list[dict] = []
        self.said: list[str] = []
        self.outlines: list[dict] = []
        self.cancelled = 0

    def health(self) -> dict:
        return {"ok": True, "mod_version": "fake", "mc_version": "fake", "world_loaded": True,
                "player": self._player["name"], "pending_chunks": 0}

    def player(self) -> dict:
        return dict(self._player)

    def setblocks(self, chunks: list[Chunk], flags: int = FORCE_FLAGS, postprocess: bool = False) -> dict:
        n = 0
        for blocks, _ in chunks:
            for x, y, z, s in blocks:
                self.world[(x, y, z)] = s
                n += 1
        self.calls.append({"chunks": [(list(b), d) for b, d in chunks], "flags": flags,
                           "postprocess": postprocess, "count": n})
        return {"queued": n, "chunks": len(chunks), "invalid": 0}

    def status(self) -> dict:
        return {"pending_chunks": 0, "pending_blocks": 0, "placed_total": len(self.world), "postprocessed": 0}

    def cancel(self) -> dict:
        self.cancelled += 1
        return {"ok": True, "cleared_chunks": 0}

    def outline(self, lo, hi, phase: str) -> None:
        self.outlines.append({"min": list(lo), "max": list(hi), "phase": phase})

    def clear_outline(self) -> None:
        self.outlines.append({"clear": True})

    def get_block(self, x: int, y: int, z: int) -> str:
        """Placed blocks first, then the procedural ground (grass on top, dirt below), else air."""
        state = self.world.get((x, y, z))
        if state is not None:
            return state
        if self.terrain is not None:
            g = self.terrain(x, z)
            if y == g:
                return "minecraft:grass_block"
            if y < g:
                return "minecraft:dirt"
        return "minecraft:air"

    def scan(self, lo, hi) -> dict[tuple[int, int, int], str]:
        return {(x, y, z): self.get_block(x, y, z)
                for x in range(lo[0], hi[0] + 1) for y in range(lo[1], hi[1] + 1)
                for z in range(lo[2], hi[2] + 1)}

    def heightmap(self, lo, hi) -> Heightmap:
        out: Heightmap = {}
        for x in range(min(lo[0], hi[0]), max(lo[0], hi[0]) + 1):
            for z in range(min(lo[1], hi[1]), max(lo[1], hi[1]) + 1):
                top = self.terrain(x, z) if self.terrain is not None else None
                for (wx, wy, wz), st in self.world.items():
                    if wx == x and wz == z and st != "minecraft:air" and (top is None or wy > top):
                        top = wy
                if top is None:
                    continue
                while self.get_block(x, top, z) == "minecraft:air" and top > -64:
                    top -= 1  # air placed into the ground
                out[(x, z)] = (top, self.get_block(x, top, z))
        return out

    def say(self, text: str) -> None:
        self.said.append(text)

    def wait_idle(self, poll_s: float = 0.0, timeout_s: float = 0.0) -> dict:
        return self.status()
