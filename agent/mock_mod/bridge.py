"""MockBridge: an in-memory world that implements the Bridge protocol without the game."""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from copilot.engine.coretypes import BlockMap

AIR = "minecraft:air"
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "copilot", "data")
FALLBACK_BLOCKS_PATH = os.path.join(_DATA_DIR, "blocks_fallback.json")

# A minimal dump so tests run before Track 4 ships copilot/data/blocks_fallback.json.
_HORIZ = ["north", "south", "west", "east"]
_MINI_BLOCKS: List[Dict[str, Any]] = [
    {"id": "minecraft:air", "properties": {}, "default": "minecraft:air"},
    {"id": "minecraft:stone", "properties": {}, "default": "minecraft:stone"},
    {"id": "minecraft:cobblestone", "properties": {}, "default": "minecraft:cobblestone"},
    {"id": "minecraft:stone_bricks", "properties": {}, "default": "minecraft:stone_bricks"},
    {"id": "minecraft:cracked_stone_bricks", "properties": {}, "default": "minecraft:cracked_stone_bricks"},
    {"id": "minecraft:mossy_stone_bricks", "properties": {}, "default": "minecraft:mossy_stone_bricks"},
    {"id": "minecraft:stone_brick_stairs", "properties": {"facing": _HORIZ, "half": ["top", "bottom"], "shape": ["straight", "inner_left", "inner_right", "outer_left", "outer_right"], "waterlogged": ["true", "false"]}, "default": "minecraft:stone_brick_stairs[facing=north,half=bottom,shape=straight,waterlogged=false]"},
    {"id": "minecraft:stone_brick_slab", "properties": {"type": ["top", "bottom", "double"], "waterlogged": ["true", "false"]}, "default": "minecraft:stone_brick_slab[type=bottom,waterlogged=false]"},
    {"id": "minecraft:stone_brick_wall", "properties": {"east": ["none", "low", "tall"], "north": ["none", "low", "tall"], "south": ["none", "low", "tall"], "up": ["true", "false"], "waterlogged": ["true", "false"], "west": ["none", "low", "tall"]}, "default": "minecraft:stone_brick_wall[east=none,north=none,south=none,up=true,waterlogged=false,west=none]"},
    {"id": "minecraft:oak_planks", "properties": {}, "default": "minecraft:oak_planks"},
    {"id": "minecraft:oak_stairs", "properties": {"facing": _HORIZ, "half": ["top", "bottom"], "shape": ["straight", "inner_left", "inner_right", "outer_left", "outer_right"], "waterlogged": ["true", "false"]}, "default": "minecraft:oak_stairs[facing=north,half=bottom,shape=straight,waterlogged=false]"},
    {"id": "minecraft:oak_slab", "properties": {"type": ["top", "bottom", "double"], "waterlogged": ["true", "false"]}, "default": "minecraft:oak_slab[type=bottom,waterlogged=false]"},
    {"id": "minecraft:oak_log", "properties": {"axis": ["x", "y", "z"]}, "default": "minecraft:oak_log[axis=y]"},
    {"id": "minecraft:oak_fence", "properties": {"east": ["true", "false"], "north": ["true", "false"], "south": ["true", "false"], "waterlogged": ["true", "false"], "west": ["true", "false"]}, "default": "minecraft:oak_fence[east=false,north=false,south=false,waterlogged=false,west=false]"},
    {"id": "minecraft:oak_door", "properties": {"facing": _HORIZ, "half": ["upper", "lower"], "hinge": ["left", "right"], "open": ["true", "false"], "powered": ["true", "false"]}, "default": "minecraft:oak_door[facing=north,half=lower,hinge=left,open=false,powered=false]"},
    {"id": "minecraft:glass", "properties": {}, "default": "minecraft:glass"},
    {"id": "minecraft:glass_pane", "properties": {"east": ["true", "false"], "north": ["true", "false"], "south": ["true", "false"], "waterlogged": ["true", "false"], "west": ["true", "false"]}, "default": "minecraft:glass_pane[east=false,north=false,south=false,waterlogged=false,west=false]"},
    {"id": "minecraft:lantern", "properties": {"hanging": ["true", "false"], "waterlogged": ["true", "false"]}, "default": "minecraft:lantern[hanging=false,waterlogged=false]"},
    {"id": "minecraft:torch", "properties": {}, "default": "minecraft:torch"},
    {"id": "minecraft:wall_torch", "properties": {"facing": _HORIZ}, "default": "minecraft:wall_torch[facing=north]"},
    {"id": "minecraft:white_banner", "properties": {"rotation": [str(i) for i in range(16)]}, "default": "minecraft:white_banner[rotation=0]"},
    {"id": "minecraft:grass_block", "properties": {"snowy": ["true", "false"]}, "default": "minecraft:grass_block[snowy=false]"},
    {"id": "minecraft:dirt", "properties": {}, "default": "minecraft:dirt"},
    {"id": "minecraft:deepslate_tiles", "properties": {}, "default": "minecraft:deepslate_tiles"},
    {"id": "minecraft:spruce_planks", "properties": {}, "default": "minecraft:spruce_planks"},
    {"id": "minecraft:dark_oak_planks", "properties": {}, "default": "minecraft:dark_oak_planks"},
]


def load_fallback_blocks() -> List[Dict[str, Any]]:
    """The block dump served by the mock: copilot/data/blocks_fallback.json if present, else a mini list."""
    if os.path.exists(FALLBACK_BLOCKS_PATH):
        try:
            with open(FALLBACK_BLOCKS_PATH) as f:
                data = json.load(f)
            blocks = data.get("blocks", data) if isinstance(data, dict) else data
            if isinstance(blocks, list) and blocks:
                return blocks
        except (OSError, ValueError):
            pass
    return [dict(b) for b in _MINI_BLOCKS]


def facing_from_yaw(yaw: float) -> str:
    """Minecraft yaw: 0 = south (+z), 90 = west (-x), 180 = north (-z), -90/270 = east (+x)."""
    y = (float(yaw) % 360 + 360) % 360
    if y < 45 or y >= 315:
        return "south"
    if y < 135:
        return "west"
    if y < 225:
        return "north"
    return "east"


class MockBridge:
    """Dict-backed world. Flat ground (grass at `ground_y`, dirt below) and a player standing on it."""

    def __init__(self, ground_y: int = 63, player_name: str = "Steve", player_pos: Optional[Sequence[float]] = None, yaw: float = 180.0, pitch: float = 0.0):
        self.ground_y = int(ground_y)
        self.world: BlockMap = {}  # explicit overrides only; everything else is procedural ground
        self.player_name = player_name
        self.pos: List[float] = list(player_pos) if player_pos is not None else [0.5, float(self.ground_y + 1), 0.5]
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.chat: List[str] = []
        self.calls: List[Dict[str, Any]] = []  # every setblocks call: {"chunks": [...], "flags": n, "count": n}
        self.camera_calls: List[Dict[str, Any]] = []
        self._blocks_cache: Optional[List[Dict[str, Any]]] = None

    # -- world access ----------------------------------------------------------------------
    def get_block(self, x: int, y: int, z: int) -> str:
        key = (int(x), int(y), int(z))
        if key in self.world:
            return self.world[key]
        if y == self.ground_y:
            return "minecraft:grass_block[snowy=false]"
        if y < self.ground_y:
            return "minecraft:dirt" if y > self.ground_y - 4 else "minecraft:stone"
        return AIR

    def set_block(self, x: int, y: int, z: int, state: str) -> None:
        self.world[(int(x), int(y), int(z))] = str(state)

    def count_non_air(self) -> int:
        """Number of explicitly placed non-air blocks (ground excluded)."""
        return sum(1 for s in self.world.values() if s != AIR)

    # -- Bridge protocol -------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return {"ok": True, "mod_version": "mock-0.1.0", "mc_version": "1.21.1", "client_jar": None, "world_loaded": True, "player": self.player_name}

    def player(self) -> Dict[str, Any]:
        return {
            "name": self.player_name,
            "pos": [float(v) for v in self.pos],
            "yaw": self.yaw,
            "pitch": self.pitch,
            "facing": facing_from_yaw(self.yaw),
            "looking_at": self._looking_at(),
            "dimension": "minecraft:overworld",
        }

    def _looking_at(self, max_dist: float = 20.0) -> Optional[Dict[str, Any]]:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        dx = -math.sin(yaw) * math.cos(pitch)
        dz = math.cos(yaw) * math.cos(pitch)
        dy = -math.sin(pitch)
        ex, ey, ez = self.pos[0], self.pos[1] + 1.62, self.pos[2]
        steps = int(max_dist * 4)
        for i in range(1, steps + 1):
            t = i / 4.0
            x, y, z = math.floor(ex + dx * t), math.floor(ey + dy * t), math.floor(ez + dz * t)
            b = self.get_block(x, y, z)
            if b != AIR:
                return {"pos": [x, y, z], "block": b}
        return None

    def scan(self, lo: Sequence[int], hi: Sequence[int]) -> BlockMap:
        lo_i = [int(min(a, b)) for a, b in zip(lo, hi)]
        hi_i = [int(max(a, b)) for a, b in zip(lo, hi)]
        out: BlockMap = {}
        for x in range(lo_i[0], hi_i[0] + 1):
            for y in range(lo_i[1], hi_i[1] + 1):
                for z in range(lo_i[2], hi_i[2] + 1):
                    out[(x, y, z)] = self.get_block(x, y, z)
        return out

    def setblocks(self, chunks, flags: int = 3) -> int:
        n = 0
        rec = []
        for blocks, delay in chunks:
            for (x, y, z, state) in blocks:
                self.set_block(x, y, z, state)
                n += 1
            rec.append({"count": len(blocks), "delay_ms": int(delay)})
        self.calls.append({"chunks": rec, "flags": int(flags), "count": n})
        return n

    def say(self, text: str) -> None:
        self.chat.append(str(text))

    def blocks(self) -> List[Dict[str, Any]]:
        if self._blocks_cache is None:
            self._blocks_cache = load_fallback_blocks()
        return self._blocks_cache

    def camera(self, **kw: Any) -> Dict[str, Any]:
        self.camera_calls.append(dict(kw))
        return {"ok": True, **{k: v for k, v in kw.items() if k == "mode"}}

    # -- test helpers ----------------------------------------------------------------------
    def placed_blocks(self) -> BlockMap:
        return {k: v for k, v in self.world.items() if v != AIR}

    def reset(self) -> None:
        self.world.clear()
        self.chat.clear()
        self.calls.clear()
        self.camera_calls.clear()
