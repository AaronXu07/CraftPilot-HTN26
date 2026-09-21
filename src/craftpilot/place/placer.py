"""Turn a rendered grid into world blocks and stream them to the mod, bottom up, one layer at a time.

With a bridge that can survey the ground (`heightmap`), the build is seated on the terrain instead of
at the player's feet, and the foundation / cut / graded apron from `craftpilot.terrain`
are streamed with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from craftpilot import terrain
from craftpilot.config import SETTINGS
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.place.anchor import Area, area_origin, plan_origin, trimmed_bbox
from craftpilot.place.bridge import FORCE_FLAGS, Block, BridgeError, Chunk

AIR = "minecraft:air"

# Blocks that need a neighbour to stay put. They go last within their layer so the build looks right
# even when someone places with neighbour updates on (flags=3).
_ATTACHABLE_SUFFIXES = ("_door", "_trapdoor", "_button", "_banner", "_pressure_plate", "_carpet",
                        "_sign", "_torch", "_candle", "_coral_fan", "_coral_wall_fan")
_ATTACHABLE_IDS = {"minecraft:lantern", "minecraft:soul_lantern", "minecraft:vine", "minecraft:glow_lichen",
                   "minecraft:ladder", "minecraft:torch", "minecraft:chain", "minecraft:bell",
                   "minecraft:flower_pot", "minecraft:lever", "minecraft:tripwire_hook",
                   "minecraft:end_rod", "minecraft:lightning_rod"}


def is_attachable(block_id: str) -> bool:
    if block_id in _ATTACHABLE_IDS:
        return True
    return block_id.endswith(_ATTACHABLE_SUFFIXES)


class Bridge(Protocol):
    def player(self) -> dict: ...
    def setblocks(self, chunks: list[Chunk], flags: int = ..., postprocess: bool = ...) -> dict: ...
    def outline(self, lo: tuple[int, int, int], hi: tuple[int, int, int], phase: str) -> None: ...


@dataclass
class PlaceOptions:
    gap: int = SETTINGS.place_gap
    sink: int = SETTINGS.place_sink
    chunk_blocks: int = SETTINGS.place_chunk          # the floor; big builds get fatter chunks (see chunk_size_for)
    chunk_max: int = SETTINGS.place_chunk_max
    target_seconds: float = SETTINGS.place_seconds
    delay_ms: int = SETTINGS.place_delay_ms
    flags: int = FORCE_FLAGS
    postprocess: bool = False
    clear: bool = True  # also set air in the empty cells of the bounding box (clears terrain and trees)
    terrain: bool = SETTINGS.place_terrain  # survey the ground and seat the build on it (needs the mod's /heightmap)
    # Which columns stand on the ground: "row0" = the row-0 footprint (a building's foundation ring, a
    # statue's plinth); "footprint" = every column holding a block (a creature on legs gets a level pad
    # under its whole body instead of a hill poking through its belly).
    base: str = "row0"


@dataclass
class PlaceContext:
    bridge: Bridge
    player: dict
    opts: PlaceOptions
    area: Area | None = None   # a base area marked with two points: the build lands on it instead of in front of the player


def build_origin(grid: SemanticGrid, ctx: PlaceContext) -> tuple[int, int, int]:
    """Where the (already rotated) grid's cell (0, 0, 0) goes: on the marked area when there is one,
    else in front of the player. Uses the full bounds box, not the trimmed one, because that is what
    the outline and the mod's hologram were anchored with."""
    if ctx.area is not None:
        return area_origin(ctx.area, grid.W, grid.D, sink=ctx.opts.sink)
    return plan_origin(ctx.player, (0, 0, 0, grid.W - 1, grid.H - 1, grid.D - 1), gap=ctx.opts.gap, sink=ctx.opts.sink)


def grid_to_blocks(grid: SemanticGrid, origin: tuple[int, int, int], clear: bool = True,
                   columns: np.ndarray | None = None) -> list[Block]:
    """World-space (x, y, z, state) for every filled cell, plus air for empty cells inside the bbox if ``clear``
    (only in the (W, D) ``columns`` mask when one is given: on a surveyed site the ground outside the
    building's base is graded, not flattened)."""
    ox, oy, oz = origin
    states = [ref.state_string() for ref in grid.palette]
    x0, y0, z0, x1, y1, z1 = trimmed_bbox(grid)
    sub = grid.block[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
    out: list[Block] = []
    for x, y, z in zip(*np.nonzero(sub >= 0)):
        out.append((int(x) + x0 + ox, int(y) + y0 + oy, int(z) + z0 + oz, states[int(sub[x, y, z])]))
    if clear:
        empty = sub < 0
        if columns is not None:
            empty &= columns[x0:x1 + 1, z0:z1 + 1][:, None, :]
        for x, y, z in zip(*np.nonzero(empty)):
            out.append((int(x) + x0 + ox, int(y) + y0 + oy, int(z) + z0 + oz, AIR))
    return out


def layer_chunks(blocks: list[Block], max_blocks: int = 1500) -> list[list[Block]]:
    """Group blocks bottom up into chunks of whole y-layers (a layer wider than ``max_blocks`` is split).

    Within a layer solids come first and attachables last, so nothing is placed before its support.
    """
    max_blocks = max(1, int(max_blocks))
    layers: dict[int, list[Block]] = {}
    for b in blocks:
        layers.setdefault(b[1], []).append(b)
    chunks: list[list[Block]] = []
    cur: list[Block] = []
    for y in sorted(layers):
        layer = layers[y]
        solids = sorted((b for b in layer if not is_attachable(b[3])), key=lambda b: (b[2], b[0]))
        attach = sorted((b for b in layer if is_attachable(b[3])), key=lambda b: (b[2], b[0]))
        ordered = solids + attach
        if len(ordered) > max_blocks:
            if cur:
                chunks.append(cur)
                cur = []
            for i in range(0, len(ordered), max_blocks):
                chunks.append(ordered[i:i + max_blocks])
            continue
        if cur and len(cur) + len(ordered) > max_blocks:
            chunks.append(cur)
            cur = []
        cur.extend(ordered)
    if cur:
        chunks.append(cur)
    return chunks


def chunk_size_for(n_blocks: int, opts: PlaceOptions) -> int:
    """Blocks per tick-chunk so the whole build lands in about ``opts.target_seconds``.

    Never below ``opts.chunk_blocks`` (small builds keep their layer-by-layer pace) and never above
    ``opts.chunk_max`` (one server tick must stay cheap)."""
    import math

    ticks_per_chunk = 1 + math.ceil(max(0, opts.delay_ms) / 50)
    max_chunks = max(1.0, opts.target_seconds * 20 / ticks_per_chunk)
    wanted = math.ceil(n_blocks / max_chunks)
    return max(opts.chunk_blocks, min(opts.chunk_max, wanted))


def show_outline(bridge: Bridge, lo: tuple[int, int, int], hi: tuple[int, int, int], phase: str) -> list[str]:
    """Best effort: the in-game outline is a courtesy and must never fail a build. Returns warnings."""
    try:
        bridge.outline(lo, hi, phase)
    except BridgeError as exc:
        return [f"outline not shown ({exc})"]
    return []


def site_blocks(grid: SemanticGrid, ctx: PlaceContext, origin: tuple[int, int, int]
                ) -> tuple[tuple[int, int, int], list[Block], terrain.Site | None]:
    """Survey the ground under ``origin`` and, when the mod can see it, re-seat the build on the terrain:
    returns the final origin, the blocks to place (building, cut air in its base columns, terrain
    edits) and the site. Without a survey the blocks are the plain ``clear`` placement at ``origin``.
    Raises ValueError when the build cannot fit on the site (height limit)."""
    opts = ctx.opts
    ox, oy, oz = origin
    heights = terrain.survey(ctx.bridge, terrain.footprint_of(grid, (ox, oz))) if opts.terrain else None
    if not heights:
        return origin, grid_to_blocks(grid, origin, clear=opts.clear), None
    mask = terrain.column_mask(grid) if opts.base == "footprint" else terrain.base_mask(grid)
    site = terrain.plan_site(grid, heights, (ox, oz), feet_y=oy, base=terrain.mask_columns(mask, (ox, oz)))
    site.ground -= int(opts.sink)
    origin = (ox, site.ground, oz)
    edits = terrain.terraform(grid, site, origin, known=_known_blocks())
    blocks = grid_to_blocks(grid, origin, clear=opts.clear, columns=mask)
    seen = {(x, y, z) for x, y, z, _ in blocks}
    blocks += [e for e in edits if (e[0], e[1], e[2]) not in seen]
    return origin, blocks, site


def _known_blocks() -> set[str] | None:
    from craftpilot.blocks.catalog import known_blocks

    return known_blocks()


def place_grid(grid: SemanticGrid, ctx: PlaceContext, label: str = "", undo_path: Path | None = None) -> dict[str, Any]:
    """Anchor the (already rotated) grid in front of the player, seat it on the terrain when the mod can
    survey the ground, and queue it on the mod. Returns immediately. With ``undo_path`` the positions
    written are recorded there (see place/undo.py) so the build can be removed again."""
    opts = ctx.opts
    origin = build_origin(grid, ctx)
    origin, blocks, site = site_blocks(grid, ctx, origin)
    chunk_blocks = chunk_size_for(len(blocks), opts)
    chunks = layer_chunks(blocks, chunk_blocks)
    ox, oy, oz = origin
    x0, y0, z0, x1, y1, z1 = trimmed_bbox(grid)
    warnings: list[str] = []
    warnings += show_outline(ctx.bridge, (x0 + ox, y0 + oy, z0 + oz), (x1 + ox, y1 + oy, z1 + oz), "placing")
    resp = ctx.bridge.setblocks([(c, opts.delay_ms) for c in chunks], flags=opts.flags, postprocess=opts.postprocess)
    air = sum(1 for b in blocks if b[3] == AIR)
    invalid = int(resp.get("invalid", 0))
    if invalid:
        samples = ", ".join(resp.get("invalid_samples", [])[:3])
        warnings.append(f"{invalid} block states were rejected by the game (version mismatch?): {samples}")
    if site is not None:
        warnings.append("Site: " + site.summary())
    if undo_path is not None:
        from craftpilot.place import undo

        try:
            undo.write_record(undo_path, label, origin, blocks)
        except OSError as exc:
            warnings.append(f"undo record not written ({exc})")
            undo_path = None
    return {
        "label": label,
        "undo_file": str(undo_path) if undo_path is not None else None,
        "origin": [ox, oy, oz],
        "world_bbox": [[x0 + ox, y0 + oy, z0 + oz], [x1 + ox, y1 + oy, z1 + oz]],
        "site": site.as_dict() if site is not None else None,
        "blocks": len(blocks) - air,
        "air": air,
        "chunks": len(chunks),
        "chunk_blocks": chunk_blocks,
        "queued": int(resp.get("queued", len(blocks))),
        "invalid": invalid,
        "invalid_samples": resp.get("invalid_samples", []),
        "estimated_seconds": round(len(chunks) * (50 + max(0, opts.delay_ms)) / 1000, 1),
        "warnings": warnings,
    }
