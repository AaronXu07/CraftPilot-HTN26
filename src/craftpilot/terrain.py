"""Site placement on real terrain: where the schematic goes, and what the ground around it needs.

The engine renders a building on a flat grid whose row y=0 (the foundation ring) sits on the ground.
On a superflat world the mod can paste that row at the player's feet. On real terrain it cannot:
half the house ends up inside the hill and the other half floats. This module is the fix, and it
runs in the service because the `.litematic` is the only contract with the mod (PLAN.md §3):

1. The mod samples a surface heightmap around the player (`TerrainSample`) and sends it with the
   `/build` request together with the player's position and yaw.
2. `place()` puts the schematic 2 blocks in front of the player, front (door side) facing them,
   and picks the **ground level from the terrain under the building** (a little below the median
   surface height: sunk a block into the slope reads better than perched on it). Tall builds are sunk
   further to fit under the height limit.
3. `terraform()` computes world-space **edits** the mod applies besides pasting the schematic:
   * cut — air for terrain inside the building's base columns above ground,
   * fill — a foundation down to the terrain (a solid plinth on gentle sites, perimeter wall +
     pillar grid where the drop is large), in the building's own foundation block,
   * grading — the apron around the base is ramped one block per column from the platform to the
     real terrain and re-topped with each column's own surface block,
   * steps — the engine's door steps continued down to the graded ground.

The mod's job stays simple: apply `edits` (bottom-up), then paste the schematic's non-air blocks at
`origin`. Nothing in `edits` overlaps a schematic block, so the order is only cosmetic. Without a
`TerrainSample` the placement falls back to the player's feet and `edits` is empty.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from craftpilot.blocks.state import BlockRef
from craftpilot.grid.enums import DIR_NAME, DIR_VEC, OPPOSITE, Role
from craftpilot.grid.semantic import SemanticGrid

Column = tuple[int, int]
Heights = dict[Column, tuple[int, str]]  # world (x, z) -> (y of the top solid block, its state)
Edit = tuple[int, int, int, str]

AIR = "minecraft:air"
MAX_BUILD_Y = 319  # highest block y (build limit 320; unchanged in 26.2)
MIN_BUILD_Y = -64
GAP = 2  # air blocks between the player and the nearest building block
GROUND_PERCENTILE = 0.4
FLAT_RANGE = 1  # terrain range up to this: only the trivial cut/fill
PLINTH_MAX_RANGE = 5  # up to this: solid fill under the whole base
STILT_DEPTH = 6  # deeper than this under a "stilts" build: perimeter/pillar fill only
PILLAR_SPACING = 5
MIN_MARGIN = 3
MAX_MARGIN = 8
MAX_DOOR_STEPS = 6
MAX_SINK = 16  # refuse a build the height limit would push further than this below its terrain
DEFAULT_FOUNDATION = "minecraft:cobblestone"
DEFAULT_TOP = "minecraft:grass_block"

SOIL_TOPS = {
    "minecraft:grass_block", "minecraft:dirt", "minecraft:coarse_dirt", "minecraft:podzol", "minecraft:mycelium",
    "minecraft:rooted_dirt", "minecraft:moss_block", "minecraft:mud", "minecraft:dirt_path", "minecraft:farmland",
    "minecraft:snow", "minecraft:snow_block",
}
FLUIDS = {"minecraft:water", "minecraft:lava", "minecraft:bubble_column"}
_STAIRS = {
    "minecraft:cobblestone": "minecraft:cobblestone_stairs",
    "minecraft:mossy_cobblestone": "minecraft:mossy_cobblestone_stairs",
    "minecraft:stone": "minecraft:stone_stairs",
    "minecraft:quartz_block": "minecraft:quartz_stairs",
    "minecraft:smooth_quartz": "minecraft:smooth_quartz_stairs",
    "minecraft:sandstone": "minecraft:sandstone_stairs",
    "minecraft:red_sandstone": "minecraft:red_sandstone_stairs",
    "minecraft:prismarine": "minecraft:prismarine_stairs",
    "minecraft:purpur_block": "minecraft:purpur_stairs",
}
_FACING_VEC = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}


# ------------------------------------------------------------------------------------- request


class TerrainSample(BaseModel):
    """Surface heightmap the mod samples around the player before a build.

    `heights[row][col]` is the y of the top motion-blocking non-leaf block of column
    (x0 + col, z0 + row) — `Heightmap.Types.MOTION_BLOCKING_NO_LEAVES` minus one, so water surfaces
    count as ground and trees do not. `null` marks a column with nothing in it. `tops` (optional,
    same shape) carries that block's id/state so graded ground is re-topped with grass, sand, ...
    """

    x0: int
    z0: int
    heights: list[list[int | None]]
    tops: list[list[str | None]] | None = None

    def to_heights(self) -> Heights:
        out: Heights = {}
        for r, row in enumerate(self.heights):
            trow = self.tops[r] if self.tops is not None and r < len(self.tops) else None
            for c, h in enumerate(row):
                if h is None:
                    continue
                top = trow[c] if trow is not None and c < len(trow) and trow[c] else DEFAULT_TOP
                out[(self.x0 + c, self.z0 + r)] = (int(h), str(top))
        return out

    @staticmethod
    def from_heights(heights: Heights) -> TerrainSample:
        """Inverse of to_heights over the bounding rectangle (missing columns become null)."""
        xs = [c[0] for c in heights]
        zs = [c[1] for c in heights]
        x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
        rows: list[list[int | None]] = []
        tops: list[list[str | None]] = []
        for z in range(z0, z1 + 1):
            rows.append([heights[(x, z)][0] if (x, z) in heights else None for x in range(x0, x1 + 1)])
            tops.append([heights[(x, z)][1] if (x, z) in heights else None for x in range(x0, x1 + 1)])
        return TerrainSample(x0=x0, z0=z0, heights=rows, tops=tops)


class PlacementRequest(BaseModel):
    """Where the player is and what the ground looks like; everything the service needs to site a build."""

    pos: tuple[float, float, float] = Field(description="Player position (feet), world coordinates")
    yaw: float = 0.0
    terrain: TerrainSample | None = None


# ---------------------------------------------------------------------------------------- site


@dataclass
class Site:
    """The survey and decisions for one placement (world coordinates)."""

    heights: Heights
    ground: int  # world y of grid row 0; the platform top is ground - 1
    base: set[Column] = field(default_factory=set)  # world columns with a block in grid row 0
    margin: int = MIN_MARGIN
    strategy: str = "flat"  # flat | fill | stilts
    range: int = 0
    lowest: int = 0
    highest: int = 0
    unknown: int = 0  # base columns outside the sample
    note: str = ""
    stats: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [f"terrain y {self.lowest}..{self.highest} (range {self.range}, {self.strategy})", f"ground y={self.ground}"]
        work = ", ".join(f"{k} {self.stats[k]}" for k in ("cut", "fill", "graded", "steps") if self.stats.get(k))
        if work:
            parts.append(work)
        if self.unknown:
            parts.append(f"{self.unknown} base columns outside the terrain sample")
        if self.note:
            parts.append(self.note)
        return "; ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ground": self.ground, "strategy": self.strategy, "range": self.range, "lowest": self.lowest,
            "highest": self.highest, "margin": self.margin, "unknown": self.unknown, "note": self.note,
            "stats": dict(self.stats), "summary": self.summary(),
        }


def facing_from_yaw(yaw: float) -> str:
    """The building faces the player: opposite of where the player looks. Yaw 0 = south, 90 = west."""
    look = ["south", "west", "north", "east"][int(((yaw % 360) + 45) // 90) % 4]
    return {"south": "north", "west": "east", "north": "south", "east": "west"}[look]


def column_mask(grid: SemanticGrid) -> np.ndarray:
    """(W, D) bool: columns holding at least one block."""
    return np.asarray((grid.block >= 0).any(axis=1))


def base_mask(grid: SemanticGrid) -> np.ndarray:
    """(W, D) bool: columns with a block in row 0 (foundation ring, floor, steps) — what stands on the ground."""
    m = np.asarray(grid.block[:, 0, :] >= 0)
    return m if m.any() else column_mask(grid)


def choose_origin_xz(pos: tuple[float, float, float], facing: str, grid: SemanticGrid, gap: int = GAP) -> tuple[int, int]:
    """World (x, z) of grid column (0, 0) so the building stands `gap` air blocks in front of the
    player, centred laterally, with its front (the `facing` side) toward them."""
    cols = column_mask(grid)
    xs, zs = np.nonzero(cols)
    cx0, cx1 = (int(xs.min()), int(xs.max())) if xs.size else (0, grid.W - 1)
    cz0, cz1 = (int(zs.min()), int(zs.max())) if zs.size else (0, grid.D - 1)
    bx, bz = math.floor(pos[0]), math.floor(pos[2])
    if facing == "north":  # door faces north, the player is north: building at larger z
        oz = bz + gap + 1 - cz0
        ox = round(pos[0] - (cx0 + cx1 + 1) / 2)
    elif facing == "south":
        oz = bz - gap - 1 - cz1
        ox = round(pos[0] - (cx0 + cx1 + 1) / 2)
    elif facing == "east":
        ox = bx - gap - 1 - cx1
        oz = round(pos[2] - (cz0 + cz1 + 1) / 2)
    else:  # west
        ox = bx + gap + 1 - cx0
        oz = round(pos[2] - (cz0 + cz1 + 1) / 2)
    return ox, oz


def grid_height(grid: SemanticGrid) -> int:
    """Rows of the schematic once trimmed above the building (matches export._trim_bounds)."""
    ys = np.nonzero(grid.block >= 0)[1]
    return int(ys.max()) + 1 if ys.size else 1


def choose_ground(heights: Heights, base: set[Column], feet_y: int, build_height: int) -> tuple[int, str]:
    """World y for grid row 0 and a note. `feet_y` is the fallback (the block the player stands in).
    Raises ValueError when the build cannot fit under the height limit."""
    hs = sorted(heights[c][0] for c in base if c in heights)
    note = ""
    if not hs:
        ground = int(feet_y)
        note = "no terrain under the base; ground at the player's feet"
    else:
        ground = hs[int(GROUND_PERCENTILE * (len(hs) - 1))] + 1
    top = ground + max(1, int(build_height)) - 1
    if top > MAX_BUILD_Y:
        sunk = top - MAX_BUILD_Y
        ground -= sunk
        if hs and ground < hs[0] - MAX_SINK:
            raise ValueError(f"build is {build_height} tall: even sunk {MAX_SINK} blocks into the ground it would pass the height limit (y {MAX_BUILD_Y})")
        note = f"sunk {sunk} blocks to fit under the height limit"
    if ground - 1 < MIN_BUILD_Y:
        raise ValueError(f"ground y={ground} is below the world floor")
    return int(ground), note


def plan_site(grid: SemanticGrid, heights: Heights, origin_xz: tuple[int, int], feet_y: int) -> Site:
    ox, oz = origin_xz
    base = {(ox + int(x), oz + int(z)) for x, z in zip(*np.nonzero(base_mask(grid)))}
    ground, note = choose_ground(heights, base, feet_y, grid_height(grid))
    hs = [heights[c][0] for c in base if c in heights]
    lowest, highest = (min(hs), max(hs)) if hs else (ground - 1, ground - 1)
    rng = highest - lowest
    strategy = "flat" if rng <= FLAT_RANGE else ("fill" if rng <= PLINTH_MAX_RANGE else "stilts")
    margin = max(MIN_MARGIN, min(MAX_MARGIN, rng))
    return Site(heights=heights, ground=ground, base=base, margin=margin, strategy=strategy, range=rng,
                lowest=lowest, highest=highest, unknown=sum(1 for c in base if c not in heights), note=note)


# ----------------------------------------------------------------------------------- materials


def _state(ref: BlockRef) -> str:
    props = ref.as_dict()
    return ref.block_id + ("[" + ",".join(f"{k}={v}" for k, v in sorted(props.items())) + "]" if props else "")


def _block_id(state: str) -> str:
    return state.split("[", 1)[0]


def _most_common_block(grid: SemanticGrid, role: int, rows: slice | None = None) -> str | None:
    sel = grid.role == role
    if rows is not None:
        keep = np.zeros_like(sel)
        keep[:, rows, :] = True
        sel &= keep
    idx = grid.block[sel]
    idx = idx[idx >= 0]
    if idx.size == 0:
        return None
    counts = Counter(grid.palette[int(i)].block_id for i in idx)
    return counts.most_common(1)[0][0]


_VARIANT_PREFIXES = ("mossy_", "cracked_", "chiseled_", "polished_")


def _plain(block_id: str, known: set[str] | None) -> str:
    """The unweathered member of a family: mossy_cobblestone -> cobblestone, cracked_stone_bricks ->
    stone_bricks. Weathering belongs at the wall base, not in a plinth three blocks deep."""
    ns, name = block_id.split(":", 1) if ":" in block_id else ("minecraft", block_id)
    for prefix in _VARIANT_PREFIXES:
        if name.startswith(prefix):
            plain = f"{ns}:{name[len(prefix):]}"
            if known is None or plain in known:
                return plain
    return block_id


def foundation_block(grid: SemanticGrid, known: set[str] | None = None) -> str:
    """The building's own foundation block (row 0 FOUNDATION cells, unweathered), else cobblestone."""
    b = _most_common_block(grid, Role.FOUNDATION, slice(0, 1)) or _most_common_block(grid, Role.FOUNDATION)
    if b:
        b = _plain(b, known)
        if known is None or b in known:
            return b
    return DEFAULT_FOUNDATION


def stairs_block(grid: SemanticGrid, foundation: str, known: set[str] | None = None) -> str:
    """The stairs the engine used for the door steps, else the stairs matching the foundation block."""
    b = _most_common_block(grid, Role.STEP)
    if b and b.endswith("_stairs") and (known is None or b in known):
        return b
    s = stairs_for(foundation)
    return s if known is None or s in known else "minecraft:cobblestone_stairs"


def stairs_for(block_id: str) -> str:
    if block_id in _STAIRS:
        return _STAIRS[block_id]
    ns, name = block_id.split(":", 1) if ":" in block_id else ("minecraft", block_id)
    for suffix, repl in (("_bricks", "_brick"), ("_tiles", "_tile"), ("_planks", ""), ("_block", "")):
        if name.endswith(suffix):
            name = name[: -len(suffix)] + repl
            break
    return f"{ns}:{name}_stairs"


def filler_for(top_state: str) -> str:
    """Under a re-topped column: dirt below soil and snow, else the top block itself (props stripped)."""
    block_id = _block_id(top_state)
    return "minecraft:dirt" if block_id in SOIL_TOPS else block_id


def is_fluid(state: str) -> bool:
    return _block_id(state) in FLUIDS


# ----------------------------------------------------------------------------------- terraform


def _perimeter_and_pillars(cols: set[Column]) -> set[Column]:
    """The columns of a base that carry a deep foundation: its outline plus a pillar grid."""
    return {
        (x, z) for (x, z) in cols
        if any((x + dx, z + dz) not in cols for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        or (x % PILLAR_SPACING == 0 and z % PILLAR_SPACING == 0)
    }


def _apron(base: set[Column], margin: int) -> dict[Column, int]:
    """Chebyshev distance 1..margin from the base for every column in the apron."""
    dist: dict[Column, int] = {}
    frontier = set(base)
    seen = set(base)
    for d in range(1, margin + 1):
        nxt = set()
        for (x, z) in frontier:
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    c = (x + dx, z + dz)
                    if c not in seen:
                        seen.add(c)
                        dist[c] = d
                        nxt.add(c)
        frontier = nxt
    return dist


def terraform(grid: SemanticGrid, site: Site, origin: tuple[int, int, int], known: set[str] | None = None) -> list[Edit]:
    """World-space edits that seat the building on its site (see the module docstring)."""
    ox, oy, oz = origin
    ground = site.ground
    platform = ground - 1
    heights = site.heights
    foundation = foundation_block(grid, known)
    H = grid.H

    def occupied(x: int, y: int, z: int) -> bool:
        gx, gy, gz = x - ox, y - oy, z - oz
        return 0 <= gx < grid.W and 0 <= gy < H and 0 <= gz < grid.D and grid.block[gx, gy, gz] >= 0

    edits: dict[tuple[int, int, int], str] = {}
    kind: dict[tuple[int, int, int], str] = {}
    surface: dict[Column, int] = {}

    def put(x: int, y: int, z: int, state: str, what: str) -> None:
        if occupied(x, y, z):
            return
        col = heights.get((x, z))
        if what == "cut" and col is not None and y > col[0] + 1:
            return
        edits[(x, y, z)] = state
        kind[(x, y, z)] = what

    # base: clear the hill, build the foundation
    deep = _perimeter_and_pillars(site.base) if site.strategy == "stilts" else site.base
    for c in site.base:
        col = heights.get(c)
        if col is None:
            continue
        h, _top = col
        x, z = c
        for y in range(ground, h + 2):  # +1 row for plants standing on the surface
            put(x, y, z, AIR, "cut")
        if h < platform:
            depth = platform - h
            if site.strategy == "stilts" and depth > STILT_DEPTH and c not in deep:
                surface[c] = h
                continue
            for y in range(h + 1, ground):
                put(x, y, z, foundation, "fill")
            surface[c] = platform
        else:
            surface[c] = min(h, platform)

    # apron: ramp the terrain from the platform to its real height
    m = site.margin
    for c, d in _apron(site.base, m).items():
        col = heights.get(c)
        if col is None:
            continue
        h, top = col
        if is_fluid(top) or (site.strategy == "stilts" and h < platform - STILT_DEPTH):
            continue  # water is not graded; a deep drop is left alone, the build stands on its foundation wall
        x, z = c
        target = round(platform + (h - platform) * d / (m + 1.0))
        if target == h:
            surface[c] = h
            continue
        if h > target:
            for y in range(target + 1, h + 2):
                put(x, y, z, AIR, "graded")
        else:
            filler = filler_for(top)
            for y in range(h + 1, target):
                put(x, y, z, filler, "graded")
        put(x, target, z, top, "graded")
        surface[c] = target

    _door_steps(grid, site, origin, surface, foundation, known, occupied, edits, kind)

    # an edit that sets a block to what the survey says is already there is not an edit
    for p in [p for p, st in edits.items() if heights.get((p[0], p[2])) is not None and _same_as_terrain(heights, p, st)]:
        del edits[p]
        del kind[p]
    site.stats = dict(Counter(kind.values()))
    return [(x, y, z, st) for (x, y, z), st in sorted(edits.items(), key=lambda e: (e[0][1], e[0][0], e[0][2]))]


def _same_as_terrain(heights: Heights, p: tuple[int, int, int], state: str) -> bool:
    """True for air above the surface, and for the surface block itself (the only two states the
    survey knows)."""
    h, top = heights[(p[0], p[2])]
    if state == AIR:
        return p[1] > h
    return p[1] == h and state == top


def _door_steps(grid: SemanticGrid, site: Site, origin: tuple[int, int, int], surface: dict[Column, int], foundation: str,
                known: set[str] | None, occupied, edits: dict, kind: dict) -> None:
    """Continue the engine's door steps down to the graded ground."""
    if grid.door is None:
        return
    ox, _oy, oz = origin
    dx, dy, dz = grid.door
    vx, _, vz = DIR_VEC[grid.front]
    if vx == 0 and vz == 0:
        return
    stairs = stairs_block(grid, foundation, known)
    toward = DIR_NAME[OPPOSITE[grid.front]]  # stairs ascend back toward the door
    # skip the foundation ring and any steps the engine laid: the first column with nothing at or
    # below the door's row is where the ground has to be reached
    k = 1
    while k < grid.W + grid.D:
        gx, gz = dx + vx * k, dz + vz * k
        if not (0 <= gx < grid.W and 0 <= gz < grid.D) or not (grid.block[gx, : dy + 1, gz] >= 0).any():
            break
        k += 1
    for i in range(1, MAX_DOOR_STEPS + 1):
        wx, wz = ox + dx + vx * (k + i - 1), oz + dz + vz * (k + i - 1)
        step_y = site.ground - i
        s = surface.get((wx, wz))
        if s is None:
            col = site.heights.get((wx, wz))
            s = col[0] if col is not None else None
        if s is None or s >= step_y or occupied(wx, step_y, wz):
            break
        edits[(wx, step_y, wz)] = f"{stairs}[facing={toward},half=bottom,shape=straight,waterlogged=false]"
        kind[(wx, step_y, wz)] = "steps"
        for fy in range(s + 1, step_y):
            if not occupied(wx, fy, wz):
                edits[(wx, fy, wz)] = foundation
                kind[(wx, fy, wz)] = "steps"
        surface[(wx, wz)] = step_y


# ------------------------------------------------------------------------------------- place


def place(grid: SemanticGrid, req: PlacementRequest, facing: str | None = None, known: set[str] | None = None) -> dict[str, Any]:
    """Site the rendered (already rotated) grid for a player: schematic origin, footprint, terrain edits.

    `facing` is the way the building's front faces (default: derived from the yaw, the same rule the
    CLI/service use to rotate the grid). Raises ValueError when the build cannot fit on the site.
    """
    facing = facing or facing_from_yaw(req.yaw)
    ox, oz = choose_origin_xz(req.pos, facing, grid)
    feet_y = math.floor(req.pos[1])
    base = base_mask(grid)
    xs, zs = np.nonzero(base)
    footprint = [ox + int(xs.min()), oz + int(zs.min()), ox + int(xs.max()), oz + int(zs.max())] if xs.size else [ox, oz, ox + grid.W - 1, oz + grid.D - 1]
    if req.terrain is None:
        return {
            "origin": [ox, feet_y, oz], "facing": facing, "footprint": footprint,
            "schematic_size": [grid.W, grid_height(grid), grid.D], "site": None, "edits": [],
        }
    site = plan_site(grid, req.terrain.to_heights(), (ox, oz), feet_y)
    origin = (ox, site.ground, oz)
    edits = terraform(grid, site, origin, known)
    return {
        "origin": [ox, site.ground, oz], "facing": facing, "footprint": footprint,
        "schematic_size": [grid.W, grid_height(grid), grid.D], "site": site.as_dict(),
        "edits": [[x, y, z, st] for (x, y, z, st) in edits],
    }
