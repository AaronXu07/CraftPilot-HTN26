"""Final detailing: inline wall details (stairs and slabs set into the wall plane) and foliage.

Nothing here protrudes from the wall. Stairs facing inward leave a notch in the outer face, slabs
leave a crack (only where a second leaf or fill sits behind them), and cracked or chiseled
variants come from the texturing pass. Foliage is vines on shaded walls and leaf bushes at the
base, with persistent leaves so they never decay.
"""

from __future__ import annotations

import numpy as np

from craftpilot.blocks import catalog
from craftpilot.engine.noise import clustered
from craftpilot.grid.enums import DIR_VEC, HORIZONTAL, Dir, Flag, OPPOSITE, Role, BShape
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.program.model import BuildProgram

_QUIET_NEIGHBOURS = {Role.WALL, Role.FRAME, Role.ROOF_FILL, Role.FOUNDATION, Role.PARAPET, Role.CHIMNEY}


def _leaf_block(program: BuildProgram) -> str:
    """Leaves that match the wood in the palette, falling back to oak."""
    for role in ("framing", "primary", "trim", "roof"):
        rp = getattr(program.palette, role)
        if rp is None:
            continue
        for fw in rp.families:
            fam = catalog.family(fw.family)
            if fam and fam.has("log") and fam.name in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak",
                                                       "mangrove", "cherry", "pale_oak"):
                return f"minecraft:{fam.name}_leaves"
    return "minecraft:oak_leaves"


def _exterior_wall_cells(grid: SemanticGrid):
    """(x, y, z, normal) for wall cells that face open air and have quiet neighbours in the wall plane."""
    out = []
    for x, y, z in zip(*np.nonzero(grid.role == Role.WALL)):
        x, y, z = int(x), int(y), int(z)
        n = int(grid.normal[x, y, z])
        if n not in HORIZONTAL or not (grid.flags[x, y, z] & Flag.PERIMETER):
            continue
        vx, _, vz = DIR_VEC[n]
        if not grid.in_bounds(x + vx, y, z + vz) or grid.role[x + vx, y, z + vz] != Role.EMPTY:
            continue
        ax, az = (1, 0) if n in (Dir.NORTH, Dir.SOUTH) else (0, 1)
        quiet = True
        for (dx, dy, dz) in ((ax, 0, az), (-ax, 0, -az), (0, 1, 0), (0, -1, 0)):
            px, py, pz = x + dx, y + dy, z + dz
            if not grid.in_bounds(px, py, pz) or int(grid.role[px, py, pz]) not in _QUIET_NEIGHBOURS:
                quiet = False
                break
        if quiet:
            out.append((x, y, z, n))
    return out


def _wall_details(grid: SemanticGrid, program: BuildProgram, density: float) -> int:
    if density <= 0:
        return 0
    rng = grid.rng
    weathering = program.palette.primary.weathering
    field = clustered((grid.W, grid.H, grid.D), 3.0, rng)
    count = 0
    for (x, y, z, n) in _exterior_wall_cells(grid):
        h = float(grid.h_norm[x, y, z])
        p = 0.07 * density * (1.0 + weathering * (1.0 - h))
        u = 0.5 * float(field[x, y, z]) + 0.5 * float(rng.random())
        if u >= p:
            continue
        vx, _, vz = DIR_VEC[n]
        behind_solid = grid.in_bounds(x - vx, y, z - vz) and int(grid.role[x - vx, y, z - vz]) in _QUIET_NEIGHBOURS
        part = int(grid.part_id[x, y, z])
        r = rng.random()
        if r < 0.45:
            grid.set(x, y, z, Role.WALL, BShape.STAIR, OPPOSITE[n], part, h, Flag.PERIMETER | Flag.NO_TEXTURE)
        elif r < 0.8:
            grid.set(x, y, z, Role.WALL, BShape.STAIR_UPSIDE, OPPOSITE[n], part, h, Flag.PERIMETER | Flag.NO_TEXTURE)
        elif behind_solid:
            shape = BShape.SLAB_TOP if rng.random() < 0.5 else BShape.SLAB_BOTTOM
            grid.set(x, y, z, Role.WALL, shape, Dir.NONE, part, h, Flag.PERIMETER | Flag.NO_TEXTURE)
        else:
            continue
        count += 1
    return count


def _vines(grid: SemanticGrid, density: float) -> int:
    if density <= 0:
        return 0
    rng = grid.rng
    field = clustered((grid.W, grid.D), 4.0, rng)
    count = 0
    ground = 0
    for (x, y, z, n) in _exterior_wall_cells(grid):
        h = float(grid.h_norm[x, y, z])
        # Vines favour the shaded north side and the lower half of the wall.
        shade = 1.4 if n == Dir.NORTH else (1.0 if n in (Dir.EAST, Dir.WEST) else 0.6)
        p = 0.10 * density * shade * (1.3 - h)
        if 0.6 * float(field[x, z]) + 0.4 * float(rng.random()) >= p:
            continue
        vx, _, vz = DIR_VEC[n]
        ox, oz = x + vx, z + vz
        # A strand hangs down from here toward the ground.
        length = int(rng.integers(1, 4))
        for i in range(length):
            yy = y - i
            if yy < ground or not grid.in_bounds(ox, yy, oz) or grid.role[ox, yy, oz] != Role.EMPTY:
                break
            if not (grid.in_bounds(x, yy, z) and grid.role[x, yy, z] in (Role.WALL, Role.FOUNDATION, Role.FRAME)):
                break
            grid.set(ox, yy, oz, Role.FOLIAGE, BShape.VINE, n, int(grid.part_id[x, y, z]), h)
            count += 1
    return count


def _bushes(grid: SemanticGrid, density: float) -> int:
    if density <= 0:
        return 0
    rng = grid.rng
    field = clustered((grid.W, grid.D), 3.0, rng)
    count = 0
    door = grid.door
    y = 0
    if not grid.in_bounds(0, y, 0):
        return 0
    base = (grid.role[:, y, :] != Role.EMPTY)
    candidates = []
    for x in range(grid.W):
        for z in range(grid.D):
            if base[x, z]:
                continue
            touching = False
            for d in HORIZONTAL:
                vx, _, vz = DIR_VEC[d]
                nx, nz = x + vx, z + vz
                if grid.in_bounds(nx, y, nz) and int(grid.role[nx, y, nz]) in (Role.FOUNDATION, Role.WALL, Role.FRAME):
                    touching = True
                    break
            if not touching:
                continue
            if door is not None:
                dx, dz = door[0], door[2]
                vx, _, vz = DIR_VEC[grid.front]
                # Keep the approach to the door clear.
                if abs(x - dx) <= 2 and 0 <= (z - dz) * vz + (x - dx) * vx <= 4:
                    continue
            if grid.in_bounds(x, y + 1, z) and grid.role[x, y + 1, z] != Role.EMPTY:
                continue
            candidates.append((x, z))
    for (x, z) in candidates:
        p = 0.25 * density
        if 0.7 * float(field[x, z]) + 0.3 * float(rng.random()) >= p:
            continue
        grid.set(x, y, z, Role.FOLIAGE, BShape.LEAVES, Dir.NONE, -1, 0.0)
        count += 1
        if rng.random() < 0.35 and grid.in_bounds(x, y + 1, z) and grid.role[x, y + 1, z] == Role.EMPTY:
            grid.set(x, y + 1, z, Role.FOLIAGE, BShape.LEAVES, Dir.NONE, -1, 0.0)
            count += 1
    return count


def detail(grid: SemanticGrid, program: BuildProgram) -> None:
    rules = program.depth
    grid.leaf_block = _leaf_block(program)
    info = {
        "wall_details": _wall_details(grid, program, rules.detail),
        "vines": _vines(grid, rules.foliage),
        "bushes": _bushes(grid, rules.foliage),
    }
    grid.stage_done("detail", **info)
