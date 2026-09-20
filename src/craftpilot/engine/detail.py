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

_QUIET_NEIGHBOURS = {Role.WALL, Role.FRAME, Role.ROOF_FILL, Role.FOUNDATION, Role.PARAPET, Role.CHIMNEY, Role.BEAM, Role.ACCENT,
                     Role.TRIM, Role.PARTITION, Role.ROOF, Role.ROOF_EDGE}
_NOISY_NEIGHBOURS = {Role.WINDOW, Role.DOOR, Role.SILL, Role.SHUTTER, Role.DETAIL, Role.LIGHT}


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
        # Not next to an opening or another detail, and a plain full block itself.
        quiet = int(grid.shape[x, y, z]) == BShape.FULL
        for (dx, dy, dz) in ((ax, 0, az), (-ax, 0, -az), (0, 1, 0), (0, -1, 0)):
            px, py, pz = x + dx, y + dy, z + dz
            if not grid.in_bounds(px, py, pz):
                quiet = False
                break
            r, sh = int(grid.role[px, py, pz]), int(grid.shape[px, py, pz])
            if r in _NOISY_NEIGHBOURS or (r == Role.WALL and sh != BShape.FULL) or r in (Role.EMPTY, Role.INTERIOR):
                quiet = False
                break
        # The cell in front of the side neighbours must be open too (no shutter or sill beside us).
        for (dx, dz) in ((ax, az), (-ax, -az)):
            px, pz = x + dx + vx, z + dz + vz
            if grid.in_bounds(px, y, pz) and int(grid.role[px, y, pz]) in _NOISY_NEIGHBOURS:
                quiet = False
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
    has_wall_shape = any(catalog.family(f.family) and catalog.family(f.family).has("wall")
                         for f in program.palette.primary.families)
    for (x, y, z, n) in _exterior_wall_cells(grid):
        h = float(grid.h_norm[x, y, z])
        # Intact builds get few notches; worn ones many, and more near the ground.
        p = 0.09 * density * (0.5 + weathering * (1.0 + (1.0 - h)))
        # The field is rank-uniform, so thresholding it gives exactly p, in patches; the local draw thins the patches.
        if float(field[x, y, z]) >= p * 1.5 or rng.random() >= 0.67:
            continue
        vx, _, vz = DIR_VEC[n]
        behind_solid = grid.in_bounds(x - vx, y, z - vz) and int(grid.role[x - vx, y, z - vz]) in _QUIET_NEIGHBOURS
        part = int(grid.part_id[x, y, z])
        ox, oz = x + vx, z + vz
        r = rng.random()
        # Inline: stairs and slabs set into the wall plane. Protruding: small textures on the face.
        fixed = Flag.PERIMETER | Flag.NO_TEXTURE | Flag.FIXED_SHAPE
        if r < 0.22:
            grid.set(x, y, z, Role.WALL, BShape.STAIR, OPPOSITE[n], part, h, fixed)
        elif r < 0.44:
            grid.set(x, y, z, Role.WALL, BShape.STAIR_UPSIDE, OPPOSITE[n], part, h, fixed)
        elif r < 0.58 and has_wall_shape:
            # A wall block set into the wall reads as a recessed post.
            grid.set(x, y, z, Role.WALL, BShape.WALLBLOCK, Dir.NONE, part, h, Flag.PERIMETER | Flag.NO_TEXTURE)
        elif r < 0.75:
            grid.set(ox, y, oz, Role.DETAIL, BShape.BUTTON, n, part, h)
        elif r < 0.9:
            # Trapdoor: open and flush against the wall as a panel. Ledges belong under windows only.
            grid.set(ox, y, oz, Role.DETAIL, BShape.TRAPDOOR, n, part, h, Flag.PROTRUDE)
        elif weathering > 0.4 and h < 0.6:
            grid.set(ox, y, oz, Role.DETAIL, BShape.LICHEN, n, part, h)
        else:
            grid.set(ox, y, oz, Role.DETAIL, BShape.BUTTON, n, part, h)
        count += 1
    return count


def _string_courses(grid: SemanticGrid) -> int:
    """A notched band along each upper floor line: upside-down stairs set into the wall, facing in."""
    count = 0
    fixed = Flag.PERIMETER | Flag.NO_TEXTURE | Flag.FIXED_SHAPE
    for part in grid.parts:
        for k in range(1, len(part.floor_heights)):
            fy = part.floor_block_y(k)
            if not (0 <= fy < grid.H):
                continue
            for x, z in zip(*np.nonzero(part.floor_masks[k])):
                if grid.role[x, fy, z] != Role.WALL or grid.shape[x, fy, z] != BShape.FULL:
                    continue
                n = int(grid.normal[x, fy, z])
                if n not in HORIZONTAL:
                    continue
                vx, _, vz = DIR_VEC[n]
                if not grid.in_bounds(x + vx, fy, z + vz) or grid.role[x + vx, fy, z + vz] != Role.EMPTY:
                    continue
                grid.set(x, fy, z, Role.WALL, BShape.STAIR_UPSIDE, OPPOSITE[n], int(grid.part_id[x, fy, z]),
                         grid.h_norm[x, fy, z], fixed)
                count += 1
    return count


def _quoins(grid: SemanticGrid) -> int:
    """Toothed corners: every other row, the cell beside an unframed corner becomes an inward stair."""
    count = 0
    fixed = Flag.PERIMETER | Flag.NO_TEXTURE | Flag.FIXED_SHAPE
    for part in grid.parts:
        if part.spec.shape.value not in ("rect", "cross", "ring"):
            continue
        for k, m in enumerate(part.floor_masks):
            fy = part.floor_block_y(k)
            y0 = fy + 1
            y1 = min(fy + part.floor_heights[k], part.top_y + 1)
            xs, zs = np.nonzero(m)
            corners = [(int(xs.min()), int(zs.min())), (int(xs.max()), int(zs.min())),
                       (int(xs.min()), int(zs.max())), (int(xs.max()), int(zs.max()))]
            if y0 >= grid.H:
                continue
            for (cx, cz) in corners:
                if grid.role[cx, y0, cz] != Role.WALL:
                    continue   # framed corners keep their logs
                sx = 1 if cx == xs.min() else -1
                sz = 1 if cz == zs.min() else -1
                for y in range(y0, y1):
                    if (y - y0) % 2 == 1:
                        continue
                    # Alternate the tooth between the two faces meeting at the corner.
                    if ((y - y0) // 2) % 2 == 0:
                        tx, tz, n = cx + sx, cz, (Dir.NORTH if sz > 0 else Dir.SOUTH)
                    else:
                        tx, tz, n = cx, cz + sz, (Dir.WEST if sx > 0 else Dir.EAST)
                    if grid.in_bounds(tx, y, tz) and grid.role[tx, y, tz] == Role.WALL and grid.shape[tx, y, tz] == BShape.FULL \
                            and int(grid.normal[tx, y, tz]) == n:
                        grid.set(tx, y, tz, Role.WALL, BShape.STAIR, OPPOSITE[n], int(grid.part_id[tx, y, tz]),
                                 grid.h_norm[tx, y, tz], fixed)
                        count += 1
    return count


def _roof_wear(grid: SemanticGrid, program: BuildProgram) -> int:
    """Missing and sunken tiles on weathered sloped roofs. A tile is only removed where something
    solid sits beneath it, so the roof body shows but the attic never opens."""
    weathering = program.palette.roof.weathering
    condition = max(0.2, program.depth.detail)
    p = 0.10 * weathering * condition
    if p <= 0.005:
        return 0
    rng = grid.rng
    field = clustered((grid.W, grid.H, grid.D), 2.5, rng)
    count = 0
    for x, y, z in zip(*np.nonzero((grid.role == Role.ROOF) & (grid.shape == BShape.STAIR))):
        x, y, z = int(x), int(y), int(z)
        part = grid.parts[int(grid.part_id[x, y, z])] if grid.part_id[x, y, z] >= 0 else None
        if part is None or part.spec.roof.type.value in ("flat", "parapet", "none"):
            continue
        if grid.flags[x, y, z] & Flag.FIXED_SHAPE:
            continue
        if float(field[x, y, z]) >= p * 1.5 or rng.random() >= 0.67:
            continue
        below = int(grid.role[x, y - 1, z]) if y > 0 else Role.EMPTY
        n = int(grid.normal[x, y, z])
        if rng.random() < 0.55 and below in (Role.ROOF_FILL, Role.WALL, Role.ROOF):
            grid.set(x, y, z, Role.EMPTY)
        else:
            grid.set(x, y, z, Role.ROOF, BShape.SLAB_BOTTOM, Dir.NONE, part.index, 1.0, int(grid.flags[x, y, z]))
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
        part = grid.parts[int(grid.part_id[x, y, z])] if grid.part_id[x, y, z] >= 0 else None
        if part is not None and y >= part.eave_y:
            continue   # not on dormer or gable walls above the eave
        # Vines favour the shaded north side and the lower half of the wall.
        shade = 1.4 if n == Dir.NORTH else (1.0 if n in (Dir.EAST, Dir.WEST) else 0.6)
        p = 0.16 * density * shade * (1.3 - h)
        if float(field[x, z]) >= min(1.0, p * 2.0) or rng.random() >= 0.5:
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
        p = 0.3 * density
        if float(field[x, z]) >= min(1.0, p * 1.5) or rng.random() >= 0.67:
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
        "string_courses": _string_courses(grid) if rules.string_courses else 0,
        "quoins": _quoins(grid) if rules.quoins else 0,
        "wall_details": _wall_details(grid, program, rules.detail),
        "roof_wear": _roof_wear(grid, program),
        "vines": _vines(grid, rules.foliage),
        "bushes": _bushes(grid, rules.foliage),
    }
    grid.stage_done("detail", **info)
