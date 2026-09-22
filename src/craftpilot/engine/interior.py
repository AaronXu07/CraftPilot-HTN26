"""Basic interiors: staircases between floors, doorways between parts, optional partitions."""

from __future__ import annotations

import numpy as np

from craftpilot.blocks.circles import perimeter
from craftpilot.engine.attic import usable_mask
from craftpilot.grid.enums import DIR_VEC, HORIZONTAL, BShape, Dir, Role
from craftpilot.grid.semantic import LayoutPart, SemanticGrid
from craftpilot.program.model import BuildProgram


def _interior_cells(grid: SemanticGrid, part: LayoutPart, y: int) -> np.ndarray:
    m = np.zeros((grid.W, grid.D), dtype=bool)
    if not (0 <= y < grid.H):
        return m
    xs, zs = np.nonzero(part.mask)
    for x, z in zip(xs, zs):
        if grid.role[x, y, z] == Role.INTERIOR and grid.part_id[x, y, z] == part.index:
            m[x, z] = True
    return m


def _try_stairs(grid: SemanticGrid, part: LayoutPart, k: int, x0: int, z0: int, dx: int, dz: int) -> bool:
    """A straight run from floor k up to k + 1 starting at (x0, z0), ascending along (dx, dz).

    A storey of height fh needs fh stair blocks: steps at fy+1 .. fy+fh, the last one set into the
    floor row of the storey above, then a solid landing cell. The floor above the middle steps opens
    for head room; a player standing on step y needs y+1 and y+2 clear.
    """
    fy = part.floor_block_y(k)
    fh = part.level_height(k)
    fy_next = part.floor_block_y(k + 1)
    steps = fh
    cells = [(x0 + dx * i, z0 + dz * i) for i in range(steps + 1)]   # last one is the landing
    # A floor cell to step onto the run from.
    ax, az = x0 - dx, z0 - dz
    if not grid.in_bounds(ax, fy + 2, az) or not part.mask[ax, az] or grid.role[ax, fy + 1, az] != Role.INTERIOR \
            or grid.role[ax, fy + 2, az] != Role.INTERIOR or grid.role[ax, fy, az] not in (Role.FLOOR, Role.FOUNDATION, Role.PARTITION):
        return False
    for i, (x, z) in enumerate(cells):
        if not grid.in_bounds(x, fy_next + 2, z) or not part.mask[x, z]:
            return False
        # Never build over another run's head room or through its opening.
        if grid.reserved[x, fy + 1:fy_next + 3, z].any():
            return False
        if i < steps:
            y = fy + 1 + i
            # The step cell, and three cells of head room: moving onto the next step's front half
            # lifts the head a block higher than standing still.
            for yy in range(fy + 1, y + 4):
                r = int(grid.role[x, yy, z])
                if yy == fy_next:
                    if r not in (Role.FLOOR, Role.INTERIOR):
                        return False
                elif yy > fy_next:
                    if r not in (Role.INTERIOR, Role.EMPTY):
                        return False
                elif r != Role.INTERIOR:
                    return False
        else:
            # Landing on the upper floor with head room.
            if int(grid.role[x, fy_next, z]) not in (Role.FLOOR, Role.INTERIOR):
                return False
            if any(int(grid.role[x, yy, z]) != Role.INTERIOR for yy in (fy_next + 1, fy_next + 2)):
                return False
    facing = Dir.EAST if dx > 0 else Dir.WEST if dx < 0 else Dir.SOUTH if dz > 0 else Dir.NORTH
    for i, (x, z) in enumerate(cells[:-1]):
        y = fy + 1 + i
        grid.set(x, y, z, Role.STAIRCASE, BShape.STAIR, facing, part.index, 0.0)
        for yy in range(fy + 1, y):
            grid.set(x, yy, z, Role.PARTITION, BShape.FULL, Dir.NONE, part.index, 0.0)
        # Head room through the floor above (the last step sits in that row itself).
        if y + 3 >= fy_next and y < fy_next:
            grid.set(x, fy_next, z, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 0.0)
    lx, lz = cells[-1]
    grid.set(lx, fy_next, lz, Role.FLOOR, BShape.FULL, Dir.UP, part.index, 0.0)
    # Reserve the run, its head room and the opening so the next storey's run goes elsewhere.
    for (x, z) in cells + [(ax, az)]:
        grid.reserved[x, fy + 1:fy_next + 4, z] = True
    return True


_RING2 = [(0, 0), (1, 0), (1, 1), (0, 1)]                                   # 2x2, clockwise from above
_RING3 = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]   # 3x3 around a centre post


def _free_column(grid: SemanticGrid, part: LayoutPart, x: int, z: int, y_from: int, fy_next: int, y_to: int) -> bool:
    if not grid.in_bounds(x, y_to, z) or not part.mask[x, z] or grid.reserved[x, y_from:y_to + 1, z].any():
        return False
    for yy in range(y_from, y_to + 1):
        r = int(grid.role[x, yy, z])
        if yy == fy_next:
            if r not in (Role.FLOOR, Role.INTERIOR):
                return False
        elif yy > fy_next:
            if r not in (Role.INTERIOR, Role.EMPTY):
                return False
        elif r != Role.INTERIOR:
            return False
    return True


def _try_spiral(grid: SemanticGrid, part: LayoutPart, k: int, x0: int, z0: int, ring: list[tuple[int, int]],
                post: tuple[int, int] | None) -> bool:
    """A spiral: one step per ring cell, rising one block per step, no fill underneath.

    A 3x3 ring gives every step its own column and lands inside the ring on the upper floor. A 2x2
    ring wraps a full turn in four steps, so it opens the whole 2x2 above and exits sideways onto the
    upper floor; landing inside it would sit three blocks over the first step.
    """
    fy = part.floor_block_y(k)
    fh = part.level_height(k)
    fy_next = part.floor_block_y(k + 1)
    cells = [(x0 + dx, z0 + dz) for dx, dz in ring]
    if fh > len(ring) and len(ring) == 8:
        return False
    for (x, z) in cells:
        if not _free_column(grid, part, x, z, fy + 1, fy_next, fy_next + 3):
            return False
    if post is not None:
        px, pz = x0 + post[0], z0 + post[1]
        if not grid.in_bounds(px, fy_next + 3, pz) or not part.mask[px, pz] or grid.reserved[px, fy + 1:fy_next + 4, pz].any():
            return False
    # Approach: a floor cell outside the ring, before the first step.
    sx, sz = cells[0]
    nx1, nz1 = cells[1]
    ax, az = sx - (nx1 - sx), sz - (nz1 - sz)
    if (ax, az) in cells or not (grid.in_bounds(ax, fy + 2, az) and part.mask[ax, az]
                                  and grid.role[ax, fy + 1, az] == Role.INTERIOR and grid.role[ax, fy + 2, az] == Role.INTERIOR
                                  and grid.role[ax, fy, az] in (Role.FLOOR, Role.FOUNDATION, Role.PARTITION)):
        return False
    # Landing: inside the ring for the 3x3, outside beside the last step for the 2x2.
    last = cells[(fh - 1) % len(ring)]
    if len(ring) == 8:
        landing = cells[fh % 8]
    else:
        landing = None
        for d in HORIZONTAL:
            vx, _, vz = DIR_VEC[d]
            lx, lz = last[0] + vx, last[1] + vz
            if (lx, lz) in cells or not grid.in_bounds(lx, fy_next + 2, lz) or not part.mask[lx, lz]:
                continue
            if grid.role[lx, fy_next, lz] in (Role.FLOOR, Role.INTERIOR) and \
                    all(int(grid.role[lx, yy, lz]) in (Role.INTERIOR, Role.EMPTY) for yy in (fy_next + 1, fy_next + 2)) \
                    and not grid.reserved[lx, fy_next:fy_next + 3, lz].any():
                landing = (lx, lz)
                break
        if landing is None:
            return False
    # Build. Each step faces the direction you arrive from (its low half toward the previous step),
    # so a turn is still a half-block step and never a jump.
    for i in range(fh):
        x, z = cells[i % len(ring)]
        px, pz = (ax, az) if i == 0 else cells[(i - 1) % len(ring)]
        facing = Dir.EAST if x > px else Dir.WEST if x < px else Dir.SOUTH if z > pz else Dir.NORTH
        y = fy + 1 + i
        grid.set(x, y, z, Role.STAIRCASE, BShape.STAIR, facing, part.index, 0.0)
    for (x, z) in cells:
        # The floor above opens wherever a step comes within three blocks of it.
        steps_here = [fy + 1 + i for i in range(fh) if cells[i % len(ring)] == (x, z)]
        if any(y + 3 >= fy_next and y < fy_next for y in steps_here) and grid.role[x, fy_next, z] == Role.FLOOR:
            grid.set(x, fy_next, z, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 0.0)
    if post is not None:
        px, pz = x0 + post[0], z0 + post[1]
        for yy in range(fy + 1, fy_next):
            grid.set(px, yy, pz, Role.PARTITION, BShape.FULL, Dir.NONE, part.index, 0.0)
    lx, lz = landing
    if grid.role[lx, fy_next, lz] == Role.INTERIOR:
        grid.set(lx, fy_next, lz, Role.FLOOR, BShape.FULL, Dir.UP, part.index, 0.0)
    for (x, z) in cells + [(ax, az), landing] + ([(x0 + post[0], z0 + post[1])] if post else []):
        grid.reserved[x, fy + 1:fy_next + 4, z] = True
    return True


def _stairs(grid: SemanticGrid, part: LayoutPart) -> int:
    """Straight run where the room is long enough, a spiral where it is tight, a ladder otherwise.

    Runs into an attic land where the roof leaves head room, so their candidates are ordered by how
    central the landing is instead of hugging the room's edge."""
    placed = 0
    for k in range(len(part.levels) - 1):
        fy = part.floor_block_y(k)
        fh = part.level_height(k)
        inner = _interior_cells(grid, part, fy + 1)
        if not inner.any():
            continue
        xs, zs = np.nonzero(inner)
        w, d = int(xs.max() - xs.min() + 1), int(zs.max() - zs.min() + 1)
        done = False
        landing_ok = usable_mask(grid, part, part.floor_block_y(k + 1)) if part.is_attic_level(k + 1) else None
        if landing_ok is not None and landing_ok.any():
            lx, lz = np.nonzero(landing_ok)
            target = (float(lx.mean()), float(lz.mean()))
        else:
            target = (float(xs.mean()), float(zs.mean()))
        # A straight run needs fh steps, a landing, and a cell to approach from, along one axis.
        if max(w, d) >= fh + 3 and min(w, d) >= 3:
            candidates = []
            if landing_ok is None:
                if w >= fh + 3:
                    for z in range(int(zs.min()), int(zs.min()) + 3):
                        row = sorted(int(x) for x in xs[zs == z])
                        for x in row[1:4]:
                            candidates.append((x, z, 1, 0))
                if d >= fh + 3:
                    for x in range(int(xs.min()), int(xs.min()) + 3):
                        col = sorted(int(z) for z in zs[xs == x])
                        for z in col[1:4]:
                            candidates.append((x, z, 0, 1))
            else:
                for x, z in zip(xs, zs):
                    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ex, ez = int(x) + dx * fh, int(z) + dz * fh
                        if grid.in_bounds(ex, 0, ez) and landing_ok[ex, ez]:
                            candidates.append((int(x), int(z), dx, dz))
                candidates.sort(key=lambda c: abs(c[0] + c[2] * fh - target[0]) + abs(c[1] + c[3] * fh - target[1]))
            for (x, z, dx, dz) in candidates:
                if _try_stairs(grid, part, k, x, z, dx, dz):
                    placed += 1
                    done = True
                    break
        if not done and w >= 3 and d >= 3:
            # 3x3 spiral around a post, in each corner of the room and at its centre; under a roof,
            # centred on the head room first.
            spots = [(int(xs.min()), int(zs.min())), (int(xs.max()) - 2, int(zs.min())),
                     (int(xs.min()), int(zs.max()) - 2), (int(xs.max()) - 2, int(zs.max()) - 2),
                     (int((xs.min() + xs.max()) // 2) - 1, int((zs.min() + zs.max()) // 2) - 1)]
            if landing_ok is not None:
                spots = [(int(round(target[0])) - 1 + ox, int(round(target[1])) - 1 + oz)
                         for ox in (0, -1, 1, -2, 2) for oz in (0, -1, 1, -2, 2)] + spots
            for (x, z) in spots:
                if _try_spiral(grid, part, k, x, z, _RING3, (1, 1)):
                    placed += 1
                    done = True
                    break
        if not done and w >= 2 and d >= 2:
            corners = [(int(xs.min()), int(zs.min())), (int(xs.max()) - 1, int(zs.min())),
                       (int(xs.min()), int(zs.max()) - 1), (int(xs.max()) - 1, int(zs.max()) - 1)]
            for (x, z) in corners:
                if _try_spiral(grid, part, k, x, z, _RING2, None):
                    placed += 1
                    done = True
                    break
        if not done and _ladder(grid, part, k, inner):
            placed += 1
            done = True
        if not done:
            grid.note(f"No room for a staircase in '{part.spec.name}' between floors {k + 1} and {k + 2}.")
    return placed


def _ladder(grid: SemanticGrid, part: LayoutPart, k: int, inner: np.ndarray) -> bool:
    """Narrow towers: a ladder against an interior wall, with a hatch in the floor above."""
    fy = part.floor_block_y(k)
    fy_next = part.floor_block_y(k + 1)
    for x, z in zip(*np.nonzero(inner)):
        for d in HORIZONTAL:
            vx, _, vz = DIR_VEC[d]
            wx, wz = x + vx, z + vz
            if not grid.in_bounds(wx, fy + 1, wz) or grid.role[wx, fy + 1, wz] not in (Role.WALL, Role.FRAME):
                continue
            if any(grid.role[x, y, z] != Role.INTERIOR for y in range(fy + 1, fy_next)) \
                    or grid.role[x, fy_next, z] not in (Role.FLOOR, Role.INTERIOR):
                continue
            # Head room where the ladder arrives.
            if any(not grid.in_bounds(x, y, z) or grid.role[x, y, z] != Role.INTERIOR for y in (fy_next + 1, fy_next + 2)):
                continue
            facing = {Dir.NORTH: Dir.SOUTH, Dir.SOUTH: Dir.NORTH, Dir.EAST: Dir.WEST, Dir.WEST: Dir.EAST}[d]
            for y in range(fy + 1, fy_next + 1):
                grid.set(x, y, z, Role.STAIRCASE, BShape.LADDER, facing, part.index, 0.0)
            return True
    return False


def _doorways(grid: SemanticGrid) -> int:
    count = 0
    for part in grid.parts:
        if part.parent is None or part.spec.attach is None or part.spec.attach.side.value == "top":
            continue
        parent = grid.parts[part.parent]
        shared = (perimeter(part.mask) | perimeter(parent.mask)) & part.mask & parent.mask
        if not shared.any():
            continue
        floors = min(len(part.floor_heights), len(parent.floor_heights))
        for k in range(floors):
            y = part.floor_block_y(k) + 1
            if parent.floor_block_y(k) != part.floor_block_y(k):
                continue
            openings = []
            for x, z in zip(*np.nonzero(shared)):
                for d in (Dir.NORTH, Dir.EAST):
                    vx, _, vz = DIR_VEC[d]
                    # Walk both ways through up to two wall cells to interior on each side.
                    a = [(x + vx * i, z + vz * i) for i in range(1, 3)]
                    b = [(x - vx * i, z - vz * i) for i in range(1, 3)]
                    def first_interior(cells):
                        walls = []
                        for (cx, cz) in cells:
                            if not grid.in_bounds(cx, y, cz):
                                return None
                            r = int(grid.role[cx, y, cz])
                            if r == Role.INTERIOR:
                                return walls
                            if r in (Role.WALL, Role.FRAME):
                                walls.append((cx, cz))
                            else:
                                return None
                        return None
                    wa, wb = first_interior(a), first_interior(b)
                    if wa is not None and wb is not None and grid.role[x, y, z] in (Role.WALL, Role.FRAME):
                        openings.append(((int(x), int(z)), wa + wb))
            if not openings:
                continue
            openings.sort(key=lambda o: (o[0][0] + o[0][1]))
            (x, z), extra = openings[len(openings) // 2]
            for (cx, cz) in [(x, z)] + extra:
                for yy in (y, y + 1):
                    if grid.in_bounds(cx, yy, cz) and grid.role[cx, yy, cz] in (Role.WALL, Role.FRAME):
                        grid.set(cx, yy, cz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 0.0)
            count += 1
    return count


def _partitions(grid: SemanticGrid, part: LayoutPart) -> int:
    count = 0
    for k in range(len(part.levels)):
        fy = part.floor_block_y(k)
        ceiling = part.level_ceiling(k)
        inner = _interior_cells(grid, part, fy + 1)
        if not inner.any():
            continue
        xs, zs = np.nonzero(inner)
        w, d = int(xs.max() - xs.min() + 1), int(zs.max() - zs.min() + 1)
        if w < 9 or d < 9:
            continue
        along_z = w >= d   # partition runs north-south when the room is wider than deep
        if along_z:
            px = int((xs.min() + xs.max()) // 2)
            line = [(px, int(z)) for z in range(int(zs.min()), int(zs.max()) + 1) if inner[px, z]]
        else:
            pz = int((zs.min() + zs.max()) // 2)
            line = [(int(x), pz) for x in range(int(xs.min()), int(xs.max()) + 1) if inner[x, pz]]
        if len(line) < 5:
            continue
        mid = len(line) // 2
        opening = {line[mid], line[mid - 1]}
        for (x, z) in line:
            for y in range(fy + 1, ceiling):
                if (x, z) in opening and y <= fy + 2:
                    continue
                if grid.role[x, y, z] == Role.INTERIOR:
                    grid.set(x, y, z, Role.PARTITION, BShape.FULL, Dir.NONE, part.index, 0.0)
        count += 1
    return count


_WALKABLE_AIR = {Role.INTERIOR, Role.DOOR, Role.STAIRCASE, Role.LIGHT}
_CARVABLE = {Role.WALL, Role.FRAME, Role.PARTITION}


def _walkable(grid: SemanticGrid, x: int, y: int, z: int) -> bool:
    if not grid.in_bounds(x, y + 1, z):
        return False
    return int(grid.role[x, y, z]) in _WALKABLE_AIR and int(grid.role[x, y + 1, z]) in _WALKABLE_AIR


def reachable(grid: SemanticGrid) -> np.ndarray:
    """Flood fill from the front door over cells a player can stand in, stepping up or down one block."""
    seen = np.zeros(grid.role.shape, dtype=bool)
    if grid.door is None:
        return seen
    x0, y0, z0 = grid.door
    stack = [(x0, y0, z0)]
    seen[x0, y0, z0] = True
    while stack:
        x, y, z = stack.pop()
        for d in HORIZONTAL:
            vx, _, vz = DIR_VEC[d]
            nx, nz = x + vx, z + vz
            for ny in (y, y + 1, y - 1):
                if grid.in_bounds(nx, ny, nz) and not seen[nx, ny, nz] and _walkable(grid, nx, ny, nz):
                    if ny == y + 1 and not _walkable(grid, x, y + 1, z) and grid.role[x, y, z] != Role.STAIRCASE:
                        continue
                    seen[nx, ny, nz] = True
                    stack.append((nx, ny, nz))
    return seen


def _carve_path(grid: SemanticGrid, seen: np.ndarray, start: tuple[int, int], y: int, limit: int = 3):
    """Shortest orthogonal path of carvable wall cells from an unreached cell to a reached one."""
    from collections import deque
    W, D = grid.W, grid.D
    q = deque([(start, [])])
    visited = {start}
    while q:
        (x, z), path = q.popleft()
        for d in HORIZONTAL:
            vx, _, vz = DIR_VEC[d]
            nx, nz = x + vx, z + vz
            if not (0 <= nx < W and 0 <= nz < D) or (nx, nz) in visited:
                continue
            visited.add((nx, nz))
            if path and seen[nx, y, nz]:
                return path
            r = int(grid.role[nx, y, nz])
            up = int(grid.role[nx, y + 1, nz]) if grid.in_bounds(nx, y + 1, nz) else Role.EMPTY
            if r in _CARVABLE and up in _CARVABLE | {Role.WINDOW} and len(path) < limit:
                q.append(((nx, nz), path + [(nx, nz)]))
    return None


def _connect(grid: SemanticGrid, max_carves: int = 12) -> tuple[int, list[str]]:
    """Carve doorways until every part's floors are reachable from the front door where a short wall separates them."""
    carved = 0
    notes: list[str] = []
    for _ in range(max_carves):
        seen = reachable(grid)
        best = None
        for part in grid.parts:
            for k in range(len(part.levels)):
                y = part.floor_block_y(k) + 1
                cells = [(int(x), int(z)) for x, z in zip(*np.nonzero(part.mask))
                         if _walkable(grid, int(x), y, int(z)) and grid.part_id[x, y, z] == part.index]
                if not cells or any(seen[x, y, z] for (x, z) in cells):
                    continue
                # Only boundary cells of the room can border a wall.
                for (x, z) in cells:
                    if not any(int(grid.role[x + vx, y, z + vz]) in _CARVABLE for vx, _, vz in
                               (DIR_VEC[d] for d in HORIZONTAL) if grid.in_bounds(x + vx, y, z + vz)):
                        continue
                    path = _carve_path(grid, seen, (x, z), y)
                    if path is not None and (best is None or len(path) < len(best[1])):
                        best = (y, path)
        if best is None:
            break
        y, path = best
        for (cx, cz) in path:
            for yy in (y, y + 1):
                grid.set(cx, yy, cz, Role.INTERIOR, BShape.FULL, Dir.NONE, int(grid.part_id[cx, yy, cz]), 0.0)
        carved += 1
    seen = reachable(grid)
    for part in grid.parts:
        if part.spec.attach is not None and part.spec.attach.side.value == "top":
            continue
        y = part.floor_block_y(0) + 1
        cells = [(int(x), int(z)) for x, z in zip(*np.nonzero(part.mask)) if _walkable(grid, int(x), y, int(z))]
        if cells and not any(seen[x, y, z] for (x, z) in cells):
            notes.append(f"Ground floor of '{part.spec.name}' is not reachable from the door.")
    return carved, notes


def interior(grid: SemanticGrid, program: BuildProgram) -> None:
    rules = program.interior
    info: dict[str, int] = {}
    if rules.stairs:
        info["stairs"] = sum(_stairs(grid, p) for p in grid.parts if len(p.levels) >= 2)
    if rules.doorways:
        info["doorways"] = _doorways(grid)
    if rules.partitions:
        info["partitions"] = sum(_partitions(grid, p) for p in grid.parts)
    if rules.doorways:
        carved, notes = _connect(grid)
        info["connect_carves"] = carved
        for n in notes:
            grid.note(n)
    grid.stage_done("interior", **info)
