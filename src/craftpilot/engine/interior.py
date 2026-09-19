"""Basic interiors: staircases between floors, doorways between parts, optional partitions."""

from __future__ import annotations

import numpy as np

from craftpilot.blocks.circles import perimeter
from craftpilot.grid.enums import DIR_VEC, HORIZONTAL, BShape, Dir, Role
from craftpilot.grid.semantic import LayoutPart, SemanticGrid
from craftpilot.program.model import BuildProgram


def _interior_cells(grid: SemanticGrid, part: LayoutPart, y: int) -> np.ndarray:
    m = np.zeros((grid.W, grid.D), dtype=bool)
    xs, zs = np.nonzero(part.mask)
    for x, z in zip(xs, zs):
        if grid.role[x, y, z] == Role.INTERIOR and grid.part_id[x, y, z] == part.index:
            m[x, z] = True
    return m


def _try_stairs(grid: SemanticGrid, part: LayoutPart, k: int, x0: int, z0: int, dx: int, dz: int) -> bool:
    """A straight run of stairs from floor k up to k + 1 starting at (x0, z0), ascending along (dx, dz)."""
    fy = part.floor_block_y(k)
    fh = part.floor_heights[k]
    fy_next = part.floor_block_y(k + 1)
    steps = fh - 1
    cells = [(x0 + dx * i, z0 + dz * i) for i in range(steps + 1)]   # last one is the landing
    for i, (x, z) in enumerate(cells):
        y = fy + 1 + i
        if not grid.in_bounds(x, y, z) or not part.mask[x, z]:
            return False
        # The step cell and the head room above it must be free interior.
        for yy in range(fy + 1, min(fy_next, y + 3)):
            if grid.role[x, yy, z] != Role.INTERIOR:
                return False
        if i >= steps - 2 and grid.role[x, fy_next, z] not in (Role.FLOOR, Role.INTERIOR):
            return False
    facing = Dir.EAST if dx > 0 else Dir.WEST if dx < 0 else Dir.SOUTH if dz > 0 else Dir.NORTH
    for i, (x, z) in enumerate(cells[:-1]):
        y = fy + 1 + i
        grid.set(x, y, z, Role.STAIRCASE, BShape.STAIR, facing, part.index, 0.0)
        for yy in range(fy + 1, y):
            grid.set(x, yy, z, Role.PARTITION, BShape.FULL, Dir.NONE, part.index, 0.0)
        # Head room: open the floor above the top steps.
        if i >= steps - 2:
            grid.set(x, fy_next, z, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 0.0)
    lx, lz = cells[-1]
    grid.set(lx, fy_next, lz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 0.0)
    # Guard rail around the opening on the upper floor where the hole meets open floor.
    return True


def _stairs(grid: SemanticGrid, part: LayoutPart) -> int:
    placed = 0
    for k in range(len(part.floor_heights) - 1):
        fy = part.floor_block_y(k)
        inner = _interior_cells(grid, part, fy + 1)
        if not inner.any():
            continue
        xs, zs = np.nonzero(inner)
        done = False
        # Along the back wall ascending east, then the west wall ascending south, then anywhere.
        candidates = []
        for z in range(int(zs.min()), int(zs.min()) + 3):
            row = sorted(int(x) for x in xs[zs == z])
            for x in row[:3]:
                candidates.append((x, z, 1, 0))
        for x in range(int(xs.min()), int(xs.min()) + 3):
            col = sorted(int(z) for z in zs[xs == x])
            for z in col[:3]:
                candidates.append((x, z, 0, 1))
        for (x, z, dx, dz) in candidates:
            if _try_stairs(grid, part, k, x, z, dx, dz):
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
    for k in range(len(part.floor_heights)):
        fy = part.floor_block_y(k)
        ceiling = min(fy + part.floor_heights[k], part.top_y + 1)
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
            for k in range(len(part.floor_heights)):
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
        info["stairs"] = sum(_stairs(grid, p) for p in grid.parts if len(p.floor_heights) >= 2)
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
