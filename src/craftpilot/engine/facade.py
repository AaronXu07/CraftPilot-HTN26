"""Facade grammar: split each wall face into frame | bay* | frame and fill bays with terminals."""

from __future__ import annotations

import math

import numpy as np

from craftpilot.blocks.circles import perimeter
from craftpilot.grid.enums import DIR_VEC, HORIZONTAL, OPPOSITE, BShape, Dir, Flag, Role
from craftpilot.grid.semantic import Anchor, LayoutPart, SemanticGrid
from craftpilot.program.model import (
    AnchorType,
    BuildProgram,
    FacadeRules,
    Framing,
    Shape,
    WindowStyle,
)

RECT_LIKE = {Shape.rect, Shape.cross, Shape.ring}


_BLOCKING = {Role.WALL, Role.FRAME, Role.ROOF, Role.ROOF_FILL, Role.CHIMNEY, Role.PILLAR, Role.FOUNDATION,
             Role.WINDOW, Role.DOOR, Role.PARAPET, Role.MERLON}


def _eligible(grid: SemanticGrid, part: LayoutPart, x: int, z: int, side: int, rows: range) -> bool:
    """A wall cell can hold a window only if the air outside it is open on every row and no other
    part touches it along the face (a wing, a tower, a lower roof)."""
    vx, _, vz = DIR_VEC[side]
    ax, az = (1, 0) if side in (Dir.NORTH, Dir.SOUTH) else (0, 1)
    for y in rows:
        if not grid.in_bounds(x, y, z) or grid.role[x, y, z] != Role.WALL or grid.part_id[x, y, z] != part.index:
            return False
        ox, oz = x + vx, z + vz
        if not grid.in_bounds(ox, y, oz):
            return False
        r = int(grid.role[ox, y, oz])
        if r != Role.EMPTY and (grid.part_id[ox, y, oz] != part.index or r in _BLOCKING):
            return False
        for o in (-1, 1):
            nx, nz = x + ax * o, z + az * o
            if grid.in_bounds(nx, y, nz) and grid.role[nx, y, nz] != Role.EMPTY \
                    and grid.part_id[nx, y, nz] not in (part.index, -1):
                return False
    return True


def _face_runs(m: np.ndarray, side: int, grid: SemanticGrid | None = None, part: LayoutPart | None = None,
               rows: range | None = None) -> list[list[tuple[int, int]]]:
    """Contiguous runs of perimeter cells whose outward side is `side`, split where a window could not go."""
    per = perimeter(m)
    W, D = m.shape
    vx, _, vz = DIR_VEC[side]
    cells = []
    for x, z in zip(*np.nonzero(per)):
        nx, nz = x + vx, z + vz
        outside = not (0 <= nx < W and 0 <= nz < D) or not m[nx, nz]
        if not outside:
            continue
        if grid is not None and part is not None and rows is not None and not _eligible(grid, part, int(x), int(z), side, rows):
            continue
        cells.append((int(x), int(z)))
    along_x = side in (Dir.NORTH, Dir.SOUTH)
    cells.sort(key=(lambda c: (c[1], c[0])) if along_x else (lambda c: (c[0], c[1])))
    runs: list[list[tuple[int, int]]] = []
    for c in cells:
        if runs:
            p = runs[-1][-1]
            contiguous = (c[1] == p[1] and c[0] == p[0] + 1) if along_x else (c[0] == p[0] and c[1] == p[1] + 1)
            if contiguous:
                runs[-1].append(c)
                continue
        runs.append([c])
    return runs


def _choose_bays(inner: int, rules: FacadeRules) -> tuple[int, int, int]:
    """Returns (bay_width, count, leftover)."""
    best = None
    lo, hi = max(1, rules.bay_width.min), max(rules.bay_width.min, rules.bay_width.max)
    for bw in range(lo, hi + 1):
        k = (inner + 1) // (bw + 1)
        if k < 1:
            continue
        total = k * bw + (k - 1)
        leftover = inner - total
        score = leftover + (0 if leftover % 2 == 0 or not rules.symmetry else 3) - 0.1 * bw
        if best is None or score < best[0]:
            best = (score, bw, k, leftover)
    if best is None:
        return (max(1, inner), 1, 0)
    return best[1], best[2], best[3]


def _set_frame_column(grid: SemanticGrid, x: int, z: int, y0: int, y1: int, part: LayoutPart, normal: int) -> None:
    for y in range(y0, y1 + 1):
        if grid.role[x, y, z] in (Role.WALL, Role.FRAME):
            hn = grid.h_norm[x, y, z]
            grid.set(x, y, z, Role.FRAME, BShape.LOG, normal, part.index, hn, Flag.PERIMETER)


def _window(grid: SemanticGrid, cells: list[tuple[int, int]], side: int, wb: int, wh: int, part: LayoutPart,
            style: WindowStyle, rules: FacadeRules, ceiling: int) -> None:
    vx, _, vz = DIR_VEC[side]
    ww = len(cells)
    if style == WindowStyle.slit:
        shape = BShape.NONE
    elif style == WindowStyle.wall or ww > 2:
        shape = BShape.FULL
    else:
        shape = BShape.PANE
    top = min(wb + wh - 1, ceiling - 1)
    for (x, z) in cells:
        for y in range(wb, top + 1):
            if grid.role[x, y, z] == Role.WALL:
                grid.set(x, y, z, Role.WINDOW, shape, side, part.index, grid.h_norm[x, y, z],
                         Flag.NO_TEXTURE | Flag.PERIMETER)
                # Thick walls: the glass stays in the outer leaf, the inner leaf is carved so light gets in.
                bx, bz = x - vx, z - vz
                if grid.in_bounds(bx, y, bz) and grid.role[bx, y, bz] == Role.WALL \
                        and grid.part_id[bx, y, bz] == part.index:
                    grid.set(bx, y, bz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, grid.h_norm[x, y, z], Flag.INSET)
    if style == WindowStyle.arched and ww == 1 and top + 1 < ceiling:
        x, z = cells[0]
        if grid.role[x, top + 1, z] == Role.WALL:
            grid.set(x, top + 1, z, Role.WINDOW, BShape.PANE, side, part.index, grid.h_norm[x, top + 1, z],
                     Flag.NO_TEXTURE | Flag.PERIMETER)
        along_x = side in (Dir.NORTH, Dir.SOUTH)
        for sgn in (-1, 1):
            fx, fz = (x + sgn, z) if along_x else (x, z + sgn)
            if grid.in_bounds(fx, top + 1, fz) and grid.role[fx, top + 1, fz] == Role.WALL:
                away = (Dir.EAST if sgn > 0 else Dir.WEST) if along_x else (Dir.SOUTH if sgn > 0 else Dir.NORTH)
                grid.set(fx, top + 1, fz, Role.TRIM, BShape.STAIR_UPSIDE, away, part.index,
                         grid.h_norm[fx, top + 1, fz], Flag.PERIMETER)
    if rules.sills and style not in (WindowStyle.wall, WindowStyle.slit):
        for (x, z) in cells:
            sx, sz = x + vx, z + vz
            if grid.in_bounds(sx, wb - 1, sz) and grid.role[sx, wb - 1, sz] == Role.EMPTY:
                grid.set(sx, wb - 1, sz, Role.SILL, BShape.STAIR_UPSIDE, OPPOSITE[side], part.index, 0.0)
    if rules.shutters > 0 and ww == 1 and style in (WindowStyle.plain, WindowStyle.tall, WindowStyle.arched):
        if grid.rng.random() < rules.shutters:
            x, z = cells[0]
            along_x = side in (Dir.NORTH, Dir.SOUTH)
            for sgn in (-1, 1):
                fx, fz = (x + sgn, z) if along_x else (x, z + sgn)
                ox, oz = fx + vx, fz + vz
                for y in range(wb, top + 1):
                    if grid.in_bounds(ox, y, oz) and grid.role[ox, y, oz] == Role.EMPTY \
                            and grid.in_bounds(fx, y, fz) and grid.role[fx, y, fz] == Role.WALL:
                        grid.set(ox, y, oz, Role.SHUTTER, BShape.TRAPDOOR, side, part.index, 0.0)


def _door(grid: SemanticGrid, x: int, z: int, side: int, y: int, part: LayoutPart, accent: bool = False) -> None:
    inward = OPPOSITE[side]
    grid.set(x, y, z, Role.DOOR, BShape.DOOR_LOWER, inward, part.index, 0.0, Flag.NO_TEXTURE)
    grid.set(x, y + 1, z, Role.DOOR, BShape.DOOR_UPPER, inward, part.index, 0.0, Flag.NO_TEXTURE)
    grid.door = (x, y, z)
    vx, _, vz = DIR_VEC[side]
    along_x = side in (Dir.NORTH, Dir.SOUTH)
    # Thick walls: carve the inner leaf so the door leads inside.
    for yy in (y, y + 1):
        bx, bz = x - vx, z - vz
        if grid.in_bounds(bx, yy, bz) and grid.role[bx, yy, bz] == Role.WALL and grid.part_id[bx, yy, bz] == part.index:
            grid.set(bx, yy, bz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 0.0)
    # Accent surround: the two jambs and the lintel.
    if accent:
        for sgn in (-1, 0, 1):
            fx, fz = (x + sgn, z) if along_x else (x, z + sgn)
            rows = (y, y + 1, y + 2) if sgn != 0 else (y + 2,)
            for yy in rows:
                if grid.in_bounds(fx, yy, fz) and grid.role[fx, yy, fz] == Role.WALL:
                    grid.set(fx, yy, fz, Role.ACCENT, BShape.FULL, side, part.index, grid.h_norm[fx, yy, fz],
                             Flag.PERIMETER | Flag.NO_TEXTURE)
    # A lantern beside the door on the outside.
    for sgn in (-1, 1):
        fx, fz = (x + sgn, z) if along_x else (x, z + sgn)
        ox, oz = fx + vx, fz + vz
        if grid.in_bounds(ox, y + 1, oz) and grid.role[ox, y + 1, oz] == Role.EMPTY \
                and grid.in_bounds(fx, y + 1, fz) and grid.role[fx, y + 1, fz] in (Role.WALL, Role.FRAME):
            grid.set(ox, y + 1, oz, Role.LIGHT, BShape.LANTERN, Dir.NONE, part.index, 0.0)
            break


def _process_run(grid: SemanticGrid, run: list[tuple[int, int]], side: int, part: LayoutPart, k: int,
                 rules: FacadeRules, is_front: bool, want_door: bool, accent: bool = False) -> bool:
    """Returns True if a door was placed."""
    fy = part.floor_block_y(k)
    fh = part.floor_heights[k]
    y_base = fy + 1
    ceiling = min(fy + fh, part.top_y + 1)
    if ceiling - y_base < 2:
        return False
    n = len(run)
    framed = rules.framing != Framing.none and n >= 5
    inner = run[1:-1] if framed else run
    L = len(inner)
    if L < 1:
        return False
    if framed:
        for (x, z) in (run[0], run[-1]):
            _set_frame_column(grid, x, z, fy if fy >= part.base_y else y_base, ceiling - 1, part, side)
    bw, count, leftover = _choose_bays(L, rules)
    start = leftover // 2
    door_placed = False
    sep_frames = rules.framing in (Framing.bays, Framing.tudor) or (bw + 1) > rules.max_flat_run
    bays = []
    for i in range(count):
        s = start + i * (bw + 1)
        bay = inner[s:s + bw]
        free = not any(grid.reserved[x, y_base:ceiling, z].any() for (x, z) in bay)
        bays.append((i, s, bay, free))
    # The door takes the free bay nearest the middle of the face.
    door_bay = None
    if want_door and is_front and k == 0:
        free_bays = [b for b in bays if b[3]]
        if free_bays:
            door_bay = min(free_bays, key=lambda b: abs(b[0] - (count - 1) / 2))[0]
    for (i, s, bay, free) in bays:
        if i > 0 and sep_frames and rules.framing != Framing.none:
            px, pz = inner[s - 1]
            _set_frame_column(grid, px, pz, y_base, ceiling - 1, part, side)
        if not free:
            continue
        center = bay[len(bay) // 2]
        grid.anchors.append(Anchor(AnchorType.wall_bay, center[0], y_base, center[1], side, part.index, bw))
        if door_bay is not None and i == door_bay and not door_placed:
            _door(grid, center[0], center[1], side, y_base, part, accent)
            door_placed = True
            continue
        style = rules.window
        if style == WindowStyle.wall:
            cells, wb, wh = bay, y_base, ceiling - y_base
        else:
            ww = max(1, min(rules.window_width, max(1, bw - 2)))
            if style == WindowStyle.round:
                ww, wh = 1, 1
            elif style == WindowStyle.tall:
                wh = max(3, rules.window_height)
            elif style == WindowStyle.slit:
                ww, wh = 1, 2
            else:
                wh = max(1, rules.window_height)
            off = (bw - ww) // 2
            cells = bay[off:off + ww]
            wb = y_base + 1
            if style == WindowStyle.round:
                wb = y_base + 2 if ceiling - y_base > 3 else y_base + 1
            wh = min(wh, ceiling - 1 - wb)
            if wh < 1:
                continue
        _window(grid, cells, side, wb, wh, part, style, rules, ceiling)
    # Pilasters on a blank run longer than the flat-run limit.
    if count == 0 and rules.framing != Framing.none and n > rules.max_flat_run:
        step = max(3, rules.max_flat_run)
        for j in range(step, n - 1, step):
            px, pz = run[j]
            _set_frame_column(grid, px, pz, y_base, ceiling - 1, part, side)
    # Tudor beams along the floor row.
    if rules.framing == Framing.tudor and fy >= part.base_y:
        along = Dir.EAST if side in (Dir.NORTH, Dir.SOUTH) else Dir.SOUTH
        for (x, z) in run:
            if grid.role[x, fy, z] == Role.WALL:
                grid.set(x, fy, z, Role.BEAM, BShape.LOG, along, part.index, grid.h_norm[x, fy, z], Flag.PERIMETER)
    return door_placed


def _round_part(grid: SemanticGrid, part: LayoutPart, k: int, rules: FacadeRules, is_root: bool,
                want_door: bool, accent: bool = False) -> bool:
    """Windows around a circular or polygonal part, one every few blocks along the perimeter."""
    m = part.floor_masks[k]
    per = perimeter(m)
    fy = part.floor_block_y(k)
    fh = part.floor_heights[k]
    y_base = fy + 1
    ceiling = min(fy + fh, part.top_y + 1)
    if ceiling - y_base < 2:
        return False
    cells = [(int(x), int(z)) for x, z in zip(*np.nonzero(per))]
    if not cells:
        return False
    cells.sort(key=lambda c: math.atan2(c[1] - part.cz, c[0] - part.cx))
    stride = max(3, rules.bay_width.max + 1)
    placed_door = False
    # Start the walk at the south-most cell so a door can sit there.
    south_i = max(range(len(cells)), key=lambda i: (cells[i][1], -abs(cells[i][0] - part.cx)))
    for j in range(0, len(cells), stride):
        x, z = cells[(south_i + j) % len(cells)]
        side = int(grid.normal[x, y_base, z])
        if side not in HORIZONTAL or grid.role[x, y_base, z] != Role.WALL:
            continue
        if want_door and not placed_door and is_root and k == 0 and j == 0 and side == Dir.SOUTH:
            _door(grid, x, z, side, y_base, part, accent)
            placed_door = True
            continue
        style = rules.window if rules.window != WindowStyle.wall else WindowStyle.plain
        wh = 3 if style == WindowStyle.tall else (1 if style == WindowStyle.round else max(1, rules.window_height))
        wb = y_base + 1
        wh = min(wh, ceiling - 1 - wb)
        if wh >= 1:
            _window(grid, [(x, z)], side, wb, wh, part, style, rules, ceiling)
    return placed_door


def facade(grid: SemanticGrid, program: BuildProgram) -> None:
    rules = program.facade
    root_name = program.root().name
    accent = program.palette.accent is not None
    door_done = grid.door is not None
    # The door goes on whichever ground part is closest to the front (a gatehouse before the keep).
    ground = [p for p in grid.parts if not p.is_attachment and (p.spec.attach is None or p.spec.attach.side.value != "top")]
    front_parts = sorted(ground, key=lambda p: (-p.z1, p.spec.name != root_name))
    door_part = front_parts[0].spec.name if front_parts else root_name
    for part in grid.parts:
        is_root = part.spec.name == door_part
        for k in range(len(part.floor_heights)):
            m = part.floor_masks[k]
            if part.spec.shape not in RECT_LIKE:
                if _round_part(grid, part, k, rules, is_root, want_door=not door_done, accent=accent):
                    door_done = True
                continue
            fy = part.floor_block_y(k)
            rows = range(fy + 1, min(fy + part.floor_heights[k], part.top_y + 1))
            # Sides in an order that puts the front first so the door lands there.
            for side in (grid.front, OPPOSITE[grid.front], Dir.EAST, Dir.WEST):
                for run in _face_runs(m, side, grid, part, rows):
                    if len(run) < 3:
                        continue
                    is_front = side == grid.front and is_root
                    if _process_run(grid, run, side, part, k, rules, is_front, want_door=not door_done, accent=accent):
                        door_done = True
    if not door_done:
        # Fallback: a front-facing wall cell with interior behind it, nearest the middle of the face.
        vx, _, vz = DIR_VEC[grid.front]
        for cand in front_parts + [grid.parts[0]]:
            y = cand.base_y
            cells = [(int(x), int(z)) for x, z in zip(*np.nonzero(perimeter(cand.floor_masks[0])))
                     if grid.normal[x, y, z] == grid.front and grid.role[x, y, z] == Role.WALL
                     and grid.in_bounds(x - vx, y, z - vz) and grid.role[x - vx, y, z - vz] == Role.INTERIOR
                     and grid.role[x - vx, y + 1, z - vz] == Role.INTERIOR
                     and grid.in_bounds(x + vx, y, z + vz) and grid.role[x + vx, y, z + vz] == Role.EMPTY]
            if cells:
                x, z = min(cells, key=lambda c: abs(c[0] - cand.cx) + abs(c[1] - cand.cz))
                _door(grid, x, z, grid.front, y, cand, accent)
                door_done = True
                break
    grid.stage_done("facade", door=grid.door)
