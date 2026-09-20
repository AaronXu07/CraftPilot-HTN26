"""Facade grammar: split each wall face into frame | bay* | frame and fill bays with terminals."""

from __future__ import annotations

import math
import re

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
            style: WindowStyle, rules: FacadeRules, ceiling: int, accent: bool = False) -> None:
    vx, _, vz = DIR_VEC[side]
    ww = len(cells)
    if style == WindowStyle.slit:
        shape = BShape.NONE
    elif style == WindowStyle.boarded:
        shape = BShape.TRAPDOOR
    elif style == WindowStyle.gate:
        shape = BShape.FENCE_GATE
    elif style == WindowStyle.bars:
        shape = BShape.BARS
    elif style == WindowStyle.fence:
        shape = BShape.FENCE
    elif style == WindowStyle.wall or ww > 2:
        shape = BShape.FULL
    else:
        shape = BShape.PANE
    top = min(wb + wh - 1, ceiling - 1)
    if style == WindowStyle.stair_slit:
        # Lower cell a stair, upper cell an upside-down stair, both facing along the wall: their
        # open quarters meet at the seam as a half-wide slit. Faces toward the middle of the face.
        along_x = side in (Dir.NORTH, Dir.SOUTH)
        for (x, z) in cells:
            if top < wb + 1:
                break
            if along_x:
                facing = Dir.EAST if x < part.cx else Dir.WEST
            else:
                facing = Dir.SOUTH if z < part.cz else Dir.NORTH
            if grid.role[x, wb, z] == Role.WALL and grid.role[x, wb + 1, z] == Role.WALL:
                hn = grid.h_norm[x, wb, z]
                fl = Flag.PERIMETER | Flag.NO_TEXTURE | Flag.FIXED_SHAPE
                grid.set(x, wb, z, Role.WALL, BShape.STAIR, facing, part.index, hn, fl)
                grid.set(x, wb + 1, z, Role.WALL, BShape.STAIR_UPSIDE, facing, part.index, hn, fl)
                # Thick walls: carve the inner leaf behind the slit so light comes through.
                bx, bz = x - vx, z - vz
                for y in (wb, wb + 1):
                    if grid.in_bounds(bx, y, bz) and grid.role[bx, y, bz] == Role.WALL and grid.part_id[bx, y, bz] == part.index:
                        grid.set(bx, y, bz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, hn, Flag.INSET)
        return
    # Glass walls, boards, gates and arched windows are never inset: the arch needs the glass in plane.
    keep_flat = Flag.INSET if style in (WindowStyle.wall, WindowStyle.boarded, WindowStyle.gate, WindowStyle.arched,
                                        WindowStyle.bars, WindowStyle.fence) else 0
    for (x, z) in cells:
        for y in range(wb, top + 1):
            if grid.role[x, y, z] == Role.WALL:
                grid.set(x, y, z, Role.WINDOW, shape, side, part.index, grid.h_norm[x, y, z],
                         Flag.NO_TEXTURE | Flag.PERIMETER | keep_flat)
                # Thick walls: the glass stays in the outer leaf, the inner leaf is carved so light gets in.
                bx, bz = x - vx, z - vz
                if grid.in_bounds(bx, y, bz) and grid.role[bx, y, bz] == Role.WALL \
                        and grid.part_id[bx, y, bz] == part.index:
                    grid.set(bx, y, bz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, grid.h_norm[x, y, z], Flag.INSET)
    if style == WindowStyle.arched and ww >= 2:
        _arch(grid, cells, side, top, part, rules)
    # Window trim, opt-in: a lintel (accent if there is one, else trim) or a full trim surround.
    trim_mode = rules.window_trim
    if trim_mode != "none" and style not in (WindowStyle.wall, WindowStyle.slit, WindowStyle.stair_slit, WindowStyle.fence):
        lintel_role = Role.ACCENT if accent else Role.TRIM
        if top + 1 <= ceiling:
            for (x, z) in cells:
                if grid.role[x, top + 1, z] == Role.WALL:
                    grid.set(x, top + 1, z, lintel_role, BShape.FULL, side, part.index, grid.h_norm[x, top + 1, z],
                             Flag.PERIMETER | Flag.NO_TEXTURE)
        if trim_mode == "surround":
            along_x = side in (Dir.NORTH, Dir.SOUTH)
            first, last = cells[0], cells[-1]
            jambs = [((first[0] - 1, first[1]) if along_x else (first[0], first[1] - 1)),
                     ((last[0] + 1, last[1]) if along_x else (last[0], last[1] + 1))]
            for (jx, jz) in jambs:
                for y in range(wb, top + 1):
                    if grid.in_bounds(jx, y, jz) and grid.role[jx, y, jz] == Role.WALL:
                        grid.set(jx, y, jz, Role.TRIM, BShape.FULL, side, part.index, grid.h_norm[jx, y, jz],
                                 Flag.PERIMETER | Flag.NO_TEXTURE)
    if rules.sills and style not in (WindowStyle.wall, WindowStyle.slit, WindowStyle.stair_slit, WindowStyle.fence):
        # Sill: an upside-down stair or a closed trapdoor ledge; sometimes a flowering window box instead.
        box = rules.window_boxes > 0 and grid.rng.random() < rules.window_boxes and wb - 1 > part.base_y
        sill_shape = BShape.STAIR_UPSIDE if grid.rng.random() < 0.6 else BShape.TRAPDOOR
        for (x, z) in cells:
            sx, sz = x + vx, z + vz
            if grid.in_bounds(sx, wb - 1, sz) and grid.role[sx, wb - 1, sz] == Role.EMPTY:
                if box:
                    grid.set(sx, wb - 1, sz, Role.FOLIAGE, BShape.LEAVES, Dir.NONE, part.index, 0.0, Flag.PROTRUDE)
                else:
                    grid.set(sx, wb - 1, sz, Role.SILL, sill_shape, OPPOSITE[side] if sill_shape == BShape.STAIR_UPSIDE else side,
                             part.index, 0.0)
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


def _arch(grid: SemanticGrid, cells: list[tuple[int, int]], side: int, top: int, part: LayoutPart,
          rules: FacadeRules) -> None:
    """A one-block relief arch standing in front of the wall, so the glass fills the whole opening.

    Row `top` (the top row of glass): upside-down stairs in front of the two end cells, their
    missing quarter toward the middle, so the glass corners are cut off by the arch shoulders.
    Row `top + 1`: bottom slabs as the crown over the inner cells (over both cells when two wide),
    rising half a block above the shoulders. Wooden arches add thin closed trapdoors over the
    shoulders so the crown steps down at the ends instead of stopping dead."""
    vx, _, vz = DIR_VEC[side]
    ww = len(cells)
    along_x = side in (Dir.NORTH, Dir.SOUTH)
    role = Role.TRIM if rules.window_trim != "none" or _trim_contrasts(grid) else Role.WALL
    flags = Flag.NO_TEXTURE | Flag.FIXED_SHAPE | Flag.PROTRUDE

    def free(fx: int, y: int, fz: int) -> bool:
        return grid.in_bounds(fx, y, fz) and grid.role[fx, y, fz] == Role.EMPTY

    front = [(x + vx, z + vz) for (x, z) in cells]
    for idx in (0, ww - 1):
        fx, fz = front[idx]
        x, z = cells[idx]
        outward = (Dir.WEST if idx == 0 else Dir.EAST) if along_x else (Dir.NORTH if idx == 0 else Dir.SOUTH)
        if free(fx, top, fz):
            grid.set(fx, top, fz, role, BShape.STAIR_UPSIDE, outward, part.index, grid.h_norm[x, top, z], flags)
    crown_y = top + 1
    crown = range(ww) if ww == 2 else range(1, ww - 1)
    for idx in crown:
        fx, fz = front[idx]
        x, z = cells[idx]
        # The crown needs something behind it (the wall row over the window), never open air.
        if free(fx, crown_y, fz) and grid.in_bounds(x, crown_y, z) \
                and grid.role[x, crown_y, z] not in (Role.EMPTY, Role.INTERIOR):
            grid.set(fx, crown_y, fz, role, BShape.SLAB_BOTTOM, side, part.index, grid.h_norm[x, top, z], flags)
        elif ww >= 3 and free(fx, top, fz):
            # No room above (eave or roof overhang): a top slab level with the shoulders makes a
            # flat-crowned arch instead of leaving a gap between the two stairs.
            grid.set(fx, top, fz, role, BShape.SLAB_TOP, side, part.index, grid.h_norm[x, top, z], flags)
    if _ARCH_WOODEN and ww >= 3:
        for idx in (0, ww - 1):
            fx, fz = front[idx]
            x, z = cells[idx]
            if free(fx, crown_y, fz) and grid.role[fx, top, fz] != Role.EMPTY and grid.in_bounds(x, crown_y, z) \
                    and grid.role[x, crown_y, z] not in (Role.EMPTY, Role.INTERIOR):
                grid.set(fx, crown_y, fz, role, BShape.TRAPDOOR, side, part.index, grid.h_norm[x, top, z], flags)


def _steps(grid: SemanticGrid, cells: list[tuple[int, int]], side: int, y: int, part: LayoutPart, outset: int) -> None:
    """Stairs down from the door across the foundation rise, one step per block of rise."""
    vx, _, vz = DIR_VEC[side]
    for k in range(1, y + 1):
        sy = y - k
        for (cx, cz) in cells:
            sx, sz = cx + vx * (outset + k), cz + vz * (outset + k)
            if grid.in_bounds(sx, sy, sz) and grid.role[sx, sy, sz] == Role.EMPTY:
                grid.set(sx, sy, sz, Role.STEP, BShape.STAIR, OPPOSITE[side], part.index, 0.0)


def _carve_behind(grid: SemanticGrid, x: int, z: int, side: int, ys: tuple[int, ...], part: LayoutPart) -> None:
    vx, _, vz = DIR_VEC[side]
    for yy in ys:
        bx, bz = x - vx, z - vz
        if grid.in_bounds(bx, yy, bz) and grid.role[bx, yy, bz] == Role.WALL and grid.part_id[bx, yy, bz] == part.index:
            grid.set(bx, yy, bz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 0.0)


def _door(grid: SemanticGrid, x: int, z: int, side: int, y: int, part: LayoutPart, accent: bool = False,
          entrance: str = "door", outset: int = 1) -> None:
    inward = OPPOSITE[side]
    vx, _, vz = DIR_VEC[side]
    along_x = side in (Dir.NORTH, Dir.SOUTH)
    ax, az = (1, 0) if along_x else (0, 1)

    def wall_at(cx, cz, yy):
        return grid.in_bounds(cx, yy, cz) and grid.role[cx, yy, cz] == Role.WALL and grid.part_id[cx, yy, cz] == part.index

    def outside_free(cx, cz, yy):
        ox, oz = cx + vx, cz + vz
        return grid.in_bounds(ox, yy, oz) and grid.role[ox, yy, oz] == Role.EMPTY

    def set_door(cx, cz):
        grid.set(cx, y, cz, Role.DOOR, BShape.DOOR_LOWER, inward, part.index, 0.0, Flag.NO_TEXTURE)
        grid.set(cx, y + 1, cz, Role.DOOR, BShape.DOOR_UPPER, inward, part.index, 0.0, Flag.NO_TEXTURE)
        _carve_behind(grid, cx, cz, side, (y, y + 1), part)

    # Fall back to narrower entrances when the wall is not wide enough.
    if entrance == "gate":
        cells = [(x - ax, z - az), (x, z), (x + ax, z + az)]
        ok = all(wall_at(cx, cz, yy) and outside_free(cx, cz, yy) for (cx, cz) in cells for yy in (y, y + 1, y + 2)) \
            and all(wall_at(cx, cz, y + 3) for (cx, cz) in cells)
        if not ok:
            entrance = "double"
    if entrance in ("double", "portal"):
        cells = [(x, z), (x + ax, z + az)]
        ok = all(wall_at(cx, cz, yy) and outside_free(cx, cz, yy) for (cx, cz) in cells for yy in (y, y + 1)) \
            and wall_at(x - ax, z - az, y) and wall_at(x + 2 * ax, z + 2 * az, y)
        if not ok:
            entrance = "door"

    if entrance == "gate":
        cells = [(x - ax, z - az), (x, z), (x + ax, z + az)]
        for (cx, cz) in cells:
            for yy in (y, y + 1):
                grid.set(cx, yy, cz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 0.0)
            grid.set(cx, y + 2, cz, Role.RAILING, BShape.BARS, side, part.index, 0.0)
            _carve_behind(grid, cx, cz, side, (y, y + 1, y + 2), part)
            if wall_at(cx, cz, y + 3):
                grid.set(cx, y + 3, cz, Role.TRIM, BShape.FULL, side, part.index, 0.0,
                         Flag.PERIMETER | Flag.NO_TEXTURE)
        for sgn in (-1, 1):
            fx, fz = x + 2 * sgn * ax, z + 2 * sgn * az
            away = (Dir.EAST if sgn > 0 else Dir.WEST) if along_x else (Dir.SOUTH if sgn > 0 else Dir.NORTH)
            if wall_at(fx, fz, y + 3):
                grid.set(fx, y + 3, fz, Role.TRIM, BShape.STAIR_UPSIDE, away, part.index, 0.0, Flag.PERIMETER)
            for yy in range(y, y + 3):
                if wall_at(fx, fz, yy):
                    grid.set(fx, yy, fz, Role.PILLAR, BShape.LOG, side, part.index, 0.0, Flag.PERIMETER)
        grid.door = (x, y, z)
        _steps(grid, [(x + o * ax, z + o * az) for o in (-2, -1, 0, 1, 2)], side, y, part, outset)
        light_cells = [(x - 3 * ax, z - 3 * az), (x + 3 * ax, z + 3 * az)]
    elif entrance in ("double", "portal"):
        set_door(x, z)
        set_door(x + ax, z + az)
        grid.door = (x, y, z)
        if entrance == "portal":
            for (cx, cz) in ((x, z), (x + ax, z + az)):
                if wall_at(cx, cz, y + 2):
                    grid.set(cx, y + 2, cz, Role.TRIM, BShape.FULL, side, part.index, 0.0,
                             Flag.PERIMETER | Flag.NO_TEXTURE)
            for sgn, fx, fz in ((-1, x - ax, z - az), (1, x + 2 * ax, z + 2 * az)):
                away = (Dir.EAST if sgn > 0 else Dir.WEST) if along_x else (Dir.SOUTH if sgn > 0 else Dir.NORTH)
                for yy in (y, y + 1):
                    if wall_at(fx, fz, yy):
                        grid.set(fx, yy, fz, Role.PILLAR, BShape.LOG, side, part.index, 0.0, Flag.PERIMETER)
                if wall_at(fx, fz, y + 2):
                    grid.set(fx, y + 2, fz, Role.TRIM, BShape.STAIR_UPSIDE, away, part.index, 0.0, Flag.PERIMETER)
            _steps(grid, [(x + o * ax, z + o * az) for o in (-1, 0, 1, 2)], side, y, part, outset)
            light_cells = [(x - 2 * ax, z - 2 * az), (x + 3 * ax, z + 3 * az)]
        else:
            light_cells = [(x - ax, z - az), (x + 2 * ax, z + 2 * az)]
    else:
        set_door(x, z)
        grid.door = (x, y, z)
        # Surround in the trim stone: jambs and lintel.
        for sgn in (-1, 0, 1):
            fx, fz = (x + sgn, z) if along_x else (x, z + sgn)
            rows = (y, y + 1, y + 2) if sgn != 0 else (y + 2,)
            for yy in rows:
                if wall_at(fx, fz, yy):
                    grid.set(fx, yy, fz, Role.TRIM, BShape.FULL, side, part.index, grid.h_norm[fx, yy, fz],
                             Flag.PERIMETER | Flag.NO_TEXTURE)
        light_cells = [(x - ax, z - az), (x + ax, z + az)]
    # Lanterns on the outside beside the entrance.
    for (fx, fz) in light_cells:
        ox, oz = fx + vx, fz + vz
        if grid.in_bounds(ox, y + 1, oz) and grid.role[ox, y + 1, oz] == Role.EMPTY \
                and grid.in_bounds(fx, y + 1, fz) and grid.role[fx, y + 1, fz] in (Role.WALL, Role.FRAME, Role.PILLAR, Role.ACCENT):
            grid.set(ox, y + 1, oz, Role.LIGHT, BShape.LANTERN, Dir.NONE, part.index, 0.0)


def _process_run(grid: SemanticGrid, run: list[tuple[int, int]], side: int, part: LayoutPart, k: int,
                 rules: FacadeRules, is_front: bool, want_door: bool, accent: bool = False,
                 entrance: str = "door", outset: int = 1) -> bool:
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
            # Wide entrances start one cell left of centre so the pair or the arch is centred.
            cx, cz = center
            if entrance in ("double", "portal") and len(bay) % 2 == 0:
                cx, cz = bay[len(bay) // 2 - 1]
            _door(grid, cx, cz, side, y_base, part, accent, entrance, outset)
            door_placed = True
            continue
        style = rules.window
        if style == WindowStyle.wall:
            cells, wb, wh = bay, y_base, ceiling - y_base
        else:
            ww = max(1, min(rules.window_width, max(1, bw - 2)))
            if style == WindowStyle.round:
                ww, wh = 1, 1
            elif style == WindowStyle.tall or (style == WindowStyle.arched and max(1, min(rules.window_width, max(1, bw - 2))) == 1):
                wh = max(3, rules.window_height)
            elif style in (WindowStyle.slit, WindowStyle.stair_slit):
                ww, wh = 1, 2
            elif style in (WindowStyle.boarded, WindowStyle.gate, WindowStyle.fence):
                wh = max(1, min(rules.window_height, 2)) if style in (WindowStyle.gate, WindowStyle.fence) \
                    else max(1, rules.window_height)
            else:
                wh = max(1, rules.window_height)
            off = (bw - ww) // 2
            cells = bay[off:off + ww]
            wb = y_base + 1
            if style == WindowStyle.round:
                wb = y_base + 2 if ceiling - y_base > 3 else y_base + 1
            wh = min(wh, ceiling - wb)
            if wh < 1:
                continue
        _window(grid, cells, side, wb, wh, part, style, rules, ceiling, accent)
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
                want_door: bool, accent: bool = False, entrance: str = "door", outset: int = 1) -> bool:
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
            _door(grid, x, z, side, y_base, part, accent, "door", outset)
            placed_door = True
            continue
        style = rules.window if rules.window != WindowStyle.wall else WindowStyle.plain
        wh = 3 if style == WindowStyle.tall else (1 if style == WindowStyle.round else max(1, rules.window_height))
        wb = y_base + 1
        wh = min(wh, ceiling - wb)
        if wh >= 1:
            _window(grid, [(x, z)], side, wb, wh, part, style, rules, ceiling, accent)
    return placed_door


_TRIM_CONTRASTS = False   # set per build in facade()
_ARCH_WOODEN = False      # set per build in facade(): the family window arches are built from is wood


def _trim_contrasts(grid: SemanticGrid) -> bool:
    return _TRIM_CONTRASTS


def _lamp_posts(grid: SemanticGrid, part: LayoutPart, outset: int) -> int:
    """Two lantern posts flanking the approach, three blocks out from the door."""
    if grid.door is None:
        return 0
    dx, dy, dz = grid.door
    side = grid.front
    vx, _, vz = DIR_VEC[side]
    ax, az = (1, 0) if side in (Dir.NORTH, Dir.SOUTH) else (0, 1)
    ground = 0
    placed = 0
    for o in (-2, 2):
        for dist in (outset + 3, outset + 2, outset + 1):
            px, pz = dx + vx * dist + ax * o, dz + vz * dist + az * o
            if not grid.in_bounds(px, ground + 2, pz):
                continue
            if any(grid.role[px, y, pz] != Role.EMPTY for y in range(ground, ground + 3)):
                continue
            grid.set(px, ground, pz, Role.PILLAR, BShape.FENCE, Dir.NONE, part.index, 0.0)
            grid.set(px, ground + 1, pz, Role.PILLAR, BShape.FENCE, Dir.NONE, part.index, 0.0)
            grid.set(px, ground + 2, pz, Role.LIGHT, BShape.LANTERN, Dir.NONE, part.index, 0.0)
            placed += 1
            break
    return placed


def _gable_windows(grid: SemanticGrid, part: LayoutPart, rules: FacadeRules, accent: bool) -> int:
    """Windows in the triangular gable-end walls above the top storey: one in the middle, a pair
    on wide gables, and a small round one near the peak when the triangle is tall."""
    from craftpilot.program.model import RoofType
    t = part.spec.roof.type
    if t not in (RoofType.gable, RoofType.gambrel) or not part.floor_masks:
        return 0
    along_x = part.spec.roof.ridge_axis == "x" or (part.spec.roof.ridge_axis == "auto" and part.width >= part.depth)
    ends = (Dir.EAST, Dir.WEST) if along_x else (Dir.NORTH, Dir.SOUTH)
    m = part.floor_masks[-1]
    placed = 0
    for side in ends:
        runs = _face_runs(m, side)
        if not runs:
            continue
        run = max(runs, key=len)
        if len(run) < 3:
            continue
        vx, _, vz = DIR_VEC[side]

        def wall_height(x: int, z: int) -> int:
            h = 0
            y = part.eave_y
            while grid.in_bounds(x, y, z) and grid.role[x, y, z] == Role.WALL and grid.part_id[x, y, z] == part.index:
                h += 1
                y += 1
            return h

        def put_window(cells: list[tuple[int, int]], wb: int, wh: int, style: WindowStyle) -> bool:
            hs = [wall_height(x, z) for (x, z) in cells]
            if min(hs) < 1:
                return False
            ceiling = part.eave_y + min(hs)          # one row of wall stays above the window
            wh = min(wh, ceiling - wb)
            if wh < 1:
                return False
            for (x, z) in cells:
                for y in range(wb, wb + wh):
                    ox, oz = x + vx, z + vz
                    if not grid.in_bounds(ox, y, oz) or grid.role[ox, y, oz] != Role.EMPTY:
                        return False
            _window(grid, cells, side, wb, wh, part, style, rules, ceiling, accent)
            return True

        centre = run[len(run) // 2]
        hc = wall_height(*centre)
        if hc < 3:
            continue
        style = rules.window if rules.window not in (WindowStyle.wall, WindowStyle.slit, WindowStyle.stair_slit) else WindowStyle.plain
        wh = 3 if style == WindowStyle.tall else max(1, min(rules.window_height, 2))
        wb = part.eave_y + 1
        if len(run) >= 9 and hc >= 4:
            for o in (-2, 2):
                idx = len(run) // 2 + o
                if 0 <= idx < len(run) and put_window([run[idx]], wb, wh, style):
                    placed += 1
            if hc >= 7:
                # Near the peak; step down until the cell outside is clear of the rake overhang's fill.
                for wb_top in range(part.eave_y + hc - 3, part.eave_y + hc - 6, -1):
                    if wb_top > wb + wh and put_window([centre], wb_top, 1, WindowStyle.round):
                        placed += 1
                        break
        else:
            if put_window([centre], wb, wh, style):
                placed += 1
    return placed


def facade(grid: SemanticGrid, program: BuildProgram) -> None:
    rules = program.facade
    root_name = program.root().name
    accent = program.palette.accent is not None
    global _TRIM_CONTRASTS
    from craftpilot.blocks import catalog as _cat
    _trim = program.palette.trim.families[0].family if program.palette.trim and program.palette.trim.families else None
    _prim = max(program.palette.primary.families, key=lambda f: f.weight).family
    _TRIM_CONTRASTS = bool(_trim and _cat.family(_trim) and _cat.family(_prim)
                           and _cat.colour_distance(_cat.family(_trim).rgb, _cat.family(_prim).rgb) > 0.25)
    global _ARCH_WOODEN
    _arch_fam = _cat.family(_trim) if (_trim and (rules.window_trim != "none" or _TRIM_CONTRASTS)) else _cat.family(_prim)
    _ARCH_WOODEN = bool(_arch_fam and _arch_fam.material == "wood" and "trapdoor" in _arch_fam.shapes)
    door_done = grid.door is not None
    # The door goes on whichever ground part is closest to the front (a gatehouse before the keep).
    ground = [p for p in grid.parts if not p.is_attachment and (p.spec.attach is None or p.spec.attach.side.value != "top")]
    front_parts = sorted(ground, key=lambda p: (-p.z1, p.spec.name != root_name))
    door_part = front_parts[0].spec.name if front_parts else root_name
    outset = max(0, program.depth.foundation_outset)
    entrance = rules.entrance
    if entrance == "auto":
        root = program.root()
        rp = grid.part_by_name(root.name) or grid.parts[0]
        label = program.label.lower()
        castle_like = root.roof.type.value == "parapet" and (root.wall_thickness >= 2 or re.search(r"castle|fort|keep|citadel|stronghold", label))
        grand = bool(re.search(r"mansion|palace|manor|cathedral|church|temple|hall|grand|estate|villa|factory|station", label))
        plain = rules.framing.value == "none"   # modern builds do not want log pillars at the door
        if castle_like and rp.width >= 15:
            entrance = "gate"
        elif (rp.width >= 25 or grand) and root.floors >= 2 and not plain:
            entrance = "portal"
        elif (rp.width >= 17 and root.floors >= 2) or (grand and plain):
            entrance = "double"
        else:
            entrance = "door"
    grid.report["entrance"] = entrance
    for part in grid.parts:
        is_root = part.spec.name == door_part
        for k in range(len(part.floor_heights)):
            m = part.floor_masks[k]
            if part.spec.shape not in RECT_LIKE:
                if _round_part(grid, part, k, rules, is_root, want_door=not door_done, accent=accent, entrance=entrance, outset=outset):
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
                    if _process_run(grid, run, side, part, k, rules, is_front, want_door=not door_done, accent=accent,
                                    entrance=entrance, outset=outset):
                        door_done = True
    gable = sum(_gable_windows(grid, p, rules, accent) for p in grid.parts)
    grid.report["gable_windows"] = gable
    if rules.lamp_posts and grid.door is not None:
        dp = grid.parts[int(grid.part_id[grid.door[0], grid.door[1], grid.door[2]])] if grid.part_id[grid.door[0], grid.door[1], grid.door[2]] >= 0 else grid.parts[0]
        grid.report["lamp_posts"] = _lamp_posts(grid, dp, outset)
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
                _door(grid, x, z, grid.front, y, cand, accent, entrance, outset)
                door_done = True
                break
    grid.stage_done("facade", door=grid.door)
