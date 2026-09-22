"""Depth pass: the anti-flatness rules, applied over tags."""

from __future__ import annotations

import numpy as np

from craftpilot.blocks.circles import perimeter
from craftpilot.grid.enums import DIR_VEC, HORIZONTAL, OPPOSITE, BShape, Dir, Flag, Role
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.program.model import BuildProgram


def corbel(grid: SemanticGrid, x: int, y: int, z: int, toward_wall: int, part: int) -> None:
    """Upside-down stair under an overhang, facing the wall it hangs from."""
    if grid.in_bounds(x, y, z) and grid.role[x, y, z] == Role.EMPTY:
        grid.set(x, y, z, Role.CORBEL, BShape.STAIR_UPSIDE, toward_wall, part, 0.0, Flag.CORBEL)


def _protrude_frames(grid: SemanticGrid, steps: int) -> int:
    count = 0
    for _ in range(steps):
        targets = []
        for x, y, z in zip(*np.nonzero((grid.role == Role.FRAME) | (grid.role == Role.BEAM))):
            role = int(grid.role[x, y, z])
            for d in HORIZONTAL:
                vx, _, vz = DIR_VEC[d]
                nx, nz = x + vx, z + vz
                bx, bz = x - vx, z - vz
                if not grid.in_bounds(nx, y, nz) or not grid.in_bounds(bx, y, bz):
                    continue
                if grid.role[nx, y, nz] == Role.EMPTY and grid.role[bx, y, bz] not in (Role.EMPTY,) \
                        and not grid.footprint[nx, nz]:
                    targets.append((nx, y, nz, role, int(grid.shape[x, y, z]), int(grid.normal[x, y, z]),
                                    int(grid.part_id[x, y, z]), float(grid.h_norm[x, y, z])))
        for (nx, y, nz, role, shape, normal, part, hn) in targets:
            if grid.role[nx, y, nz] == Role.EMPTY:
                grid.set(nx, y, nz, role, shape, normal, part, hn, Flag.PROTRUDE)
                count += 1
    return count


def _diagonal_cells(w: int, h: int, rising: int) -> list[tuple[int, int, int, int]]:
    """One diagonal across a w x h panel as (column, row, shape, ascent) with row 0 at the bottom.
    `rising` +1 climbs toward higher columns, -1 toward lower. Stairs carry the line, facing the way
    it climbs; a top slab fills the half-step where the line rises less than a block per column, and
    sits under each stair of a 45 degree run so the timber reads as one piece."""
    out: list[tuple[int, int, int, int]] = []
    if w < 2 or h < 2:
        return out
    prev: tuple[int, int] | None = None    # (row, was_stair) of the previous column
    for i in range(w):
        c = i if rising > 0 else w - 1 - i
        f = i * (h - 1) / (w - 1)
        r = int(f)
        frac = f - r
        if frac < 0.5:
            if prev is not None and prev[1] and r == prev[0] + 1:
                out.append((c, r - 1, BShape.SLAB_TOP, 0))
            out.append((c, r, BShape.STAIR, rising))
            prev = (r, True)
        else:
            out.append((c, r, BShape.SLAB_TOP, 0))
            prev = (r, False)
    return out


def _x_brace(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Two diagonals crossing; where they share a cell the crossing becomes a full block."""
    cells: dict[tuple[int, int], tuple[int, int]] = {}
    for c, r, shape, asc in _diagonal_cells(w, h, +1) + _diagonal_cells(w, h, -1):
        if (c, r) in cells and cells[(c, r)] != (shape, asc):
            cells[(c, r)] = (BShape.FULL, 0)
        else:
            cells[(c, r)] = (shape, asc)
    return [(c, r, shape, asc) for (c, r), (shape, asc) in cells.items()]


def _braces(grid: SemanticGrid, steps: int) -> int:
    """Diagonal timber in the panels the facade left blank, standing proud with the posts: stairs
    and slabs of the framing family one block in front of the wall. Flush framing gets no braces,
    since a stair set into the wall would leave a hole."""
    panels = grid.report.get("braced_panels", [])
    if not panels:
        return 0
    if steps < 1:
        grid.note("Braces skipped: framing is flush (depth.frame_protrude is 0).")
        return 0
    count = 0
    for p in panels:
        side = int(p["side"])
        vx, _, vz = DIR_VEC[side]
        cells = p["cells"]
        along_x = side in (Dir.NORTH, Dir.SOUTH)
        w, h, y0 = len(cells), int(p["h"]), int(p["y0"])
        for c, r, shape, asc in _x_brace(w, h):
            x, z = cells[c]
            ox, oz, y = x + vx * steps, z + vz * steps, y0 + r
            if not grid.in_bounds(ox, y, oz) or grid.role[ox, y, oz] != Role.EMPTY:
                continue
            if asc > 0:
                normal = Dir.EAST if along_x else Dir.SOUTH
            elif asc < 0:
                normal = Dir.WEST if along_x else Dir.NORTH
            else:
                normal = side
            grid.set(ox, y, oz, Role.FRAME, shape, normal, int(p["part"]), grid.h_norm[x, y, z],
                     Flag.PROTRUDE | Flag.FIXED_SHAPE | Flag.NO_TEXTURE)
            count += 1
    return count


def _inset_windows(grid: SemanticGrid) -> int:
    moves = []
    for x, y, z in zip(*np.nonzero(grid.role == Role.WINDOW)):
        shape = int(grid.shape[x, y, z])
        # Panes are thin bars: inset them and they read as a bar in a cavity. Only full glass is inset.
        if shape != BShape.FULL:
            continue
        n = int(grid.normal[x, y, z])
        if n not in HORIZONTAL or grid.flags[x, y, z] & Flag.INSET:
            continue
        vx, _, vz = DIR_VEC[n]
        ix, iz = x - vx, z - vz
        if grid.in_bounds(ix, y, iz) and grid.role[ix, y, iz] == Role.INTERIOR:
            moves.append((int(x), int(y), int(z), ix, iz, n, shape))
    for (x, y, z, ix, iz, n, shape) in moves:
        part = int(grid.part_id[x, y, z])
        grid.set(ix, y, iz, Role.WINDOW, shape, n, part, grid.h_norm[x, y, z], Flag.NO_TEXTURE | Flag.INSET)
        grid.set(x, y, z, Role.INTERIOR, BShape.FULL, Dir.NONE, part, grid.h_norm[x, y, z], Flag.INSET)
    return len(moves)


def _floor_lips(grid: SemanticGrid) -> int:
    count = 0
    for part in grid.parts:
        for k in range(1, len(part.floor_heights)):
            fy = part.floor_block_y(k)
            m = part.floor_masks[k]
            for x, z in zip(*np.nonzero(perimeter(m))):
                n = int(grid.normal[x, fy, z])
                if n not in HORIZONTAL:
                    continue
                vx, _, vz = DIR_VEC[n]
                ox, oz = x + vx, z + vz
                if grid.in_bounds(ox, fy, oz) and grid.role[ox, fy, oz] == Role.EMPTY:
                    grid.set(ox, fy, oz, Role.FLOOR_LIP, BShape.SLAB_TOP, Dir.NONE, part.index, 0.0)
                    count += 1
    return count


def _stacked_overhang_support(grid: SemanticGrid, corbels: bool) -> int:
    """Under a stacked part that overhangs its parent: corbels, or posts at the corners without them."""
    count = 0
    for part in grid.parts:
        if part.parent is None or part.spec.attach is None or part.spec.attach.side.value != "top":
            continue
        parent = grid.parts[part.parent]
        over = part.mask & ~parent.mask
        if not over.any():
            continue
        fy = part.floor_block_y(0)
        if fy - 1 < 0 or part.base_y >= grid.H:
            continue
        if corbels:
            for x, z in zip(*np.nonzero(over & perimeter(part.mask))):
                n = int(grid.normal[x, part.base_y, z]) if grid.role[x, part.base_y, z] == Role.WALL else Dir.NONE
                if n in HORIZONTAL and grid.role[x, fy - 1, z] == Role.EMPTY:
                    grid.set(x, fy - 1, z, Role.CORBEL, BShape.STAIR_UPSIDE, OPPOSITE[n], part.index, 0.0, Flag.CORBEL)
                    count += 1
        else:
            xs, zs = np.nonzero(over)
            for (x, z) in ((xs.min(), zs.min()), (xs.max(), zs.min()), (xs.min(), zs.max()), (xs.max(), zs.max())):
                if not over[x, z]:
                    continue
                for y in range(parent.base_y, fy):
                    if grid.role[x, y, z] == Role.EMPTY:
                        grid.set(x, y, z, Role.PILLAR, BShape.PILLAR, Dir.UP, part.index, 0.0)
                        count += 1
    return count


def _interior_lighting(grid: SemanticGrid) -> int:
    """Hanging lanterns under ceilings so interiors do not spawn mobs."""
    count = 0
    for part in grid.parts:
        for k in range(len(part.levels)):
            fy = part.floor_block_y(k)
            ceiling = part.level_ceiling(k)
            attic = part.is_attic_level(k)
            m = part.floor_masks[min(k, len(part.floor_masks) - 1)]
            xs, zs = np.nonzero(m)
            if xs.size == 0:
                continue
            for x in range(int(xs.min()) + 2, int(xs.max()), 5):
                for z in range(int(zs.min()) + 2, int(zs.max()), 5):
                    y = ceiling - 1
                    if attic:
                        # Hang from whatever is overhead: the roof underside or the next attic floor.
                        y = fy + 3
                        while grid.in_bounds(x, y + 1, z) and grid.role[x, y + 1, z] == Role.INTERIOR:
                            y += 1
                    if grid.in_bounds(x, y, z) and grid.role[x, y, z] == Role.INTERIOR and grid.in_bounds(x, y + 1, z) \
                            and grid.role[x, y + 1, z] not in (Role.EMPTY, Role.INTERIOR):
                        grid.set(x, y, z, Role.LIGHT, BShape.LANTERN, Dir.UP, part.index, 0.0)
                        count += 1
    return count


def depth(grid: SemanticGrid, program: BuildProgram) -> None:
    rules = program.depth
    info = {}
    if rules.window_inset > 0:
        info["insets"] = _inset_windows(grid)
    if rules.frame_protrude > 0:
        info["protrusions"] = _protrude_frames(grid, min(2, rules.frame_protrude))
    info["braces"] = _braces(grid, min(2, rules.frame_protrude))
    if rules.floor_lips:
        info["lips"] = _floor_lips(grid)
    info["supports"] = _stacked_overhang_support(grid, rules.corbels)
    if program.interior.lighting:
        info["lights"] = _interior_lighting(grid)
    grid.stage_done("depth", **info)
