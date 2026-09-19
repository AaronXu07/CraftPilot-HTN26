"""Attachment implementations. Each is a small parametric generator on the semantic grid."""

from __future__ import annotations

import math

import numpy as np

from craftpilot.blocks.circles import circle_mask, dilate, perimeter
from craftpilot.engine.attachments.base import (
    is_reserved,
    register,
    reserve,
    spread,
    target_part,
)
from craftpilot.engine.depth import corbel
from craftpilot.grid.enums import DIR_VEC, HORIZONTAL, OPPOSITE, BShape, Dir, Flag, Role
from craftpilot.grid.semantic import LayoutPart, SemanticGrid
from craftpilot.program.model import (
    Attach,
    AttachmentKind,
    AttachmentRequest,
    BuildProgram,
    PartSpec,
    RoofSpec,
    RoofType,
    Shape,
    Side,
    Size,
)

OVERWRITABLE = {Role.EMPTY, Role.INTERIOR, Role.ROOF, Role.ROOF_FILL, Role.ROOF_TRIM}


def _axis_dirs(side: int) -> tuple[int, int, int, int]:
    """(vx, vz) outward for side, and the two along-face directions as (ax, az)."""
    vx, _, vz = DIR_VEC[side]
    ax, az = (1, 0) if side in (Dir.NORTH, Dir.SOUTH) else (0, 1)
    return vx, vz, ax, az


def _face_cells(part: LayoutPart, side: int, k: int = 0) -> list[tuple[int, int]]:
    m = part.floor_masks[k] if part.floor_masks else part.mask
    per = perimeter(m)
    vx, _, vz = DIR_VEC[side]
    cells = []
    W, D = m.shape
    for x, z in zip(*np.nonzero(per)):
        nx, nz = x + vx, z + vz
        if not (0 <= nx < W and 0 <= nz < D) or not m[nx, nz]:
            cells.append((int(x), int(z)))
    along_x = side in (Dir.NORTH, Dir.SOUTH)
    cells.sort(key=(lambda c: c[0]) if along_x else (lambda c: c[1]))
    # Longest contiguous run only.
    runs: list[list[tuple[int, int]]] = []
    for c in cells:
        if runs and ((along_x and c[0] == runs[-1][-1][0] + 1 and c[1] == runs[-1][-1][1]) or
                     (not along_x and c[1] == runs[-1][-1][1] + 1 and c[0] == runs[-1][-1][0])):
            runs[-1].append(c)
        else:
            runs.append([c])
    return max(runs, key=len) if runs else []


def _roof_top_y(grid: SemanticGrid, x: int, z: int) -> int | None:
    s = grid.roof_surface[x, z]
    if not np.isfinite(s):
        return None
    return int(math.ceil(float(s))) - 1


# ---------------------------------------------------------------- towers (mass level)

@register(AttachmentKind.tower, "mass")
def tower(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    p = req.params
    short = min(part.width, part.depth)
    r = p.radius if p.radius else max(2, round(short / 8) + 1)
    diameter = 2 * r + 1
    embed = max(1, diameter // 3, part.spec.wall_thickness + 1)
    corners = [(part.x0, part.z0, -1, -1), (part.x1, part.z0, 1, -1), (part.x0, part.z1, -1, 1), (part.x1, part.z1, 1, 1)]
    count = req.count if req.count is not None else 4
    if count < 4:
        # Prefer the front corners, then the back.
        corners = corners[2:] + corners[:2]
    corners = corners[:max(0, count)]
    placed = 0
    floors = p.floors if p.floors else part.spec.floors + 1
    roof_type = p.roof if p.roof else RoofType.cone
    for i, (cx0, cz0, sx, sz) in enumerate(corners):
        cx = cx0 + sx * (r - embed + 1)
        cz = cz0 + sz * (r - embed + 1)
        cx = max(r + 1, min(grid.W - r - 2, cx))
        cz = max(r + 1, min(grid.D - r - 2, cz))
        mask = circle_mask(grid.W, grid.D, cx, cz, r)
        if not mask.any():
            continue
        spec = PartSpec(name=f"tower_{i + 1}", shape=Shape.circle, size=Size(width=diameter / max(1, part.width),
                                                                          depth=diameter / max(1, part.depth)),
                        floors=floors, floor_height=part.spec.floor_height, taper=0.0, odd_dims=True,
                        roof=RoofSpec(type=roof_type, pitch=part.spec.roof.pitch if roof_type != RoofType.spire else 2.0,
                                      overhang=1 if roof_type in (RoofType.cone, RoofType.spire) else 0,
                                      crenellated=roof_type == RoofType.parapet),
                        attach=Attach(to=part.spec.name, side=Side.south), role_hint="tower")
        xs, zs = np.nonzero(mask)
        lp = LayoutPart(index=len(grid.parts), spec=spec, mask=mask, x0=int(xs.min()), z0=int(zs.min()),
                        x1=int(xs.max()), z1=int(zs.max()), parent=part.index, is_attachment=True)
        grid.parts.append(lp)
        grid.footprint |= mask
        placed += 1
    return placed


# ---------------------------------------------------------------- chimney (roof level)

@register(AttachmentKind.chimney, "roof")
def chimney(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    if part.roof_surface is None:
        return 0
    count = req.count if req.count is not None else 1
    # Gable ends first (sides perpendicular to the ridge), never the front.
    if part.spec.roof.type in (RoofType.gable, RoofType.gambrel):
        along_x = part.spec.roof.ridge_axis == "x" or (part.spec.roof.ridge_axis == "auto" and part.width >= part.depth)
        sides = [Dir.EAST, Dir.WEST] if along_x else [Dir.NORTH, Dir.SOUTH]
    else:
        sides = [Dir.EAST, Dir.WEST, Dir.NORTH]
    sides = [s for s in sides if s != grid.front] + [s for s in sides if s == grid.front]
    placed = 0
    clear = req.params.height if req.params.height else 3
    for side in sides:
        if placed >= count:
            break
        cells = _face_cells(part, side)
        if len(cells) < 5:
            continue
        vx, vz, ax, az = _axis_dirs(side)
        mid = len(cells) // 2
        # Off-centre by a bay so it does not sit over the middle window.
        order = [mid + 2, mid - 2, mid + 3, mid - 3, mid, mid + 4, mid - 4]
        for oi in order:
            if placed >= count or not (1 <= oi < len(cells) - 2):
                continue
            (x0, z0), (x1, z1) = cells[oi], cells[oi + 1]
            cols = [(x0, z0), (x1, z1), (x0 + vx, z0 + vz), (x1 + vx, z1 + vz)]
            if any(not (0 <= cx < grid.W and 0 <= cz < grid.D) for cx, cz in cols):
                continue
            tops = [_roof_top_y(grid, cx, cz) for cx, cz in cols]
            tops = [t for t in tops if t is not None]
            if not tops:
                continue
            y_top = max(tops) + clear
            if y_top + 1 >= grid.H:
                y_top = grid.H - 2
            if is_reserved(grid, min(c[0] for c in cols), 0, min(c[1] for c in cols),
                           max(c[0] for c in cols), y_top, max(c[1] for c in cols)):
                continue
            for cx, cz in cols:
                for y in range(y_top):
                    r = int(grid.role[cx, y, cz])
                    if r == Role.DOOR:
                        continue
                    hn = min(1.0, y / max(1, part.top_y))
                    grid.set(cx, y, cz, Role.CHIMNEY, BShape.FULL, side, part.index, hn, Flag.RESERVED)
                grid.set(cx, y_top, cz, Role.CHIMNEY_CAP, BShape.SLAB_BOTTOM, Dir.NONE, part.index, 1.0)
            reserve(grid, min(c[0] for c in cols) - 1, 0, min(c[1] for c in cols) - 1,
                    max(c[0] for c in cols) + 1, y_top, max(c[1] for c in cols) + 1)
            placed += 1
            break
    return placed


# ---------------------------------------------------------------- dormers (roof level)

def _slope_sides(part: LayoutPart) -> list[int]:
    t = part.spec.roof.type
    if t in (RoofType.gable, RoofType.gambrel):
        along_x = part.spec.roof.ridge_axis == "x" or (part.spec.roof.ridge_axis == "auto" and part.width >= part.depth)
        return [Dir.SOUTH, Dir.NORTH] if along_x else [Dir.EAST, Dir.WEST]
    if t in (RoofType.hip, RoofType.mansard):
        return [Dir.SOUTH, Dir.NORTH, Dir.EAST, Dir.WEST]
    return []


def _place_dormer(grid: SemanticGrid, part: LayoutPart, side: int, cx: int, cz: int) -> bool:
    """cx, cz is the wall cell on `side` under the dormer's centre column."""
    vx, vz, ax, az = _axis_dirs(side)
    ridge_y = int(math.ceil(float(part.roof_surface[part.roof_mask].max()))) - 1
    # Walk uphill from the wall column until the roof top block is two above the eave.
    x, z = cx, cz
    for _ in range(4):
        t = _roof_top_y(grid, x, z)
        if t is not None and t >= part.eave_y + 2:
            break
        x, z = x - vx, z - vz
        if not grid.in_bounds(x, 0, z):
            return False
    fx, fz = x, z
    yb = _roof_top_y(grid, fx, fz)
    if yb is None:
        return False
    for wall_h in (3, 2):
        y_w0, y_w1 = yb + 1, yb + wall_h
        y_r = y_w1 + 1
        if y_r + 1 <= ridge_y:
            break
    else:
        return False
    # Occlusion check across the dormer footprint.
    if is_reserved(grid, fx - 2 * ax - vx * 1, y_w0, fz - 2 * az - vz * 1, fx + 2 * ax + vx * 1, y_r + 1, fz + 2 * az + vz * 1):
        return False
    depth_cells = []
    x, z = fx, fz
    while grid.in_bounds(x, 0, z) and part.roof_mask[x, z]:
        t = _roof_top_y(grid, x, z)
        if t is None or t >= y_r:
            break
        depth_cells.append((x, z))
        x, z = x - vx, z - vz
        if len(depth_cells) > 12:
            break
    if len(depth_cells) < 2:
        return False

    def put(px, pz, py, role, shape=BShape.FULL, normal=Dir.NONE, flags=0):
        if grid.in_bounds(px, py, pz) and int(grid.role[px, py, pz]) in OVERWRITABLE:
            grid.set(px, py, pz, role, shape, normal, part.index, 1.0, flags)

    # Front wall with window.
    for d in (-1, 0, 1):
        wx, wz = fx + d * ax, fz + d * az
        for y in range(y_w0, y_w1 + 1):
            put(wx, wz, y, Role.WALL, BShape.FULL, side, Flag.PERIMETER)
        if d == 0:
            for y in range(y_w0 + 1, y_w1 + 1):
                put(wx, wz, y, Role.WINDOW, BShape.PANE, side, Flag.NO_TEXTURE | Flag.PERIMETER)
            sx, sz = wx + vx, wz + vz
            if grid.in_bounds(sx, y_w0, sz) and grid.role[sx, y_w0, sz] == Role.EMPTY:
                grid.set(sx, y_w0, sz, Role.SILL, BShape.STAIR_UPSIDE, OPPOSITE[side], part.index, 0.0)
    put(fx, fz, y_r, Role.WALL, BShape.FULL, side, Flag.PERIMETER)
    # Side walls and interior along the depth.
    for i, (dx, dz) in enumerate(depth_cells):
        if i == 0:
            continue
        t = _roof_top_y(grid, dx, dz)
        for d in (-1, 1):
            wx, wz = dx + d * ax, dz + d * az
            tw = _roof_top_y(grid, wx, wz)
            base = (tw if tw is not None else t) + 1
            for y in range(base, y_w1 + 1):
                nrm = (Dir.EAST if d > 0 else Dir.WEST) if ax else (Dir.SOUTH if d > 0 else Dir.NORTH)
                put(wx, wz, y, Role.WALL, BShape.FULL, nrm, Flag.PERIMETER)
        for y in range((t or yb) + 1, y_w1 + 1):
            if grid.in_bounds(dx, y, dz) and grid.role[dx, y, dz] == Role.EMPTY:
                grid.set(dx, y, dz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 1.0)
    # Roof: a three-wide gable running up the slope, with a one block overhang at the front.
    left = (Dir.EAST if ax else Dir.SOUTH)   # uphill direction for the -1 side stair
    right = (Dir.WEST if ax else Dir.NORTH)
    roof_cells = [(fx + vx, fz + vz)] + depth_cells
    for i, (dx, dz) in enumerate(roof_cells):
        for d, facing in ((-1, left), (1, right)):
            wx, wz = dx + d * ax, dz + d * az
            tw = _roof_top_y(grid, wx, wz)
            if tw is not None and tw >= y_r and i > 0:
                continue
            put(wx, wz, y_r, Role.ROOF, BShape.STAIR, facing, Flag.OVERHANG if i == 0 else 0)
        t = _roof_top_y(grid, dx, dz)
        if i == 0 or t is None or t < y_r:
            put(dx, dz, y_r, Role.ROOF_FILL if i > 0 else Role.ROOF, BShape.FULL, Dir.NONE)
        if i == 0 or t is None or t < y_r + 1:
            put(dx, dz, y_r + 1, Role.ROOF, BShape.FULL, Dir.NONE)
    reserve(grid, min(fx - 2 * ax, fx + 2 * ax), y_w0 - 1, min(fz - 2 * az, fz + 2 * az),
            max(fx - 2 * ax, fx + 2 * ax), y_r + 1, max(fz - 2 * az, fz + 2 * az))
    return True


@register(AttachmentKind.dormer, "roof")
def dormer(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    if part.roof_surface is None or part.roof_mask is None:
        return 0
    sides = _slope_sides(part)
    if not sides:
        return 0
    placed = 0
    total_wanted = req.count if req.count is not None else None
    for side in sides:
        cells = _face_cells(part, side, len(part.floor_masks) - 1)
        if len(cells) < 7:
            continue
        n_side = total_wanted if total_wanted is not None else max(1, min(3, (len(cells) - 3) // 6))
        if total_wanted is not None:
            n_side = max(0, total_wanted - placed)
            if n_side == 0:
                break
            n_side = min(n_side, max(1, (len(cells) - 3) // 6))
        idxs = spread(n_side, 2, len(cells) - 3, grid.rng, req.spacing == "regular", 5)
        for i in idxs:
            for shift in (0, -1, 1, -2, 2, -3, 3):
                j = i + shift
                if not (2 <= j <= len(cells) - 3):
                    continue
                cx, cz = cells[j]
                if _place_dormer(grid, part, side, cx, cz):
                    placed += 1
                    break
        if total_wanted is None and placed >= 4:
            break
    return placed


# ---------------------------------------------------------------- balcony (roof level, before facade)

@register(AttachmentKind.balcony, "roof")
def balcony(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    if len(part.floor_heights) < 2:
        return 0
    count = req.count if req.count is not None else 1
    width = req.params.width if req.params.width else 3
    depth_out = req.params.depth if req.params.depth else 2
    placed = 0
    sides = [grid.front, Dir.EAST, Dir.WEST, OPPOSITE[grid.front]]
    for k in range(1, len(part.floor_heights)):
        for side in sides:
            if placed >= count:
                return placed
            cells = _face_cells(part, side, k)
            if len(cells) < width + 2:
                continue
            vx, vz, ax, az = _axis_dirs(side)
            mid = len(cells) // 2
            fy = part.floor_block_y(k)
            half = width // 2
            span = cells[mid - half:mid + half + 1]
            x0 = min(c[0] for c in span) + min(0, vx * depth_out)
            x1 = max(c[0] for c in span) + max(0, vx * depth_out)
            z0 = min(c[1] for c in span) + min(0, vz * depth_out)
            z1 = max(c[1] for c in span) + max(0, vz * depth_out)
            if is_reserved(grid, x0, fy - 1, z0, x1, fy + 3, z1):
                continue
            ok = True
            for (cx, cz) in span:
                for d in range(1, depth_out + 1):
                    px, pz = cx + vx * d, cz + vz * d
                    if not grid.in_bounds(px, fy, pz) or grid.role[px, fy, pz] not in (Role.EMPTY,):
                        ok = False
            if not ok:
                continue
            for (cx, cz) in span:
                for d in range(1, depth_out + 1):
                    px, pz = cx + vx * d, cz + vz * d
                    grid.set(px, fy, pz, Role.TRIM, BShape.SLAB_TOP, Dir.NONE, part.index, 0.0)
                    edge = d == depth_out or (cx, cz) in (span[0], span[-1])
                    if edge and grid.role[px, fy + 1, pz] == Role.EMPTY:
                        grid.set(px, fy + 1, pz, Role.RAILING, BShape.FENCE, Dir.NONE, part.index, 0.0)
                    if d == 1 and program.depth.corbels:
                        corbel(grid, px, fy - 1, pz, OPPOSITE[side], part.index)
            # Door in the middle of the span.
            dx, dz = span[len(span) // 2]
            inward = OPPOSITE[side]
            grid.set(dx, fy + 1, dz, Role.DOOR, BShape.DOOR_LOWER, inward, part.index, 0.0, Flag.NO_TEXTURE)
            grid.set(dx, fy + 2, dz, Role.DOOR, BShape.DOOR_UPPER, inward, part.index, 0.0, Flag.NO_TEXTURE)
            reserve(grid, min(c[0] for c in span), fy + 1, min(c[1] for c in span),
                    max(c[0] for c in span), fy + 3, max(c[1] for c in span))
            reserve(grid, x0, fy, z0, x1, fy + 2, z1)
            placed += 1
    return placed


# ---------------------------------------------------------------- porch (after facade, needs the door)

@register(AttachmentKind.porch, "facade")
def porch(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    if grid.door is None:
        return 0
    dx, dy, dz = grid.door
    side = grid.front
    vx, vz, ax, az = _axis_dirs(side)
    part = grid.parts[int(grid.part_id[dx, dy, dz])] if grid.part_id[dx, dy, dz] >= 0 else grid.parts[0]
    width = req.params.width if req.params.width else 5
    half = width // 2
    depth_out = req.params.depth if req.params.depth else 2
    y_roof = dy + 3
    # Everything must be free.
    for d in range(1, depth_out + 1):
        for o in range(-half, half + 1):
            px, pz = dx + vx * d + ax * o, dz + vz * d + az * o
            for y in range(dy, y_roof + 2):
                if not grid.in_bounds(px, y, pz) or grid.role[px, y, pz] not in (Role.EMPTY, Role.LIGHT, Role.SILL):
                    return 0
    # Re-hang the door lantern from the porch roof instead of the wall.
    for o in (-1, 1):
        lx, lz = dx + vx + ax * o, dz + vz + az * o
        if grid.in_bounds(lx, dy + 1, lz) and grid.role[lx, dy + 1, lz] == Role.LIGHT:
            grid.set(lx, dy + 1, lz, Role.EMPTY)
            grid.set(lx, y_roof + (depth_out - 1) - 1, lz, Role.LIGHT, BShape.LANTERN, Dir.UP, part.index, 0.0)
    for d in range(1, depth_out + 1):
        for o in range(-half, half + 1):
            px, pz = dx + vx * d + ax * o, dz + vz * d + az * o
            # Roof rises one block per row toward the wall.
            yr = y_roof + (depth_out - d)
            if d == depth_out:
                grid.set(px, yr, pz, Role.ROOF, BShape.STAIR, OPPOSITE[side], part.index, 1.0, Flag.OVERHANG)
            else:
                grid.set(px, yr, pz, Role.ROOF, BShape.STAIR, OPPOSITE[side], part.index, 1.0, Flag.OVERHANG)
                grid.set(px, yr - 1, pz, Role.ROOF_FILL, BShape.FULL, Dir.NONE, part.index, 1.0, Flag.OVERHANG)
            # Deck at ground level where nothing is there.
            if dy - 1 >= 0 and grid.role[px, dy - 1, pz] == Role.EMPTY:
                grid.set(px, dy - 1, pz, Role.STEP, BShape.FULL, Dir.NONE, part.index, 0.0)
    # Posts at the outer corners.
    for o in (-half, half):
        px, pz = dx + vx * depth_out + ax * o, dz + vz * depth_out + az * o
        for y in range(dy, y_roof):
            grid.set(px, y, pz, Role.PILLAR, BShape.FENCE, Dir.NONE, part.index, 0.0)
    reserve(grid, dx - half * ax + min(0, vx * depth_out), dy - 1, dz - half * az + min(0, vz * depth_out),
            dx + half * ax + max(0, vx * depth_out), y_roof + depth_out, dz + half * az + max(0, vz * depth_out))
    return 1


# ---------------------------------------------------------------- buttress (after facade)

@register(AttachmentKind.buttress, "facade")
def buttress(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    if part.spec.shape not in (Shape.rect, Shape.cross, Shape.ring):
        return 0
    height = req.params.height if req.params.height else min(part.wall_height - 1, max(3, part.wall_height * 2 // 3))
    placed = 0
    corners = [(part.x0, part.z0), (part.x1, part.z0), (part.x0, part.z1), (part.x1, part.z1)]
    per = perimeter(part.mask)
    for (cx, cz) in corners:
        if not per[cx, cz]:
            continue
        for side in HORIZONTAL:
            vx, _, vz = DIR_VEC[side]
            px, pz = cx + vx, cz + vz
            if not grid.in_bounds(px, 0, pz) or part.mask[px, pz]:
                continue
            # Only along faces (not diagonals): the neighbour must be outside the part.
            if is_reserved(grid, px, part.base_y, pz, px, part.base_y + height, pz):
                continue
            free = all(grid.role[px, y, pz] == Role.EMPTY for y in range(part.base_y, part.base_y + height + 1)
                       if grid.in_bounds(px, y, pz))
            if not free:
                continue
            for y in range(part.base_y, part.base_y + height):
                grid.set(px, y, pz, Role.STEP, BShape.FULL, side, part.index, 0.0)
            grid.set(px, part.base_y + height, pz, Role.STEP, BShape.STAIR, OPPOSITE[side], part.index, 0.0)
            reserve(grid, px, part.base_y, pz, px, part.base_y + height, pz)
            placed += 1
    return placed


# ---------------------------------------------------------------- cupola (roof level)

@register(AttachmentKind.cupola, "roof")
def cupola(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    if part.roof_surface is None or part.spec.roof.type in (RoofType.cone, RoofType.spire, RoofType.dome, RoofType.none):
        return 0
    ridge = [a for a in grid.anchors if a.part == part.index and a.type.value == "ridge"]
    if not ridge:
        return 0
    ridge.sort(key=lambda a: (a.x, a.z))
    a = ridge[len(ridge) // 2]
    cx, cz, y0 = a.x, a.z, a.y + 1
    if y0 + 5 >= grid.H or is_reserved(grid, cx - 2, y0, cz - 2, cx + 2, y0 + 4, cz + 2):
        return 0
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            x, z = cx + dx, cz + dz
            if not grid.in_bounds(x, y0 + 4, z):
                return 0
            edge = abs(dx) == 1 or abs(dz) == 1
            for y in (y0, y0 + 1):
                if edge:
                    if (dx == 0 or dz == 0) and y == y0 + 1:
                        n = Dir.EAST if dx > 0 else Dir.WEST if dx < 0 else Dir.SOUTH if dz > 0 else Dir.NORTH
                        grid.set(x, y, z, Role.WINDOW, BShape.PANE, n, part.index, 1.0, Flag.NO_TEXTURE)
                    else:
                        grid.set(x, y, z, Role.PILLAR, BShape.LOG, Dir.UP, part.index, 1.0)
                elif grid.role[x, y, z] == Role.EMPTY:
                    grid.set(x, y, z, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 1.0)
            # Little hip roof.
            if edge:
                facing = Dir.WEST if dx > 0 else Dir.EAST if dx < 0 else Dir.NORTH if dz > 0 else Dir.SOUTH
                grid.set(x, y0 + 2, z, Role.ROOF, BShape.STAIR, facing, part.index, 1.0)
            else:
                grid.set(x, y0 + 2, z, Role.ROOF_FILL, BShape.FULL, Dir.NONE, part.index, 1.0)
                grid.set(x, y0 + 3, z, Role.ROOF, BShape.FULL, Dir.NONE, part.index, 1.0)
                grid.set(x, y0 + 4, z, Role.LIGHT, BShape.LANTERN, Dir.DOWN, part.index, 1.0)
    reserve(grid, cx - 2, y0 - 1, cz - 2, cx + 2, y0 + 4, cz + 2)
    return 1


# ---------------------------------------------------------------- bartizan (roof level)

RECT_LIKE = {Shape.rect, Shape.cross, Shape.ring}


def _corners(part: LayoutPart, count: int) -> list[tuple[int, int, int, int]]:
    corners = [(part.x0, part.z0, -1, -1), (part.x1, part.z0, 1, -1), (part.x0, part.z1, -1, 1), (part.x1, part.z1, 1, 1)]
    if count < 4:
        corners = corners[2:] + corners[:2]     # front corners first
    return corners[:max(0, count)]


@register(AttachmentKind.bartizan, "roof")
def bartizan(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    if part.spec.shape not in RECT_LIKE or part.wall_height < 6:
        return 0
    count = req.count if req.count is not None else 4
    h = req.params.height if req.params.height else 4
    y1 = part.top_y
    y0 = max(part.base_y + 2, y1 - h + 1)
    placed = 0
    for (cx, cz, sx, sz) in _corners(part, count):
        mx, mz = cx + sx, cz + sz
        cells = [(mx + dx, mz + dz) for dx in (-1, 0, 1) for dz in (-1, 0, 1)]
        outside = [(x, z) for (x, z) in cells if grid.in_bounds(x, 0, z) and not grid.footprint[x, z]]
        if len(outside) < 8:
            continue
        if is_reserved(grid, mx - 1, y0 - 2, mz - 1, mx + 1, y1 + 3, mz + 1):
            continue
        if any(grid.role[x, y, z] != Role.EMPTY for (x, z) in outside for y in range(y0 - 2, min(grid.H, y1 + 4))):
            continue
        for (x, z) in outside:
            dx, dz = x - mx, z - mz
            if dx == 0 and dz == 0:
                for y in range(y0, y1 + 1):
                    grid.set(x, y, z, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 1.0)
                continue
            n = (Dir.EAST if dx > 0 else Dir.WEST) if abs(dx) >= abs(dz) and dx != 0 else (Dir.SOUTH if dz > 0 else Dir.NORTH)
            for y in range(y0, y1 + 1):
                grid.set(x, y, z, Role.WALL, BShape.FULL, n, part.index, 1.0, Flag.PERIMETER)
            # Corbel courses under the overhang, stepping in toward the corner.
            toward = (Dir.WEST if sx > 0 else Dir.EAST) if abs(dx) >= abs(dz) else (Dir.NORTH if sz > 0 else Dir.SOUTH)
            corbel(grid, x, y0 - 1, z, toward, part.index)
            if (dx == 0 or dz == 0) and (x, z) != (mx, mz) and abs(x - cx) + abs(z - cz) == 1:
                corbel(grid, x, y0 - 2, z, toward, part.index)
            # Roof: stairs leaning to the centre.
            facing = (Dir.WEST if dx > 0 else Dir.EAST) if dx != 0 and dz == 0 else \
                     (Dir.NORTH if dz > 0 else Dir.SOUTH) if dz != 0 and dx == 0 else \
                     (Dir.WEST if dx > 0 else Dir.EAST)
            grid.set(x, y1 + 1, z, Role.ROOF, BShape.STAIR, facing, part.index, 1.0)
        # Arrow slits on the two outward faces.
        for (x, z, n) in ((mx + sx, mz, Dir.EAST if sx > 0 else Dir.WEST), (mx, mz + sz, Dir.SOUTH if sz > 0 else Dir.NORTH)):
            if grid.in_bounds(x, y0 + 1, z) and not grid.footprint[x, z]:
                for y in (y0 + 1, y0 + 2):
                    if y <= y1:
                        grid.set(x, y, z, Role.WINDOW, BShape.NONE, n, part.index, 1.0, Flag.NO_TEXTURE)
        if grid.in_bounds(mx, y1 + 2, mz) and not grid.footprint[mx, mz]:
            grid.set(mx, y1 + 1, mz, Role.ROOF_FILL, BShape.FULL, Dir.NONE, part.index, 1.0)
            grid.set(mx, y1 + 2, mz, Role.ROOF, BShape.FULL, Dir.NONE, part.index, 1.0)
            grid.set(mx, y1 + 3, mz, Role.ROOF, BShape.SLAB_BOTTOM, Dir.NONE, part.index, 1.0)
        reserve(grid, mx - 1, y0 - 2, mz - 1, mx + 1, y1 + 3, mz + 1)
        placed += 1
    return placed


# ---------------------------------------------------------------- gallery (roof level)

@register(AttachmentKind.gallery, "roof")
def gallery(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    if not part.floor_masks:
        return 0
    k = len(part.floor_masks) - 1
    m = part.floor_masks[k]
    fy = part.floor_block_y(k)
    if fy < part.base_y:
        # Single storey: hang the gallery just under the eave instead.
        fy = part.top_y - 1
    others = grid.footprint & ~part.mask
    ring = dilate(m, 1) & ~m & ~others
    cells = [(int(x), int(z)) for x, z in zip(*np.nonzero(ring))]
    if len(cells) < 6:
        return 0
    # The ring may sit on a taper ledge (solid wall of this part) or in the air.
    for (x, z) in cells:
        if not grid.in_bounds(x, fy + 1, z) or grid.role[x, fy + 1, z] != Role.EMPTY:
            return 0
        if grid.role[x, fy, z] not in (Role.EMPTY,) and not (grid.role[x, fy, z] == Role.WALL and grid.part_id[x, fy, z] == part.index):
            return 0
    for (x, z) in cells:
        if grid.role[x, fy, z] == Role.EMPTY:
            grid.set(x, fy, z, Role.TRIM, BShape.SLAB_TOP, Dir.NONE, part.index, 1.0)
        grid.set(x, fy + 1, z, Role.RAILING, BShape.FENCE, Dir.NONE, part.index, 1.0)
        # Corbel facing the wall it hangs from.
        toward = Dir.NONE
        for d in HORIZONTAL:
            vx, _, vz = DIR_VEC[d]
            if grid.in_bounds(x + vx, fy, z + vz) and m[x + vx, z + vz]:
                toward = d
                break
        if toward != Dir.NONE and program.depth.corbels:
            corbel(grid, x, fy - 1, z, toward, part.index)
    glazed = req.params.glazed if req.params.glazed is not None else True
    if glazed:
        ceiling = min(fy + part.floor_heights[k], part.top_y + 1) if fy >= part.base_y else part.top_y + 1
        for x, z in zip(*np.nonzero(perimeter(m))):
            for y in range(fy + 1, ceiling):
                if grid.role[x, y, z] == Role.WALL:
                    n = int(grid.normal[x, y, z])
                    grid.set(x, y, z, Role.WINDOW, BShape.FULL, n, part.index, grid.h_norm[x, y, z],
                             Flag.NO_TEXTURE | Flag.PERIMETER)
        # A door onto the gallery on the front.
        for x, z in zip(*np.nonzero(perimeter(m))):
            if int(grid.normal[x, fy + 1, z]) == grid.front and grid.role[x, fy + 1, z] == Role.WINDOW and abs(x - part.cx) < 1:
                inward = OPPOSITE[grid.front]
                grid.set(x, fy + 1, z, Role.DOOR, BShape.DOOR_LOWER, inward, part.index, 0.0, Flag.NO_TEXTURE)
                grid.set(x, fy + 2, z, Role.DOOR, BShape.DOOR_UPPER, inward, part.index, 0.0, Flag.NO_TEXTURE)
                break
    xs = [c[0] for c in cells]
    zs = [c[1] for c in cells]
    reserve(grid, min(xs), fy - 1, min(zs), max(xs), fy + 1, max(zs))
    return 1


# ---------------------------------------------------------------- spire and flagpole (roof level)

def _ridge_centre(grid: SemanticGrid, part: LayoutPart) -> tuple[int, int, int] | None:
    ridge = [a for a in grid.anchors if a.part == part.index and a.type.value == "ridge"]
    if ridge:
        ridge.sort(key=lambda a: (a.x, a.z))
        a = ridge[len(ridge) // 2]
        return a.x, a.y + 1, a.z
    if part.roof_surface is not None and part.roof_mask is not None and part.roof_mask.any():
        x, z = int(round(part.cx)), int(round(part.cz))
        t = _roof_top_y(grid, x, z)
        if t is not None:
            return x, t + 1, z
    return None


@register(AttachmentKind.spire, "roof")
def spire(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    c = _ridge_centre(grid, part)
    if c is None:
        return 0
    cx, y0, cz = c
    h = req.params.height if req.params.height else 3
    h = max(1, min(h, grid.H - y0 - 4))      # shrink to fit the bounds
    if y0 + h + 3 >= grid.H or is_reserved(grid, cx - 1, y0, cz - 1, cx + 1, y0 + h + 3, cz + 1):
        return 0
    if any(grid.role[cx + dx, y0, cz + dz] != Role.EMPTY for dx in (-1, 0, 1) for dz in (-1, 0, 1)
           if grid.in_bounds(cx + dx, y0, cz + dz)):
        return 0
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            x, z = cx + dx, cz + dz
            if not grid.in_bounds(x, y0, z):
                continue
            if dx == 0 and dz == 0:
                grid.set(x, y0, z, Role.ROOF_FILL, BShape.FULL, Dir.NONE, part.index, 1.0)
            else:
                facing = (Dir.WEST if dx > 0 else Dir.EAST) if dx != 0 else (Dir.NORTH if dz > 0 else Dir.SOUTH)
                grid.set(x, y0, z, Role.ROOF, BShape.STAIR, facing, part.index, 1.0)
    grid.set(cx, y0 + 1, cz, Role.ROOF, BShape.FULL, Dir.NONE, part.index, 1.0)
    for y in range(y0 + 2, y0 + 2 + h):
        grid.set(cx, y, cz, Role.PILLAR, BShape.FENCE, Dir.NONE, part.index, 1.0)
    grid.set(cx, y0 + 2 + h, cz, Role.LIGHT, BShape.LANTERN, Dir.DOWN, part.index, 1.0)
    reserve(grid, cx - 1, y0, cz - 1, cx + 1, y0 + h + 2, cz + 1)
    return 1


@register(AttachmentKind.flagpole, "roof")
def flagpole(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    h = req.params.height if req.params.height else 5
    spots: list[tuple[int, int, int]] = []
    if part.spec.roof.type in (RoofType.flat, RoofType.parapet):
        m = part.floor_masks[-1]
        xs, zs = np.nonzero(m)
        for (x, z) in ((int(xs.max()) - 1, int(zs.max()) - 1), (int(xs.min()) + 1, int(zs.max()) - 1),
                       (int(round(part.cx)), int(round(part.cz)))):
            t = _roof_top_y(grid, x, z)
            if t is not None:
                spots.append((x, t + 1, z))
    else:
        c = _ridge_centre(grid, part)
        if c is not None:
            ridge = sorted((a for a in grid.anchors if a.part == part.index and a.type.value == "ridge"),
                           key=lambda a: (a.x, a.z))
            if ridge:
                a = ridge[-1]
                spots.append((a.x, a.y + 1, a.z))
            spots.append(c)
    count = req.count if req.count is not None else 1
    placed = 0
    for (x, y0, z) in spots:
        if placed >= count:
            break
        h = max(2, min(h, grid.H - y0 - 2))
        if y0 + h + 1 >= grid.H or is_reserved(grid, x - 1, y0, z - 1, x + 1, y0 + h + 1, z + 1):
            continue
        if any(grid.role[x, y, z] != Role.EMPTY for y in range(y0, y0 + h + 1)):
            continue
        for y in range(y0, y0 + h):
            grid.set(x, y, z, Role.PILLAR, BShape.FENCE, Dir.NONE, part.index, 1.0)
        # Banner hangs off the pole on the side away from the building centre.
        side = Dir.EAST if x >= part.cx else Dir.WEST
        vx, _, vz = DIR_VEC[side]
        bx, bz = x + vx, z + vz
        if grid.in_bounds(bx, y0 + h - 1, bz) and grid.role[bx, y0 + h - 1, bz] == Role.EMPTY:
            grid.set(bx, y0 + h - 1, bz, Role.ACCENT, BShape.BANNER, side, part.index, 1.0)
        reserve(grid, x - 1, y0, z - 1, x + 1, y0 + h, z + 1)
        placed += 1
    return placed


# ---------------------------------------------------------------- awning, colonnade, arch (after facade)

def _ground_windows(grid: SemanticGrid, part: LayoutPart, side: int) -> dict[tuple[int, int], list[int]]:
    """Ground floor window columns on a face: (x, z) -> rows."""
    fy = part.floor_block_y(0)
    y0, y1 = fy + 1, min(fy + part.floor_heights[0], part.top_y + 1)
    cols: dict[tuple[int, int], list[int]] = {}
    for x, y, z in zip(*np.nonzero(grid.role == Role.WINDOW)):
        if y0 <= y < y1 and int(grid.part_id[x, y, z]) == part.index and int(grid.normal[x, y, z]) == side \
                and grid.flags[x, y, z] & Flag.PERIMETER:
            cols.setdefault((int(x), int(z)), []).append(int(y))
    return cols


@register(AttachmentKind.awning, "facade")
def awning(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    side = grid.front
    vx, vz, ax, az = _axis_dirs(side)
    placed = 0
    for (x, z), rows in _ground_windows(grid, part, side).items():
        top = max(rows)
        # Windows may have been inset by one; the wall plane is one block outward then.
        wx, wz = x, z
        if grid.in_bounds(x + vx, top, z + vz) and grid.role[x + vx, top, z + vz] in (Role.INTERIOR, Role.EMPTY) \
                and grid.footprint[x + vx, z + vz]:
            wx, wz = x + vx, z + vz
        ok = True
        for o in (-1, 0, 1):
            px, pz = wx + vx + ax * o, wz + vz + az * o
            if not grid.in_bounds(px, top + 1, pz) or grid.role[px, top + 1, pz] != Role.EMPTY:
                ok = False
        if not ok:
            continue
        for o in (-1, 0, 1):
            px, pz = wx + vx + ax * o, wz + vz + az * o
            grid.set(px, top + 1, pz, Role.ACCENT, BShape.STAIR, OPPOSITE[side], part.index, 0.0)
        placed += 1
    return placed


@register(AttachmentKind.colonnade, "facade")
def colonnade(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    side = grid.front
    cells = _face_cells(part, side, 0)
    if len(cells) < 7:
        return 0
    vx, vz, ax, az = _axis_dirs(side)
    d = req.params.depth if req.params.depth else 2
    fy0 = part.floor_block_y(0)
    y_r = part.floor_block_y(1) if len(part.floor_heights) > 1 else part.top_y
    run = cells[1:-1]
    for (cx, cz) in run:
        for k in range(1, d + 1):
            px, pz = cx + vx * k, cz + vz * k
            for y in range(fy0 + 1, y_r + 1):
                if not grid.in_bounds(px, y, pz) or grid.role[px, y, pz] not in (Role.EMPTY, Role.LIGHT, Role.SILL, Role.SHUTTER):
                    return 0
    for i, (cx, cz) in enumerate(run):
        for k in range(1, d + 1):
            px, pz = cx + vx * k, cz + vz * k
            for y in range(fy0 + 1, y_r):
                if grid.role[px, y, pz] != Role.EMPTY:
                    grid.set(px, y, pz, Role.EMPTY)
            grid.set(px, y_r, pz, Role.TRIM, BShape.FULL, Dir.NONE, part.index, 0.0)
            if k == d and len(part.floor_heights) > 1 and grid.in_bounds(px, y_r + 1, pz) \
                    and grid.role[px, y_r + 1, pz] == Role.EMPTY:
                grid.set(px, y_r + 1, pz, Role.RAILING, BShape.FENCE, Dir.NONE, part.index, 0.0)
            if fy0 >= 0 and grid.role[px, fy0, pz] == Role.EMPTY:
                grid.set(px, fy0, pz, Role.STEP, BShape.FULL, Dir.NONE, part.index, 0.0)
        if i % 3 == 0 or i == len(run) - 1:
            px, pz = cx + vx * d, cz + vz * d
            for y in range(fy0 + 1, y_r):
                grid.set(px, y, pz, Role.PILLAR, BShape.PILLAR, Dir.UP, part.index, 0.0)
    xs = [c[0] + vx * k for c in run for k in range(1, d + 1)]
    zs = [c[1] + vz * k for c in run for k in range(1, d + 1)]
    reserve(grid, min(xs), fy0, min(zs), max(xs), y_r + 1, max(zs))
    return 1


@register(AttachmentKind.arch, "facade")
def arch(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    """Ground floor windows on the front become arched openings (arcade)."""
    part = target_part(grid, program, req)
    side = grid.front
    vx, vz, ax, az = _axis_dirs(side)
    fy = part.floor_block_y(0)
    y_base = fy + 1
    placed = 0
    for (x, z), rows in _ground_windows(grid, part, side).items():
        top = max(rows)
        # Locate the wall plane (windows may be inset).
        wx, wz = x, z
        if grid.footprint[x + vx, z + vz] and grid.role[x + vx, y_base, z + vz] in (Role.WALL, Role.INTERIOR):
            wx, wz = x + vx, z + vz
        if grid.door is not None and (wx, wz) == (grid.door[0], grid.door[2]):
            continue
        for y in range(y_base, top + 1):
            for (px, pz) in ((x, z), (wx, wz)):
                if grid.role[px, y, pz] in (Role.WALL, Role.WINDOW, Role.INTERIOR):
                    grid.set(px, y, pz, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, 0.0)
        for o in (-1, 1):
            px, pz = wx + ax * o, wz + az * o
            if grid.in_bounds(px, top, pz) and grid.role[px, top, pz] == Role.WALL:
                away = (Dir.EAST if o > 0 else Dir.WEST) if ax else (Dir.SOUTH if o > 0 else Dir.NORTH)
                grid.set(px, top, pz, Role.TRIM, BShape.STAIR_UPSIDE, away, part.index, 0.0, Flag.PERIMETER)
        # Remove sill and shutters in front of the opening.
        for o in (-1, 0, 1):
            px, pz = wx + vx + ax * o, wz + vz + az * o
            for y in range(y_base - 1, top + 1):
                if grid.in_bounds(px, y, pz) and grid.role[px, y, pz] in (Role.SILL, Role.SHUTTER):
                    grid.set(px, y, pz, Role.EMPTY)
        placed += 1
    return placed


# ---------------------------------------------------------------- jetty and cantilever (mass level)

@register(AttachmentKind.jetty, "mass")
def jetty(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    part = target_part(grid, program, req)
    if part.spec.floors < 2 or part.spec.shape not in RECT_LIKE:
        return 0
    part.jetty = max(1, min(2, req.params.depth or 1))
    return 1


@register(AttachmentKind.cantilever, "mass")
def cantilever(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest) -> int:
    """Slide a stacked part toward the front so it overhangs its parent."""
    part = target_part(grid, program, req)
    if part.spec.attach is None or part.spec.attach.side != Side.top or part.parent is None:
        # Fall back to the first stacked part.
        stacked = [p for p in grid.parts if p.spec.attach is not None and p.spec.attach.side == Side.top]
        if not stacked:
            return 0
        part = stacked[0]
    d = max(1, min(6, req.params.depth or 3))
    parent = grid.parts[part.parent]
    current = part.z1 - parent.z1
    shift = max(0, d - current)
    if shift == 0 or part.z1 + shift >= grid.D - 1:
        return 0
    part.mask = np.roll(part.mask, shift, axis=1)
    part.mask[:, :shift] = False
    part.z0 += shift
    part.z1 += shift
    grid.footprint |= part.mask
    return 1
