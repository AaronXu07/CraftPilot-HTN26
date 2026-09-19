"""Massing: extrude footprints into walls, floors, and foundation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from craftpilot.blocks.circles import circle_mask, dilate, perimeter
from craftpilot.engine.roof import roof_height_estimate
from craftpilot.grid.enums import HORIZONTAL, OPPOSITE, BShape, Dir, Flag, Role
from craftpilot.grid.semantic import Anchor, LayoutPart, SemanticGrid
from craftpilot.program.model import AnchorType, BuildProgram, RoofType, Shape, Side


def _outward_normal(mask: np.ndarray, x: int, z: int, cx: float, cz: float) -> int:
    W, D = mask.shape
    dirs = []
    if z - 1 < 0 or not mask[x, z - 1]:
        dirs.append(Dir.NORTH)
    if z + 1 >= D or not mask[x, z + 1]:
        dirs.append(Dir.SOUTH)
    if x + 1 >= W or not mask[x + 1, z]:
        dirs.append(Dir.EAST)
    if x - 1 < 0 or not mask[x - 1, z]:
        dirs.append(Dir.WEST)
    if len(dirs) == 1:
        return dirs[0]
    if not dirs:
        return Dir.NONE
    # Corner or curved wall: pick the direction most aligned with the vector from the centre.
    dx, dz = x - cx, z - cz
    if abs(dx) >= abs(dz):
        pref = Dir.EAST if dx > 0 else Dir.WEST
    else:
        pref = Dir.SOUTH if dz > 0 else Dir.NORTH
    return pref if pref in dirs else dirs[0]


def _shrunk(part: LayoutPart, n: int) -> np.ndarray:
    if n <= 0:
        return part.mask
    if part.spec.shape == Shape.circle:
        r = (min(part.width, part.depth) - 1) / 2 - n
        if r < 1:
            return np.zeros_like(part.mask)
        return circle_mask(part.mask.shape[0], part.mask.shape[1], part.cx, part.cz, r) & part.mask
    m = ndimage.binary_erosion(part.mask, iterations=n)
    return m if m.any() else part.mask


def massing(grid: SemanticGrid, program: BuildProgram) -> None:
    rise = max(0, program.depth.foundation_rise)
    outset = max(0, program.depth.foundation_outset)
    has_attachments = bool(program.attachments)
    tall_reserve = {"chimney": 3, "cupola": 5, "spire": 5, "flagpole": 6}
    tall = max((tall_reserve.get(a.kind.value, 0) for a in program.attachments), default=0)
    root_name = program.root().name

    for part in grid.parts:
        spec = part.spec
        fh = max(3, spec.floor_height)
        floors = max(1, spec.floors)
        extra = program.facade.ground_floor_taller if spec.name == root_name else 0
        fh_list = [fh + max(0, extra)] + [fh] * (floors - 1)
        est_roof = roof_height_estimate(spec.roof, part.width, part.depth)
        reserve = max(tall, 2 if has_attachments else 0)
        base_y = rise
        stacked = spec.attach is not None and spec.attach.side == Side.top and part.parent is not None
        if stacked:
            base_y = grid.parts[part.parent].eave_y + 1
        # Height budget: flatten a steep roof first, then drop floors, then shorten storeys.
        while base_y + sum(fh_list) + est_roof + reserve > grid.H and spec.roof.pitch > 1.0:
            spec.roof.pitch = max(1.0, spec.roof.pitch - 0.5)
            if spec.roof.type == RoofType.spire and spec.roof.pitch < 2.0:
                spec.roof.type = RoofType.cone
            est_roof = roof_height_estimate(spec.roof, part.width, part.depth)
            grid.note(f"Part '{spec.name}': roof pitch lowered to {spec.roof.pitch} to fit the height.")
        while base_y + sum(fh_list) + est_roof + reserve > grid.H and len(fh_list) > 1:
            fh_list.pop()
        if base_y + sum(fh_list) + est_roof + reserve > grid.H and fh > 3:
            fh_list = [max(3, h - 1) for h in fh_list]
        if len(fh_list) < floors:
            grid.note(f"Part '{spec.name}' reduced to {len(fh_list)} floors to fit the height.")
        part.floor_heights = fh_list
        part.base_y = base_y
        part.top_y = min(grid.H - 1, base_y + sum(fh_list) - 1)
        wall_h = part.top_y - part.base_y + 1

        # Floor masks with taper, or a jetty (upper floors step out).
        part.floor_masks = []
        for k in range(len(fh_list)):
            n = int(round(part.width * max(0.0, spec.taper) * k / 2))
            m = _shrunk(part, n)
            if part.jetty > 0 and k > 0:
                m = dilate(part.mask, part.jetty)
            part.floor_masks.append(m)
        thick = max(1, min(2, spec.wall_thickness))

        y = part.base_y
        for k, h in enumerate(fh_list):
            m = part.floor_masks[k]
            per = perimeter(m)
            if thick > 1:
                inner = ndimage.binary_erosion(m, iterations=thick)
                per = m & ~inner if inner.any() else per
            xs, zs = np.nonzero(m)
            for yy in range(y, min(y + h, part.top_y + 1)):
                hn = (yy - part.base_y) / max(1, wall_h - 1)
                for x, z in zip(xs, zs):
                    if per[x, z]:
                        n = _outward_normal(m, x, z, part.cx, part.cz)
                        # Do not overwrite a wall from an earlier part with interior air.
                        grid.set(x, yy, z, Role.WALL, BShape.FULL, n, part.index, hn, Flag.PERIMETER)
                    elif grid.role[x, yy, z] in (Role.EMPTY, Role.INTERIOR) or \
                            (part.is_attachment and grid.role[x, yy, z] == Role.WALL):
                        grid.set(x, yy, z, Role.INTERIOR, BShape.FULL, Dir.NONE, part.index, hn)
            # Floor block row for this storey (k = 0 is the top of the foundation).
            fy = part.floor_block_y(k)
            if fy >= 0:
                inner = m & ~per if k > 0 else m
                if k > 0:
                    # Ledge left by taper stays solid wall.
                    ledge = part.floor_masks[k - 1] & ~m
                    for x, z in zip(*np.nonzero(ledge)):
                        grid.set(x, fy, z, Role.WALL, BShape.FULL, Dir.UP, part.index, 1.0, Flag.PERIMETER)
                    # Jetty overhang: the outer ring of the floor row is wall, with corbels under it.
                    over = m & ~part.floor_masks[k - 1]
                    for x, z in zip(*np.nonzero(over)):
                        n = _outward_normal(m, x, z, part.cx, part.cz)
                        grid.set(x, fy, z, Role.WALL, BShape.FULL, n, part.index, 0.0, Flag.PERIMETER)
                        if program.depth.corbels and fy - 1 >= 0 and grid.role[x, fy - 1, z] == Role.EMPTY:
                            grid.set(x, fy - 1, z, Role.CORBEL, BShape.STAIR_UPSIDE, OPPOSITE[n] if n in HORIZONTAL else Dir.NORTH,
                                     part.index, 0.0, Flag.CORBEL)
                for x, z in zip(*np.nonzero(inner)):
                    if grid.role[x, fy, z] not in (Role.WALL, Role.FRAME):
                        grid.set(x, fy, z, Role.FLOOR, BShape.FULL, Dir.UP, part.index, 0.0)
            y += h

        # Foundation ring (ground parts only).
        if rise > 0 and not stacked:
            dil = dilate(part.mask, outset) if outset > 0 else part.mask
            per0 = perimeter(part.mask)
            ring = dil & ~(part.mask & ~per0)
            for yy in range(rise):
                for x, z in zip(*np.nonzero(ring)):
                    if grid.role[x, yy, z] in (Role.EMPTY, Role.INTERIOR, Role.FLOOR):
                        n = _outward_normal(dil, x, z, part.cx, part.cz)
                        grid.set(x, yy, z, Role.FOUNDATION, BShape.FULL, n, part.index, 0.0, Flag.PERIMETER)
                # Solid under the interior too, except the top row which is floor.
                if yy < rise - 1:
                    for x, z in zip(*np.nonzero(part.mask & ~per0)):
                        if grid.role[x, yy, z] == Role.EMPTY:
                            grid.set(x, yy, z, Role.FOUNDATION, BShape.FULL, Dir.NONE, part.index, 0.0)

        # Corner anchors for rectangular parts.
        if spec.shape in (Shape.rect, Shape.cross, Shape.ring):
            for (x, z, n) in ((part.x0, part.z0, Dir.NORTH), (part.x1, part.z0, Dir.NORTH),
                              (part.x0, part.z1, Dir.SOUTH), (part.x1, part.z1, Dir.SOUTH)):
                grid.anchors.append(Anchor(AnchorType.corner, x, part.base_y, z, n, part.index, wall_h))

    grid.stage_done("massing", parts=[(p.spec.name, p.base_y, p.top_y, p.floor_heights) for p in grid.parts])
