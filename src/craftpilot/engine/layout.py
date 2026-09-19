"""Layout: resolve the part tree into absolute footprints inside the bounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from craftpilot.blocks.circles import circle_mask, dilate, ngon_mask
from craftpilot.grid.semantic import LayoutPart, SemanticGrid
from craftpilot.program.model import Align, BuildProgram, PartSpec, Shape, Side


@dataclass
class _Placed:
    spec: PartSpec
    w: int
    d: int
    x0: int = 0
    z0: int = 0
    parent: int | None = None

    @property
    def x1(self) -> int:
        return self.x0 + self.w - 1

    @property
    def z1(self) -> int:
        return self.z0 + self.d - 1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cz(self) -> float:
        return (self.z0 + self.z1) / 2


def _snap(v: int, odd: bool, minimum: int = 5) -> int:
    v = max(minimum, int(v))
    if odd and v % 2 == 0:
        v -= 1
    return max(minimum, v)


def _order(program: BuildProgram) -> list[PartSpec]:
    """Root first, then breadth-first by attachment; drops parts whose parent is unknown."""
    by_name = {p.name: p for p in program.parts}
    root = program.root()
    ordered = [root]
    seen = {root.name}
    frontier = [root.name]
    while frontier:
        nxt = []
        for parent in frontier:
            for p in program.parts:
                if p.name in seen or p.attach is None:
                    continue
                if p.attach.to == parent:
                    ordered.append(p)
                    seen.add(p.name)
                    nxt.append(p.name)
        frontier = nxt
    # Orphans: attach to root.
    for p in program.parts:
        if p.name not in seen:
            ordered.append(p)
            seen.add(p.name)
    return ordered


def _place(program: BuildProgram, avail_w: float, avail_d: float, scale: float) -> list[_Placed]:
    ordered = _order(program)
    root = ordered[0]
    root_w = _snap(round(avail_w * scale * min(root.size.width, 1.5)), root.odd_dims)
    root_d = _snap(round(avail_d * scale * min(root.size.depth, 1.5)), root.odd_dims)
    if root.shape in (Shape.circle, Shape.ngon):
        root_w = root_d = min(root_w, root_d)
    placed: list[_Placed] = [_Placed(root, root_w, root_d)]
    index = {root.name: 0}
    for spec in ordered[1:]:
        w = _snap(round(root_w * spec.size.width), spec.odd_dims)
        d = _snap(round(root_d * spec.size.depth), spec.odd_dims)
        if spec.shape in (Shape.circle, Shape.ngon):
            w = d = min(w, d)
        att = spec.attach
        parent_idx = index.get(att.to, 0) if att else 0
        parent = placed[parent_idx]
        side = att.side if att else Side.south
        align = att.align if att else Align.center
        overlap = max(0, att.overlap) if att else 1
        offset = att.offset if att else 0
        p = _Placed(spec, w, d, parent=parent_idx)
        if side == Side.top:
            oz = att.offset_z if att else 0
            if align == Align.center:
                p.x0 = int(round(parent.cx + offset - (w - 1) / 2))
                p.z0 = int(round(parent.cz + oz - (d - 1) / 2))
            elif align == Align.start:
                p.x0, p.z0 = parent.x0 + offset, parent.z0 + oz
            else:
                p.x0, p.z0 = parent.x1 - w + 1 + offset, parent.z1 - d + 1 + oz
        elif side in (Side.south, Side.north):
            if side == Side.south:
                p.z0 = parent.z1 + 1 - overlap
            else:
                p.z0 = parent.z0 - 1 + overlap - d + 1
            if align == Align.center:
                p.x0 = int(round(parent.cx + offset - (w - 1) / 2))
            elif align == Align.start:
                p.x0 = parent.x0 + offset
            else:
                p.x0 = parent.x1 - offset - w + 1
        else:
            if side == Side.east:
                p.x0 = parent.x1 + 1 - overlap
            else:
                p.x0 = parent.x0 - 1 + overlap - w + 1
            if align == Align.center:
                p.z0 = int(round(parent.cz + offset - (d - 1) / 2))
            elif align == Align.start:
                p.z0 = parent.z0 + offset
            else:
                p.z0 = parent.z1 - offset - d + 1
        index[spec.name] = len(placed)
        placed.append(p)
    return placed


def _mask_for(p: _Placed, W: int, D: int) -> np.ndarray:
    m = np.zeros((W, D), dtype=bool)
    x0, z0 = max(0, p.x0), max(0, p.z0)
    x1, z1 = min(W - 1, p.x1), min(D - 1, p.z1)
    if x1 < x0 or z1 < z0:
        return m
    spec = p.spec
    if spec.shape == Shape.rect:
        m[x0:x1 + 1, z0:z1 + 1] = True
    elif spec.shape == Shape.circle:
        r = (min(p.w, p.d) - 1) / 2
        m = circle_mask(W, D, p.cx, p.cz, r)
    elif spec.shape == Shape.ngon:
        r = min(p.w, p.d) / 2 - 0.2
        m = ngon_mask(W, D, p.cx, p.cz, r, spec.sides)
    elif spec.shape == Shape.cross:
        aw = max(3, p.w // 2) | 1
        ad = max(3, p.d // 2) | 1
        ax0 = int(round(p.cx - (aw - 1) / 2))
        az0 = int(round(p.cz - (ad - 1) / 2))
        m[x0:x1 + 1, max(0, az0):min(D, az0 + ad)] = True
        m[max(0, ax0):min(W, ax0 + aw), z0:z1 + 1] = True
    elif spec.shape == Shape.ring:
        m[x0:x1 + 1, z0:z1 + 1] = True
        t = max(3, min(p.w, p.d) // 4)
        ix0, iz0, ix1, iz1 = x0 + t, z0 + t, x1 - t, z1 - t
        if ix1 - ix0 >= 2 and iz1 - iz0 >= 2:
            m[ix0:ix1 + 1, iz0:iz1 + 1] = False
    # Clip to the grid.
    clip = np.zeros((W, D), dtype=bool)
    clip[x0:x1 + 1, z0:z1 + 1] = True
    return m & clip


def layout(grid: SemanticGrid, program: BuildProgram) -> None:
    W, D = grid.W, grid.D
    outset = max(0, program.depth.foundation_outset)
    max_over = max((p.roof.overhang for p in program.parts), default=1)
    margin = max(outset, max_over, 0) + 1
    avail_w = max(3, W - 2 * margin)
    avail_d = max(3, D - 2 * margin)

    scale = 1.0
    placed = _place(program, avail_w, avail_d, scale)
    for _ in range(8):
        minx = min(p.x0 for p in placed)
        maxx = max(p.x1 for p in placed)
        minz = min(p.z0 for p in placed)
        maxz = max(p.z1 for p in placed)
        bw, bd = maxx - minx + 1, maxz - minz + 1
        if bw <= avail_w and bd <= avail_d:
            break
        scale *= min(avail_w / bw, avail_d / bd) * 0.97
        placed = _place(program, avail_w, avail_d, scale)
    if scale < 1.0:
        grid.note(f"Composition scaled to {scale:.2f} to fit the bounds.")

    minx = min(p.x0 for p in placed)
    maxx = max(p.x1 for p in placed)
    minz = min(p.z0 for p in placed)
    maxz = max(p.z1 for p in placed)
    bw, bd = maxx - minx + 1, maxz - minz + 1
    shift_x = margin - minx + (avail_w - bw) // 2
    shift_z = margin - minz + (avail_d - bd) // 2
    for p in placed:
        p.x0 += shift_x
        p.z0 += shift_z

    grid.parts = []
    for i, p in enumerate(placed):
        mask = _mask_for(p, W, D)
        if not mask.any():
            grid.note(f"Part '{p.spec.name}' fell outside the bounds and was dropped.")
            continue
        xs, zs = np.nonzero(mask)
        part = LayoutPart(index=len(grid.parts), spec=p.spec, mask=mask,
                          x0=int(xs.min()), z0=int(zs.min()), x1=int(xs.max()), z1=int(zs.max()),
                          parent=p.parent if p.parent is not None and p.parent < len(grid.parts) else None)
        grid.parts.append(part)
        grid.footprint |= mask

    if outset > 0:
        dil = dilate(grid.footprint, outset)
    else:
        dil = grid.footprint
    grid.exterior = ~dil
    grid.stage_done("layout", parts=[(p.spec.name, p.x0, p.z0, p.x1, p.z1) for p in grid.parts], scale=scale)
