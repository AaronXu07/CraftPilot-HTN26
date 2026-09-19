"""Roofs: per-part height maps merged by max, then quantized into stairs, slabs, and fills.

For rectilinear footprints the straight-skeleton roof height equals the
chessboard distance to the boundary times the pitch, so hip roofs are a
distance transform. Every other type is a variation on the same idea.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from craftpilot.blocks.circles import dilate, perimeter
from craftpilot.grid.enums import DIR_VEC, HORIZONTAL, OPPOSITE, BShape, Dir, Flag, Role
from craftpilot.grid.semantic import Anchor, LayoutPart, SemanticGrid
from craftpilot.program.model import (
    AnchorType,
    BuildProgram,
    RoofSpec,
    RoofType,
    Shape,
    Side,
)

FULL_ONLY_TYPES = {RoofType.flat, RoofType.parapet}


def roof_height_estimate(roof: RoofSpec, w: int, d: int) -> int:
    p = max(0.5, roof.pitch)
    o = max(0, roof.overhang)
    short = min(w, d) + 2 * o
    r = short / 2
    t = roof.type
    if t == RoofType.none:
        return 0
    if t in (RoofType.flat,):
        return 1
    if t == RoofType.parapet:
        return 3
    if t in (RoofType.hip, RoofType.gable):
        return int(math.ceil(p * r))
    if t in (RoofType.cone,):
        return int(math.ceil(p * (r + 1)))
    if t == RoofType.spire:
        return int(math.ceil(max(p, 2.0) * (r + 1)))
    if t == RoofType.dome:
        return int(math.ceil(r + 1))
    if t == RoofType.pagoda:
        return max(1, roof.tiers) * 3 + 2
    if t in (RoofType.mansard, RoofType.gambrel):
        return int(math.ceil(3 + 0.5 * max(0, r - 2)))
    if t == RoofType.shed:
        return int(math.ceil(p * short / 2))
    return int(math.ceil(p * r))


def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    return dilate(mask, n)


def _chessboard(mask: np.ndarray) -> np.ndarray:
    return ndimage.distance_transform_cdt(mask, metric="chessboard").astype(np.float32)


def _halves(h: np.ndarray) -> np.ndarray:
    return np.round(h * 2.0) / 2.0


def _axis_distance(dil: np.ndarray, along_x: bool) -> np.ndarray:
    """Distance to the two edges perpendicular to the ridge, per column, 1 at the edge."""
    W, D = dil.shape
    out = np.zeros((W, D), dtype=np.float32)
    if along_x:
        for x in range(W):
            zs = np.nonzero(dil[x])[0]
            if zs.size == 0:
                continue
            lo, hi = zs.min(), zs.max()
            z = np.arange(lo, hi + 1)
            out[x, lo:hi + 1] = np.minimum(z - lo, hi - z) + 1
    else:
        for z in range(D):
            xs = np.nonzero(dil[:, z])[0]
            if xs.size == 0:
                continue
            lo, hi = xs.min(), xs.max()
            x = np.arange(lo, hi + 1)
            out[lo:hi + 1, z] = np.minimum(x - lo, hi - x) + 1
    return out * dil


def _ridge_along_x(part: LayoutPart, spec: RoofSpec) -> bool:
    if spec.ridge_axis == "x":
        return True
    if spec.ridge_axis == "z":
        return False
    return part.width >= part.depth


def height_map(part: LayoutPart, spec: RoofSpec, top_mask: np.ndarray):
    """Returns (h, dil, closed_edge, top_role, fill_role, top_shape) for one part."""
    over = max(0, spec.overhang)
    pitch = max(0.5, spec.pitch)
    dil = _dilate(top_mask, over)
    W, D = dil.shape
    ring = perimeter(dil)
    top_role = np.full((W, D), int(Role.ROOF), dtype=np.uint8)
    fill_role = np.full((W, D), int(Role.ROOF_FILL), dtype=np.uint8)
    top_shape = np.full((W, D), 255, dtype=np.uint8)   # 255 = let the quantizer decide
    closed = ring.copy()
    t = spec.type

    if t == RoofType.hip:
        h = pitch * _chessboard(dil)
    elif t == RoofType.gable:
        ax = _ridge_along_x(part, spec)
        dz = _axis_distance(dil, ax)
        h = pitch * dz
        closed = ring & (dz == 1)
    elif t == RoofType.gambrel:
        ax = _ridge_along_x(part, spec)
        dz = _axis_distance(dil, ax)
        h = 1.5 * np.minimum(dz, 2) + 0.5 * np.maximum(0, dz - 2)
        closed = ring & (dz == 1)
    elif t == RoofType.mansard:
        dist = _chessboard(dil)
        h = 1.5 * np.minimum(dist, 2) + 0.5 * np.maximum(0, dist - 2)
    elif t == RoofType.shed:
        # Low side faces away from the parent, or south for the root.
        low = Side.south
        if part.spec.attach is not None:
            low = part.spec.attach.side
        xs = np.arange(W)[:, None]
        zs = np.arange(D)[None, :]
        if low == Side.south:
            d = (part.z1 + over) - zs + 1
        elif low == Side.north:
            d = zs - (part.z0 - over) + 1
        elif low == Side.east:
            d = (part.x1 + over) - xs + 1
        else:
            d = xs - (part.x0 - over) + 1
        h = pitch * np.clip(np.broadcast_to(d, (W, D)).astype(np.float32), 1, None) * dil
        closed = ring & (h <= pitch + 0.01)
    elif t == RoofType.flat:
        h = np.ones((W, D), dtype=np.float32) * dil
    elif t == RoofType.parapet:
        h = np.ones((W, D), dtype=np.float32) * dil
        h[ring] = 2
        top_role[ring] = int(Role.PARAPET)
        if spec.crenellated:
            xs, zs = np.nonzero(ring)
            merlon = ((xs + zs) % 2 == 0)
            h[xs[merlon], zs[merlon]] = 3
            top_role[xs[merlon], zs[merlon]] = int(Role.MERLON)
    elif t in (RoofType.cone, RoofType.spire) and part.spec.shape in (Shape.rect, Shape.cross, Shape.ring):
        # A cone over a square footprint is a pyramid: steep chessboard distance.
        p = max(pitch, 2.0) if t == RoofType.spire else pitch
        h = p * _chessboard(dil)
    elif t in (RoofType.cone, RoofType.spire, RoofType.dome):
        r = (min(part.width, part.depth) - 1) / 2 + over
        xs = np.arange(W)[:, None]
        zs = np.arange(D)[None, :]
        e = np.sqrt((xs - part.cx) ** 2 + (zs - part.cz) ** 2)
        if t == RoofType.dome:
            # Starts one block high at the eave and follows a circular profile to the centre.
            R = r + 1.0
            edge = math.sqrt(max(0.0, R * R - r * r))
            h = 1.0 + pitch * (np.sqrt(np.clip(R * R - e * e, 0, None)) - edge)
            h = np.maximum(h, 1.0)
        else:
            p = max(pitch, 2.0) if t == RoofType.spire else pitch
            h = p * np.maximum(1.0, r + 1.0 - e)
        h = h * dil
    elif t == RoofType.pagoda:
        tiers = max(1, spec.tiers)
        tier_h = 5          # eave to eave
        cap = 3             # each lower roof rises this much before the next tier's wall band
        short = min(part.width, part.depth)
        k = max(1, (short // 2 - 2) // max(1, tiers))
        h = np.zeros((W, D), dtype=np.float32)
        for ti in range(tiers):
            m = ndimage.binary_erosion(top_mask, iterations=ti * k) if ti > 0 else top_mask
            if not m.any():
                break
            dil_t = _dilate(m, over)
            dist = _chessboard(dil_t)
            this_cap = cap if ti < tiers - 1 else 1e9
            ht = ti * tier_h + np.minimum(pitch * dist, this_cap)
            h = np.where(dil_t, np.maximum(h, ht), h)
            if ti > 0:
                # The vertical band under this tier's eave is wall, not roof.
                fill_role[perimeter(m)] = int(Role.WALL)
            top_shape[perimeter(dil_t)] = int(BShape.STAIR_UPSIDE)
        h = h * dil
    else:  # none
        h = np.zeros((W, D), dtype=np.float32)
        dil = np.zeros_like(dil)
        closed = np.zeros_like(dil)

    h = _halves(h.astype(np.float32))
    return h, dil, closed, top_role, fill_role, top_shape


_PROTECTED = {Role.WALL, Role.FRAME, Role.WINDOW, Role.DOOR, Role.FLOOR}


def _protected(grid: SemanticGrid, x: int, y: int, z: int, part: int) -> bool:
    """A cell that belongs to a different part's walls must not be roofed over."""
    if not grid.in_bounds(x, y, z):
        return True
    r = int(grid.role[x, y, z])
    return r in _PROTECTED and int(grid.part_id[x, y, z]) != part


def roof(grid: SemanticGrid, program: BuildProgram) -> None:
    W, D = grid.W, grid.D
    top_role_global = np.full((W, D), int(Role.ROOF), dtype=np.uint8)
    fill_role_global = np.full((W, D), int(Role.ROOF_FILL), dtype=np.uint8)
    top_shape_global = np.full((W, D), 255, dtype=np.uint8)
    closed_global = np.zeros((W, D), dtype=bool)
    edge_global = np.zeros((W, D), dtype=bool)
    part_max: dict[int, float] = {}

    for part in grid.parts:
        spec = part.spec.roof
        if spec.type == RoofType.none or not part.floor_masks:
            continue
        top_mask = part.floor_masks[-1]
        h, dil, closed, top_role, fill_role, top_shape = height_map(part, spec, top_mask)
        surface = np.where(dil, part.eave_y + h, -np.inf).astype(np.float32)
        part.roof_surface = surface
        part.roof_mask = dil
        better = surface > grid.roof_surface
        grid.roof_surface = np.where(better, surface, grid.roof_surface)
        grid.roof_part[better] = part.index
        top_role_global[better] = top_role[better]
        fill_role_global[better] = fill_role[better]
        top_shape_global[better] = top_shape[better]
        closed_global[better] = closed[better]
        edge_global[better] = perimeter(dil)[better]
        part_max[part.index] = float(surface[dil].max()) if dil.any() else -np.inf

    S = grid.roof_surface
    valid = S > -np.inf
    eave_trim = program.depth.eave_trim

    for x, z in zip(*np.nonzero(valid)):
        s = float(S[x, z])
        pi = int(grid.roof_part[x, z])
        part = grid.parts[pi]
        eave_y = part.eave_y
        yb = int(math.ceil(s)) - 1
        frac = s - math.floor(s)
        full_only = part.spec.roof.type in FULL_ONLY_TYPES

        best_dir, best_diff, min_nb = Dir.NONE, 0.0, None
        nb: dict[int, float | None] = {}
        for d in HORIZONTAL:
            vx, _, vz = DIR_VEC[d]
            nx, nz = x + vx, z + vz
            if not (0 <= nx < W and 0 <= nz < D) or not valid[nx, nz]:
                nb[d] = None
                continue
            sn = float(S[nx, nz])
            nb[d] = sn
            diff = sn - s
            if diff > best_diff:
                best_dir, best_diff = d, diff
            min_nb = sn if min_nb is None else min(min_nb, sn)

        if full_only:
            shape, normal = BShape.FULL, Dir.NONE
        elif abs(frac - 0.5) < 1e-3:
            shape, normal = BShape.SLAB_BOTTOM, Dir.NONE
        elif best_dir != Dir.NONE and best_diff >= 0.75:
            shape, normal = BShape.STAIR, best_dir
        else:
            shape, normal = BShape.FULL, Dir.NONE
            # Two-wide ridge: a pair of stairs facing each other instead of a flat strip.
            for d in HORIZONTAL:
                o = OPPOSITE[d]
                if nb.get(d) is not None and abs(nb[d] - s) < 1e-3 and nb.get(o) is not None \
                        and 0.75 <= s - nb[o] <= 1.25:
                    shape, normal = BShape.STAIR, d
                    break
        override = int(top_shape_global[x, z])
        if override != 255 and not full_only:
            shape = override
            if normal == Dir.NONE and best_dir != Dir.NONE:
                normal = best_dir

        if yb >= grid.H:
            yb = grid.H - 1
            shape, normal = BShape.FULL, Dir.NONE
        role = int(top_role_global[x, z])
        in_mask = bool(part.mask[x, z])
        flags = 0 if in_mask else Flag.OVERHANG
        if not _protected(grid, x, yb, z, pi):
            grid.set(x, yb, z, role, shape, normal, pi, 1.0, flags)

        # Body under the surface.
        if closed_global[x, z]:
            y_lo = eave_y
        elif min_nb is not None:
            y_lo = max(eave_y, int(math.ceil(min_nb)))
        else:
            y_lo = eave_y
        fill = int(fill_role_global[x, z])
        for y in range(y_lo, yb):
            if not _protected(grid, x, y, z, pi):
                grid.set(x, y, z, fill, BShape.FULL, Dir.NONE, pi, 1.0, flags)
        if in_mask:
            per = perimeter(part.floor_masks[-1])[x, z]
            for y in range(eave_y, min(y_lo, yb)):
                if per:
                    n = int(grid.normal[x, part.top_y, z]) if grid.role[x, part.top_y, z] == Role.WALL else Dir.NONE
                    grid.set(x, y, z, Role.WALL, BShape.FULL, n, pi, 1.0, Flag.PERIMETER)
                elif grid.role[x, y, z] == Role.EMPTY:
                    grid.set(x, y, z, Role.INTERIOR, BShape.FULL, Dir.NONE, pi, 1.0)

        # Bracket under the eave.
        if eave_trim and not in_mask and edge_global[x, z] and eave_y - 1 >= 0:
            ty = eave_y - 1
            if grid.role[x, ty, z] == Role.EMPTY:
                for d in HORIZONTAL:
                    vx, _, vz = DIR_VEC[d]
                    nx, nz = x + vx, z + vz
                    if grid.in_bounds(nx, ty, nz) and grid.role[nx, ty, nz] in (Role.WALL, Role.FRAME):
                        grid.set(x, ty, z, Role.ROOF_TRIM, BShape.STAIR_UPSIDE, d, pi, 1.0, Flag.OVERHANG)
                        break

    # Ridge anchors: highest cells of each part's own surface.
    for part in grid.parts:
        if part.roof_surface is None or part.spec.roof.type in FULL_ONLY_TYPES | {RoofType.none}:
            continue
        mx = part_max.get(part.index, -np.inf)
        cells = np.argwhere((part.roof_surface >= mx - 0.01) & (grid.roof_part == part.index))
        for x, z in cells:
            y = int(math.ceil(mx)) - 1
            grid.anchors.append(Anchor(AnchorType.ridge, int(x), y, int(z), Dir.UP, part.index))

    grid.stage_done("roof", parts=[p.spec.name for p in grid.parts if p.roof_surface is not None])
