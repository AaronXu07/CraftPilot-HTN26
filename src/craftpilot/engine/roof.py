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


def _axis_distance(dil: np.ndarray, along_x: bool, offset: float = 0.0) -> np.ndarray:
    """Distance-like units to the two edges perpendicular to the ridge, 1 at the edge, rising to the
    same ridge height on both sides. With `offset` the ridge moves toward one eave and that side
    gets the steeper slope (units per cell scale so both sides still meet at the ridge)."""
    W, D = dil.shape
    out = np.zeros((W, D), dtype=np.float32)

    def profile(lo: int, hi: int) -> np.ndarray:
        n = hi - lo + 1
        half = (n - 1) / 2 + 1                       # ridge height in units for a symmetric roof
        ridge = lo + (n - 1) * (0.5 + offset)
        a = max(1.0, ridge - lo + 1)
        b = max(1.0, hi - ridge + 1)
        idx = np.arange(lo, hi + 1, dtype=np.float32)
        return np.minimum((idx - lo + 1) * half / a, (hi - idx + 1) * half / b)

    if along_x:
        for x in range(W):
            zs = np.nonzero(dil[x])[0]
            if zs.size == 0:
                continue
            lo, hi = int(zs.min()), int(zs.max())
            out[x, lo:hi + 1] = profile(lo, hi)
    else:
        for z in range(D):
            xs = np.nonzero(dil[:, z])[0]
            if xs.size == 0:
                continue
            lo, hi = int(xs.min()), int(xs.max())
            out[lo:hi + 1, z] = profile(lo, hi)
    return out * dil


def _shape_profile(units: np.ndarray, dil: np.ndarray, pitch: float, profile: str) -> np.ndarray:
    """Turn distance units into heights. Straight is linear; concave and convex bend the slope."""
    top = float(units[dil].max()) if dil.any() else 1.0
    H = pitch * top
    t = np.clip(units / max(top, 1e-6), 0.0, 1.0)
    if profile == "concave":
        h = H * t ** 1.7
    elif profile == "convex":
        h = H * t ** 0.55
    else:
        return pitch * units
    # Keep at least half a block at the eave so the ring is a slab, not nothing.
    return np.where(dil, np.maximum(h, 0.5), 0.0)


def _corner_lift(dil: np.ndarray, dist: np.ndarray, curl: float) -> np.ndarray:
    """Upward sweep of the eave at the corners: strongest on the corner ring cells, fading along the
    eave and inward."""
    if curl <= 0 or not dil.any():
        return np.zeros_like(dist)
    W, D = dil.shape
    xs, zs = np.nonzero(dil)
    x0, x1, z0, z1 = int(xs.min()), int(xs.max()), int(zs.min()), int(zs.max())
    gx, gz = np.meshgrid(np.arange(W), np.arange(D), indexing="ij")
    along = np.full((W, D), 1e9, dtype=np.float32)
    for cx, cz in ((x0, z0), (x1, z0), (x0, z1), (x1, z1)):
        along = np.minimum(along, np.maximum(np.abs(gx - cx), np.abs(gz - cz)).astype(np.float32))
    reach = max(2.0, min(x1 - x0, z1 - z0) / 3.0)
    fade_along = np.clip(1.0 - along / reach, 0.0, 1.0)
    fade_in = np.clip(1.0 - (dist - 1.0) / 2.0, 0.0, 1.0)
    lift = curl * (fade_along ** 1.5) * fade_in
    return (lift * dil).astype(np.float32)


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
        dist = _chessboard(dil)
        h = _shape_profile(dist, dil, pitch, spec.profile) + _corner_lift(dil, dist, spec.curl)
    elif t == RoofType.gable:
        ax = _ridge_along_x(part, spec)
        dz = _axis_distance(dil, ax, spec.ridge_offset)
        h = _shape_profile(dz, dil, pitch, spec.profile) + _corner_lift(dil, _chessboard(dil), spec.curl)
        closed = ring & (dz <= 1.0 + 1e-3)
    elif t == RoofType.gambrel:
        ax = _ridge_along_x(part, spec)
        dz = _axis_distance(dil, ax, spec.ridge_offset)
        h = 1.5 * np.minimum(dz, 2) + 0.5 * np.maximum(0, dz - 2)
        closed = ring & (dz <= 1.0 + 1e-3)
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
            units = np.maximum(1.0, r + 1.0 - e).astype(np.float32)
            h = _shape_profile(units, dil, p, spec.profile)
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
            base_h = _shape_profile(dist, dil_t, pitch, spec.profile) if ti == tiers - 1 else pitch * dist
            ht = ti * tier_h + np.minimum(base_h, this_cap) + _corner_lift(dil_t, dist, spec.curl)
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


def _quantize_column(grid: SemanticGrid, part: LayoutPart, S: np.ndarray, valid: np.ndarray, x: int, z: int,
                     closed: bool, edge: bool, top_role: int, fill_role: int, top_shape: int, top_perims: dict,
                     eave_trim: bool, under: bool) -> None:
    """Place one column of a part's roof from its own surface S. With `under` the column belongs to a
    lower roof running beneath a higher one, so only empty cells and this part's attic air are written."""
    W, D = grid.W, grid.D
    s = float(S[x, z])
    pi = part.index
    eave_y = part.eave_y
    yb = int(math.ceil(s)) - 1
    frac = s - math.floor(s)
    full_only = part.spec.roof.type in FULL_ONLY_TYPES

    def writable(y: int) -> bool:
        if not grid.in_bounds(x, y, z):
            return False
        if _protected(grid, x, y, z, pi):
            return False
        if under:
            r = int(grid.role[x, y, z])
            return r == Role.EMPTY or (r == Role.INTERIOR and int(grid.part_id[x, y, z]) == pi)
        return True

    best_dir, best_diff, min_nb = Dir.NONE, 0.0, None
    nb: dict[int, float | None] = {}
    min_nb_block = None
    halves = abs(frac - 0.5) < 1e-3
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
        nb_block = int(math.ceil(sn)) - 1
        min_nb_block = nb_block if min_nb_block is None else min(min_nb_block, nb_block)
        if abs((sn - math.floor(sn)) - 0.5) < 1e-3:
            halves = True

    if full_only:
        shape, normal = BShape.FULL, Dir.NONE
    elif abs(frac - 0.5) < 1e-3:
        shape, normal = BShape.SLAB_BOTTOM, Dir.NONE
    elif best_dir != Dir.NONE and best_diff >= 0.75:
        shape, normal = BShape.STAIR, best_dir
    else:
        shape, normal = BShape.FULL, Dir.NONE
        for d in HORIZONTAL:
            o = OPPOSITE[d]
            if nb.get(d) is not None and abs(nb[d] - s) < 1e-3 and nb.get(o) is not None \
                    and 0.75 <= s - nb[o] <= 1.25:
                shape, normal = BShape.STAIR, d
                break
    if top_shape != 255 and not full_only:
        shape = top_shape
        if normal == Dir.NONE and best_dir != Dir.NONE:
            normal = best_dir

    if yb >= grid.H:
        yb = grid.H - 1
        shape, normal = BShape.FULL, Dir.NONE
    in_mask = bool(part.mask[x, z])
    flags = 0 if in_mask else Flag.OVERHANG
    if writable(yb):
        grid.set(x, yb, z, top_role, shape, normal, pi, 1.0, flags)

    if closed:
        y_lo = eave_y
    elif min_nb_block is not None:
        y_lo = max(eave_y, min_nb_block)
    else:
        y_lo = eave_y
    for y in range(y_lo, yb):
        if writable(y):
            grid.set(x, y, z, fill_role, BShape.FULL, Dir.NONE, pi, 1.0, flags)

    # Walls and attic air under the surface, for every part whose top storey sits in this column.
    for other in grid.parts:
        if not other.floor_masks or not other.floor_masks[-1][x, z] or other.eave_y > yb:
            continue
        per = top_perims[other.index][x, z]
        for y in range(other.eave_y, min(y_lo, yb)):
            r = int(grid.role[x, y, z])
            if per and r in (Role.EMPTY, Role.INTERIOR, Role.ROOF_FILL):
                n = int(grid.normal[x, other.top_y, z]) if grid.role[x, other.top_y, z] == Role.WALL else Dir.NONE
                grid.set(x, y, z, Role.WALL, BShape.FULL, n, other.index, 1.0, Flag.PERIMETER)
            elif r == Role.EMPTY:
                grid.set(x, y, z, Role.INTERIOR, BShape.FULL, Dir.NONE, other.index, 1.0)

    # Bracket under the eave.
    if eave_trim and not in_mask and edge and eave_y - 1 >= 0:
        ty = eave_y - 1
        if grid.role[x, ty, z] == Role.EMPTY:
            for d in HORIZONTAL:
                vx, _, vz = DIR_VEC[d]
                nx, nz = x + vx, z + vz
                if grid.in_bounds(nx, ty, nz) and grid.role[nx, ty, nz] in (Role.WALL, Role.FRAME):
                    grid.set(x, ty, z, Role.ROOF_TRIM, BShape.STAIR_UPSIDE, d, pi, 1.0, Flag.OVERHANG)
                    break


def roof(grid: SemanticGrid, program: BuildProgram) -> None:
    W, D = grid.W, grid.D
    per_part: dict[int, dict] = {}
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
        per_part[part.index] = {"S": surface, "valid": dil, "closed": closed, "edge": perimeter(dil),
                                "top_role": top_role, "fill_role": fill_role, "top_shape": top_shape}
        part_max[part.index] = float(surface[dil].max()) if dil.any() else -np.inf

    eave_trim = program.depth.eave_trim
    top_perims = {p.index: perimeter(p.floor_masks[-1]) if p.floor_masks else np.zeros((W, D), dtype=bool)
                  for p in grid.parts}

    # Pass 1: the highest roof at every column.
    for x, z in zip(*np.nonzero(grid.roof_surface > -np.inf)):
        pi = int(grid.roof_part[x, z])
        part = grid.parts[pi]
        pp = per_part[pi]
        _quantize_column(grid, part, pp["S"], pp["valid"], int(x), int(z), bool(pp["closed"][x, z]),
                         bool(pp["edge"][x, z]), int(pp["top_role"][x, z]), int(pp["fill_role"][x, z]),
                         int(pp["top_shape"][x, z]), top_perims, eave_trim, under=False)

    # Pass 2: lower roofs continue underneath higher ones (a wing's roof under the parent's eave)
    # wherever the column is not inside another part's footprint.
    for pi, pp in per_part.items():
        part = grid.parts[pi]
        others = np.zeros((W, D), dtype=bool)
        for o in grid.parts:
            if o.index != pi:
                others |= o.mask
        cols = pp["valid"] & (grid.roof_part != pi) & ~others
        for x, z in zip(*np.nonzero(cols)):
            _quantize_column(grid, part, pp["S"], pp["valid"], int(x), int(z), bool(pp["closed"][x, z]),
                             bool(pp["edge"][x, z]), int(pp["top_role"][x, z]), int(pp["fill_role"][x, z]),
                             int(pp["top_shape"][x, z]), top_perims, eave_trim, under=True)

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
