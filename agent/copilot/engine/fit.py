"""Sub-voxel surface fitting: full block / slab / stairs / wall per surface voxel. plan.md §5.2–5.3.

Every candidate voxel (occupied with an exposed face, or air face-adjacent to a fittable occupied
voxel) is sampled at its 8 sub-centres (±0.25). The 8-bit inside pattern selects the block kind.

FACING CONVENTION (Minecraft): a stair's `facing` is the direction of its FULL-HEIGHT (high) side,
i.e. the direction the player looked when placing it. A stair whose top half is missing on its
east side therefore gets facing=west. On a cone or roof slope the facings point *toward* the
axis/ridge, exactly like hand-built roof stairs. `FACING_CALIBRATION` records that mapping; it is
verified in-game by scripts/calibrate_facing.py (Track 2).

`fit_surface` updates `raster.material` / `raster.owner` in place so that afterwards
`(raster.material > 0) == (fit.kind > 0)` (voxels added or removed by fitting stay consistent).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .coretypes import Facing, FitResult, Half, Kind, RasterResult

# pattern side -> stair facing (away from the missing pair, i.e. the full side)
FACING_CALIBRATION: Dict[str, str] = {
    "missing_top_pair_on_west": "east",
    "missing_top_pair_on_east": "west",
    "missing_top_pair_on_north": "south",
    "missing_top_pair_on_south": "north",
    "slab_bottom_half_inside": "type=bottom",
    "slab_top_half_inside": "type=top",
}

FIT_MODES = ("none", "slab", "stairs", "stairs+slab", "walls")

# sub-point index b = ix*4 + iy*2 + iz ; ix/iy/iz = 1 means the +0.25 offset
_OFFSETS = np.array([[(-0.25 if ix == 0 else 0.25), (-0.25 if iy == 0 else 0.25), (-0.25 if iz == 0 else 0.25)] for ix in (0, 1) for iy in (0, 1) for iz in (0, 1)])
_BOTTOM = np.array([0, 1, 4, 5])
_TOP = np.array([2, 3, 6, 7])
_SIDES = {  # side name -> (all 4 indices on that side, top pair, bottom pair, facing when that side is the LOW side)
    "west": (np.array([0, 1, 2, 3]), np.array([2, 3]), np.array([0, 1]), Facing.EAST),
    "east": (np.array([4, 5, 6, 7]), np.array([6, 7]), np.array([4, 5]), Facing.WEST),
    "north": (np.array([0, 2, 4, 6]), np.array([2, 6]), np.array([0, 4]), Facing.SOUTH),
    "south": (np.array([1, 3, 5, 7]), np.array([3, 7]), np.array([1, 5]), Facing.NORTH),
}
_NEIGHBOUR_PRIORITY = [(0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0)]


def _shift(a: np.ndarray, d: Tuple[int, int, int], fill) -> np.ndarray:
    """Array shifted so out[i] = a[i - d] (neighbour lookup), padded with `fill`."""
    out = np.full_like(a, fill)
    X, Y, Z = a.shape
    dx, dy, dz = d
    src = (slice(max(0, -dx), X - max(0, dx)), slice(max(0, -dy), Y - max(0, dy)), slice(max(0, -dz), Z - max(0, dz)))
    dst = (slice(max(0, dx), X - max(0, -dx)), slice(max(0, dy), Y - max(0, -dy)), slice(max(0, dz), Z - max(0, -dz)))
    out[dst] = a[src]
    return out


def exposed_faces(occupied: np.ndarray) -> np.ndarray:
    """bool[6,X,Y,Z]: face exposed to air/out-of-bounds; order -x,+x,-y,+y,-z,+z."""
    occ = occupied.astype(bool)
    dirs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    out = np.zeros((6,) + occ.shape, dtype=bool)
    for i, d in enumerate(dirs):
        neighbour = _shift(occ, (-d[0], -d[1], -d[2]), False)  # neighbour[i] = occ[i + d]
        out[i] = occ & ~neighbour
    return out


def _neighbour_count(mask: np.ndarray) -> np.ndarray:
    """Number of occupied 6-neighbours per voxel."""
    m = mask.astype(np.int8)
    total = np.zeros(mask.shape, dtype=np.int8)
    for d in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        total += _shift(m, (-d[0], -d[1], -d[2]), 0)
    return total


def gravity_check(kind: np.ndarray, origin: Tuple[int, int, int] = (0, 0, 0)) -> List[Tuple[int, int, int]]:
    """World coords of floating voxels: nothing below and no horizontal neighbours."""
    occ = kind > 0
    below = _shift(occ, (0, 1, 0), False)  # below[i] = occ[i - (0,1,0)]
    horiz = np.zeros(occ.shape, dtype=np.int8)
    for d in [(-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)]:
        horiz += _shift(occ.astype(np.int8), (-d[0], -d[1], -d[2]), 0)
    floating = occ & ~below & (horiz == 0)
    floating[:, 0, :] = False  # bottom layer sits on the ground
    idx = np.argwhere(floating)
    return [(int(i + origin[0]), int(j + origin[1]), int(k + origin[2])) for i, j, k in idx]


def _mode_flags(mode: str) -> Tuple[bool, bool, bool]:
    """(slab allowed, stairs allowed, walls allowed) for a fit mode."""
    if mode == "none":
        return False, False, False
    if mode == "slab":
        return True, False, False
    if mode == "stairs":
        return False, True, False
    if mode == "walls":
        return True, True, True
    return True, True, False


def _classify(inside: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pattern table -> (kind, facing, half) arrays for inside patterns [K,8] (before mode filtering)."""
    K = len(inside)
    count = inside.sum(axis=1)
    kind = np.where(count >= 4, np.uint8(Kind.FULL), np.uint8(Kind.AIR)).astype(np.uint8)
    facing = np.zeros(K, dtype=np.uint8)
    half = np.zeros(K, dtype=np.uint8)
    bottom_all = inside[:, _BOTTOM].all(axis=1)
    top_all = inside[:, _TOP].all(axis=1)
    bottom_none = ~inside[:, _BOTTOM].any(axis=1)
    top_none = ~inside[:, _TOP].any(axis=1)
    slab_b = bottom_all & top_none
    slab_t = top_all & bottom_none
    kind[slab_b] = Kind.SLAB
    half[slab_b] = Half.BOTTOM
    kind[slab_t] = Kind.SLAB
    half[slab_t] = Half.TOP
    six = count == 6
    for name, (all_idx, top_pair, bot_pair, face_when_low) in _SIDES.items():
        miss_top = six & ~inside[:, top_pair].any(axis=1)
        kind[miss_top] = Kind.STAIRS
        half[miss_top] = Half.BOTTOM
        facing[miss_top] = int(face_when_low)
        miss_bot = six & ~inside[:, bot_pair].any(axis=1)
        kind[miss_bot] = Kind.STAIRS
        half[miss_bot] = Half.TOP
        facing[miss_bot] = int(face_when_low)
    four = count == 4
    vertical_half = np.zeros(K, dtype=bool)
    for name, (all_idx, _, _, _) in _SIDES.items():
        vertical_half |= four & inside[:, all_idx].all(axis=1)
    kind[vertical_half] = Kind.WALL  # downgraded to FULL later unless the material allows walls
    kind[count == 8] = Kind.FULL
    return kind, facing, half


def fit_surface(raster: RasterResult, fit_modes: Optional[Dict[str, str]] = None, default_mode: str = "stairs+slab") -> FitResult:
    """Fit slabs/stairs/walls to every surface voxel of `raster` according to per-material fit modes."""
    fit_modes = fit_modes or {}
    occ = raster.material > 0
    X, Y, Z = occ.shape
    kind = np.where(occ, np.uint8(Kind.FULL), np.uint8(Kind.AIR)).astype(np.uint8)
    facing = np.zeros_like(kind)
    half = np.zeros_like(kind)
    result = FitResult(kind=kind, facing=facing, half=half)
    if raster.scene_sdf is None or not np.any(occ):
        return result

    # per material-index mode
    modes = [default_mode] * len(raster.material_names)
    for i, name in enumerate(raster.material_names):
        if i == 0:
            modes[i] = "none"
        else:
            modes[i] = fit_modes.get(name, default_mode)
    modes_arr = np.array([m if m in FIT_MODES else default_mode for m in modes])
    flags = np.array([_mode_flags(m) for m in modes_arr], dtype=bool)  # [M,3]
    mat = raster.material
    fittable = occ & (modes_arr[mat] != "none")

    # candidates: occupied+exposed+fittable, and air adjacent to fittable occupied voxels
    exp = exposed_faces(occ).any(axis=0)
    cand_occ = fittable & exp
    adj = np.zeros(occ.shape, dtype=bool)
    for d in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        adj |= _shift(fittable, d, False)
    cand_air = adj & ~occ
    # material for air candidates: neighbour priority (-y, -x, +x, -z, +z, +y)
    air_mat = np.zeros(occ.shape, dtype=np.int16)
    air_owner = np.full(occ.shape, -1, dtype=np.int32)
    for d in reversed(_NEIGHBOUR_PRIORITY):
        nm = _shift(np.where(fittable, mat, 0), (-d[0], -d[1], -d[2]), np.int16(0))
        no = _shift(np.where(fittable, raster.owner, -1), (-d[0], -d[1], -d[2]), np.int32(-1))
        pick = cand_air & (nm > 0)
        air_mat[pick] = nm[pick]
        air_owner[pick] = no[pick]
    cand_air &= air_mat > 0

    cands = np.argwhere(cand_occ | cand_air)
    if len(cands) == 0:
        return result
    is_occ = occ[cands[:, 0], cands[:, 1], cands[:, 2]]
    cmat = np.where(is_occ, mat[cands[:, 0], cands[:, 1], cands[:, 2]], air_mat[cands[:, 0], cands[:, 1], cands[:, 2]])
    centres = cands.astype(np.float64) + np.asarray(raster.origin, dtype=np.float64) + 0.5
    pts = (centres[:, None, :] + _OFFSETS[None, :, :]).reshape(-1, 3)
    f, owner = raster.scene_sdf(pts)
    inside = (f <= 0.0).reshape(-1, 8)
    owner = owner.reshape(-1, 8)
    fmin = f.reshape(-1, 8).min(axis=1)

    k, fc, hf = _classify(inside)
    slab_ok = flags[cmat, 0]
    stairs_ok = flags[cmat, 1]
    walls_ok = flags[cmat, 2]
    # mode filtering
    k = k.copy()
    k[(k == Kind.WALL) & ~walls_ok] = Kind.FULL
    bad_slab = (k == Kind.SLAB) & ~slab_ok
    bad_stairs = (k == Kind.STAIRS) & ~stairs_ok
    # occupied voxels fall back to FULL; air voxels fall back to slab (if allowed) or AIR.
    # Air candidates may only gain partial blocks (slab/stairs), never a full block or wall.
    k[(bad_slab | bad_stairs) & is_occ] = Kind.FULL
    air_bad = ((bad_slab | bad_stairs) | (k == Kind.FULL) | (k == Kind.WALL)) & ~is_occ
    bottom_all = inside[:, _BOTTOM].all(axis=1)
    top_all = inside[:, _TOP].all(axis=1)
    fb = air_bad & slab_ok & bottom_all
    ft = air_bad & slab_ok & top_all & ~bottom_all
    k[air_bad] = Kind.AIR
    k[fb] = Kind.SLAB
    hf[fb] = Half.BOTTOM
    k[ft] = Kind.SLAB
    hf[ft] = Half.TOP

    # props: keep the voxel directly below a prop if fitting would delete it
    if raster.props:
        support = np.zeros(occ.shape, dtype=bool)
        for (px, py, pz) in raster.props:
            i, j, kk = raster.world_to_index(px, py - 1, pz)
            if 0 <= i < X and 0 <= j < Y and 0 <= kk < Z:
                support[i, j, kk] = True
        sup = support[cands[:, 0], cands[:, 1], cands[:, 2]] & is_occ
        k[sup & (k == Kind.AIR)] = Kind.FULL
    else:
        support = None

    # write back
    ci, cj, ck = cands[:, 0], cands[:, 1], cands[:, 2]
    kind[ci, cj, ck] = k
    facing[ci, cj, ck] = fc
    half[ci, cj, ck] = hf
    new_air = ~is_occ & (k > 0)
    if np.any(new_air):
        # owner of added voxels: majority owner among inside sub-points, else the neighbour's owner
        own = owner[new_air]
        ins = inside[new_air]
        best = np.array([np.bincount(o[i][o[i] >= 0]).argmax() if np.any(o[i] >= 0) else -1 for o, i in zip(own, ins)])
        fallback = air_owner[ci[new_air], cj[new_air], ck[new_air]]
        best = np.where(best >= 0, best, fallback)
        raster.material[ci[new_air], cj[new_air], ck[new_air]] = cmat[new_air]
        raster.owner[ci[new_air], cj[new_air], ck[new_air]] = best.astype(np.int32)
        raster.sdf[ci[new_air], cj[new_air], ck[new_air]] = fmin[new_air].astype(np.float32)

    # structural sanity: drop isolated partial/new voxels (keep prop supports and solid originals)
    full_inside = np.zeros(occ.shape, dtype=bool)
    full_inside[ci, cj, ck] = inside.all(axis=1) & is_occ
    solid_original = occ & ~(cand_occ) | full_inside
    present = kind > 0
    isolated = present & (_neighbour_count(present) == 0) & ~solid_original
    if support is not None:
        isolated &= ~support
    kind[isolated] = Kind.AIR

    removed = (kind == Kind.AIR)
    raster.material[removed] = 0
    raster.owner[removed] = -1
    return result


def fit_summary(fit: FitResult) -> str:
    """One-line count summary."""
    c = fit.counts()
    return ", ".join(f"{k}={v}" for k, v in c.items())
