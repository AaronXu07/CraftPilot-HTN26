"""Resolver: material id per voxel + fit result -> concrete block states (plan.md §4.4).

`resolve(raster, fit, scene, registry, seed) -> BlockMap` is the only public entry point.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .coretypes import FACING_NAMES, BlockMap, FitResult, Kind, RasterResult, format_state, parse_state, short_id
from .materials import MaterialSpec, choose_blocks, resolve_material
from .registry import Registry

_CONNECTIVITY_PROPS = {"shape", "north", "east", "south", "west", "up", "down"}


def resolve(raster: RasterResult, fit: FitResult, scene: Any, registry: Registry, seed: int = 0,
            warnings: Optional[List[str]] = None) -> BlockMap:
    """Turn a rasterised scene into `{(x,y,z): "minecraft:id[props]"}` (world coords, no air).

    Steps per voxel: material name (from raster.material, or the nearest occupied neighbour for
    slab/stair candidates that fit.py added in previously-air voxels) -> palette entry via coherent
    noise -> gradient / face override -> stairs/slab/wall substitution from the block's family (or
    the material base's family) when fit.kind says so -> registry validation. Explicit `block`
    props from raster.props are merged last (doors get their upper half, leaves are made persistent).
    Emitted states carry only the properties we set; the game computes connectivity (stairs shape,
    fence/wall/pane sides) on placement.
    """
    if warnings is None:
        warnings = []
    kind = np.asarray(fit.kind)
    occ = kind > 0
    out: BlockMap = {}
    if not occ.any() and not raster.props:
        return out
    idx = np.argwhere(occ)
    if len(idx):
        mat = np.asarray(raster.material)
        mat_idx = mat[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int64)
        missing = mat_idx == 0
        if missing.any():
            mat_idx[missing] = _nearest_material(mat, idx[missing])
        ox, oy, oz = raster.origin
        centres = idx.astype(np.float64) + 0.5
        centres[:, 0] += ox
        centres[:, 1] += oy
        centres[:, 2] += oz
        top_exp, side_exp, bot_exp = _exposure(kind, idx)
        facing = np.asarray(fit.facing)[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int64)
        half = np.asarray(fit.half)[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int64)
        kinds = kind[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int64)
        chosen = np.empty(len(idx), dtype=object)
        base_of = np.empty(len(idx), dtype=object)
        fit_of = np.empty(len(idx), dtype=object)
        scene_mats = getattr(scene, "materials", {}) or {}
        specs: Dict[int, MaterialSpec] = {}
        for mi in np.unique(mat_idx):
            name = raster.material_names[int(mi)] if 0 <= int(mi) < len(raster.material_names) else "default"
            specs[int(mi)] = resolve_material(name or "default", scene_mats, warnings, registry)
        for mi, spec in specs.items():
            m = mat_idx == mi
            chosen[m] = choose_blocks(spec, centres[m], seed, top_exp[m], side_exp[m], bot_exp[m])
            base_of[m] = spec.base
            fit_of[m] = spec.fit
        states = _assemble_states(chosen, base_of, fit_of, kinds, facing, half, registry, warnings)
        xs = idx[:, 0] + ox
        ys = idx[:, 1] + oy
        zs = idx[:, 2] + oz
        for x, y, z, st in zip(xs.tolist(), ys.tolist(), zs.tolist(), states):
            if st:
                out[(x, y, z)] = st
    # explicit props (block solids) win over fitted geometry
    for pos, state in (raster.props or {}).items():
        for p, st in _expand_prop(pos, state, registry, warnings):
            out[p] = st
    return out


def _nearest_material(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Material index of the first occupied 6-neighbour (down, up, -x, +x, -z, +z), else the most
    common material in the grid, else 1."""
    X, Y, Z = mat.shape
    res = np.zeros(len(idx), dtype=np.int64)
    for d in ((0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)):
        todo = res == 0
        if not todo.any():
            break
        n = idx[todo] + np.asarray(d)
        ok = (n[:, 0] >= 0) & (n[:, 0] < X) & (n[:, 1] >= 0) & (n[:, 1] < Y) & (n[:, 2] >= 0) & (n[:, 2] < Z)
        vals = np.zeros(len(n), dtype=np.int64)
        vals[ok] = mat[n[ok, 0], n[ok, 1], n[ok, 2]]
        sub = res[todo]
        sub[vals > 0] = vals[vals > 0]
        res[todo] = sub
    still = res == 0
    if still.any():
        nz = mat[mat > 0]
        common = int(np.bincount(nz).argmax()) if nz.size else 1
        res[still] = common
    return res


def _exposure(kind: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(top, side, bottom) exposure booleans for the given voxel indices (out of bounds = exposed)."""
    X, Y, Z = kind.shape

    def air_at(d: Tuple[int, int, int]) -> np.ndarray:
        n = idx + np.asarray(d)
        inb = (n[:, 0] >= 0) & (n[:, 0] < X) & (n[:, 1] >= 0) & (n[:, 1] < Y) & (n[:, 2] >= 0) & (n[:, 2] < Z)
        res = np.ones(len(idx), dtype=bool)
        res[inb] = kind[n[inb, 0], n[inb, 1], n[inb, 2]] == 0
        return res

    top = air_at((0, 1, 0))
    bottom = air_at((0, -1, 0))
    side = air_at((1, 0, 0)) | air_at((-1, 0, 0)) | air_at((0, 0, 1)) | air_at((0, 0, -1))
    return top, side, bottom


def _assemble_states(chosen: np.ndarray, base_of: np.ndarray, fit_of: np.ndarray, kinds: np.ndarray,
                     facing: np.ndarray, half: np.ndarray, registry: Registry, warnings: List[str]) -> List[str]:
    """Vectorised-ish state assembly: unique (block, base, fit, kind, facing, half) combos -> state."""
    keys = [(c, b, f, int(k), int(fa), int(h)) for c, b, f, k, fa, h in zip(chosen, base_of, fit_of, kinds, facing, half)]
    cache: Dict[Tuple[str, str, str, int, int, int], str] = {}
    out: List[str] = []
    for key in keys:
        st = cache.get(key)
        if st is None:
            st = _state_for(key, registry, warnings)
            cache[key] = st
        out.append(st)
    return out


def _state_for(key: Tuple[str, str, str, int, int, int], registry: Registry, warnings: List[str]) -> str:
    block, base, fit, kind, facing, half = key
    props: Dict[str, str] = {}
    bid = block
    if kind == int(Kind.SLAB) and fit in ("slab", "stairs+slab", "walls"):
        slab = _form(block, base, "slab", registry)
        if slab:
            bid = slab
            props = {"type": "top" if half == 1 else "bottom"}
    elif kind == int(Kind.STAIRS) and fit in ("stairs", "stairs+slab", "walls"):
        st = _form(block, base, "stairs", registry)
        if st:
            bid = st
            props = {"facing": FACING_NAMES[facing % 4], "half": "top" if half == 1 else "bottom"}
        else:
            slab = _form(block, base, "slab", registry) if fit != "stairs" else None
            if slab:
                bid = slab
                props = {"type": "top" if half == 1 else "bottom"}
    elif kind == int(Kind.WALL) and fit == "walls":
        wall = _form(block, base, "wall", registry)
        if wall:
            bid = wall
    ok, res = registry.validate_state(format_state(bid, props), fill_defaults=False)
    if ok:
        return res
    ok2, res2 = registry.validate_state(block, fill_defaults=False)
    if ok2:
        warnings.append(f"{res}; used {short_id(block)}")
        return res2
    warnings.append(f"{res}; used stone")
    return "minecraft:stone"


def _form(block: str, base: str, form: str, registry: Registry) -> Optional[str]:
    fam = registry.family(block)
    if form in fam:
        return fam[form]
    fam_b = registry.family(base)
    return fam_b.get(form)


def _expand_prop(pos: Tuple[int, int, int], state: str, registry: Registry, warnings: List[str]) -> List[Tuple[Tuple[int, int, int], str]]:
    """Validate a prop state and add companion blocks (door upper half, tall plant upper half)."""
    ok, res = registry.validate_state(state, fill_defaults=False)
    if not ok:
        warnings.append(f"prop at {pos} dropped: {res}")
        return []
    bid, props = parse_state(res)
    bdef = registry.get(bid)
    if bdef is not None and "persistent" in bdef.properties and "persistent" not in props:
        props["persistent"] = "true"
    out = [(tuple(int(v) for v in pos), format_state(bid, props))]
    x, y, z = out[0][0]
    if bdef is not None and "half" in bdef.properties:
        vals = bdef.properties["half"]
        if "lower" in vals and "upper" in vals and props.get("half", "lower") == "lower":
            up = dict(props)
            up["half"] = "upper"
            out.append(((x, y + 1, z), format_state(bid, up)))
    return out
