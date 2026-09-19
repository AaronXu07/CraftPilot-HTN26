"""Scene -> voxel grid. See plan.md §5.1 and CONTRACTS.md §5.

Voxel `v` spans world [v, v+1); SDFs are sampled at voxel centres `v + 0.5`. Objects are evaluated
in list order within their own voxel range only (CSG painter's stack):
  add       f = min(f, g); the later object takes ownership/material where g <= 0
  subtract  f = max(f, -g); voxels with f > 0 are cleared (carved surfaces keep a valid sdf)
  intersect f = max(f, g) inside the object's range; everything outside the range is cleared
  paint     material := paint material where g <= 0 and the voxel is occupied
`block` shapes are placed directly into `props` at floor(pos) and never touch the material grid.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .coretypes import Bbox, RasterResult
from .modifiers import build_object_sdf
from .scene import Scene, SceneObject

MAX_VOXELS = 200 * 120 * 200
MAX_DIM = 256


class RasterError(ValueError):
    pass


class RasterCache(dict):
    """dict[key -> (sdf_array, range)] of per-object evaluations; safe to store in Session.cache."""


def _object_key(scene: Scene, obj: SceneObject, origin: Tuple[int, int, int], rng) -> str:
    """Cache key: object json + boolean targets' json + grid origin + voxel range."""
    payload = {"o": obj.to_dict(), "origin": list(origin), "rng": [list(map(int, rng[0])), list(map(int, rng[1]))]}
    for m in obj.modifiers:
        if m["type"] == "boolean" and scene.has(m["target"]):
            payload.setdefault("t", []).append(scene.get(m["target"]).to_dict())
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _voxel_range(bb: Bbox, origin: np.ndarray, dims: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Grid index range [lo, hi) covered by a world bbox, clipped to the grid; None when empty."""
    if bb.is_empty():
        return None
    lo = np.floor(bb.lo - 1e-9).astype(int) - origin
    hi = np.ceil(bb.hi + 1e-9).astype(int) - origin
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, dims)
    if np.any(hi <= lo):
        return None
    return lo, hi


def _centres(origin: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """World-space voxel centres for grid index range [lo, hi), in C order (x, y, z)."""
    xs = np.arange(lo[0], hi[0]) + origin[0] + 0.5
    ys = np.arange(lo[1], hi[1]) + origin[1] + 0.5
    zs = np.arange(lo[2], hi[2]) + origin[2] + 0.5
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)


def scene_grid_bounds(scene: Scene, pad: int = 1) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Integer grid bounds [lo, hi) covering every visible `add` object (incl. props), padded."""
    bb = Bbox.empty()
    for o in scene.objects:
        if not o.visible or o.op != "add":
            continue
        bb = bb.union(scene.object_bbox(o))
    if bb.is_empty():
        return None
    lo = np.floor(bb.lo + 1e-9).astype(int) - pad
    hi = np.ceil(bb.hi - 1e-9).astype(int) + pad
    return lo, hi


def rasterize(scene: Scene, pad: int = 1, cache: Optional[dict] = None) -> RasterResult:
    """Rasterize the scene's CSG stack into a RasterResult (material/owner/sdf grids + props)."""
    t0 = time.perf_counter()
    bounds = scene_grid_bounds(scene, pad)
    if bounds is None:
        empty = np.zeros((1, 1, 1), dtype=np.int16)
        res = RasterResult(origin=(0, 0, 0), material=empty, material_names=[""], owner=np.full((1, 1, 1), -1, np.int32), sdf=np.full((1, 1, 1), np.inf, np.float32))
        res.scene_sdf = lambda pts: (np.full(len(np.asarray(pts).reshape(-1, 3)), np.inf), np.full(len(np.asarray(pts).reshape(-1, 3)), -1))
        res.scene_hash = scene.hash()
        return res
    lo, hi = bounds
    dims = hi - lo
    if np.any(dims > MAX_DIM) or int(np.prod(dims)) > MAX_VOXELS:
        raise RasterError(
            f"scene grid {dims[0]}x{dims[1]}x{dims[2]} exceeds the cap ({MAX_DIM} per axis, {MAX_VOXELS:,} voxels); "
            "shrink the build or check for a units mistake (an object dimension > 120 blocks?)"
        )
    origin = lo
    X, Y, Z = (int(v) for v in dims)
    f = np.full((X, Y, Z), np.inf, dtype=np.float32)
    owner = np.full((X, Y, Z), -1, dtype=np.int32)
    material = np.zeros((X, Y, Z), dtype=np.int16)
    material_names: List[str] = [""]
    props: Dict[Tuple[int, int, int], str] = {}
    timings: Dict[str, float] = {}
    sdf_fns: List[Optional[Callable]] = [None] * len(scene.objects)
    obj_bboxes: List[Optional[Bbox]] = [None] * len(scene.objects)
    evaluated = 0
    cached_hits = 0

    def mat_index(name: str) -> int:
        name = name or "default"
        if name not in material_names:
            material_names.append(name)
        return material_names.index(name)

    for idx, obj in enumerate(scene.objects):
        if not obj.visible:
            continue
        if obj.shape["type"] == "block":
            if obj.op == "add":
                p = np.floor(obj.transform.pos + 1e-9).astype(int)
                props[(int(p[0]), int(p[1]), int(p[2]))] = obj.shape["state"]
            continue
        bb = scene.object_bbox(obj)
        obj_bboxes[idx] = bb
        rng = _voxel_range(bb, origin, dims)
        fn = build_object_sdf(scene, obj)
        sdf_fns[idx] = fn
        if obj.op == "intersect":
            # everything outside the object's range is gone
            keep = np.zeros((X, Y, Z), dtype=bool)
            if rng is not None:
                keep[rng[0][0]:rng[1][0], rng[0][1]:rng[1][1], rng[0][2]:rng[1][2]] = True
            f[~keep] = np.inf
            owner[~keep] = -1
            material[~keep] = 0
        if rng is None:
            continue
        rlo, rhi = rng
        key = _object_key(scene, obj, tuple(int(v) for v in origin), rng)
        g = None
        if cache is not None and key in cache:
            g = cache[key]
            cached_hits += 1
        if g is None:
            ts = time.perf_counter()
            pts = _centres(origin, rlo, rhi)
            g = fn(pts).reshape(tuple(int(v) for v in (rhi - rlo))).astype(np.float32)
            timings[f"obj:{obj.id}"] = time.perf_counter() - ts
            evaluated += 1
            if cache is not None:
                cache[key] = g
        sl = (slice(rlo[0], rhi[0]), slice(rlo[1], rhi[1]), slice(rlo[2], rhi[2]))
        fs = f[sl]
        if obj.op == "add":
            inside = g <= 0.0
            f[sl] = np.minimum(fs, g)
            owner[sl][inside] = idx
            material[sl][inside] = mat_index(obj.material_name())
        elif obj.op == "subtract":
            fn_new = np.maximum(fs, -g)
            f[sl] = fn_new
            cleared = fn_new > 0.0
            owner[sl][cleared] = -1
            material[sl][cleared] = 0
        elif obj.op == "intersect":
            fn_new = np.maximum(fs, g)
            f[sl] = fn_new
            cleared = fn_new > 0.0
            owner[sl][cleared] = -1
            material[sl][cleared] = 0
        elif obj.op == "paint":
            mask = (g <= 0.0) & (material[sl] > 0)
            material[sl][mask] = mat_index(obj.material_name())
    # any voxel with f <= 0 but no material (should not happen) -> clear for consistency
    bad = (f <= 0) & (material == 0)
    if np.any(bad):
        f[bad] = np.inf
        owner[bad] = -1

    def scene_sdf(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the CSG stack at arbitrary world points -> (f, owner index)."""
        P = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        n = len(P)
        ff = np.full(n, np.inf)
        oo = np.full(n, -1, dtype=np.int64)
        for i, obj in enumerate(scene.objects):
            fn_i = sdf_fns[i]
            if fn_i is None:
                continue
            bb = obj_bboxes[i]
            m = np.all((P >= bb.lo - 0.01) & (P <= bb.hi + 0.01), axis=1)
            if obj.op == "intersect":
                ff[~m] = np.inf
                oo[~m] = -1
            if not np.any(m):
                continue
            g = fn_i(P[m])
            if obj.op == "add":
                cur = ff[m]
                ins = g <= 0.0
                ff[m] = np.minimum(cur, g)
                sub = oo[m]
                sub[ins] = i
                oo[m] = sub
            elif obj.op == "subtract":
                cur = np.maximum(ff[m], -g)
                ff[m] = cur
                sub = oo[m]
                sub[cur > 0.0] = -1
                oo[m] = sub
            elif obj.op == "intersect":
                cur = np.maximum(ff[m], g)
                ff[m] = cur
                sub = oo[m]
                sub[cur > 0.0] = -1
                oo[m] = sub
        oo[ff > 0.0] = -1
        return ff, oo

    res = RasterResult(
        origin=(int(origin[0]), int(origin[1]), int(origin[2])),
        material=material,
        material_names=material_names,
        owner=owner,
        sdf=f,
        props=props,
        scene_sdf=scene_sdf,
        scene_hash=scene.hash(),
    )
    timings["total"] = time.perf_counter() - t0
    timings["objects_evaluated"] = float(evaluated)
    timings["cache_hits"] = float(cached_hits)
    res.timings = timings
    return res


def material_index_of(raster: RasterResult, name: str) -> int:
    """Index of a material name in the raster (appending it if new)."""
    if name not in raster.material_names:
        raster.material_names.append(name)
    return raster.material_names.index(name)
