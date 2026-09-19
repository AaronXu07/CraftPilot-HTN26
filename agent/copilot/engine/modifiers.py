"""Modifier stack -> world-space SDF callable for a scene object. See CONTRACTS.md §4.

Evaluation order for `build_object_sdf(scene, obj)(P)`:
  1. array / mirror expand the world query set (instances are unioned with `min`),
  2. world -> local via the object's Transform and anchor,
  3. taper / twist warp the local points,
  4. base primitive SDF (rounded via `shrunk` when a `round` modifier is present), scaled by
     the transform's isotropic scale factor,
  5. shell / noise_displace / boolean applied in stack order.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from .noise import noise3
from .solids import Solid, make_solid

SdfFn = Callable[[np.ndarray], np.ndarray]
MAX_BOOLEAN_DEPTH = 4


def build_object_sdf(scene, obj, _depth: int = 0) -> SdfFn:
    """Return f(P_world[N,3]) -> signed distance [N] for `obj` including its modifier stack."""
    solid: Solid = make_solid(obj.shape)
    mods = list(obj.modifiers)
    lo, hi = solid.local_bbox()
    anchor_pt = obj.anchor_point()
    transform = obj.transform
    sf = transform.scale_factor()

    # -- round: replace the base primitive by its shrunk version -------------------------
    round_r = sum(float(m["radius"]) for m in mods if m["type"] == "round")
    base = solid
    if round_r > 0:
        shrunk = solid.shrunk(round_r)
        if shrunk is not None:
            base = shrunk
        else:
            round_r = 0.0

    # -- world-space point expansion (array / mirror) --------------------------------------
    expanders: List[Tuple[str, dict]] = [(m["type"], m) for m in mods if m["type"] in ("array", "mirror")]

    # -- local warps (taper / twist) --------------------------------------------------------
    warps: List[dict] = [m for m in mods if m["type"] in ("taper", "twist")]
    cx, cz = float((lo[0] + hi[0]) / 2.0), float((lo[2] + hi[2]) / 2.0)
    y0, y1 = float(lo[1]), float(hi[1])
    yspan = max(y1 - y0, 1e-9)

    # -- post ops (shell / noise / boolean) in stack order ----------------------------------
    post: List[dict] = [m for m in mods if m["type"] in ("shell", "noise_displace", "boolean")]
    boolean_fns: Dict[str, SdfFn] = {}
    for m in post:
        if m["type"] == "boolean" and _depth < MAX_BOOLEAN_DEPTH and scene.has(m["target"]) and m["target"] != obj.id:
            boolean_fns[m["target"]] = build_object_sdf(scene, scene.get(m["target"]), _depth + 1)

    def _expand(P: np.ndarray) -> Tuple[np.ndarray, int]:
        """Apply array/mirror: returns stacked instance points [k*N,3] and k."""
        pts = [P]
        for kind, m in expanders:
            if kind == "array":
                off = np.asarray(m["offset"], dtype=np.float64)
                n = int(m["count"])
                pts = [p - off * i for p in pts for i in range(n)]
            else:
                ax = "xyz".index(m["axis"])
                plane = float(m["plane"])
                refl = []
                for p in pts:
                    q = p.copy()
                    q[:, ax] = 2.0 * plane - q[:, ax]
                    refl.append(q)
                pts = pts + refl if m.get("keep_original", True) else refl
        return np.concatenate(pts, axis=0), len(pts)

    def _warp(q: np.ndarray) -> np.ndarray:
        for m in warps:
            t = np.clip((q[:, 1] - y0) / yspan, 0.0, 1.0)
            if m["type"] == "taper":
                s = 1.0 + (float(m["top_scale"]) - 1.0) * t
                s = np.where(np.abs(s) < 1e-6, 1e-6, s)
                q = q.copy()
                q[:, 0] = (q[:, 0] - cx) / s + cx
                q[:, 2] = (q[:, 2] - cz) / s + cz
            else:
                ang = np.deg2rad(float(m["deg_per_block"]) * (q[:, 1] - y0))
                c, s_ = np.cos(ang), np.sin(ang)
                x = q[:, 0] - cx
                z = q[:, 2] - cz
                q = q.copy()
                q[:, 0] = c * x + s_ * z + cx
                q[:, 2] = -s_ * x + c * z + cz
        return q

    def f(P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        n = len(P)
        if n == 0:
            return np.zeros(0)
        pts, k = _expand(P)
        q = transform.world_to_local(pts, anchor_pt)
        q = _warp(q)
        d = base.sdf(q) * sf - round_r
        for m in post:
            t = m["type"]
            if t == "shell":
                th = float(m["thickness"])
                d = np.maximum(d, -(d + th))
            elif t == "noise_displace":
                d = d + float(m["amplitude"]) * noise3(pts, float(m["scale"]), int(m["seed"]))
            elif t == "boolean":
                g_fn = boolean_fns.get(m["target"])
                if g_fn is None:
                    continue
                g = g_fn(pts)
                if m["op"] == "union":
                    d = np.minimum(d, g)
                elif m["op"] == "subtract":
                    d = np.maximum(d, -g)
                else:
                    d = np.maximum(d, g)
        if k > 1:
            d = d.reshape(k, n).min(axis=0)
        return d

    return f
