"""Shape catalog: parameter validation, analytic local bboxes and one-line summaries.

Shared by scene.py (ops, outline) and solids.py (SDFs). Local-space conventions are in
CONTRACTS.md §3: shapes sit on y=0 and are centred in x/z, except sphere/ellipsoid/torus/capsule
which are centred at the origin. `block` occupies [0,1)^3 and is placed at floor(pos).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

SHAPE_TYPES = (
    "box", "cylinder", "sphere", "ellipsoid", "cone", "pyramid", "wedge", "prism", "torus",
    "capsule", "extrude", "revolve", "sweep", "plane_cut", "block", "line",
)

MODIFIER_TYPES: Dict[str, Dict[str, Any]] = {
    "shell": {"thickness": 1.0},
    "round": {"radius": 0.5},
    "array": {"count": 2, "offset": [1.0, 0.0, 0.0]},
    "mirror": {"axis": "x", "plane": 0.0, "keep_original": True},
    "taper": {"top_scale": 0.5},
    "twist": {"deg_per_block": 5.0},
    "noise_displace": {"amplitude": 1.0, "scale": 4.0, "seed": 0},
    "boolean": {"target": "", "op": "union"},
}

PLANE_CUT_EXTENT = 512.0


class ShapeError(ValueError):
    pass


def _f(v, name: str, positive: bool = False, nonneg: bool = False) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ShapeError(f"{name} must be a number, got {v!r}")
    if math.isnan(x) or math.isinf(x):
        raise ShapeError(f"{name} must be finite")
    if positive and x <= 0:
        raise ShapeError(f"{name} must be > 0, got {x}")
    if nonneg and x < 0:
        raise ShapeError(f"{name} must be >= 0, got {x}")
    return x


def _vec(v, name: str, n: int, positive: bool = False) -> List[float]:
    if isinstance(v, (int, float)):
        v = [v] * n
    try:
        vals = [float(x) for x in v]
    except (TypeError, ValueError):
        raise ShapeError(f"{name} must be a list of {n} numbers, got {v!r}")
    if len(vals) != n:
        raise ShapeError(f"{name} must have {n} components, got {len(vals)}")
    if positive and any(x <= 0 for x in vals):
        raise ShapeError(f"{name} components must be > 0, got {vals}")
    return vals


def _pts(v, name: str, dim: int, min_count: int) -> List[List[float]]:
    try:
        pts = [[float(c) for c in p] for p in v]
    except (TypeError, ValueError):
        raise ShapeError(f"{name} must be a list of {dim}-D points")
    for p in pts:
        if len(p) != dim:
            raise ShapeError(f"{name} points must have {dim} components, got {p}")
    if len(pts) < min_count:
        raise ShapeError(f"{name} needs at least {min_count} points, got {len(pts)}")
    return pts


SHAPE_PARAMS: Dict[str, Tuple[str, ...]] = {
    "box": ("size",),
    "cylinder": ("radius", "height", "axis", "radius_top"),
    "sphere": ("radius", "half"),
    "ellipsoid": ("radii",),
    "cone": ("radius", "height", "radius_top"),
    "pyramid": ("base", "height", "top"),
    "wedge": ("size", "slope_axis"),
    "prism": ("sides", "radius", "height"),
    "torus": ("major", "minor", "axis"),
    "capsule": ("radius", "height"),
    "extrude": ("profile", "height"),
    "revolve": ("profile",),
    "sweep": ("radius", "path", "closed", "thickness"),
    "plane_cut": ("normal", "offset"),
    "block": ("state",),
    "line": ("from", "to", "thickness"),
}
# Convenience aliases the model will reach for: width/height/depth on box-like shapes, etc.
_SIZE_ALIASES = {"width": 0, "height": 1, "depth": 2, "length": 2}


def _apply_aliases(shape: Dict[str, Any]) -> Dict[str, Any]:
    """Translate convenience keys (width/height/depth, diameter, ...) into canonical params."""
    t = shape.get("type")
    out = dict(shape)
    if t in ("box", "wedge"):
        if any(k in out for k in _SIZE_ALIASES):
            size = list(out.get("size", [1, 1, 1]))
            if len(size) != 3:
                size = [1, 1, 1]
            for k, i in _SIZE_ALIASES.items():
                if k in out:
                    size[i] = out.pop(k)
            out["size"] = size
    if t in ("cylinder", "cone", "sphere", "prism", "capsule") and "diameter" in out:
        out["radius"] = float(out.pop("diameter")) / 2.0
    if t == "ellipsoid" and "size" in out and "radii" not in out:
        out["radii"] = [float(v) / 2.0 for v in out.pop("size")]
    if t == "pyramid" and "size" in out and "base" not in out:
        sz = list(out.pop("size"))
        if len(sz) == 3:
            out["base"] = [sz[0], sz[2]]
            out.setdefault("height", sz[1])
        elif len(sz) == 2:
            out["base"] = sz
    if t == "sweep" and "thickness" in out and "radius" not in out:
        out["radius"] = float(out.pop("thickness")) / 2.0
    return out


def validate_shape(shape: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalised copy of the shape dict (defaults filled, numbers coerced, aliases applied).

    Unknown parameters raise a ShapeError naming the valid ones, so `set_shape(keep, height=10)` on a
    cylinder works while a typo never silently no-ops.
    """
    if not isinstance(shape, dict):
        raise ShapeError(f"shape must be an object like {{'type': 'box', 'size': [4,3,4]}}, got {shape!r}")
    t = shape.get("type")
    if t not in SHAPE_TYPES:
        raise ShapeError(f"unknown shape type {t!r}; expected one of {', '.join(SHAPE_TYPES)}")
    shape = _apply_aliases(shape)
    unknown = [k for k in shape if k != "type" and k not in SHAPE_PARAMS[t]]
    if unknown:
        hint = ""
        if t in ("box", "wedge"):
            hint = " (box/wedge take size=[sx,sy,sz]; width/height/depth aliases are accepted)"
        raise ShapeError(f"{t} has no parameter {unknown[0]!r}; valid: {', '.join(SHAPE_PARAMS[t])}{hint}")
    s: Dict[str, Any] = {"type": t}
    if t == "box":
        s["size"] = _vec(shape.get("size", [1, 1, 1]), "size", 3, positive=True)
    elif t == "cylinder":
        s["radius"] = _f(shape.get("radius", 1), "radius", positive=True)
        s["height"] = _f(shape.get("height", 1), "height", positive=True)
        s["axis"] = _axis(shape.get("axis", "y"))
        if shape.get("radius_top") is not None:
            s["radius_top"] = _f(shape["radius_top"], "radius_top", nonneg=True)
    elif t == "sphere":
        s["radius"] = _f(shape.get("radius", 1), "radius", positive=True)
        if shape.get("half"):
            s["half"] = True
    elif t == "ellipsoid":
        s["radii"] = _vec(shape.get("radii", [1, 1, 1]), "radii", 3, positive=True)
    elif t == "cone":
        s["radius"] = _f(shape.get("radius", 1), "radius", positive=True)
        s["height"] = _f(shape.get("height", 1), "height", positive=True)
        s["radius_top"] = _f(shape.get("radius_top", 0), "radius_top", nonneg=True)
    elif t == "pyramid":
        s["base"] = _vec(shape.get("base", [1, 1]), "base", 2, positive=True)
        s["height"] = _f(shape.get("height", 1), "height", positive=True)
        top = shape.get("top", [0, 0])
        s["top"] = [max(0.0, v) for v in _vec(top, "top", 2)]
    elif t == "wedge":
        s["size"] = _vec(shape.get("size", [1, 1, 1]), "size", 3, positive=True)
        sa = str(shape.get("slope_axis", "x"))
        if sa not in ("x", "-x", "z", "-z"):
            raise ShapeError("slope_axis must be one of x, -x, z, -z")
        s["slope_axis"] = sa
    elif t == "prism":
        s["sides"] = int(shape.get("sides", 6))
        if s["sides"] < 3:
            raise ShapeError("prism needs sides >= 3")
        s["radius"] = _f(shape.get("radius", 1), "radius", positive=True)
        s["height"] = _f(shape.get("height", 1), "height", positive=True)
    elif t == "torus":
        s["major"] = _f(shape.get("major", 2), "major", positive=True)
        s["minor"] = _f(shape.get("minor", 0.5), "minor", positive=True)
        s["axis"] = _axis(shape.get("axis", "y"))
    elif t == "capsule":
        s["radius"] = _f(shape.get("radius", 1), "radius", positive=True)
        s["height"] = _f(shape.get("height", 2), "height", positive=True)
        if s["height"] < 2 * s["radius"]:
            s["height"] = 2 * s["radius"]
    elif t == "extrude":
        s["profile"] = _pts(shape.get("profile"), "profile", 2, 3)
        s["height"] = _f(shape.get("height", 1), "height", positive=True)
    elif t == "revolve":
        s["profile"] = _pts(shape.get("profile"), "profile", 2, 3)
        if any(p[0] < 0 for p in s["profile"]):
            raise ShapeError("revolve profile points are [r, y] with r >= 0")
    elif t == "sweep":
        s["radius"] = _f(shape.get("radius", shape.get("thickness", 1)), "radius", positive=True)
        s["path"] = _pts(shape.get("path"), "path", 3, 2)
        s["closed"] = bool(shape.get("closed", False))
    elif t == "plane_cut":
        n = _vec(shape.get("normal", [0, 1, 0]), "normal", 3)
        norm = math.sqrt(sum(c * c for c in n))
        if norm == 0:
            raise ShapeError("normal must be non-zero")
        s["normal"] = [c / norm for c in n]
        s["offset"] = _f(shape.get("offset", 0), "offset")
    elif t == "block":
        st = shape.get("state")
        if not st or not isinstance(st, str):
            raise ShapeError("block needs state, e.g. 'minecraft:lantern[hanging=true]'")
        s["state"] = st if ":" in st.split("[")[0] else "minecraft:" + st
    elif t == "line":
        s["from"] = _vec(shape.get("from", [0, 0, 0]), "from", 3)
        s["to"] = _vec(shape.get("to", [1, 1, 1]), "to", 3)
        s["thickness"] = _f(shape.get("thickness", 1), "thickness", positive=True)
    return s


def _axis(a) -> str:
    a = str(a).lower()
    if a not in ("x", "y", "z"):
        raise ShapeError("axis must be x, y or z")
    return a


def shape_local_bbox(shape: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic local-space AABB (lo, hi) of a validated shape."""
    t = shape["type"]
    if t == "box" or t == "wedge":
        sx, sy, sz = shape["size"]
        return _bb(-sx / 2, 0, -sz / 2, sx / 2, sy, sz / 2)
    if t == "cylinder":
        r = max(shape["radius"], shape.get("radius_top", 0.0))
        h = shape["height"]
        return _axis_bbox(shape["axis"], r, h)
    if t == "prism":
        r, h = shape["radius"], shape["height"]
        return _bb(-r, 0, -r, r, h, r)
    if t == "cone":
        r = max(shape["radius"], shape["radius_top"])
        return _bb(-r, 0, -r, r, shape["height"], r)
    if t == "pyramid":
        sx = max(shape["base"][0], shape["top"][0])
        sz = max(shape["base"][1], shape["top"][1])
        return _bb(-sx / 2, 0, -sz / 2, sx / 2, shape["height"], sz / 2)
    if t == "sphere":
        r = shape["radius"]
        if shape.get("half"):
            return _bb(-r, 0, -r, r, r, r)
        return _bb(-r, -r, -r, r, r, r)
    if t == "ellipsoid":
        rx, ry, rz = shape["radii"]
        return _bb(-rx, -ry, -rz, rx, ry, rz)
    if t == "torus":
        R = shape["major"] + shape["minor"]
        m = shape["minor"]
        ax = shape["axis"]
        if ax == "y":
            return _bb(-R, -m, -R, R, m, R)
        if ax == "x":
            return _bb(-m, -R, -R, m, R, R)
        return _bb(-R, -R, -m, R, R, m)
    if t == "capsule":
        r, h = shape["radius"], shape["height"]
        return _bb(-r, -h / 2, -r, r, h / 2, r)
    if t == "extrude":
        p = np.asarray(shape["profile"], dtype=np.float64)
        return (np.array([p[:, 0].min(), 0.0, p[:, 1].min()]), np.array([p[:, 0].max(), shape["height"], p[:, 1].max()]))
    if t == "revolve":
        p = np.asarray(shape["profile"], dtype=np.float64)
        R = float(p[:, 0].max())
        return (np.array([-R, float(p[:, 1].min()), -R]), np.array([R, float(p[:, 1].max()), R]))
    if t == "sweep":
        p = np.asarray(shape["path"], dtype=np.float64)
        r = shape["radius"]
        return (p.min(axis=0) - r, p.max(axis=0) + r)
    if t == "plane_cut":
        e = PLANE_CUT_EXTENT
        return _bb(-e, -e, -e, e, e, e)
    if t == "block":
        return _bb(0, 0, 0, 1, 1, 1)
    if t == "line":
        a = np.asarray(shape["from"], dtype=np.float64)
        b = np.asarray(shape["to"], dtype=np.float64)
        r = shape["thickness"] / 2.0
        return (np.minimum(a, b) - r, np.maximum(a, b) + r)
    raise ShapeError(f"no bbox for shape {t}")


def _axis_bbox(axis: str, r: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
    if axis == "y":
        return _bb(-r, 0, -r, r, h, r)
    if axis == "x":
        return _bb(-h / 2, 0, -r, h / 2, 2 * r, r)
    return _bb(-r, 0, -h / 2, r, 2 * r, h / 2)


def _bb(x0, y0, z0, x1, y1, z1) -> Tuple[np.ndarray, np.ndarray]:
    return np.array([x0, y0, z0], dtype=np.float64), np.array([x1, y1, z1], dtype=np.float64)


def _n(v: float) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:g}"


def shape_summary(shape: Dict[str, Any]) -> str:
    """One-line human/LLM-readable summary used by the scene outline."""
    t = shape["type"]
    if t == "box":
        return "box " + "x".join(_n(v) for v in shape["size"])
    if t == "cylinder":
        s = f"cylinder r{_n(shape['radius'])} h{_n(shape['height'])}"
        if shape.get("radius_top") is not None:
            s += f" rtop{_n(shape['radius_top'])}"
        if shape["axis"] != "y":
            s += f" axis={shape['axis']}"
        return s
    if t == "sphere":
        return f"sphere r{_n(shape['radius'])}" + (" half" if shape.get("half") else "")
    if t == "ellipsoid":
        return "ellipsoid r" + "x".join(_n(v) for v in shape["radii"])
    if t == "cone":
        s = f"cone r{_n(shape['radius'])} h{_n(shape['height'])}"
        if shape["radius_top"]:
            s += f" rtop{_n(shape['radius_top'])}"
        return s
    if t == "pyramid":
        s = f"pyramid {_n(shape['base'][0])}x{_n(shape['base'][1])} h{_n(shape['height'])}"
        if any(shape["top"]):
            s += f" top{_n(shape['top'][0])}x{_n(shape['top'][1])}"
        return s
    if t == "wedge":
        return "wedge " + "x".join(_n(v) for v in shape["size"]) + f" slope={shape['slope_axis']}"
    if t == "prism":
        return f"prism n{shape['sides']} r{_n(shape['radius'])} h{_n(shape['height'])}"
    if t == "torus":
        s = f"torus R{_n(shape['major'])} r{_n(shape['minor'])}"
        if shape["axis"] != "y":
            s += f" axis={shape['axis']}"
        return s
    if t == "capsule":
        return f"capsule r{_n(shape['radius'])} h{_n(shape['height'])}"
    if t == "extrude":
        p = np.asarray(shape["profile"])
        w = p[:, 0].max() - p[:, 0].min()
        d = p[:, 1].max() - p[:, 1].min()
        return f"extrude {len(p)}pts plan {_n(w)}x{_n(d)} h{_n(shape['height'])}"
    if t == "revolve":
        p = np.asarray(shape["profile"])
        return f"revolve {len(p)}pts rmax{_n(p[:, 0].max())} y{_n(p[:, 1].min())}..{_n(p[:, 1].max())}"
    if t == "sweep":
        return f"sweep r{_n(shape['radius'])} path {len(shape['path'])}pts" + (" closed" if shape.get("closed") else "")
    if t == "plane_cut":
        n = shape["normal"]
        return f"plane_cut n=({_n(n[0])},{_n(n[1])},{_n(n[2])}) off={_n(shape['offset'])}"
    if t == "block":
        return f"block {shape['state']}"
    if t == "line":
        a, b = shape["from"], shape["to"]
        return f"line ({','.join(_n(v) for v in a)})->({','.join(_n(v) for v in b)}) t{_n(shape['thickness'])}"
    return t


def validate_modifier(mod: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(mod, dict) or "type" not in mod:
        raise ShapeError(f"modifier must be an object with a 'type', got {mod!r}")
    t = mod["type"]
    if t not in MODIFIER_TYPES:
        raise ShapeError(f"unknown modifier {t!r}; expected one of {', '.join(MODIFIER_TYPES)}")
    m: Dict[str, Any] = {"type": t}
    if t == "shell":
        m["thickness"] = _f(mod.get("thickness", 1), "thickness", positive=True)
    elif t == "round":
        m["radius"] = _f(mod.get("radius", 0.5), "radius", positive=True)
    elif t == "array":
        m["count"] = int(mod.get("count", 2))
        if m["count"] < 1:
            raise ShapeError("array count must be >= 1")
        m["offset"] = _vec(mod.get("offset", [1, 0, 0]), "offset", 3)
    elif t == "mirror":
        m["axis"] = _axis(mod.get("axis", "x"))
        m["plane"] = _f(mod.get("plane", 0), "plane")
        m["keep_original"] = bool(mod.get("keep_original", True))
    elif t == "taper":
        m["top_scale"] = _f(mod.get("top_scale", 0.5), "top_scale", nonneg=True)
    elif t == "twist":
        m["deg_per_block"] = _f(mod.get("deg_per_block", 5), "deg_per_block")
    elif t == "noise_displace":
        m["amplitude"] = _f(mod.get("amplitude", 1), "amplitude", nonneg=True)
        m["scale"] = _f(mod.get("scale", 4), "scale", positive=True)
        m["seed"] = int(mod.get("seed", 0))
    elif t == "boolean":
        tgt = mod.get("target")
        if not tgt or not isinstance(tgt, str):
            raise ShapeError("boolean modifier needs a target object id")
        m["target"] = tgt
        op = str(mod.get("op", "union"))
        if op not in ("union", "subtract", "intersect"):
            raise ShapeError("boolean op must be union, subtract or intersect")
        m["op"] = op
    return m


def modifier_summary(mod: Dict[str, Any]) -> str:
    t = mod["type"]
    if t == "shell":
        return f"shell({_n(mod['thickness'])})"
    if t == "round":
        return f"round({_n(mod['radius'])})"
    if t == "array":
        return f"array({mod['count']},[{','.join(_n(v) for v in mod['offset'])}])"
    if t == "mirror":
        return f"mirror({mod['axis']}@{_n(mod['plane'])})"
    if t == "taper":
        return f"taper({_n(mod['top_scale'])})"
    if t == "twist":
        return f"twist({_n(mod['deg_per_block'])})"
    if t == "noise_displace":
        return f"noise({_n(mod['amplitude'])},{_n(mod['scale'])})"
    if t == "boolean":
        return f"boolean({mod['op']}:{mod['target']})"
    return t
