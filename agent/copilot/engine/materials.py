"""Materials are rules, not blocks (plan.md §4.2, CONTRACTS.md §6).

A material maps a voxel to a *block id* using a weighted palette chosen with coherent noise, an
optional gradient (a second palette between two coordinates), optional per-face overrides, and a
`fit` mode that tells fit.py which sub-voxel forms (stairs/slab/wall) the resolver may substitute.
"""
from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .coretypes import normalize_id, short_id
from .registry import Registry, default_registry

FIT_MODES = ("none", "slab", "stairs", "stairs+slab", "walls")
FACE_KEYS = ("top", "side", "bottom")
BLEND_WIDTH = 1.5  # blocks of noisy blending at a gradient boundary


class MaterialError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------
# Presets: materials, not templates. Any of these can be referenced by name or copied and tweaked.
# ----------------------------------------------------------------------------------------------
def _p(base: str, palette: Sequence[Tuple[str, float]], gradient: Optional[dict] = None, fit: str = "stairs+slab",
       faces: Optional[dict] = None, noise_scale: float = 3.0) -> dict:
    d: Dict[str, Any] = {"base": base, "palette": [[b, w] for b, w in palette], "fit": fit, "noise": {"scale": noise_scale, "seed": 0}}
    if gradient:
        d["gradient"] = gradient
    if faces:
        d["faces"] = faces
    return d


def _g(frm: float, to: float, palette: Sequence[Tuple[str, float]], axis: str = "y") -> dict:
    return {"axis": axis, "from": frm, "to": to, "palette": [[b, w] for b, w in palette]}


PRESETS: Dict[str, dict] = {
    "default": _p("stone", [("stone", 1.0)]),
    "medieval_stone": _p("stone_bricks", [("stone_bricks", 0.62), ("cracked_stone_bricks", 0.16), ("mossy_stone_bricks", 0.10), ("cobblestone", 0.12)],
                         _g(0, 3, [("cobblestone", 0.5), ("mossy_cobblestone", 0.5)])),
    "castle_grey": _p("stone_bricks", [("stone_bricks", 0.5), ("andesite", 0.25), ("cobblestone", 0.15), ("polished_andesite", 0.10)],
                      _g(0, 2, [("cobblestone", 0.7), ("mossy_cobblestone", 0.3)])),
    "cobble_rustic": _p("cobblestone", [("cobblestone", 0.7), ("mossy_cobblestone", 0.2), ("stone", 0.1)]),
    "mossy_ruin": _p("mossy_stone_bricks", [("mossy_stone_bricks", 0.4), ("stone_bricks", 0.3), ("cracked_stone_bricks", 0.2), ("mossy_cobblestone", 0.1)],
                     _g(0, 2, [("mossy_cobblestone", 0.8), ("cobblestone", 0.2)])),
    "andesite_grey": _p("polished_andesite", [("polished_andesite", 0.7), ("andesite", 0.3)]),
    "diorite_white": _p("polished_diorite", [("polished_diorite", 0.7), ("diorite", 0.3)]),
    "granite_warm": _p("polished_granite", [("polished_granite", 0.7), ("granite", 0.3)]),
    "tuff_grey": _p("tuff_bricks", [("tuff_bricks", 0.7), ("polished_tuff", 0.2), ("chiseled_tuff", 0.1)]),
    "smooth_stone_modern": _p("smooth_stone", [("smooth_stone", 0.85), ("stone", 0.15)], fit="slab"),
    "spruce_timber": _p("spruce_planks", [("spruce_planks", 0.8), ("stripped_spruce_log", 0.2)]),
    "oak_timber": _p("oak_planks", [("oak_planks", 0.8), ("stripped_oak_log", 0.2)]),
    "dark_oak_timber": _p("dark_oak_planks", [("dark_oak_planks", 0.85), ("stripped_dark_oak_log", 0.15)]),
    "birch_light": _p("birch_planks", [("birch_planks", 0.9), ("stripped_birch_log", 0.1)]),
    "cherry_pink": _p("cherry_planks", [("cherry_planks", 0.8), ("stripped_cherry_log", 0.2)]),
    "bamboo_zen": _p("bamboo_planks", [("bamboo_planks", 0.6), ("bamboo_mosaic", 0.4)]),
    "acacia_orange": _p("acacia_planks", [("acacia_planks", 0.85), ("stripped_acacia_log", 0.15)]),
    "sandstone_desert": _p("sandstone", [("sandstone", 0.6), ("smooth_sandstone", 0.25), ("cut_sandstone", 0.15)],
                           _g(0, 2, [("sandstone", 0.5), ("cut_sandstone", 0.5)])),
    "red_sandstone": _p("red_sandstone", [("red_sandstone", 0.6), ("smooth_red_sandstone", 0.25), ("cut_red_sandstone", 0.15)]),
    "blackstone_dark": _p("polished_blackstone_bricks", [("polished_blackstone_bricks", 0.6), ("polished_blackstone", 0.2), ("cracked_polished_blackstone_bricks", 0.12), ("blackstone", 0.08)]),
    "deepslate_dark": _p("deepslate_bricks", [("deepslate_bricks", 0.6), ("deepslate_tiles", 0.2), ("cracked_deepslate_bricks", 0.1), ("polished_deepslate", 0.1)],
                         _g(0, 3, [("cobbled_deepslate", 0.6), ("deepslate_bricks", 0.4)])),
    "quartz_modern": _p("quartz_block", [("quartz_block", 0.7), ("smooth_quartz", 0.3)]),
    "white_concrete_modern": _p("white_concrete", [("white_concrete", 0.8), ("light_gray_concrete", 0.2)], fit="slab"),
    "copper_roof": _p("cut_copper", [("cut_copper", 0.7), ("exposed_cut_copper", 0.3)]),
    "oxidized_copper_roof": _p("oxidized_cut_copper", [("oxidized_cut_copper", 0.7), ("weathered_cut_copper", 0.3)]),
    "slate_roof": _p("deepslate_tiles", [("deepslate_tiles", 0.85), ("cracked_deepslate_tiles", 0.15)]),
    "terracotta_roof": _p("bricks", [("bricks", 0.75), ("red_terracotta", 0.15), ("orange_terracotta", 0.10)]),
    "thatch_roof": _p("oak_planks", [("hay_block", 0.8), ("oak_planks", 0.2)]),
    "shingle_dark": _p("dark_oak_planks", [("dark_oak_planks", 0.8), ("spruce_planks", 0.2)]),
    "brick_red": _p("bricks", [("bricks", 0.85), ("red_terracotta", 0.15)]),
    "mud_brick": _p("mud_bricks", [("mud_bricks", 0.85), ("packed_mud", 0.15)]),
    "nether_gothic": _p("nether_bricks", [("nether_bricks", 0.7), ("red_nether_bricks", 0.15), ("cracked_nether_bricks", 0.15)]),
    "prismarine_sea": _p("prismarine_bricks", [("prismarine_bricks", 0.6), ("prismarine", 0.3), ("dark_prismarine", 0.1)]),
    "end_stone_pale": _p("end_stone_bricks", [("end_stone_bricks", 0.8), ("end_stone", 0.2)]),
    "purpur_violet": _p("purpur_block", [("purpur_block", 0.8), ("purpur_pillar", 0.2)]),
    # T4: ten presets tuned on a test wall in the renderer (walls with grounding, roofs, trims)
    "limestone_pale": _p("smooth_sandstone", [("smooth_sandstone", 0.55), ("sandstone", 0.25), ("calcite", 0.12), ("cut_sandstone", 0.08)],
                         _g(0, 2, [("cut_sandstone", 0.6), ("sandstone", 0.4)])),
    "tudor_plaster": _p("mushroom_stem", [("mushroom_stem", 0.65), ("white_concrete_powder", 0.2), ("light_gray_wool", 0.15)],
                        _g(0, 2, [("cobblestone", 0.6), ("stone_bricks", 0.4)]), fit="none"),
    "dark_slate_wall": _p("deepslate_tiles", [("deepslate_tiles", 0.5), ("deepslate_bricks", 0.3), ("polished_deepslate", 0.12), ("cracked_deepslate_tiles", 0.08)],
                          _g(0, 3, [("cobbled_deepslate", 0.7), ("deepslate_bricks", 0.3)])),
    "red_brick_victorian": _p("bricks", [("bricks", 0.78), ("red_terracotta", 0.08), ("granite", 0.07), ("polished_granite", 0.07)],
                              _g(0, 2, [("stone_bricks", 0.6), ("cobblestone", 0.4)])),
    "weathered_wood": _p("spruce_planks", [("spruce_planks", 0.5), ("dark_oak_planks", 0.3), ("stripped_spruce_log", 0.2)],
                         _g(0, 1, [("dark_oak_planks", 0.7), ("stripped_dark_oak_log", 0.3)])),
    "turf_roof": _p("moss_block", [("moss_block", 0.7), ("grass_block", 0.2), ("mossy_cobblestone", 0.1)], fit="none"),
    "spruce_shingle_roof": _p("spruce_planks", [("spruce_planks", 0.7), ("dark_oak_planks", 0.2), ("stripped_spruce_log", 0.1)]),
    "stone_trim_light": _p("polished_andesite", [("polished_andesite", 0.55), ("smooth_stone", 0.3), ("polished_diorite", 0.15)], fit="slab"),
    "quartz_trim": _p("smooth_quartz", [("smooth_quartz", 0.7), ("quartz_bricks", 0.3)], fit="slab"),
    "sandstone_trim": _p("chiseled_sandstone", [("chiseled_sandstone", 0.6), ("cut_sandstone", 0.4)], fit="slab"),
    "glass_clear": _p("glass", [("glass", 1.0)], fit="none"),
    "glass_tinted": _p("gray_stained_glass", [("gray_stained_glass", 0.7), ("light_gray_stained_glass", 0.3)], fit="none"),
    "stone_path": _p("cobblestone", [("gravel", 0.4), ("cobblestone", 0.3), ("andesite", 0.2), ("stone", 0.1)], fit="none"),
    "gravel_path": _p("gravel", [("gravel", 0.7), ("coarse_dirt", 0.3)], fit="none"),
    "grass_ground": _p("grass_block", [("grass_block", 1.0)], fit="none"),
    "dirt_ground": _p("dirt", [("dirt", 0.7), ("coarse_dirt", 0.3)], fit="none"),
    "snow_white": _p("snow_block", [("snow_block", 1.0)], fit="none"),
    "ice_blue": _p("packed_ice", [("packed_ice", 0.8), ("blue_ice", 0.2)], fit="none"),
    "oak_leaves": _p("oak_leaves", [("oak_leaves", 1.0)], fit="none"),
    "spruce_leaves": _p("spruce_leaves", [("spruce_leaves", 1.0)], fit="none"),
    "water": _p("water", [("water", 1.0)], fit="none"),
    "lava": _p("lava", [("lava", 1.0)], fit="none"),
}


# ----------------------------------------------------------------------------------------------
# Spec validation / dataclass
# ----------------------------------------------------------------------------------------------
def _norm_palette(pal: Any, registry: Registry, what: str) -> List[List[Any]]:
    if isinstance(pal, str):
        pal = [[pal, 1.0]]
    if not isinstance(pal, (list, tuple)) or not pal:
        raise MaterialError(f"{what} must be a non-empty list like [['stone_bricks', 0.7], ['cobblestone', 0.3]]")
    out: List[List[Any]] = []
    for entry in pal:
        bid: Any
        w: Any
        if isinstance(entry, str):
            bid, w = entry, 1.0
        elif isinstance(entry, dict):
            bid, w = entry.get("block") or entry.get("id"), entry.get("weight", 1.0)
        elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
            bid = entry[0]
            w = entry[1] if len(entry) > 1 else 1.0
        else:
            raise MaterialError(f"{what} entry {entry!r} must be [block_id, weight]")
        if not isinstance(bid, str):
            raise MaterialError(f"{what} entry {entry!r}: block id must be a string")
        bid = normalize_id(bid.split("[")[0])
        if not registry.has(bid):
            sugg = registry.search(short_id(bid), limit=3)
            hint = f" (did you mean {', '.join(short_id(s) for s in sugg)}?)" if sugg else ""
            raise MaterialError(f"{what}: unknown block {short_id(bid)!r}{hint}")
        try:
            w = float(w)
        except (TypeError, ValueError):
            raise MaterialError(f"{what} entry {bid}: weight must be a number")
        if w <= 0:
            raise MaterialError(f"{what} entry {bid}: weight must be > 0")
        out.append([bid, w])
    total = sum(w for _, w in out)
    return [[b, round(w / total, 6)] for b, w in out]


def validate_material_spec(spec: Dict[str, Any], registry: Optional[Registry] = None) -> Dict[str, Any]:
    """Validate and normalise a material spec dict (raises MaterialError with an actionable message).

    Fills defaults: fit="stairs+slab", noise={"scale":3,"seed":0}, palette=[[base,1]] if missing,
    base=first palette entry if missing. Block ids are normalised to `minecraft:` form and checked
    against the registry. A `preset` key merges that preset first, then the given overrides.
    """
    registry = registry or default_registry()
    if not isinstance(spec, dict):
        raise MaterialError("material spec must be an object like {'base': 'stone_bricks', 'palette': [...]}")
    spec = _copy.deepcopy(spec)
    if "preset" in spec:
        pname = spec.pop("preset")
        if pname not in PRESETS:
            raise MaterialError(f"unknown preset {pname!r}; presets: {', '.join(sorted(PRESETS))}")
        merged = _copy.deepcopy(PRESETS[pname])
        merged.update(spec)
        spec = merged
    out: Dict[str, Any] = {}
    palette = spec.get("palette")
    base = spec.get("base")
    if base is None and palette:
        first = palette[0]
        base = first if isinstance(first, str) else (first.get("block") if isinstance(first, dict) else first[0])
    if base is None:
        raise MaterialError("material spec needs a 'base' block id (the full block used for stairs/slab fitting)")
    if not isinstance(base, str):
        raise MaterialError("'base' must be a block id string")
    base = normalize_id(base.split("[")[0])
    if not registry.has(base):
        sugg = registry.search(short_id(base), limit=3)
        hint = f" (did you mean {', '.join(short_id(s) for s in sugg)}?)" if sugg else ""
        raise MaterialError(f"unknown base block {short_id(base)!r}{hint}")
    out["base"] = base
    out["palette"] = _norm_palette(palette if palette else [[base, 1.0]], registry, "palette")
    grad = spec.get("gradient")
    if grad:
        if not isinstance(grad, dict):
            raise MaterialError("gradient must be an object {axis, from, to, palette}")
        axis = str(grad.get("axis", "y")).lower()
        if axis not in ("x", "y", "z"):
            raise MaterialError("gradient axis must be x, y or z")
        try:
            frm, to = float(grad.get("from", 0)), float(grad.get("to", 0))
        except (TypeError, ValueError):
            raise MaterialError("gradient from/to must be numbers")
        if to <= frm:
            raise MaterialError(f"gradient needs to > from (got from={frm}, to={to})")
        if "palette" not in grad:
            raise MaterialError("gradient needs its own palette")
        out["gradient"] = {"axis": axis, "from": frm, "to": to, "palette": _norm_palette(grad["palette"], registry, "gradient palette")}
    faces = spec.get("faces")
    if faces:
        if not isinstance(faces, dict):
            raise MaterialError("faces must be an object like {'top': 'stone_brick_slab'}")
        fo: Dict[str, Optional[str]] = {}
        for k, v in faces.items():
            if k not in FACE_KEYS:
                raise MaterialError(f"faces key {k!r} must be one of {FACE_KEYS}")
            if v is None:
                fo[k] = None
                continue
            if not isinstance(v, str):
                raise MaterialError(f"faces.{k} must be a block id or null")
            v = normalize_id(v.split("[")[0])
            if not registry.has(v):
                raise MaterialError(f"faces.{k}: unknown block {short_id(v)!r}")
            fo[k] = v
        if any(fo.values()):
            out["faces"] = fo
    fit = str(spec.get("fit", "stairs+slab")).lower().replace(" ", "")
    if fit == "slab+stairs":
        fit = "stairs+slab"
    if fit not in FIT_MODES:
        raise MaterialError(f"fit must be one of {FIT_MODES}")
    out["fit"] = fit
    noise = spec.get("noise") or {}
    if not isinstance(noise, dict):
        raise MaterialError("noise must be an object {scale, seed}")
    try:
        scale = float(noise.get("scale", 3.0))
        seed = int(noise.get("seed", 0))
    except (TypeError, ValueError):
        raise MaterialError("noise.scale must be a number and noise.seed an integer")
    if scale <= 0:
        raise MaterialError("noise.scale must be > 0")
    out["noise"] = {"scale": scale, "seed": seed}
    for extra in ("name", "description"):
        if extra in spec:
            out[extra] = spec[extra]
    return out


@dataclass
class MaterialSpec:
    name: str
    base: str
    palette: List[Tuple[str, float]]
    gradient: Optional[Dict[str, Any]] = None
    faces: Dict[str, Optional[str]] = field(default_factory=dict)
    fit: str = "stairs+slab"
    noise: Dict[str, Any] = field(default_factory=lambda: {"scale": 3.0, "seed": 0})

    @staticmethod
    def from_dict(d: Dict[str, Any], name: str = "material", registry: Optional[Registry] = None) -> "MaterialSpec":
        """Build from a (possibly un-normalised) spec dict; validates against the registry."""
        n = validate_material_spec(d, registry)
        return MaterialSpec(
            name=n.get("name", name) if isinstance(n.get("name"), str) else name,
            base=n["base"],
            palette=[(b, w) for b, w in n["palette"]],
            gradient=n.get("gradient"),
            faces=dict(n.get("faces", {})),
            fit=n["fit"],
            noise=dict(n["noise"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"base": self.base, "palette": [[b, w] for b, w in self.palette], "fit": self.fit, "noise": dict(self.noise)}
        if self.gradient:
            d["gradient"] = _copy.deepcopy(self.gradient)
        if self.faces:
            d["faces"] = dict(self.faces)
        return d

    def palette_ids(self) -> List[str]:
        return [b for b, _ in self.palette]

    def weights(self) -> np.ndarray:
        return np.asarray([w for _, w in self.palette], dtype=np.float64)

    def gradient_ids(self) -> List[str]:
        return [b for b, _ in self.gradient["palette"]] if self.gradient else []

    def all_block_ids(self) -> List[str]:
        ids = [self.base] + self.palette_ids() + self.gradient_ids() + [v for v in self.faces.values() if v]
        seen: List[str] = []
        for i in ids:
            if i not in seen:
                seen.append(i)
        return seen


def resolve_material(name_or_dict: Any, scene_materials: Dict[str, Any], warnings: Optional[List[str]] = None,
                     registry: Optional[Registry] = None) -> MaterialSpec:
    """Resolve a material reference: scene table first, then presets, then an inline dict.

    Unknown names fall back to the `default` preset and append a warning to `warnings`.
    """
    registry = registry or default_registry()
    if isinstance(name_or_dict, dict):
        try:
            return MaterialSpec.from_dict(name_or_dict, name=str(name_or_dict.get("name", "inline")), registry=registry)
        except MaterialError as e:
            if warnings is not None:
                warnings.append(f"inline material invalid ({e}); using default")
            return MaterialSpec.from_dict(PRESETS["default"], name="default", registry=registry)
    name = str(name_or_dict or "default")
    if name in scene_materials:
        try:
            return MaterialSpec.from_dict(scene_materials[name], name=name, registry=registry)
        except MaterialError as e:
            if warnings is not None:
                warnings.append(f"material {name!r} invalid ({e}); using default")
            return MaterialSpec.from_dict(PRESETS["default"], name=name, registry=registry)
    if name in PRESETS:
        return MaterialSpec.from_dict(PRESETS[name], name=name, registry=registry)
    # allow a bare block id as a material ("stone_bricks")
    if registry.has(name):
        return MaterialSpec.from_dict({"base": name, "palette": [[name, 1.0]]}, name=name, registry=registry)
    if warnings is not None:
        warnings.append(f"unknown material {name!r}; using default (stone). Define it with define_material or use a preset.")
    return MaterialSpec.from_dict(PRESETS["default"], name=name, registry=registry)


def material_fit_modes(scene: Any, registry: Optional[Registry] = None) -> Dict[str, str]:
    """fit mode per material name used in the scene (for fit.py)."""
    out: Dict[str, str] = {}
    mats = getattr(scene, "materials", {}) or {}
    names = set(mats)
    for o in getattr(scene, "objects", []):
        mn = o.material_name() if hasattr(o, "material_name") else (o.get("material") if isinstance(o, dict) else None)
        if mn:
            names.add(mn)
    for n in names:
        out[n] = resolve_material(n, mats, registry=registry).fit
    return out


# ----------------------------------------------------------------------------------------------
# Coherent noise + palette selection
# ----------------------------------------------------------------------------------------------
def _hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic lattice hash -> float in [0,1)."""
    h = (ix.astype(np.uint32) * np.uint32(374761393)) ^ (iy.astype(np.uint32) * np.uint32(668265263)) ^ (iz.astype(np.uint32) * np.uint32(2246822519)) ^ np.uint32(seed * 1013904223 & 0xFFFFFFFF)
    h ^= h >> np.uint32(13)
    h *= np.uint32(1274126177)
    h ^= h >> np.uint32(16)
    return h.astype(np.float64) / 4294967296.0


def _value_noise(pts: np.ndarray, scale: float, seed: int) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64) / float(scale)
    i = np.floor(p)
    f = p - i
    u = f * f * (3.0 - 2.0 * f)
    i = i.astype(np.int64)
    ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
    out = np.zeros(len(p), dtype=np.float64)
    for dx in (0, 1):
        wx = u[:, 0] if dx else 1.0 - u[:, 0]
        for dy in (0, 1):
            wy = u[:, 1] if dy else 1.0 - u[:, 1]
            for dz in (0, 1):
                wz = u[:, 2] if dz else 1.0 - u[:, 2]
                out += wx * wy * wz * _hash01(ix + dx, iy + dy, iz + dz, seed)
    return out


def noise3(pts: np.ndarray, scale: float = 3.0, seed: int = 0, octaves: int = 2) -> np.ndarray:
    """Coherent value noise in [0,1) at world points [N,3]. Private fallback; prefers Track 3's
    `copilot.engine.noise.noise3` when it exists."""
    try:  # Track 3 owns engine/noise.py; use it when present so materials and noise_displace agree
        from . import noise as _n  # type: ignore

        fn = getattr(_n, "noise3", None)
        if fn is not None and fn is not noise3:
            v = np.asarray(fn(np.asarray(pts, dtype=np.float64), scale, seed), dtype=np.float64)
            if v.size and v.min() < 0:  # Track 3's noise3 is in [-1, 1]; materials want [0, 1)
                v = (v + 1.0) / 2.0
            return np.clip(v, 0.0, 1.0 - 1e-9)
    except Exception:  # noqa: BLE001
        pass
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0)
    total = np.zeros(len(pts))
    amp, norm = 1.0, 0.0
    for o in range(max(1, octaves)):
        total += amp * _value_noise(pts + 17.3 * o, scale / (2 ** o), seed + 101 * o)
        norm += amp
        amp *= 0.5
    return total / norm


def _equalize(v: np.ndarray) -> np.ndarray:
    """Rank-transform to a uniform [0,1) distribution (monotonic, so coherence is preserved)."""
    n = len(v)
    if n == 0:
        return v
    order = np.argsort(v, kind="stable")
    u = np.empty(n, dtype=np.float64)
    u[order] = (np.arange(n) + 0.5) / n
    return u


def pick_palette_entry(spec: MaterialSpec, pts: np.ndarray, seed: int = 0, palette: Optional[Sequence[Tuple[str, float]]] = None) -> np.ndarray:
    """Index into `spec.palette` (or `palette`) for each world point [N,3], using coherent noise so
    variation clumps like weathering rather than white noise. Weights are respected in aggregate."""
    pal = list(palette if palette is not None else spec.palette)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if len(pal) == 1:
        return np.zeros(n, dtype=np.int64)
    w = np.asarray([x[1] for x in pal], dtype=np.float64)
    w = w / w.sum()
    cum = np.cumsum(w)
    cum[-1] = 1.0 + 1e-9
    u = _equalize(noise3(pts, float(spec.noise.get("scale", 3.0)), int(spec.noise.get("seed", 0)) + seed))
    return np.searchsorted(cum, u, side="right").clip(0, len(pal) - 1)


def choose_blocks(spec: MaterialSpec, pts: np.ndarray, seed: int = 0,
                  top_exposed: Optional[np.ndarray] = None, side_exposed: Optional[np.ndarray] = None,
                  bottom_exposed: Optional[np.ndarray] = None) -> np.ndarray:
    """Full material evaluation: palette + gradient + face overrides -> block id per point (object array).

    `pts` are voxel centres in scene coordinates. Exposure masks are optional booleans [N].
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    ids = np.asarray(spec.palette_ids(), dtype=object)
    chosen = ids[pick_palette_entry(spec, pts, seed)]
    if spec.gradient:
        g = spec.gradient
        ax = "xyz".index(g["axis"])
        c = pts[:, ax]
        frm, to = float(g["from"]), float(g["to"])
        # w = 1 inside [from, to]; the noisy blend extends OUTWARD by BLEND_WIDTH so the band itself is
        # never eroded (a gradient from 0 guarantees a full ground row)
        w = np.where(c < frm, 1.0 - (frm - c) / BLEND_WIDTH, np.where(c > to, 1.0 - (c - to) / BLEND_WIDTH, 1.0))
        w = np.clip(w, 0.0, 1.0)
        u = _equalize(noise3(pts, 2.0, int(spec.noise.get("seed", 0)) + seed + 7))
        in_grad = u < w
        if in_grad.any():
            gpal = [(b, float(wt)) for b, wt in g["palette"]]
            gids = np.asarray([b for b, _ in gpal], dtype=object)
            sub = pick_palette_entry(spec, pts[in_grad], seed + 13, palette=gpal)
            chosen[in_grad] = gids[sub]
    if spec.faces:
        for key, mask in (("side", side_exposed), ("bottom", bottom_exposed), ("top", top_exposed)):
            ov = spec.faces.get(key)
            if ov and mask is not None:
                m = np.asarray(mask, dtype=bool).reshape(-1)
                if m.shape[0] == n and m.any():
                    chosen[m] = ov
    return chosen


def preset_names() -> List[str]:
    return sorted(PRESETS)


def presets_text() -> str:
    """One line per preset for the prompt."""
    lines = []
    for name in sorted(PRESETS):
        p = PRESETS[name]
        pal = "/".join(short_id(b) for b, _ in p["palette"])
        g = ""
        if p.get("gradient"):
            gg = p["gradient"]
            g = f" base({gg['from']:g}..{gg['to']:g}: {'/'.join(short_id(b) for b, _ in gg['palette'])})"
        lines.append(f"{name}: {pal}{g} fit={p['fit']}")
    return "\n".join(lines)
