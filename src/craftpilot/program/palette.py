"""Palette harmony: colour (60-30-10), context, and noisiness checks with repair.

The rules encode what builders do by eye:
- The wall (60) is a calm block: low to medium texture noise.
- The 30 group (roof, foundation, framing, secondary) is darker than the wall, or clearly a
  different hue, so the massing reads. Roofs and framing are darker than walls.
- The 10 group (trim, accent) may be saturated, but the whole palette holds at most two hue
  families plus neutrals, and at most two loud blocks.
- Every block must make sense next to the wall: it shares a style with the primary, or is generic.
"""

from __future__ import annotations

import colorsys
import re

from craftpilot.blocks import catalog
from craftpilot.blocks.palette_data import LIBRARY
from craftpilot.program.model import FamilyWeight, PaletteSpec, RolePalette

ROLES_30 = ("roof", "foundation", "secondary", "framing")
ROLES_10 = ("trim", "accent")
NEUTRAL_SAT = 0.22


def _hsl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (c / 255.0 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360.0, s, l


class _Info:
    __slots__ = ("rgb", "noise", "material", "styles")

    def __init__(self, fam):
        self.rgb, self.noise, self.material, self.styles = fam.rgb, fam.noise, fam.material, set(fam.styles)


def _fam_info(name: str):
    fam = catalog.family(name)
    if fam is None:
        return None, None
    return fam, _Info(fam)


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _dominant(rp: RolePalette | None) -> str | None:
    if rp is None or not rp.families:
        return None
    return max(rp.families, key=lambda f: f.weight).family


def _shares_style(name: str, primary_styles: set[str]) -> bool:
    """A generic wall accepts anything; a themed wall accepts generic blocks and blocks of its theme."""
    if "generic" in primary_styles:
        return True
    fam, fi = _fam_info(name)
    if fi is None:
        return True
    return "generic" in fi.styles or bool(fi.styles & primary_styles)


def _candidates(material: str | None, shapes: tuple[str, ...], styles: set[str]) -> list[str]:
    out = []
    for fam in catalog.FAMILIES.values():
        if fam.tone == "glass" or fam.loud or fam.use != catalog.USE_STRUCTURAL:
            continue
        fi = _Info(fam)
        if material and fi.material != material:
            continue
        if any(not fam.has(s) for s in shapes):
            continue
        if not ("generic" in fi.styles or fi.styles & styles):
            continue
        out.append(fam.name)
    return out


def _closest(name: str, options: list[str], want_darker_than: float | None = None, hue_of: float | None = None) -> str | None:
    _, fi = _fam_info(name)
    if fi is None or not options:
        return None
    h0, s0, l0 = _hsl(fi.rgb)
    best, best_d = None, 1e9
    for o in options:
        _, oi = _fam_info(o)
        if oi is None:
            continue
        h, s, l = _hsl(oi.rgb)
        if want_darker_than is not None and l > want_darker_than:
            continue
        d = abs(l - l0) * 2 + abs(s - s0) + (_hue_distance(h, hue_of if hue_of is not None else h0) / 180.0) * (s + s0) / 2
        if d < best_d:
            best, best_d = o, d
    return best


def check_and_repair(spec: PaletteSpec, label: str = "") -> list[str]:
    """Mutates the palette to satisfy the rules; returns human readable notes of what changed."""
    notes: list[str] = []
    prim_name = _dominant(spec.primary)
    if prim_name is None:
        return notes
    prim_fam, prim_info = _fam_info(prim_name)
    if prim_info is None:
        return notes
    ph, ps, pl = _hsl(prim_info.rgb)
    primary_styles = set(prim_info.styles)
    rustic = bool(re.search(r"rustic|ruin|cabin|hut|cottage|barn|medieval|castle|fort|old|ancient|abandoned", label.lower()))

    # 1. Primary noisiness: the dominant wall block must be calm unless the building is rustic.
    #    (Derived texture noise: concrete ~0.02, stone bricks ~0.27, planks ~0.42, cobblestone ~0.57.)
    limit = 0.65 if rustic else 0.5
    if prim_info.noise > limit:
        options = _candidates(prim_info.material, ("full",), primary_styles)
        options = [o for o in options if _Info(catalog.family(o)).noise <= limit]
        swap = _closest(prim_name, options)
        if swap and swap != prim_name:
            for fw in spec.primary.families:
                if fw.family == prim_name:
                    fw.family = swap
            notes.append(f"Primary '{prim_name}' is too busy for a wall; used '{swap}' (moved '{prim_name}' to foundation).")
            if spec.foundation is None:
                spec.foundation = RolePalette(families=[FamilyWeight(family=prim_name, weight=1.0)], weathering=0.6, texture_rate=0.3)
            prim_name = swap
            prim_fam, prim_info = _fam_info(prim_name)
            ph, ps, pl = _hsl(prim_info.rgb)
            primary_styles = set(prim_info.styles)

    # 1a. Precious and functional blocks never make up walls, roofs, foundations or framing; as an
    #     accent or trim they may stay, in small amounts.
    for role in ("primary", "roof", "foundation", "secondary", "framing"):
        rp = getattr(spec, role)
        if rp is None:
            continue
        for fw in list(rp.families):
            fam = catalog.family(fw.family)
            if fam is None or fam.use not in (catalog.USE_PRECIOUS, catalog.USE_FUNCTIONAL, catalog.USE_NATURAL):
                continue
            shapes = ("stairs", "slab") if role == "roof" else ("full",)
            options = _candidates(fam.material if fam.material != "metal" else None, shapes, primary_styles) or \
                _candidates(None, shapes, primary_styles)
            swap = _closest(fw.family, options)
            if swap:
                notes.append(f"'{fw.family}' is a {fam.use} block, not a building material; used '{swap}' for {role}.")
                fw.family = swap
            elif len(rp.families) > 1:
                rp.families.remove(fw)
    prim_name = _dominant(spec.primary) or prim_name
    prim_fam, prim_info = _fam_info(prim_name)
    if prim_info is None:
        return notes
    ph, ps, pl = _hsl(prim_info.rgb)
    primary_styles = set(prim_info.styles)

    # 1b. A mixed wall must mix kin: same material and close in colour to the dominant block.
    #     Ordered gradients and bands are deliberate (lighthouse stripes), so they are exempt.
    for fw in list(spec.primary.families) if spec.primary.gradient.value == "none" else []:
        if fw.family == prim_name:
            continue
        _, fi = _fam_info(fw.family)
        if fi is None:
            continue
        h, s, l = _hsl(fi.rgb)
        far = abs(l - pl) > 0.22 or (s > NEUTRAL_SAT and ps > NEUTRAL_SAT and _hue_distance(h, ph) > 40)
        kin = {"plaster": "painted", "terracotta": "painted"}
        same_kin = kin.get(fi.material, fi.material) == kin.get(prim_info.material, prim_info.material)
        # Different materials still mix when the colours nearly match (white concrete with calcite).
        close = abs(l - pl) <= 0.1 and (s <= NEUTRAL_SAT or ps <= NEUTRAL_SAT or _hue_distance(h, ph) <= 25)
        if far or (not same_kin and not close):
            options = [o for o in _candidates(prim_info.material, ("full",), primary_styles) if o != fw.family]
            options = [o for o in options if abs(_hsl(_Info(catalog.family(o)).rgb)[2] - pl) <= 0.22]
            swap = _closest(prim_name, options)
            if swap and swap != fw.family and swap != prim_name:
                notes.append(f"Wall mix '{fw.family}' clashes with '{prim_name}'; used '{swap}'.")
                fw.family = swap
            elif len(spec.primary.families) > 1:
                notes.append(f"Wall mix '{fw.family}' clashes with '{prim_name}' and was dropped.")
                spec.primary.families.remove(fw)

    # 1c. No foundation given: derive a rougher, slightly darker stone relative of the wall.
    if spec.foundation is None or not spec.foundation.families:
        options = [o for o in _candidates("stone", ("full",), primary_styles)
                   if _Info(catalog.family(o)).noise >= 0.4]
        options = [o for o in options if _hsl(_Info(catalog.family(o)).rgb)[2] <= pl + 0.05]
        swap = _closest(prim_name, options, want_darker_than=pl + 0.05)
        if swap:
            spec.foundation = RolePalette(families=[FamilyWeight(family=swap, weight=1.0)], weathering=0.6, texture_rate=0.3)
            notes.append(f"Foundation derived from the wall: '{swap}'.")

    # 2. Context: every family shares a style with the primary or is generic.
    for role in ROLES_30 + ROLES_10:
        rp = getattr(spec, role)
        if rp is None:
            continue
        for fw in rp.families:
            if not _shares_style(fw.family, primary_styles):
                fam, fi = _fam_info(fw.family)
                shapes = ("stairs", "slab") if role in ("roof", "trim") else ("full",)
                options = _candidates(fi.material if fi else None, shapes, primary_styles) or _candidates(None, shapes, primary_styles)
                swap = _closest(fw.family, options)
                if swap and swap != fw.family:
                    notes.append(f"'{fw.family}' does not fit a {'/'.join(sorted(primary_styles - {'generic'})) or 'generic'} building; used '{swap}' for {role}.")
                    fw.family = swap

    # 3. Lightness relationships: on a light wall, the roof and framing must not be lighter than the wall
    #    (a clearly different hue also reads). Dark walls take any roof; foundations only need to be stone.
    for role, min_wall_l, tol in (("roof", 0.45, 0.02), ("framing", 0.55, 0.05)):
        rp = getattr(spec, role)
        dom = _dominant(rp)
        if dom is None or pl < min_wall_l:
            continue
        fam, fi = _fam_info(dom)
        if fi is None:
            continue
        h, s, l = _hsl(fi.rgb)
        different_hue = s > NEUTRAL_SAT and ps > NEUTRAL_SAT and _hue_distance(h, ph) > 60
        if l > pl + tol and not different_hue:
            shapes = ("stairs", "slab") if role == "roof" else ("full",)
            options = _candidates(fi.material, shapes, primary_styles) or _candidates(None, shapes, primary_styles)
            swap = _closest(dom, options, want_darker_than=pl - 0.05, hue_of=h)
            if swap and swap != dom:
                for fw in rp.families:
                    if fw.family == dom:
                        fw.family = swap
                notes.append(f"{role.capitalize()} '{dom}' is not darker than the wall; used '{swap}'.")

    # 4. Hue discipline: at most two saturated hue groups; extras become neutrals of the same material.
    saturated: list[tuple[str, str, float]] = []
    for role in ("primary",) + ROLES_30 + ROLES_10:
        rp = getattr(spec, role)
        if rp is None:
            continue
        for fw in rp.families:
            _, fi = _fam_info(fw.family)
            if fi is None:
                continue
            h, s, l = _hsl(fi.rgb)
            if s > NEUTRAL_SAT:
                saturated.append((role, fw.family, h))
    groups: list[float] = []
    for role, name, h in saturated:
        if not any(_hue_distance(h, g) <= 35 for g in groups):
            groups.append(h)
    if len(groups) > 2:
        keep = groups[:2]
        for role in ROLES_10 + ROLES_30[::-1]:
            rp = getattr(spec, role)
            if rp is None:
                continue
            for fw in rp.families:
                _, fi = _fam_info(fw.family)
                if fi is None:
                    continue
                h, s, l = _hsl(fi.rgb)
                if s > NEUTRAL_SAT and not any(_hue_distance(h, g) <= 35 for g in keep):
                    shapes = ("stairs", "slab") if role in ("roof", "trim") else ("full",)
                    options = [o for o in _candidates(fi.material, shapes, primary_styles)
                               if _hsl(_Info(catalog.family(o)).rgb)[1] <= NEUTRAL_SAT]
                    swap = _closest(fw.family, options)
                    if swap:
                        notes.append(f"Too many colours: '{fw.family}' in {role} replaced by neutral '{swap}'.")
                        fw.family = swap

    # 5. Loud blocks: at most two, and never as the primary of a non-fantasy building.
    loud = [(role, fw) for role in ("primary",) + ROLES_30 + ROLES_10 if getattr(spec, role) is not None
            for fw in getattr(spec, role).families if catalog.family(fw.family) and catalog.family(fw.family).loud]
    for role, fw in loud[2:]:
        fam, fi = _fam_info(fw.family)
        options = _candidates(fi.material, ("full",), primary_styles)
        swap = _closest(fw.family, options)
        if swap:
            notes.append(f"Too many loud blocks: '{fw.family}' in {role} replaced by '{swap}'.")
            fw.family = swap
    return notes


def library_score(text: str, name: str) -> int:
    """How well a curated palette's tags fit a description. Building-type words outrank
    feature words such as "glass" or "roof" (see exemplars.GENERIC)."""
    from craftpilot.program.exemplars import tag_score

    words = set(re.findall(r"[a-z]+", text.lower()))
    entry = LIBRARY[name]
    return tag_score(words, set(entry["tags"].split())) + (2 if name in text.lower() else 0)


def library_match(text: str, min_score: int = 1) -> tuple[str, dict] | None:
    """Best curated palette for a description, by weighted tag overlap."""
    best, best_score = None, min_score - 1
    for name in LIBRARY:
        score = library_score(text, name)
        if score > best_score:
            best, best_score = name, score
    return (best, LIBRARY[best]) if best else None


def library_palette(name: str) -> PaletteSpec | None:
    entry = LIBRARY.get(name)
    if entry is None:
        return None

    def rp(key: str, **kw) -> RolePalette | None:
        fams = [f for f in entry.get(key, []) if catalog.family(f)]
        if not fams:
            return None
        n = len(fams)
        weights = [1.0] + [0.5] * (n - 1)
        return RolePalette(families=[FamilyWeight(family=f, weight=w) for f, w in zip(fams, weights)], **kw)

    return PaletteSpec(
        primary=rp("primary", texture_rate=0.15, weathering=0.3),
        roof=rp("roof", texture_rate=0.05),
        secondary=rp("secondary", texture_rate=0.2, weathering=0.4),
        accent=rp("accent"),
        framing=rp("framing"),
        trim=rp("trim"),
        foundation=rp("foundation", texture_rate=0.3, weathering=0.7),
        glass=None,
    )


def library_summary() -> str:
    lines = []
    for name, e in LIBRARY.items():
        lines.append(f"- {name}: wall {'/'.join(e['primary'])}, roof {'/'.join(e['roof'])}, framing {'/'.join(e.get('framing', []))}, "
                     f"accent {'/'.join(e.get('accent', []))}  ({e['tags'].split()[0]}, {e['tags'].split()[1]})")
    return "\n".join(lines)
