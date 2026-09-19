"""Semantic validation and repair of a BuildProgram. Never fails: repairs and reports."""

from __future__ import annotations

from craftpilot.blocks import catalog
from craftpilot.program.model import (
    Attach,
    Bounds,
    BuildProgram,
    FamilyWeight,
    PartSpec,
    RolePalette,
    RoofSpec,
    RoofType,
    Shape,
    Side,
    Size,
)

DEFAULT_ROOF_FAMILY = "dark_oak"
DEFAULT_PRIMARY_FAMILY = "oak"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# A tone name, or a loose material word, given where a family was expected.
_ALIASES = {
    "dark_wood": "dark_oak", "light_wood": "oak", "warm_wood": "spruce", "pink_wood": "cherry",
    "grey_stone": "stone_bricks", "gray_stone": "stone_bricks", "dark_stone": "deepslate_bricks",
    "warm_stone": "bricks", "sand": "sandstone", "white": "quartz", "metal": "iron", "copper_block": "copper",
    "brick": "bricks", "stone_brick": "stone_bricks", "wood": "oak", "stone": "stone_bricks", "concrete": "white_concrete",
    "glass_pane": "glass", "planks": "oak", "log": "oak", "logs": "oak", "cobble": "cobblestone",
    "deepslate": "deepslate_bricks", "terracotta": "terracotta", "dark_oak_planks": "dark_oak", "oak_planks": "oak",
    "spruce_planks": "spruce", "stone_bricks_family": "stone_bricks",
}


def _resolve_family(name: str) -> str | None:
    n = name.strip().lower().replace("minecraft:", "").replace(" ", "_")
    if catalog.family(n):
        return n
    if n in _ALIASES and catalog.family(_ALIASES[n]):
        return _ALIASES[n]
    for suffix in ("_planks", "_block", "_bricks_family", "_log"):
        if n.endswith(suffix) and catalog.family(n[: -len(suffix)]):
            return n[: -len(suffix)]
    # A tone name: pick the first quiet family of that tone.
    for fam in catalog.FAMILIES.values():
        if fam.tone == n and not fam.loud:
            return fam.name
    return None


def _repair_palette(rp: RolePalette | None, role: str, notes: list[str], required_shapes: tuple[str, ...] = ()) -> RolePalette | None:
    if rp is None:
        return None
    kept: list[FamilyWeight] = []
    for fw in rp.families:
        resolved = _resolve_family(fw.family)
        if resolved is None:
            notes.append(f"Unknown block family '{fw.family}' in {role} palette was dropped.")
            continue
        if resolved != fw.family:
            notes.append(f"Block family '{fw.family}' in {role} palette was read as '{resolved}'.")
            fw.family = resolved
        fam = catalog.family(resolved)
        if role != "glass" and fam.tone == "glass":
            notes.append(f"Glass ('{fw.family}') is only for windows; dropped from the {role} palette.")
            continue
        if fw.weight <= 0:
            continue
        kept.append(FamilyWeight(family=fw.family, weight=float(fw.weight)))
    if not kept:
        return None
    rp.families = kept
    rp.weathering = _clamp(rp.weathering, 0.0, 1.0)
    rp.texture_rate = _clamp(rp.texture_rate, 0.0, 0.6)
    return rp


def repair(program: BuildProgram, safety_limit: tuple[int, int, int]) -> tuple[BuildProgram, list[str]]:
    notes: list[str] = []

    # Parts: at least one, unique names, one root, valid attach targets.
    if not program.parts:
        program.parts = [PartSpec(name="main", size=Size(width=1.0, depth=1.0), roof=RoofSpec(type=RoofType.gable))]
        notes.append("No parts given; a single gable-roofed block was used.")
    seen: set[str] = set()
    for i, p in enumerate(program.parts):
        if p.name in seen or not p.name:
            p.name = f"{p.name or 'part'}_{i}"
        seen.add(p.name)
    roots = [p for p in program.parts if p.attach is None]
    if not roots:
        program.parts[0].attach = None
        notes.append(f"No root part; '{program.parts[0].name}' was made the root.")
    elif len(roots) > 1:
        root = roots[0]
        for p in roots[1:]:
            p.attach = Attach(to=root.name, side=Side.east)
            notes.append(f"Extra root '{p.name}' was attached to '{root.name}'.")
    names = {p.name for p in program.parts}
    root = program.root()
    for p in program.parts:
        if p.attach is not None and (p.attach.to not in names or p.attach.to == p.name):
            p.attach.to = root.name
            notes.append(f"Part '{p.name}' attached to unknown part; reattached to the root.")
        p.floors = int(_clamp(p.floors, 1, 12))
        p.floor_height = int(_clamp(p.floor_height, 3, 8))
        p.taper = _clamp(p.taper, 0.0, 0.3)
        p.wall_thickness = int(_clamp(p.wall_thickness, 1, 2))
        p.sides = int(_clamp(p.sides, 3, 12))
        p.size.width = _clamp(p.size.width, 0.1, 1.5)
        p.size.depth = _clamp(p.size.depth, 0.1, 1.5)
        if p.attach is None:
            p.size.width = _clamp(p.size.width, 0.3, 1.0)
            p.size.depth = _clamp(p.size.depth, 0.3, 1.0)
        r = p.roof
        r.pitch = float(round(_clamp(r.pitch, 0.5, 3.0) * 2) / 2)
        r.overhang = int(_clamp(r.overhang, 0, 3))
        r.curl = _clamp(r.curl, 0.0, 3.0)
        r.edge_width = int(_clamp(r.edge_width, 0, 3))
        r.band_spacing = int(_clamp(r.band_spacing, 2, 6))
        r.ridge_offset = _clamp(r.ridge_offset, -0.4, 0.4)
        r.tiers = int(_clamp(r.tiers, 1, 5))
        if r.crenellated and r.type != RoofType.parapet:
            r.type = RoofType.parapet
            notes.append(f"Part '{p.name}': crenellation requires a parapet roof; roof type changed.")
        if r.type == RoofType.pagoda and r.tiers == 1:
            r.tiers = 3
        if p.shape == Shape.ring and p.attach is not None:
            pass

    # Facade and depth clamps.
    f = program.facade
    f.bay_width.min = int(_clamp(f.bay_width.min, 2, 9))
    f.bay_width.max = int(_clamp(f.bay_width.max, f.bay_width.min, 9))
    f.window_width = int(_clamp(f.window_width, 1, 5))
    f.window_height = int(_clamp(f.window_height, 1, 5))
    f.shutters = _clamp(f.shutters, 0.0, 1.0)
    f.max_flat_run = int(_clamp(f.max_flat_run, 3, 20))
    f.ground_floor_taller = int(_clamp(f.ground_floor_taller, 0, 4))
    f.window_boxes = _clamp(f.window_boxes, 0.0, 1.0)
    d = program.depth
    d.frame_protrude = int(_clamp(d.frame_protrude, 0, 2))
    d.window_inset = int(_clamp(d.window_inset, 0, 1))
    d.foundation_rise = int(_clamp(d.foundation_rise, 0, 3))
    d.foundation_outset = int(_clamp(d.foundation_outset, 0, 2))
    d.detail = _clamp(d.detail, 0.0, 1.0)
    d.foliage = _clamp(d.foliage, 0.0, 1.0)
    d.shading = _clamp(d.shading, 0.0, 1.0)

    b = program.budget
    b.dominant = int(_clamp(b.dominant, 0, 3))
    b.medium_min = int(_clamp(b.medium_min, 0, 6))
    b.medium_max = int(_clamp(b.medium_max, b.medium_min, 8))
    b.small_min = int(_clamp(b.small_min, 0, 6))
    b.small_max = int(_clamp(b.small_max, b.small_min, 8))
    # Attachments.
    for a in program.attachments:
        if a.count is not None:
            a.count = int(_clamp(a.count, 0, 12))
        if a.on is not None and a.on not in names:
            a.on = None

    # Palette: start from a curated palette when one is named, then repair what the model composed.
    from craftpilot.program.palette import check_and_repair, library_palette

    if program.palette_name:
        lib = library_palette(program.palette_name.strip().lower())
        if lib is None:
            notes.append(f"Unknown palette '{program.palette_name}'; composed palette kept.")
        else:
            for role in ("primary", "roof", "secondary", "accent", "framing", "trim", "foundation"):
                if getattr(program.palette, role) is None or not getattr(program.palette, role).families:
                    setattr(program.palette, role, getattr(lib, role))
    pal = program.palette
    pal.primary = _repair_palette(pal.primary, "primary", notes) or RolePalette(
        families=[FamilyWeight(family=DEFAULT_PRIMARY_FAMILY, weight=1.0)])
    pal.roof = _repair_palette(pal.roof, "roof", notes) or RolePalette(
        families=[FamilyWeight(family=DEFAULT_ROOF_FAMILY, weight=1.0)])
    for role in ("secondary", "accent", "framing", "trim", "foundation", "glass", "roof_edge"):
        setattr(pal, role, _repair_palette(getattr(pal, role), role, notes))
    if pal.glass is not None and any(catalog.family(fw.family).tone != "glass" for fw in pal.glass.families
                                     if catalog.family(fw.family)):
        pal.glass = None
        notes.append("Glass palette used a non-glass family and was reset.")
    notes.extend(check_and_repair(pal, program.label))
    if pal.roof is not None:
        for fw in pal.roof.families:
            fam = catalog.family(fw.family)
            if fam is not None and not fam.has("stairs"):
                notes.append(f"Roof family '{fw.family}' has no stairs; slopes will use full blocks.")

    # A roof edge palette implies an edge of one row when none was given.
    if pal.roof_edge is not None:
        for p_ in program.parts:
            if p_.roof.edge_width == 0 and p_.roof.edge_lines == "none":
                p_.roof.edge_width = 1
    # Bounds.
    b = program.bounds
    b.width = int(_clamp(b.width, 7, safety_limit[0]))
    b.height = int(_clamp(b.height, 8, safety_limit[1]))
    b.depth = int(_clamp(b.depth, 7, safety_limit[2]))
    return program, notes


def clamp_bounds(bounds: Bounds, safety_limit: tuple[int, int, int]) -> tuple[Bounds, str | None]:
    if bounds.width > safety_limit[0] or bounds.height > safety_limit[1] or bounds.depth > safety_limit[2]:
        return bounds, (f"Bounds {bounds.as_tuple()} exceed the safety limit {safety_limit}; "
                        f"raise CRAFTPILOT_SAFETY_LIMIT to allow it.")
    return Bounds(width=max(7, bounds.width), height=max(8, bounds.height), depth=max(7, bounds.depth)), None
