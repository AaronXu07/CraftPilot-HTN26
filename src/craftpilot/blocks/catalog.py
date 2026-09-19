"""Block family catalog.

A family is a set of blocks that share a material and come in several shapes
(full block, stairs, slab, fence, wall, log, trapdoor, door, pane) plus
same-tone variants used for texturing. The catalog only lists blocks that
exist in Minecraft Java 1.20.2 and later so schematics paste into old and new
worlds alike. A generated catalog from the game jar can replace this later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Variant:
    block_id: str
    tags: frozenset[str] = frozenset()   # "weathered", "log", "smooth", "chiseled"


@dataclass
class Family:
    name: str
    tone: str
    loud: bool = False
    shapes: dict[str, str] = field(default_factory=dict)   # shape name -> block id
    variants: list[Variant] = field(default_factory=list)  # alternative full blocks
    # Appearance and context, filled from palette_data at build time.
    rgb: tuple[int, int, int] = (128, 128, 128)
    noise: float = 0.4                                     # 0 flat colour .. 1 very busy texture
    material: str = "stone"                                # wood, stone, brick, plaster, terracotta, metal, glass, organic
    styles: frozenset[str] = frozenset({"generic"})

    def has(self, shape: str) -> bool:
        return shape in self.shapes

    def colour_words(self) -> str:
        return colour_words(self.rgb)

    def noise_word(self) -> str:
        return "smooth" if self.noise <= 0.2 else "textured" if self.noise <= 0.45 else "rough" if self.noise <= 0.7 else "busy"


def colour_words(rgb: tuple[int, int, int]) -> str:
    """A short human colour name: lightness word plus hue word, e.g. 'light warm grey', 'dark red'."""
    import colorsys
    h, l, sat = colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))
    h *= 360
    if l < 0.12:
        return "black"
    if l > 0.88 and sat < 0.25:
        return "white"
    light = "very dark" if l < 0.22 else "dark" if l < 0.4 else "mid" if l < 0.6 else "light" if l < 0.8 else "very light"
    if sat < 0.1:
        return f"{light} grey"
    if sat < 0.25:
        tint = "warm" if h < 70 or h > 320 else "cool"
        return f"{light} {tint} grey"
    if h < 15 or h >= 340:
        hue = "red"
    elif h < 40:
        hue = "orange-brown" if l < 0.55 else "orange"
    elif h < 65:
        hue = "brown" if l < 0.5 else "tan" if sat < 0.5 else "yellow"
    elif h < 160:
        hue = "green"
    elif h < 200:
        hue = "teal"
    elif h < 260:
        hue = "blue"
    elif h < 300:
        hue = "purple"
    else:
        hue = "pink"
    return f"{light} {hue}"


def _mc(block: str) -> str:
    return f"minecraft:{block}"


def _wood(name: str, tone: str, log: str | None = None, planks: str | None = None) -> Family:
    log = log or f"{name}_log"
    planks = planks or f"{name}_planks"
    stripped = f"stripped_{log}"
    return Family(
        name=name,
        tone=tone,
        shapes={
            "full": _mc(planks),
            "stairs": _mc(f"{name}_stairs"),
            "slab": _mc(f"{name}_slab"),
            "fence": _mc(f"{name}_fence"),
            "log": _mc(log),
            "stripped_log": _mc(stripped),
            "trapdoor": _mc(f"{name}_trapdoor"),
            "door": _mc(f"{name}_door"),
            "button": _mc(f"{name}_button"),
            "fence_gate": _mc(f"{name}_fence_gate"),
        },
        variants=[Variant(_mc(stripped), frozenset({"log"}))],
    )


def _stone(
    name: str,
    tone: str,
    full: str | None = None,
    stairs: str | None = None,
    slab: str | None = None,
    wall: str | None = None,
    variants: list[tuple[str, set[str]]] | None = None,
    loud: bool = False,
    pillar: str | None = None,
) -> Family:
    full = full or name
    shapes = {"full": _mc(full)}
    if stairs:
        shapes["stairs"] = _mc(stairs)
    if slab:
        shapes["slab"] = _mc(slab)
    if wall:
        shapes["wall"] = _mc(wall)
    if pillar:
        shapes["pillar"] = _mc(pillar)
    shapes["button"] = _mc("polished_blackstone_button" if "blackstone" in name else "stone_button")
    if name in ("copper", "exposed_copper", "weathered_copper", "oxidized_copper"):
        base = "waxed_" + (name if name != "copper" else "copper")
        shapes["trapdoor"] = _mc(f"{base}_trapdoor")
        shapes["door"] = _mc(f"{base}_door")
    return Family(
        name=name,
        tone=tone,
        loud=loud,
        shapes=shapes,
        variants=[Variant(_mc(v), frozenset(t)) for v, t in (variants or [])],
    )


def _plain(name: str, tone: str, block: str, loud: bool = False) -> Family:
    return Family(name=name, tone=tone, loud=loud, shapes={"full": _mc(block)})


def _build() -> dict[str, Family]:
    fams: list[Family] = [
        # Woods
        _wood("oak", "light_wood"),
        _wood("birch", "light_wood"),
        _wood("spruce", "dark_wood"),
        _wood("dark_oak", "dark_wood"),
        _wood("jungle", "warm_wood"),
        _wood("acacia", "warm_wood"),
        _wood("mangrove", "warm_wood"),
        _wood("cherry", "pink_wood"),
        _wood("bamboo", "light_wood", log="bamboo_block", planks="bamboo_planks"),
        _wood("crimson", "red", log="crimson_stem"),
        _wood("warped", "teal", log="warped_stem"),
        _wood("pale_oak", "white"),
        _stone("bamboo_mosaic", "light_wood", stairs="bamboo_mosaic_stairs", slab="bamboo_mosaic_slab"),
        # Grey stone
        _stone("cobblestone", "grey_stone", stairs="cobblestone_stairs", slab="cobblestone_slab",
               wall="cobblestone_wall", variants=[("mossy_cobblestone", {"weathered"})]),
        _stone("mossy_cobblestone", "grey_stone", stairs="mossy_cobblestone_stairs",
               slab="mossy_cobblestone_slab", wall="mossy_cobblestone_wall"),
        _stone("mossy_stone_bricks", "grey_stone", stairs="mossy_stone_brick_stairs", slab="mossy_stone_brick_slab",
               wall="mossy_stone_brick_wall"),
        _stone("stone_bricks", "grey_stone", stairs="stone_brick_stairs", slab="stone_brick_slab",
               wall="stone_brick_wall",
               variants=[("cracked_stone_bricks", {"weathered"}), ("mossy_stone_bricks", {"weathered"}),
                         ("chiseled_stone_bricks", {"chiseled"})]),
        _stone("stone", "grey_stone", stairs="stone_stairs", slab="stone_slab",
               variants=[("smooth_stone", {"smooth"})]),
        _stone("smooth_stone", "grey_stone", slab="smooth_stone_slab"),
        _stone("andesite", "grey_stone", stairs="andesite_stairs", slab="andesite_slab", wall="andesite_wall",
               variants=[("polished_andesite", {"smooth"})]),
        _stone("polished_andesite", "grey_stone", stairs="polished_andesite_stairs",
               slab="polished_andesite_slab"),
        _stone("tuff", "grey_stone", stairs="tuff_stairs", slab="tuff_slab", wall="tuff_wall",
               variants=[("chiseled_tuff", {"chiseled"})]),
        _stone("tuff_bricks", "grey_stone", stairs="tuff_brick_stairs", slab="tuff_brick_slab", wall="tuff_brick_wall",
               variants=[("chiseled_tuff_bricks", {"chiseled"})]),
        _stone("polished_tuff", "grey_stone", stairs="polished_tuff_stairs", slab="polished_tuff_slab",
               wall="polished_tuff_wall"),
        _stone("resin_bricks", "copper", stairs="resin_brick_stairs", slab="resin_brick_slab", wall="resin_brick_wall",
               variants=[("chiseled_resin_bricks", {"chiseled"})]),
        _stone("cinnabar", "dark_red", stairs="cinnabar_stairs", slab="cinnabar_slab", wall="cinnabar_wall",
               variants=[("chiseled_cinnabar", {"chiseled"})]),
        _stone("cinnabar_bricks", "dark_red", stairs="cinnabar_brick_stairs", slab="cinnabar_brick_slab",
               wall="cinnabar_brick_wall"),
        _stone("sulfur", "sand", stairs="sulfur_stairs", slab="sulfur_slab", wall="sulfur_wall", loud=True,
               variants=[("chiseled_sulfur", {"chiseled"})]),
        _stone("sulfur_bricks", "sand", stairs="sulfur_brick_stairs", slab="sulfur_brick_slab",
               wall="sulfur_brick_wall", loud=True),
        _stone("smooth_red_sandstone", "red", stairs="smooth_red_sandstone_stairs", slab="smooth_red_sandstone_slab"),
        # Dark stone
        _stone("deepslate_bricks", "dark_stone", stairs="deepslate_brick_stairs", slab="deepslate_brick_slab",
               wall="deepslate_brick_wall", variants=[("cracked_deepslate_bricks", {"weathered"})]),
        _stone("deepslate_tiles", "dark_stone", stairs="deepslate_tile_stairs", slab="deepslate_tile_slab",
               wall="deepslate_tile_wall", variants=[("cracked_deepslate_tiles", {"weathered"})]),
        _stone("polished_deepslate", "dark_stone", stairs="polished_deepslate_stairs",
               slab="polished_deepslate_slab", wall="polished_deepslate_wall"),
        _stone("cobbled_deepslate", "dark_stone", stairs="cobbled_deepslate_stairs",
               slab="cobbled_deepslate_slab", wall="cobbled_deepslate_wall"),
        _stone("blackstone", "dark_stone", stairs="blackstone_stairs", slab="blackstone_slab",
               wall="blackstone_wall"),
        _stone("polished_blackstone_bricks", "dark_stone", stairs="polished_blackstone_brick_stairs",
               slab="polished_blackstone_brick_slab", wall="polished_blackstone_brick_wall",
               variants=[("cracked_polished_blackstone_bricks", {"weathered"})]),
        _stone("polished_blackstone", "dark_stone", stairs="polished_blackstone_stairs",
               slab="polished_blackstone_slab", wall="polished_blackstone_wall"),
        _stone("basalt", "dark_stone", pillar="basalt", variants=[("smooth_basalt", {"smooth"})]),
        # Warm stone
        _stone("bricks", "warm_stone", stairs="brick_stairs", slab="brick_slab", wall="brick_wall"),
        _stone("mud_bricks", "warm_stone", stairs="mud_brick_stairs", slab="mud_brick_slab",
               wall="mud_brick_wall", variants=[("packed_mud", {"weathered"})]),
        _stone("granite", "warm_stone", stairs="granite_stairs", slab="granite_slab", wall="granite_wall",
               variants=[("polished_granite", {"smooth"})]),
        _stone("polished_granite", "warm_stone", stairs="polished_granite_stairs", slab="polished_granite_slab"),
        _stone("sandstone", "sand", stairs="sandstone_stairs", slab="sandstone_slab", wall="sandstone_wall",
               variants=[("cut_sandstone", {"smooth"}), ("chiseled_sandstone", {"chiseled"})]),
        _stone("smooth_sandstone", "sand", stairs="smooth_sandstone_stairs", slab="smooth_sandstone_slab"),
        _stone("red_sandstone", "red", stairs="red_sandstone_stairs", slab="red_sandstone_slab",
               wall="red_sandstone_wall", variants=[("cut_red_sandstone", {"smooth"})]),
        _stone("terracotta", "warm_stone"),
        # Nether / end
        _stone("nether_bricks", "dark_red", stairs="nether_brick_stairs", slab="nether_brick_slab",
               wall="nether_brick_wall", variants=[("cracked_nether_bricks", {"weathered"})]),
        _stone("red_nether_bricks", "dark_red", stairs="red_nether_brick_stairs",
               slab="red_nether_brick_slab", wall="red_nether_brick_wall", loud=True),
        _stone("end_stone_bricks", "white", stairs="end_stone_brick_stairs", slab="end_stone_brick_slab",
               wall="end_stone_brick_wall"),
        _stone("purpur", "purple", full="purpur_block", stairs="purpur_stairs", slab="purpur_slab",
               pillar="purpur_pillar", loud=True),
        _stone("prismarine", "teal", stairs="prismarine_stairs", slab="prismarine_slab",
               wall="prismarine_wall", loud=True),
        _stone("prismarine_bricks", "teal", stairs="prismarine_brick_stairs", slab="prismarine_brick_slab",
               loud=True),
        _stone("dark_prismarine", "teal", stairs="dark_prismarine_stairs", slab="dark_prismarine_slab"),
        # White
        _stone("quartz", "white", full="quartz_block", stairs="quartz_stairs", slab="quartz_slab",
               pillar="quartz_pillar", variants=[("smooth_quartz", {"smooth"}), ("chiseled_quartz_block", {"chiseled"})]),
        _stone("smooth_quartz", "white", full="smooth_quartz", stairs="smooth_quartz_stairs",
               slab="smooth_quartz_slab"),
        _stone("diorite", "white", stairs="diorite_stairs", slab="diorite_slab", wall="diorite_wall",
               variants=[("polished_diorite", {"smooth"})]),
        _stone("polished_diorite", "white", stairs="polished_diorite_stairs", slab="polished_diorite_slab"),
        _stone("calcite", "white"),
        _stone("bone", "white", full="bone_block", pillar="bone_block"),
        # Copper
        _stone("copper", "copper", full="waxed_copper_block", stairs="waxed_cut_copper_stairs",
               slab="waxed_cut_copper_slab", variants=[("waxed_cut_copper", {"smooth"})]),
        _stone("exposed_copper", "copper", full="waxed_exposed_copper", stairs="waxed_exposed_cut_copper_stairs",
               slab="waxed_exposed_cut_copper_slab", variants=[("waxed_exposed_cut_copper", {"smooth"})]),
        _stone("weathered_copper", "teal", full="waxed_weathered_copper",
               stairs="waxed_weathered_cut_copper_stairs", slab="waxed_weathered_cut_copper_slab",
               variants=[("waxed_weathered_cut_copper", {"smooth"})]),
        _stone("oxidized_copper", "teal", full="waxed_oxidized_copper",
               stairs="waxed_oxidized_cut_copper_stairs", slab="waxed_oxidized_cut_copper_slab",
               variants=[("waxed_oxidized_cut_copper", {"smooth"})]),
        # Metal and misc
        Family("iron", "metal", shapes={"full": _mc("iron_block"), "trapdoor": _mc("iron_trapdoor"), "door": _mc("iron_door"),
                                        "button": _mc("stone_button")}),
        Family("iron_bars", "metal", shapes={"full": _mc("iron_bars"), "pane": _mc("iron_bars")}),
        _plain("obsidian", "dark_stone", "obsidian"),
        _plain("dark_oak_wood", "dark_wood", "dark_oak_wood"),
        _plain("moss", "green", "moss_block"),
        _plain("hay", "sand", "hay_block", loud=True),
        _plain("bookshelf", "warm_wood", "bookshelf"),
        # Glass
        Family("glass", "glass", shapes={"full": _mc("glass"), "pane": _mc("glass_pane")}),
        Family("tinted_glass", "glass", shapes={"full": _mc("tinted_glass"), "pane": _mc("tinted_glass")}),
    ]
    # Concrete and terracotta colours: full blocks only, most are loud.
    colours = ["white", "light_gray", "gray", "black", "brown", "red", "orange", "yellow", "lime",
               "green", "cyan", "light_blue", "blue", "purple", "magenta", "pink"]
    quiet = {"white", "light_gray", "gray", "black", "brown"}
    for c in colours:
        tone = "white" if c == "white" else ("dark_stone" if c == "black" else c)
        fams.append(_plain(f"{c}_concrete", tone, f"{c}_concrete", loud=c not in quiet))
        fams.append(_plain(f"{c}_terracotta", "warm_stone" if c in quiet else c, f"{c}_terracotta",
                           loud=c not in quiet))
        fams.append(_plain(f"{c}_wool", tone, f"{c}_wool", loud=c not in quiet))
        fams.append(Family(f"{c}_stained_glass", "glass", loud=c not in quiet,
                           shapes={"full": _mc(f"{c}_stained_glass"), "pane": _mc(f"{c}_stained_glass_pane")}))
    return {f.name: f for f in fams}


def _known_blocks() -> set[str] | None:
    """Block ids of the target Minecraft version, from data/blocks_<version>.json, if generated."""
    import json
    import os
    from pathlib import Path

    version = os.environ.get("CRAFTPILOT_MC_VERSION", "26.2")
    path = Path(__file__).resolve().parents[3] / "data" / f"blocks_{version}.json"
    if not path.exists():
        return None
    try:
        return set(json.loads(path.read_text())["blocks"].keys())
    except Exception:
        return None


def _filter(fams: dict[str, Family], known: set[str] | None) -> dict[str, Family]:
    """Drop shapes, variants, and whole families whose blocks the target version does not have,
    and attach appearance data to every family."""
    from craftpilot.blocks.palette_data import info

    out: dict[str, Family] = {}
    for name, f in fams.items():
        shapes = {k: v for k, v in f.shapes.items() if known is None or v in known}
        if "full" not in shapes:
            continue
        variants = [v for v in f.variants if known is None or v.block_id in known]
        fi = info(name, f.tone)
        out[name] = Family(name=f.name, tone=f.tone, loud=f.loud, shapes=shapes, variants=variants,
                           rgb=fi.rgb, noise=fi.noise, material=fi.material, styles=frozenset(fi.styles))
    return out


KNOWN_BLOCKS: set[str] | None = _known_blocks()
FAMILIES: dict[str, Family] = _filter(_build(), KNOWN_BLOCKS)

# Roles that need a specific shape. Materials falls back through this chain.
SHAPE_FALLBACK: dict[str, list[str]] = {
    "stairs": ["stairs", "slab", "full"],
    "slab": ["slab", "full"],
    "fence": ["fence", "wall", "full"],
    "wall": ["wall", "fence", "full"],
    "log": ["log", "pillar", "full"],
    "stripped_log": ["stripped_log", "log", "pillar", "full"],
    "pillar": ["pillar", "log", "full"],
    "trapdoor": ["trapdoor", "slab", "full"],
    "door": ["door"],
    "pane": ["pane", "full"],
    "button": ["button"],
    "full": ["full"],
}

DOOR_FALLBACK = _mc("oak_door")
LANTERN = _mc("lantern")
CAMPFIRE = _mc("campfire")
AIR = _mc("air")


def family(name: str) -> Family | None:
    return FAMILIES.get(name)


def resolve_shape(fam: Family, shape: str) -> str | None:
    """Block id for a shape in a family, following the fallback chain."""
    for s in SHAPE_FALLBACK.get(shape, [shape, "full"]):
        if s in fam.shapes:
            return fam.shapes[s]
    return None


def families_by_tone() -> dict[str, list[Family]]:
    out: dict[str, list[Family]] = {}
    for f in FAMILIES.values():
        out.setdefault(f.tone, []).append(f)
    return out


def catalog_summary() -> str:
    """One line per family for the LLM prompt: name, shapes, colour, texture, material, where it belongs."""
    by_material: dict[str, list[Family]] = {}
    for f in FAMILIES.values():
        by_material.setdefault(f.material, []).append(f)
    lines = []
    for material in ("wood", "stone", "brick", "plaster", "terracotta", "metal", "glass", "organic", "cloth"):
        fams = by_material.get(material)
        if not fams:
            continue
        lines.append(f"## {material}")
        for f in sorted(fams, key=lambda f: f.rgb[0] + f.rgb[1] + f.rgb[2]):
            shapes = "".join(
                ch for ch, s in (("S", "stairs"), ("s", "slab"), ("F", "fence"), ("W", "wall"),
                                 ("L", "log"), ("T", "trapdoor"), ("D", "door"), ("P", "pane"), ("B", "button"), ("G", "fence_gate"))
                if s in f.shapes
            )
            styles = " ".join(sorted(f.styles - {"generic"})) or "any"
            loud = " LOUD" if f.loud else ""
            lines.append(f"{f.name} [{shapes}] {f.colour_words()}, {f.noise_word()}; fits: {styles}{loud}")
    return "\n".join(lines)
