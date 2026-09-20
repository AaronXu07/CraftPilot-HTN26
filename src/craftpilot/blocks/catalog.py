"""Block catalog, built from the game's own block list.

Every block in `data/blocks_<version>.json` gets three layers of information:

- appearance: an average colour and a texture-noise value derived from its texture (hand values
  from palette_data fill in when the data file has none);
- a use class: `structural` (the engine may pick it on its own), `decorative` (pieces such as
  trapdoors, fences, lanterns, leaves), `thematic` (belongs to a setting: mushroom, prismarine,
  nether, end, ice; only when a program names it), `precious` and `functional` (ores, mineral
  blocks, redstone, containers, technical blocks; last resort or tiny accent only), `natural`
  (dirt, sand, crops; never used);
- style tags, from the hand table where one exists, else inferred from the name.

Families group a block with its shapes (stairs, slab, wall, fence, gate, trapdoor, door, button,
log, pane) and its texture variants (cracked, mossy, chiseled, smooth, polished, cut), all found
by naming pattern. A block with no relatives is a family of one. Pieces are resolved colour-first
across every family through `nearest_with_shape`, because builders pick a trapdoor by colour,
not by wood type.
"""

from __future__ import annotations

import colorsys
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from craftpilot.blocks.palette_data import _OVERRIDES, info as _hand_info

USE_STRUCTURAL = "structural"
USE_DECORATIVE = "decorative"
USE_THEMATIC = "thematic"
USE_PRECIOUS = "precious"
USE_FUNCTIONAL = "functional"
USE_NATURAL = "natural"
AUTO_USES = (USE_STRUCTURAL, USE_DECORATIVE)


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
    rgb: tuple[int, int, int] = (128, 128, 128)
    noise: float = 0.3                                     # 0 flat colour .. 1 very busy texture
    material: str = "stone"
    styles: frozenset[str] = frozenset({"generic"})
    use: str = USE_STRUCTURAL

    def has(self, shape: str) -> bool:
        return shape in self.shapes

    def colour_words(self) -> str:
        return colour_words(self.rgb)

    def noise_word(self) -> str:
        return "smooth" if self.noise <= 0.15 else "textured" if self.noise <= 0.35 else "rough" if self.noise <= 0.5 else "busy"


def colour_words(rgb: tuple[int, int, int]) -> str:
    """A short human colour name: lightness word plus hue word, e.g. 'light warm grey', 'dark red'."""
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


LANTERN = _mc("lantern")
CAMPFIRE = _mc("campfire")
AIR = _mc("air")
DOOR_FALLBACK = _mc("oak_door")

# Roles that need a specific shape; the fallback chain runs before colour-first lookup kicks in.
SHAPE_FALLBACK: dict[str, list[str]] = {
    "stairs": ["stairs"],
    "slab": ["slab"],
    "fence": ["fence", "wall"],
    "wall": ["wall", "fence"],
    "log": ["log", "pillar", "full"],
    "stripped_log": ["stripped_log", "log", "pillar", "full"],
    "pillar": ["pillar", "log", "full"],
    "trapdoor": ["trapdoor"],
    "door": ["door"],
    "pane": ["pane", "full"],
    "button": ["button"],
    "fence_gate": ["fence_gate"],
    "full": ["full"],
}

# ------------------------------------------------------------------------------------------------
# Raw block data


def _load_blocks() -> tuple[dict[str, dict], str]:
    version = os.environ.get("CRAFTPILOT_MC_VERSION", "26.2")
    path = Path(__file__).resolve().parents[3] / "data" / f"blocks_{version}.json"
    if not path.exists():
        return {}, version
    try:
        return json.loads(path.read_text())["blocks"], version
    except Exception:
        return {}, version


BLOCKS, MC_VERSION = _load_blocks()
KNOWN_BLOCKS: set[str] | None = set(BLOCKS) if BLOCKS else None


def _short(block_id: str) -> str:
    return block_id.split(":", 1)[-1]


def block_rgb(block_id: str) -> tuple[int, int, int] | None:
    c = BLOCKS.get(block_id, {}).get("color")
    return (int(c[0]), int(c[1]), int(c[2])) if c else None


def block_noise(block_id: str) -> float | None:
    n = BLOCKS.get(block_id, {}).get("noise")
    return float(n) if n is not None else None


# ------------------------------------------------------------------------------------------------
# Use classification

_FUNCTIONAL = {
    "furnace", "blast_furnace", "smoker", "crafting_table", "chest", "trapped_chest", "ender_chest", "barrel",
    "hopper", "dropper", "dispenser", "piston", "sticky_piston", "piston_head", "moving_piston", "observer", "lever",
    "comparator", "repeater", "redstone_wire", "redstone_torch", "redstone_wall_torch", "redstone_lamp", "tnt",
    "jukebox", "note_block", "bell", "anvil", "chipped_anvil", "damaged_anvil", "grindstone", "stonecutter", "loom",
    "cartography_table", "smithing_table", "fletching_table", "enchanting_table", "brewing_stand", "cauldron",
    "water_cauldron", "lava_cauldron", "powder_snow_cauldron", "composter", "beehive", "bee_nest", "lectern",
    "respawn_anchor", "lodestone", "beacon", "conduit", "spawner", "trial_spawner", "vault", "command_block",
    "chain_command_block", "repeating_command_block", "structure_block", "structure_void", "jigsaw", "barrier",
    "light", "bedrock", "end_portal", "end_portal_frame", "end_gateway", "nether_portal", "dragon_egg",
    "daylight_detector", "target", "tripwire", "tripwire_hook", "rail", "powered_rail", "detector_rail",
    "activator_rail", "scaffolding", "slime_block", "honey_block", "crafter", "chiseled_bookshelf", "sculk_sensor",
    "calibrated_sculk_sensor", "sculk_shrieker", "sculk_catalyst", "infested_stone", "frosted_ice", "bubble_column",
    "fire", "soul_fire", "cake", "flower_pot", "cobweb", "sponge", "wet_sponge", "test_block", "test_instance_block",
}
_FUNCTIONAL_PATTERNS = ("shulker_box", "_bed", "_head", "_skull", "_sign", "command_block", "cauldron",
                        "potted_", "candle_cake", "copper_bulb", "infested_")
_PRECIOUS_PATTERNS = ("_ore", "raw_", "diamond_block", "gold_block", "emerald_block", "netherite_block", "lapis_block",
                      "redstone_block", "ancient_debris", "gilded_blackstone", "budding_amethyst")
_NATURAL = {
    "dirt", "coarse_dirt", "rooted_dirt", "grass_block", "podzol", "mycelium", "farmland", "dirt_path", "sand",
    "red_sand", "gravel", "suspicious_sand", "suspicious_gravel", "snow", "ice", "water", "lava", "powder_snow",
    "melon", "pumpkin", "cactus", "sugar_cane", "bamboo", "kelp", "kelp_plant", "seagrass", "tall_seagrass",
    "lily_pad", "dead_bush", "short_grass", "tall_grass", "fern", "large_fern", "hanging_roots", "spore_blossom",
    "big_dripleaf", "small_dripleaf", "azalea", "flowering_azalea", "pointed_dripstone", "cave_vines",
    "cave_vines_plant", "weeping_vines", "twisting_vines", "sculk_vein", "frogspawn", "turtle_egg", "sniffer_egg",
    "brown_mushroom", "red_mushroom", "cocoa", "wheat", "carrots", "potatoes", "beetroots", "nether_wart",
    "sweet_berry_bush", "torchflower_crop", "pitcher_crop", "mud", "magma_block", "netherrack", "soul_sand",
    "soul_soil", "end_stone", "moss_carpet", "pale_moss_carpet",
}
_NATURAL_PATTERNS = ("_sapling", "_propagule", "coral", "_crop", "attached_", "_bush", "_flower", "tulip", "orchid",
                     "allium", "azure_bluet", "daisy", "cornflower", "lily_of", "poppy", "dandelion", "wither_rose",
                     "torchflower", "pitcher_plant", "sunflower", "lilac", "rose_bush", "peony", "pink_petals",
                     "wildflowers", "leaf_litter", "firefly_bush", "cactus_flower", "dry_", "seagrass", "pickle",
                     "_egg", "_roots", "sprouts", "_fungus", "chorus_", "eyeblossom", "_grass", "concrete_powder",
                     "_stem" if False else "cave_vines")
_DECORATIVE = {
    "lantern", "soul_lantern", "torch", "wall_torch", "soul_torch", "soul_wall_torch", "chain", "iron_bars", "ladder",
    "campfire", "soul_campfire", "glowstone", "sea_lantern", "shroomlight", "bookshelf", "hay_block", "moss_block",
    "pale_moss_block", "carved_pumpkin", "jack_o_lantern", "lightning_rod", "end_rod", "candle", "decorated_pot",
    "glow_lichen", "vine", "resin_clump", "copper_grate", "exposed_copper_grate", "weathered_copper_grate",
    "oxidized_copper_grate", "iron_chain", "copper_chain", "copper_lantern", "copper_torch", "shelf", "cushion",
}
_DECORATIVE_PATTERNS = ("_leaves", "_carpet", "_pane", "_trapdoor", "_button", "_fence", "_door", "_candle",
                        "_lantern", "_chain", "_grate", "_banner", "_shelf", "_cushion", "_torch", "_bars")
_THEMATIC_PATTERNS = ("mushroom", "prismarine", "purpur", "end_stone", "nether", "crimson", "warped", "sculk",
                      "froglight", "packed_ice", "blue_ice", "snow_block", "amethyst", "bone_block", "honeycomb",
                      "dried_kelp", "warped_wart", "glazed_terracotta", "crying_obsidian", "quartz_bricks",
                      "reinforced_deepslate", "chorus", "magma", "cinnabar", "sulfur", "resin", "_wool")
_STRUCTURAL_MATERIALS = {"wood", "plaster", "terracotta", "brick"}
_STRUCTURAL_NAMES = {
    "calcite", "tuff", "dripstone_block", "clay", "smooth_basalt", "basalt", "polished_basalt", "blackstone",
    "obsidian", "iron_block", "coal_block", "smooth_stone", "stone", "cobblestone", "mossy_cobblestone", "andesite",
    "diorite", "granite", "deepslate", "cobbled_deepslate", "sandstone", "red_sandstone", "quartz_block",
    "smooth_quartz", "bricks", "mud_bricks", "packed_mud", "stone_bricks", "mossy_stone_bricks",
    "cracked_stone_bricks", "chiseled_stone_bricks", "terracotta", "bamboo_mosaic", "waxed_copper_block",
    "copper_block", "cut_copper", "chiseled_copper", "bamboo_block", "stripped_bamboo_block", "white_wool",
    "light_gray_wool", "gray_wool", "black_wool", "brown_wool", "dark_prismarine",
}


def use_class(block_id: str, material: str, has_shapes: bool) -> str:
    n = _short(block_id)
    if n == "air" or n.endswith("_air") or n in _FUNCTIONAL or any(p in n for p in _FUNCTIONAL_PATTERNS):
        return USE_FUNCTIONAL
    if any(p in n for p in _PRECIOUS_PATTERNS):
        return USE_PRECIOUS
    if n in _STRUCTURAL_NAMES:
        return USE_STRUCTURAL
    if n in _NATURAL or any(p in n for p in _NATURAL_PATTERNS):
        return USE_NATURAL
    if n in _DECORATIVE or any(p in n for p in _DECORATIVE_PATTERNS) or material == "glass":
        return USE_DECORATIVE
    if any(p in n for p in _THEMATIC_PATTERNS):
        return USE_THEMATIC
    if material in _STRUCTURAL_MATERIALS or has_shapes:
        return USE_STRUCTURAL
    for pre in ("cracked_", "mossy_", "chiseled_", "smooth_", "polished_", "cut_", "waxed_", "exposed_",
                "weathered_", "oxidized_"):
        if n.startswith(pre):
            return USE_STRUCTURAL
    return USE_THEMATIC


# ------------------------------------------------------------------------------------------------
# Material and style inference


def _material_for(name: str) -> str:
    if name.endswith("_planks") or any(t in name for t in ("_log", "_wood", "_stem", "hyphae", "bamboo_block",
                                                            "bamboo_mosaic", "stripped_")):
        return "wood"
    if "concrete" in name:
        return "plaster"
    if "terracotta" in name:
        return "terracotta"
    if "glass" in name:
        return "glass"
    if any(t in name for t in ("copper", "iron_", "gold_", "netherite", "diamond", "emerald", "lapis",
                                "redstone_block", "chain")):
        return "metal"
    if any(t in name for t in ("leaves", "mushroom", "moss", "hay", "wart", "pumpkin", "melon", "shroomlight",
                                "sponge", "kelp", "honey", "azalea", "bush", "grass", "vine", "roots", "fungus")):
        return "organic"
    if "wool" in name or "carpet" in name:
        return "cloth"
    if "brick" in name:
        return "brick"
    return "stone"


def _styles_for(name: str, material: str) -> frozenset[str]:
    if any(t in name for t in ("nether", "crimson", "warped", "blackstone", "basalt", "soul_", "magma", "gilded")):
        return frozenset({"nether", "dark", "fantasy"})
    if any(t in name for t in ("end_stone", "purpur", "chorus")):
        return frozenset({"end", "fantasy"})
    if any(t in name for t in ("prismarine", "sea_lantern", "sponge", "kelp")):
        return frozenset({"ocean", "fantasy"})
    if "sculk" in name:
        return frozenset({"dark", "fantasy", "sculk"})
    if any(t in name for t in ("mushroom", "shroomlight", "amethyst", "froglight")):
        return frozenset({"fantasy", "natural"})
    if any(t in name for t in ("packed_ice", "blue_ice", "snow")):
        return frozenset({"winter"})
    if any(t in name for t in ("sandstone", "terracotta", "mud")):
        return frozenset({"desert", "generic"})
    return frozenset({"generic"})


def _tone_for(rgb: tuple[int, int, int], material: str) -> str:
    if material == "glass":
        return "glass"
    h, l, s = colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))
    h *= 360
    if material == "metal":
        return "copper" if (s > 0.25 and h < 60) else "metal"
    if material == "wood":
        if l > 0.75:
            return "white"
        if (h > 300 or h < 20) and s > 0.2:
            return "pink_wood" if l > 0.6 else "warm_wood"
        return "light_wood" if l > 0.45 else "dark_wood"
    if s < 0.12:
        return "white" if l > 0.75 else "dark_stone" if l < 0.3 else "grey_stone"
    if l > 0.7 and 30 < h < 70:
        return "sand"
    if 15 <= h < 70:
        return "warm_stone"
    if h < 15 or h >= 340:
        return "red" if l > 0.3 else "dark_red"
    if 70 <= h < 160:
        return "green"
    if 160 <= h < 220:
        return "teal"
    if 220 <= h < 340:
        return "purple"
    return "grey_stone"


# ------------------------------------------------------------------------------------------------
# Family construction by naming pattern

_SHAPE_SUFFIXES = [
    ("_fence_gate", "fence_gate"), ("_stairs", "stairs"), ("_slab", "slab"), ("_wall", "wall"), ("_fence", "fence"),
    ("_trapdoor", "trapdoor"), ("_door", "door"), ("_button", "button"),
]
_NOT_WALL = ("_wall_banner", "_wall_sign", "_wall_head", "_wall_skull", "_wall_fan", "_wall_hanging_sign",
             "_wall_torch")
_VARIANT_PREFIXES = {"cracked_": "weathered", "mossy_": "weathered", "chiseled_": "chiseled", "smooth_": "smooth",
                     "polished_": "smooth", "cut_": "smooth"}
_CUBE_PROPS = {"axis", "north", "east", "south", "west", "up", "down", "waterlogged", "distance", "persistent"}


def _base_candidates(base: str) -> list[str]:
    return [base, base + "_planks", base + "_block", base + "s", re.sub(r"brick$", "bricks", base),
            re.sub(r"tile$", "tiles", base)]


def _family_key(base: str) -> str:
    for suf in ("_planks", "_block"):
        if base.endswith(suf) and base != "bamboo_block":
            return base[: -len(suf)]
    return base


def _build() -> dict[str, Family]:
    names = {_short(b) for b in BLOCKS} if BLOCKS else set()
    if not names:
        return {}
    fams: dict[str, Family] = {}
    consumed: set[str] = set()

    def get_family(base: str) -> Family:
        key = _family_key(base)
        if key not in fams:
            fams[key] = Family(name=key, tone="grey_stone", shapes={"full": _mc(base)})
            consumed.add(base)
        return fams[key]

    # Shaped blocks attach to their base block.
    for n in sorted(names):
        if any(bad in n for bad in _NOT_WALL):
            continue
        for suffix, shape in _SHAPE_SUFFIXES:
            if not n.endswith(suffix):
                continue
            base = n[: -len(suffix)]
            if base == "petrified_oak":
                break
            found = next((c for c in _base_candidates(base) if c in names and c != n), None)
            if found is None:
                fam = fams.setdefault(base, Family(name=base, tone="grey_stone", shapes={}))
                fam.shapes[shape] = _mc(n)
            else:
                get_family(found).shapes.setdefault(shape, _mc(n))
            consumed.add(n)
            break

    # Wood: logs, stripped logs and bark; glass panes and bars.
    for n in sorted(names):
        if n.endswith("_planks"):
            wood = n[: -len("_planks")]
            fam = get_family(n)
            for log in (f"{wood}_log", f"{wood}_stem", "bamboo_block" if wood == "bamboo" else ""):
                if log and log in names:
                    fam.shapes.setdefault("log", _mc(log))
                    consumed.add(log)
                    stripped = f"stripped_{log}"
                    if stripped in names:
                        fam.shapes.setdefault("stripped_log", _mc(stripped))
                        fam.variants.append(Variant(_mc(stripped), frozenset({"log"})))
                        consumed.add(stripped)
                    break
            for bark in (f"{wood}_wood", f"{wood}_hyphae", f"stripped_{wood}_wood", f"stripped_{wood}_hyphae"):
                if bark in names:
                    consumed.add(bark)
        if n.endswith("_pane"):
            base = n[: -len("_pane")]
            if base in names:
                get_family(base).shapes.setdefault("pane", _mc(n))
                consumed.add(n)
    if "iron_bars" in names:
        fams["iron_bars"] = Family(name="iron_bars", tone="metal", shapes={"full": _mc("iron_bars"), "pane": _mc("iron_bars")})
        consumed.add("iron_bars")
    for n, base in (("quartz_pillar", "quartz_block"), ("purpur_pillar", "purpur_block"), ("basalt", "basalt"),
                    ("polished_basalt", "polished_basalt"), ("bone_block", "bone_block"), ("hay_block", "hay_block")):
        if n in names and base in names:
            get_family(base).shapes.setdefault("pillar", _mc(n))
            if n != base:
                consumed.add(n)

    # Variants: prefixed relatives of a family base.
    for n in sorted(names):
        for prefix, tag in _VARIANT_PREFIXES.items():
            if n.startswith(prefix):
                base = n[len(prefix):]
                for cand in (base, base + "_block", base + "s"):
                    if cand in names and _family_key(cand) in fams:
                        fams[_family_key(cand)].variants.append(Variant(_mc(n), frozenset({tag})))
                        break

    # Every other cube-like block becomes a family of one (plants and other natural blocks are skipped).
    taken = {f.shapes.get("full") for f in fams.values()}
    for n in sorted(names):
        if n in consumed or _mc(n) in taken:
            continue
        if use_class(_mc(n), _material_for(n), False) == USE_NATURAL:
            continue
        if any(n.endswith(suf) for suf, _ in _SHAPE_SUFFIXES) or n.endswith(("_pane", "_carpet", "_pressure_plate",
                                                                             "_sign", "_hanging_sign", "_banner")):
            continue
        props = set(BLOCKS.get(_mc(n), {}).get("properties", {}))
        if props and not props <= _CUBE_PROPS:
            continue
        key = _family_key(n) if _family_key(n) not in fams else n
        fam = Family(name=key, tone="grey_stone", shapes={"full": _mc(n)})
        if props == {"axis"}:
            fam.shapes["pillar"] = _mc(n)
        fams[key] = fam

    # Attributes.
    for fam in fams.values():
        full = fam.shapes.get("full") or next(iter(fam.shapes.values()))
        short = _short(full)
        hand = _OVERRIDES.get(fam.name)
        material = hand.material if hand else _material_for(short)
        rgb = block_rgb(full) or (hand.rgb if hand else None)
        noise = block_noise(full)
        if rgb is None or noise is None:
            fallback = _hand_info(fam.name, _tone_for(rgb or (128, 128, 128), material))
            rgb = rgb or fallback.rgb
            noise = noise if noise is not None else min(1.0, fallback.noise * 0.7)
        styles = frozenset(hand.styles) if hand else _styles_for(short, material)
        has_shapes = any(s in fam.shapes for s in ("stairs", "slab", "wall"))
        use = use_class(full, material, has_shapes)
        if "full" not in fam.shapes:
            use = USE_DECORATIVE
        h, l, s = colorsys.rgb_to_hls(*(c / 255.0 for c in rgb))
        loud = (s > 0.5 and 0.2 < l < 0.85 and material != "wood") or "glazed_terracotta" in short
        fam.rgb, fam.noise, fam.material, fam.styles, fam.use, fam.loud = tuple(rgb), float(noise), material, styles, use, loud
        fam.tone = _tone_for(fam.rgb, material)
        fam.variants = [v for v in fam.variants
                        if use_class(v.block_id, material, True) in (USE_STRUCTURAL, USE_THEMATIC)]
    return fams


def _hand_fallback() -> dict[str, Family]:
    """Without a data file, build families from the hand tables only (fresh checkouts)."""
    fams: dict[str, Family] = {}
    for name, hi in _OVERRIDES.items():
        fams[name] = Family(name=name, tone=_tone_for(hi.rgb, hi.material), shapes={"full": _mc(name)},
                            rgb=hi.rgb, noise=hi.noise * 0.7, material=hi.material, styles=frozenset(hi.styles))
    return fams


FAMILIES: dict[str, Family] = _build() or _hand_fallback()

# Unwaxed copper oxidises in the world, so the waxed family stands in for it everywhere.
_ALIASES: dict[str, str] = {}
for _name in list(FAMILIES):
    if "copper" in _name and not _name.startswith("waxed_") and f"waxed_{_name}" in FAMILIES:
        FAMILIES[_name].use = USE_NATURAL
        _ALIASES[_name] = f"waxed_{_name}"
        for _bid in FAMILIES[_name].shapes.values():
            _ALIASES.setdefault(_short(_bid), f"waxed_{_name}")
# Block names resolve to their family too: "netherite_block" -> "netherite", "oak_planks" -> "oak".
for _f in FAMILIES.values():
    if _f.use == USE_NATURAL:
        continue
    for _bid in list(_f.shapes.values()) + [v.block_id for v in _f.variants]:
        _ALIASES.setdefault(_short(_bid), _f.name)
del _f, _bid, _name


def family(name: str) -> Family | None:
    f = FAMILIES.get(name)
    if f is None or f.use == USE_NATURAL:
        alias = _ALIASES.get(_short(name))
        f = FAMILIES.get(alias) if alias else f
    return f


def families_by_tone() -> dict[str, list[Family]]:
    out: dict[str, list[Family]] = {}
    for f in FAMILIES.values():
        out.setdefault(f.tone, []).append(f)
    return out


# ------------------------------------------------------------------------------------------------
# Colour-first shape lookup

_SHAPE_INDEX: dict[str, list[tuple[Family, str]]] = {}
for _f in FAMILIES.values():
    for _shape, _bid in _f.shapes.items():
        _SHAPE_INDEX.setdefault(_shape, []).append((_f, _bid))
del _f, _shape, _bid

_NEAREST_CACHE: dict[tuple, tuple[str, str] | None] = {}


def colour_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    h0, l0, s0 = colorsys.rgb_to_hls(*(c / 255.0 for c in a))
    h1, l1, s1 = colorsys.rgb_to_hls(*(c / 255.0 for c in b))
    dh = min(abs(h0 - h1), 1 - abs(h0 - h1))
    return 2.0 * abs(l0 - l1) + abs(s0 - s1) + dh * 4.0 * max(s0, s1)


def nearest_with_shape(shape: str, rgb: tuple[int, int, int], material: str | None = None,
                       allow: tuple[str, ...] = AUTO_USES, last_resort: bool = True,
                       exclude: str | None = None) -> tuple[Family, str] | None:
    """The block of `shape` closest in colour to `rgb`. Same material is a small bonus, never a
    gate. Precious, functional and thematic blocks are considered only when nothing allowed is
    anywhere near, so a grey stone can still get an iron trapdoor but never a netherite one."""
    key = (shape, rgb, material, allow, last_resort, exclude)
    if key in _NEAREST_CACHE:
        hit = _NEAREST_CACHE[key]
        return (FAMILIES[hit[0]], hit[1]) if hit else None
    best, best_d = None, 1e9
    fallback, fallback_d = None, 1e9
    for fam, bid in _SHAPE_INDEX.get(shape, []):
        if fam.name == exclude:
            continue
        d = colour_distance(rgb, block_rgb(bid) or fam.rgb)
        if material and fam.material != material:
            d += 0.08
        if fam.loud:
            d += 0.1
        if fam.use in allow:
            if d < best_d:
                best, best_d = (fam, bid), d
        elif fam.use in (USE_PRECIOUS, USE_FUNCTIONAL, USE_THEMATIC) and d < fallback_d:
            fallback, fallback_d = (fam, bid), d
    if best is None or (last_resort and fallback is not None and best_d > 0.45 and fallback_d < best_d - 0.25):
        best = fallback
    _NEAREST_CACHE[key] = (best[0].name, best[1]) if best else None
    return best


def resolve_shape(fam: Family, shape: str) -> str | None:
    """Block id for a shape in a family: the family's own, a fallback shape, else the closest
    colour from any family."""
    for s in SHAPE_FALLBACK.get(shape, [shape, "full"]):
        if s in fam.shapes:
            return fam.shapes[s]
    if shape == "full":
        return None
    hit = nearest_with_shape(shape, fam.rgb, fam.material)
    return hit[1] if hit else None


# ------------------------------------------------------------------------------------------------
# Prompt summary

_MATERIAL_ORDER = ("wood", "stone", "brick", "plaster", "terracotta", "metal", "cloth", "organic", "glass")


def catalog_summary() -> str:
    """The vocabulary the model may use: structural families grouped by material, then decorative
    and thematic families under their own headings, with colour and texture words."""

    def line(f: Family) -> str:
        shapes = "".join(
            ch for ch, s in (("S", "stairs"), ("s", "slab"), ("F", "fence"), ("W", "wall"), ("L", "log"),
                             ("T", "trapdoor"), ("D", "door"), ("P", "pane"), ("B", "button"), ("G", "fence_gate"))
            if s in f.shapes
        )
        styles = " ".join(sorted(f.styles - {"generic"})) or "any"
        loud = " LOUD" if f.loud else ""
        return f"{f.name} [{shapes}] {f.colour_words()}, {f.noise_word()}; fits: {styles}{loud}"

    lines = []
    for material in _MATERIAL_ORDER:
        fams = [f for f in FAMILIES.values() if f.material == material and f.use == USE_STRUCTURAL]
        if not fams:
            continue
        lines.append(f"## {material}")
        lines.extend(line(f) for f in sorted(fams, key=lambda f: sum(f.rgb)))
    deco = [f for f in FAMILIES.values() if f.use == USE_DECORATIVE and "full" in f.shapes]
    if deco:
        lines.append("## decorative (leaves, lights, carpets; trim and accent only)")
        lines.extend(line(f) for f in sorted(deco, key=lambda f: sum(f.rgb)))
    them = [f for f in FAMILIES.values() if f.use == USE_THEMATIC]
    if them:
        lines.append("## thematic (only when the building's setting calls for it)")
        lines.extend(line(f) for f in sorted(them, key=lambda f: (sorted(f.styles)[0], sum(f.rgb))))
    lines.append("## precious and functional blocks (gold, diamond, netherite, ores, redstone, containers) are "
                 "not listed; use one only as a tiny accent where the design truly calls for it.")
    return "\n".join(lines)
