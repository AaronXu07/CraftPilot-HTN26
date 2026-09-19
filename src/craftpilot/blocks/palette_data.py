"""Per-family palette data: colour, texture noisiness, material, and style context.

Colour drives the 60-30-10 checks (lightness and hue relationships between the wall, the
secondary group, and the accents). Noise limits what may be a primary wall block. Styles say
where a block makes sense: a family must share a style with the primary, or be generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Tone defaults, used when a family has no explicit entry.
_TONE_DEFAULTS: dict[str, tuple[tuple[int, int, int], float, str, set[str]]] = {
    "light_wood": ((177, 144, 86), 0.35, "wood", {"generic"}),
    "dark_wood": ((92, 66, 38), 0.35, "wood", {"generic"}),
    "warm_wood": ((160, 100, 60), 0.35, "wood", {"generic"}),
    "pink_wood": ((226, 178, 172), 0.3, "wood", {"asian", "fantasy", "cottage"}),
    "grey_stone": ((125, 125, 125), 0.5, "stone", {"generic"}),
    "dark_stone": ((70, 70, 72), 0.45, "stone", {"generic"}),
    "warm_stone": ((150, 100, 80), 0.4, "stone", {"generic"}),
    "sand": ((216, 203, 155), 0.3, "stone", {"desert", "classical", "generic"}),
    "white": ((225, 225, 220), 0.15, "stone", {"modern", "classical", "generic"}),
    "red": ((142, 33, 33), 0.15, "stone", {"generic"}),
    "dark_red": ((60, 20, 24), 0.4, "stone", {"nether", "fantasy", "dark"}),
    "purple": ((160, 120, 160), 0.3, "stone", {"end", "fantasy"}),
    "teal": ((90, 150, 140), 0.3, "stone", {"ocean", "fantasy"}),
    "copper": ((192, 107, 79), 0.3, "metal", {"generic", "steampunk", "modern"}),
    "metal": ((200, 200, 200), 0.1, "metal", {"modern", "industrial", "generic"}),
    "green": ((89, 110, 45), 0.5, "organic", {"generic", "natural"}),
    "glass": ((200, 230, 240), 0.0, "glass", {"generic"}),
}


@dataclass
class FamilyInfo:
    rgb: tuple[int, int, int]
    noise: float                       # 0 flat colour .. 1 very busy texture
    material: str                      # wood, stone, brick, plaster, metal, glass, organic, terracotta
    styles: set[str] = field(default_factory=lambda: {"generic"})


_OVERRIDES: dict[str, FamilyInfo] = {
    # Woods (planks)
    "oak": FamilyInfo((177, 144, 86), 0.35, "wood", {"generic", "cottage", "medieval", "rustic", "farm"}),
    "birch": FamilyInfo((196, 179, 123), 0.3, "wood", {"generic", "modern", "cottage", "nordic"}),
    "spruce": FamilyInfo((114, 84, 48), 0.35, "wood", {"generic", "medieval", "rustic", "nordic", "farm", "cottage"}),
    "dark_oak": FamilyInfo((66, 43, 20), 0.3, "wood", {"generic", "medieval", "rustic", "fantasy", "asian"}),
    "jungle": FamilyInfo((160, 115, 80), 0.35, "wood", {"generic", "tropical", "rustic"}),
    "acacia": FamilyInfo((168, 90, 50), 0.35, "wood", {"generic", "desert", "savanna", "modern"}),
    "mangrove": FamilyInfo((117, 54, 48), 0.35, "wood", {"generic", "tropical", "rustic"}),
    "cherry": FamilyInfo((226, 178, 172), 0.3, "wood", {"asian", "fantasy", "cottage", "generic"}),
    "bamboo": FamilyInfo((194, 173, 84), 0.4, "wood", {"asian", "tropical", "generic"}),
    "bamboo_mosaic": FamilyInfo((190, 170, 85), 0.5, "wood", {"asian", "tropical"}),
    "pale_oak": FamilyInfo((225, 218, 206), 0.3, "wood", {"generic", "modern", "nordic", "fantasy"}),
    "crimson": FamilyInfo((110, 50, 70), 0.35, "wood", {"nether", "fantasy", "dark"}),
    "warped": FamilyInfo((50, 120, 115), 0.35, "wood", {"nether", "fantasy", "ocean"}),
    # Grey stone
    "cobblestone": FamilyInfo((127, 127, 127), 0.85, "stone", {"generic", "medieval", "rustic", "cottage", "farm"}),
    "mossy_cobblestone": FamilyInfo((104, 118, 92), 0.9, "stone", {"medieval", "rustic", "ruin", "natural"}),
    "stone_bricks": FamilyInfo((122, 122, 122), 0.45, "stone", {"generic", "medieval", "castle", "classical"}),
    "mossy_stone_bricks": FamilyInfo((105, 120, 100), 0.55, "stone", {"medieval", "ruin", "natural"}),
    "stone": FamilyInfo((125, 125, 125), 0.4, "stone", {"generic", "modern", "medieval"}),
    "smooth_stone": FamilyInfo((160, 160, 160), 0.1, "stone", {"generic", "modern", "industrial"}),
    "andesite": FamilyInfo((136, 136, 137), 0.6, "stone", {"generic", "medieval", "rustic"}),
    "polished_andesite": FamilyInfo((132, 135, 134), 0.2, "stone", {"generic", "modern", "classical"}),
    "tuff": FamilyInfo((108, 110, 102), 0.6, "stone", {"generic", "medieval", "rustic"}),
    "tuff_bricks": FamilyInfo((100, 103, 96), 0.45, "stone", {"generic", "medieval", "castle"}),
    "polished_tuff": FamilyInfo((98, 102, 96), 0.2, "stone", {"generic", "modern"}),
    # Dark stone
    "deepslate_bricks": FamilyInfo((70, 70, 71), 0.45, "stone", {"generic", "medieval", "castle", "dark", "modern"}),
    "deepslate_tiles": FamilyInfo((55, 55, 56), 0.5, "stone", {"generic", "medieval", "castle", "dark", "roof"}),
    "polished_deepslate": FamilyInfo((72, 72, 73), 0.15, "stone", {"generic", "modern", "dark"}),
    "cobbled_deepslate": FamilyInfo((77, 77, 80), 0.85, "stone", {"generic", "medieval", "castle", "dark", "rustic"}),
    "blackstone": FamilyInfo((42, 36, 41), 0.7, "stone", {"dark", "nether", "fantasy", "castle"}),
    "polished_blackstone_bricks": FamilyInfo((48, 42, 48), 0.45, "stone", {"dark", "nether", "fantasy", "castle"}),
    "polished_blackstone": FamilyInfo((52, 46, 52), 0.15, "stone", {"dark", "nether", "modern"}),
    "basalt": FamilyInfo((80, 80, 84), 0.5, "stone", {"dark", "nether", "modern"}),
    "obsidian": FamilyInfo((15, 10, 24), 0.3, "stone", {"dark", "fantasy", "end"}),
    # Warm stone and brick
    "bricks": FamilyInfo((150, 97, 83), 0.4, "brick", {"generic", "urban", "victorian", "industrial", "colonial"}),
    "mud_bricks": FamilyInfo((137, 104, 79), 0.5, "brick", {"desert", "rustic", "generic", "tropical"}),
    "granite": FamilyInfo((149, 103, 86), 0.65, "stone", {"generic", "rustic", "desert"}),
    "polished_granite": FamilyInfo((154, 106, 89), 0.2, "stone", {"generic", "classical"}),
    "sandstone": FamilyInfo((216, 203, 155), 0.35, "stone", {"desert", "classical", "generic"}),
    "smooth_sandstone": FamilyInfo((223, 214, 170), 0.1, "stone", {"desert", "classical", "modern"}),
    "smooth_red_sandstone": FamilyInfo((190, 100, 30), 0.1, "stone", {"desert", "modern"}),
    "red_sandstone": FamilyInfo((186, 99, 29), 0.35, "stone", {"desert", "generic"}),
    "terracotta": FamilyInfo((152, 94, 68), 0.2, "terracotta", {"desert", "generic", "modern", "colonial"}),
    "resin_bricks": FamilyInfo((200, 90, 40), 0.4, "brick", {"fantasy", "generic", "autumn"}),
    "cinnabar": FamilyInfo((150, 50, 45), 0.5, "stone", {"fantasy", "generic"}),
    "cinnabar_bricks": FamilyInfo((140, 48, 44), 0.45, "brick", {"fantasy", "generic"}),
    "sulfur": FamilyInfo((210, 190, 60), 0.5, "stone", {"fantasy", "desert"}),
    "sulfur_bricks": FamilyInfo((200, 180, 60), 0.45, "brick", {"fantasy", "desert"}),
    # Nether and end
    "nether_bricks": FamilyInfo((44, 22, 26), 0.4, "brick", {"nether", "dark", "fantasy"}),
    "red_nether_bricks": FamilyInfo((70, 7, 9), 0.4, "brick", {"nether", "dark", "fantasy"}),
    "end_stone_bricks": FamilyInfo((219, 222, 158), 0.35, "stone", {"end", "fantasy", "desert"}),
    "purpur": FamilyInfo((170, 126, 170), 0.3, "stone", {"end", "fantasy"}),
    "prismarine": FamilyInfo((99, 156, 151), 0.5, "stone", {"ocean", "fantasy"}),
    "prismarine_bricks": FamilyInfo((99, 171, 158), 0.35, "stone", {"ocean", "fantasy"}),
    "dark_prismarine": FamilyInfo((51, 91, 75), 0.3, "stone", {"ocean", "fantasy", "dark"}),
    # White
    "quartz": FamilyInfo((236, 230, 223), 0.1, "stone", {"classical", "modern", "generic", "fantasy"}),
    "smooth_quartz": FamilyInfo((236, 230, 223), 0.05, "stone", {"classical", "modern", "generic"}),
    "diorite": FamilyInfo((188, 188, 188), 0.6, "stone", {"generic", "classical"}),
    "polished_diorite": FamilyInfo((192, 192, 192), 0.2, "stone", {"generic", "classical", "modern"}),
    "calcite": FamilyInfo((223, 224, 220), 0.1, "stone", {"modern", "classical", "generic"}),
    "bone": FamilyInfo((225, 221, 200), 0.3, "stone", {"fantasy", "desert"}),
    # Copper and metal
    "copper": FamilyInfo((192, 107, 79), 0.3, "metal", {"generic", "steampunk", "modern", "roof"}),
    "exposed_copper": FamilyInfo((160, 125, 100), 0.35, "metal", {"generic", "steampunk", "roof"}),
    "weathered_copper": FamilyInfo((108, 153, 110), 0.35, "metal", {"generic", "steampunk", "roof", "classical"}),
    "oxidized_copper": FamilyInfo((82, 162, 132), 0.35, "metal", {"generic", "steampunk", "roof", "classical"}),
    "iron": FamilyInfo((220, 220, 220), 0.1, "metal", {"modern", "industrial"}),
    "iron_bars": FamilyInfo((140, 140, 140), 0.2, "metal", {"generic"}),
    # Misc
    "dark_oak_wood": FamilyInfo((60, 40, 20), 0.4, "wood", {"generic", "rustic"}),
    "moss": FamilyInfo((89, 110, 45), 0.5, "organic", {"natural", "ruin", "fantasy"}),
    "hay": FamilyInfo((200, 170, 60), 0.6, "organic", {"farm", "rustic"}),
    "bookshelf": FamilyInfo((150, 120, 80), 0.7, "wood", {"generic"}),
    "glass": FamilyInfo((200, 230, 240), 0.0, "glass", {"generic"}),
    "tinted_glass": FamilyInfo((40, 35, 45), 0.0, "glass", {"modern", "dark"}),
}

_DYE_RGB = {
    "white": (207, 213, 214), "light_gray": (125, 125, 115), "gray": (55, 58, 62), "black": (8, 10, 15),
    "brown": (96, 60, 32), "red": (142, 33, 33), "orange": (224, 97, 1), "yellow": (241, 175, 21),
    "lime": (94, 169, 24), "green": (73, 91, 36), "cyan": (21, 119, 136), "light_blue": (36, 137, 199),
    "blue": (45, 47, 143), "purple": (100, 32, 156), "magenta": (169, 48, 159), "pink": (214, 101, 143),
}


def info(name: str, tone: str) -> FamilyInfo:
    if name in _OVERRIDES:
        return _OVERRIDES[name]
    for dye, rgb in _DYE_RGB.items():
        if name == f"{dye}_concrete":
            # Concrete is a painted surface: it goes with anything, so it is generic in every colour.
            return FamilyInfo(rgb, 0.0, "plaster", {"modern", "generic"})
        if name == f"{dye}_terracotta":
            r, g, b = rgb
            muted = (int(r * 0.75 + 40), int(g * 0.75 + 35), int(b * 0.75 + 30))
            return FamilyInfo(muted, 0.15, "terracotta", {"desert", "modern", "colonial", "generic"})
        if name == f"{dye}_wool":
            return FamilyInfo(rgb, 0.2, "cloth", {"cottage", "fantasy"})
        if name == f"{dye}_stained_glass":
            return FamilyInfo(rgb, 0.0, "glass", {"generic"})
    rgb, noise, material, styles = _TONE_DEFAULTS.get(tone, ((128, 128, 128), 0.4, "stone", {"generic"}))
    return FamilyInfo(rgb, noise, material, set(styles))


# ---------------------------------------------------------------------------------------------
# Curated palettes. Roles follow 60-30-10: primary is the 60, roof/foundation/secondary/framing the 30,
# trim/accent the 10. Every one of these is a combination Minecraft builders use.

LIBRARY: dict[str, dict] = {
    "oak cottage": dict(tags="cottage house rustic village medieval farm cozy", primary=["oak"], framing=["dark_oak"],
                        roof=["dark_oak"], foundation=["cobblestone", "stone_bricks"], secondary=["stone_bricks"],
                        trim=["spruce"], accent=["dark_oak"]),
    "spruce and stone": dict(tags="nordic cabin cottage mountain rustic tavern inn medieval", primary=["spruce"],
                             framing=["dark_oak"], roof=["dark_oak"], foundation=["cobblestone", "andesite"],
                             secondary=["stone_bricks"], trim=["spruce"], accent=["stripped_spruce"]),
    "tudor": dict(tags="tudor half timber medieval town inn manor english", primary=["white_concrete", "calcite"],
                  framing=["dark_oak"], roof=["dark_oak"], foundation=["cobblestone", "stone_bricks"],
                  secondary=["stone_bricks"], trim=["dark_oak"], accent=["dark_oak"]),
    "castle grey": dict(tags="castle keep fortress stone medieval tower fort", primary=["cobblestone", "stone_bricks", "andesite"],
                        framing=["dark_oak"], roof=["deepslate_tiles"], foundation=["cobbled_deepslate", "cobblestone"],
                        secondary=["stone_bricks"], trim=["stone_bricks"], accent=["red_concrete"]),
    "dark castle": dict(tags="castle dark evil fortress gothic vampire", primary=["deepslate_bricks", "polished_deepslate"],
                        framing=["blackstone"], roof=["polished_blackstone_bricks"], foundation=["cobbled_deepslate"],
                        secondary=["deepslate_tiles"], trim=["deepslate_tiles"], accent=["purple_concrete"]),
    "sandstone desert": dict(tags="desert sandstone oasis arabian egyptian temple market", primary=["sandstone", "smooth_sandstone"],
                             framing=["acacia"], roof=["smooth_sandstone"], foundation=["red_sandstone"],
                             secondary=["terracotta"], trim=["acacia"], accent=["cyan_terracotta"]),
    "adobe": dict(tags="adobe pueblo desert southwest mud clay village", primary=["terracotta", "mud_bricks"],
                  framing=["spruce"], roof=["spruce"], foundation=["mud_bricks"], secondary=["mud_bricks"],
                  trim=["spruce"], accent=["orange_terracotta"]),
    "red brick": dict(tags="brick townhouse urban victorian city warehouse factory school", primary=["bricks"],
                      framing=["stone_bricks"], roof=["deepslate_tiles"], foundation=["stone_bricks"],
                      secondary=["stone_bricks"], trim=["stone_bricks"], accent=["dark_oak"]),
    "georgian brick": dict(tags="mansion georgian manor estate grand brick colonial", primary=["bricks"],
                           framing=["quartz"], roof=["deepslate_tiles"], foundation=["stone_bricks"],
                           secondary=["quartz"], trim=["quartz"], accent=["dark_oak"]),
    "white mansion": dict(tags="mansion palace classical quartz white marble grand villa", primary=["quartz", "smooth_quartz"],
                          framing=["quartz"], roof=["deepslate_tiles"], foundation=["polished_andesite"],
                          secondary=["polished_diorite"], trim=["polished_andesite"], accent=["oxidized_copper"]),
    "modern white": dict(tags="modern minimalist contemporary villa flat concrete glass", primary=["white_concrete"],
                         framing=["dark_oak"], roof=["polished_deepslate"], foundation=["gray_concrete"],
                         secondary=["gray_concrete"], trim=["dark_oak"], accent=["dark_oak"]),
    "modern dark": dict(tags="modern dark contemporary sleek black office", primary=["gray_concrete", "polished_deepslate"],
                        framing=["polished_blackstone"], roof=["polished_deepslate"], foundation=["black_concrete"],
                        secondary=["light_gray_concrete"], trim=["spruce"], accent=["copper"]),
    "modern warm": dict(tags="modern scandinavian warm wood contemporary house", primary=["birch", "pale_oak"],
                        framing=["spruce"], roof=["polished_deepslate"], foundation=["polished_andesite"],
                        secondary=["white_concrete"], trim=["spruce"], accent=["copper"]),
    "japanese": dict(tags="japanese pagoda temple shrine asian zen tea house dojo", primary=["white_concrete", "calcite"],
                     framing=["dark_oak"], roof=["dark_oak"], foundation=["stone_bricks", "andesite"],
                     secondary=["stone_bricks"], trim=["dark_oak"], accent=["red_concrete"]),
    "chinese": dict(tags="chinese pagoda temple palace asian imperial", primary=["red_terracotta", "terracotta"],
                    framing=["dark_oak"], roof=["deepslate_tiles"], foundation=["stone_bricks"],
                    secondary=["stone_bricks"], trim=["dark_oak"], accent=["yellow_terracotta"]),
    "church stone": dict(tags="church chapel cathedral abbey monastery gothic stone", primary=["stone_bricks", "cobblestone"],
                         framing=["stone_bricks"], roof=["spruce"], foundation=["cobblestone"],
                         secondary=["polished_andesite"], trim=["stone_bricks"], accent=["dark_oak"]),
    "barn red": dict(tags="barn farm stable red rural agricultural silo", primary=["red_concrete", "mangrove"],
                     framing=["spruce"], roof=["spruce"], foundation=["cobblestone", "andesite"],
                     secondary=["smooth_stone"], trim=["spruce"], accent=["hay"]),
    "lighthouse": dict(tags="lighthouse beacon coastal harbour striped", primary=["white_concrete", "red_concrete"],
                       framing=["stone_bricks"], roof=["copper"], foundation=["stone_bricks", "cobblestone"],
                       secondary=["stone_bricks"], trim=["dark_oak"], accent=["red_concrete"]),
    "fantasy elven": dict(tags="elven fantasy forest magical elegant treehouse", primary=["birch", "pale_oak"],
                          framing=["dark_oak"], roof=["weathered_copper"], foundation=["polished_andesite", "moss"],
                          secondary=["polished_diorite"], trim=["oxidized_copper"], accent=["moss"]),
    "wizard tower": dict(tags="wizard tower magic fantasy arcane mage", primary=["stone_bricks", "mossy_stone_bricks"],
                         framing=["dark_oak"], roof=["purple_concrete", "purpur"], foundation=["cobbled_deepslate"],
                         secondary=["deepslate_bricks"], trim=["dark_oak"], accent=["purpur"]),
    "nether fortress": dict(tags="nether fortress hell dark demonic blackstone", primary=["nether_bricks", "polished_blackstone_bricks"],
                            framing=["blackstone"], roof=["red_nether_bricks"], foundation=["blackstone"],
                            secondary=["basalt"], trim=["polished_blackstone"], accent=["crimson"]),
    "ocean monument": dict(tags="ocean sea atlantis underwater prismarine aquatic", primary=["prismarine_bricks", "prismarine"],
                           framing=["dark_prismarine"], roof=["dark_prismarine"], foundation=["dark_prismarine"],
                           secondary=["prismarine"], trim=["dark_prismarine"], accent=["sea_lantern"]),
    "steampunk": dict(tags="steampunk industrial copper factory workshop victorian", primary=["bricks", "mud_bricks"],
                      framing=["spruce"], roof=["exposed_copper"], foundation=["stone_bricks"],
                      secondary=["polished_deepslate"], trim=["copper"], accent=["copper"]),
    "ruined": dict(tags="ruin ruined abandoned overgrown ancient crumbling", primary=["mossy_stone_bricks", "stone_bricks", "cobblestone"],
                   framing=["mossy_cobblestone"], roof=["cobbled_deepslate"], foundation=["mossy_cobblestone"],
                   secondary=["andesite"], trim=["stone_bricks"], accent=["moss"]),
    "mediterranean": dict(tags="mediterranean villa greek italian coastal terracotta white", primary=["white_concrete", "smooth_quartz"],
                          framing=["spruce"], roof=["terracotta"], foundation=["sandstone"],
                          secondary=["sandstone"], trim=["spruce"], accent=["light_blue_terracotta"]),
    "cherry blossom": dict(tags="cherry blossom pink spring asian garden", primary=["pale_oak", "white_concrete"],
                           framing=["cherry"], roof=["cherry"], foundation=["stone_bricks"],
                           secondary=["polished_andesite"], trim=["cherry"], accent=["pink_terracotta"]),
    "colonial": dict(tags="colonial farmhouse american clapboard white shutters", primary=["white_concrete", "birch"],
                     framing=["spruce"], roof=["deepslate_tiles"], foundation=["stone_bricks"],
                     secondary=["stone_bricks"], trim=["spruce"], accent=["dark_oak"]),
    "swamp witch": dict(tags="swamp witch hut spooky halloween dark rickety", primary=["spruce", "dark_oak"],
                        framing=["dark_oak"], roof=["dark_oak"], foundation=["mossy_cobblestone"],
                        secondary=["mossy_cobblestone"], trim=["spruce"], accent=["moss"]),
}
