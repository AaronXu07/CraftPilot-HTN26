"""Isometric PNG preview of a resolved grid, for eyeballing and golden tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from craftpilot.grid.semantic import SemanticGrid

_COLORS: dict[str, tuple[int, int, int]] = {
    "oak_planks": (177, 144, 86), "spruce_planks": (114, 84, 48), "birch_planks": (196, 179, 123),
    "dark_oak_planks": (66, 43, 20), "jungle_planks": (160, 115, 80), "acacia_planks": (168, 90, 50),
    "mangrove_planks": (117, 54, 48), "cherry_planks": (226, 178, 172), "bamboo_planks": (194, 173, 84),
    "oak_log": (109, 85, 50), "spruce_log": (58, 37, 16), "dark_oak_log": (48, 32, 16), "birch_log": (216, 215, 210),
    "stripped_oak_log": (180, 145, 85), "stripped_spruce_log": (116, 88, 51), "stripped_dark_oak_log": (72, 56, 36),
    "cobblestone": (127, 127, 127), "mossy_cobblestone": (104, 118, 92), "stone_bricks": (122, 122, 122),
    "cracked_stone_bricks": (110, 110, 110), "mossy_stone_bricks": (105, 120, 100), "stone": (125, 125, 125),
    "smooth_stone": (160, 160, 160), "andesite": (136, 136, 137), "polished_andesite": (132, 135, 134),
    "deepslate_bricks": (70, 70, 71), "deepslate_tiles": (55, 55, 56), "polished_deepslate": (72, 72, 73),
    "cobbled_deepslate": (77, 77, 80), "blackstone": (42, 36, 41), "polished_blackstone_bricks": (48, 42, 48),
    "bricks": (150, 97, 83), "mud_bricks": (137, 104, 79), "sandstone": (216, 203, 155), "smooth_sandstone": (223, 214, 170),
    "red_sandstone": (186, 99, 29), "quartz_block": (236, 230, 223), "smooth_quartz": (236, 230, 223),
    "diorite": (188, 188, 188), "calcite": (223, 224, 220), "white_concrete": (207, 213, 214),
    "light_gray_concrete": (125, 125, 115), "gray_concrete": (55, 58, 62), "black_concrete": (8, 10, 15),
    "glass": (120, 190, 230), "glass_pane": (120, 190, 230), "tinted_glass": (40, 35, 45),
    "lantern": (255, 200, 90), "campfire": (255, 150, 60), "waxed_copper_block": (192, 107, 79),
    "waxed_oxidized_copper": (82, 162, 132), "waxed_weathered_copper": (108, 153, 110),
    "nether_bricks": (44, 22, 26), "red_nether_bricks": (70, 7, 9), "prismarine": (99, 156, 151),
    "terracotta": (152, 94, 68), "granite": (149, 103, 86), "purpur_block": (170, 126, 170),
    "iron_bars": (140, 140, 140), "iron_block": (220, 220, 220), "moss_block": (89, 110, 45),
    "end_stone_bricks": (219, 222, 158), "packed_mud": (142, 106, 79), "obsidian": (15, 10, 24),
    "oak_leaves": (60, 110, 30), "spruce_leaves": (45, 90, 45), "birch_leaves": (90, 130, 50), "dark_oak_leaves": (50, 95, 25),
    "cherry_leaves": (230, 160, 190), "pale_oak_leaves": (110, 150, 80), "acacia_leaves": (90, 120, 40),
    "jungle_leaves": (55, 120, 30), "mangrove_leaves": (60, 105, 35), "vine": (50, 100, 40), "ladder": (150, 120, 70),
}


_DYES = {
    "white": (207, 213, 214), "light_gray": (125, 125, 115), "gray": (55, 58, 62), "black": (8, 10, 15),
    "brown": (96, 60, 32), "red": (142, 33, 33), "orange": (224, 97, 1), "yellow": (241, 175, 21),
    "lime": (94, 169, 24), "green": (73, 91, 36), "cyan": (21, 119, 136), "light_blue": (36, 137, 199),
    "blue": (45, 47, 143), "purple": (100, 32, 156), "magenta": (169, 48, 159), "pink": (214, 101, 143),
}


def _color(block_id: str) -> tuple[int, int, int]:
    name = block_id.split(":")[-1]
    if name in _COLORS:
        return _COLORS[name]
    for dye, rgb in _DYES.items():
        if name.startswith(dye + "_") and name.endswith(("concrete", "terracotta", "wool", "stained_glass",
                                                          "stained_glass_pane")):
            return rgb
    if name.startswith(("cracked_", "mossy_", "chiseled_", "polished_", "smooth_", "cut_")):
        base = name.split("_", 1)[1]
        if base in _COLORS:
            r, g, b = _COLORS[base]
            k = 0.85 if name.startswith(("cracked_", "cut_")) else 1.05
            return (min(255, int(r * k)), min(255, int(g * k)), min(255, int(b * k)))
    base = name
    for suffix in ("_stairs", "_slab", "_fence", "_wall", "_trapdoor", "_door", "_pane"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            break
    for candidate in (base, base + "_planks", base + "_block", base + "s", base.replace("brick", "bricks")):
        if candidate in _COLORS:
            return _COLORS[candidate]
    h = hashlib.md5(name.encode()).digest()
    return (90 + h[0] % 120, 90 + h[1] % 120, 90 + h[2] % 120)


def render(grid: SemanticGrid, path: Path, scale: int = 8) -> Path:
    W, H, D = grid.W, grid.H, grid.D
    ys_all = np.nonzero(grid.block >= 0)[1]
    H = int(ys_all.max()) + 1 if ys_all.size else 1
    a = scale
    hh = scale

    def proj(x: float, y: float, z: float) -> tuple[float, float]:
        return ((x - z) * a, (x + z) * a / 2 - y * hh)

    corners = [proj(0, 0, 0), proj(W, 0, 0), proj(0, 0, D), proj(W, 0, D), proj(0, H, 0), proj(W, H, 0),
               proj(0, H, D), proj(W, H, D)]
    minu = min(c[0] for c in corners)
    maxu = max(c[0] for c in corners)
    minv = min(c[1] for c in corners)
    maxv = max(c[1] for c in corners)
    img = Image.new("RGB", (int(maxu - minu) + 20, int(maxv - minv) + 20), (30, 32, 38))
    draw = ImageDraw.Draw(img)
    ox, oy = 10 - minu, 10 - minv

    def p(x: float, y: float, z: float) -> tuple[float, float]:
        u, v = proj(x, y, z)
        return (u + ox, v + oy)

    blocks = grid.block
    solid = blocks >= 0
    xs, ys, zs = np.nonzero(solid)
    order = np.argsort(xs + ys + zs, kind="stable")
    for i in order:
        x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
        ref = grid.palette[int(blocks[x, y, z])]
        top_exposed = y + 1 >= H or not solid[x, y + 1, z]
        east_exposed = x + 1 >= W or not solid[x + 1, y, z]
        south_exposed = z + 1 >= D or not solid[x, y, z + 1]
        if not (top_exposed or east_exposed or south_exposed):
            continue
        r, g, b = _color(ref.block_id)
        is_slab = ref.block_id.endswith("_slab") and ref.prop("type") == "bottom"
        y1 = y + (0.5 if is_slab else 1.0)
        if top_exposed:
            draw.polygon([p(x, y1, z), p(x + 1, y1, z), p(x + 1, y1, z + 1), p(x, y1, z + 1)], fill=(r, g, b))
        if south_exposed:
            k = 0.78
            draw.polygon([p(x, y, z + 1), p(x + 1, y, z + 1), p(x + 1, y1, z + 1), p(x, y1, z + 1)],
                         fill=(int(r * k), int(g * k), int(b * k)))
        if east_exposed:
            k = 0.6
            draw.polygon([p(x + 1, y, z), p(x + 1, y1, z), p(x + 1, y1, z + 1), p(x + 1, y, z + 1)],
                         fill=(int(r * k), int(g * k), int(b * k)))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path
