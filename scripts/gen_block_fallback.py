#!/usr/bin/env python3
"""Generate copilot/data/blocks_fallback.json: a Minecraft 1.21.1 block registry dump in the exact
shape of the mod's GET /blocks response (CONTRACTS.md §7). Used when the game is not running.

    python scripts/gen_block_fallback.py            # writes agent/copilot/data/blocks_fallback.json

Every entry: {"id": "minecraft:oak_stairs", "properties": {"facing": [...], ...}, "default": "minecraft:oak_stairs[...]"}.
Only ids we are confident exist in vanilla 1.21.1 are emitted; when in doubt a block is skipped.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "copilot", "data", "blocks_fallback.json")

BOOL = ["true", "false"]
HFACING = ["north", "south", "west", "east"]
FACING6 = ["north", "east", "south", "west", "up", "down"]
AXIS = ["x", "y", "z"]
ROT16 = [str(i) for i in range(16)]
LEVEL16 = [str(i) for i in range(16)]

WOODS = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "bamboo", "crimson", "warped"]
COLORS = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"]

# property set templates: name -> (properties, defaults)
P: Dict[str, Tuple[Dict[str, List[str]], Dict[str, str]]] = {
    "none": ({}, {}),
    "stairs": ({"facing": HFACING, "half": ["top", "bottom"], "shape": ["straight", "inner_left", "inner_right", "outer_left", "outer_right"], "waterlogged": BOOL},
               {"facing": "north", "half": "bottom", "shape": "straight", "waterlogged": "false"}),
    "slab": ({"type": ["top", "bottom", "double"], "waterlogged": BOOL}, {"type": "bottom", "waterlogged": "false"}),
    "wall": ({"east": ["none", "low", "tall"], "north": ["none", "low", "tall"], "south": ["none", "low", "tall"], "west": ["none", "low", "tall"], "up": BOOL, "waterlogged": BOOL},
             {"east": "none", "north": "none", "south": "none", "west": "none", "up": "true", "waterlogged": "false"}),
    "fence": ({"east": BOOL, "north": BOOL, "south": BOOL, "west": BOOL, "waterlogged": BOOL},
              {"east": "false", "north": "false", "south": "false", "west": "false", "waterlogged": "false"}),
    "pane": ({"east": BOOL, "north": BOOL, "south": BOOL, "west": BOOL, "waterlogged": BOOL},
             {"east": "false", "north": "false", "south": "false", "west": "false", "waterlogged": "false"}),
    "fence_gate": ({"facing": HFACING, "in_wall": BOOL, "open": BOOL, "powered": BOOL},
                   {"facing": "north", "in_wall": "false", "open": "false", "powered": "false"}),
    "door": ({"facing": HFACING, "half": ["upper", "lower"], "hinge": ["left", "right"], "open": BOOL, "powered": BOOL},
             {"facing": "north", "half": "lower", "hinge": "left", "open": "false", "powered": "false"}),
    "trapdoor": ({"facing": HFACING, "half": ["top", "bottom"], "open": BOOL, "powered": BOOL, "waterlogged": BOOL},
                 {"facing": "north", "half": "bottom", "open": "false", "powered": "false", "waterlogged": "false"}),
    "button": ({"face": ["floor", "wall", "ceiling"], "facing": HFACING, "powered": BOOL}, {"face": "wall", "facing": "north", "powered": "false"}),
    "pressure_plate": ({"powered": BOOL}, {"powered": "false"}),
    "weighted_plate": ({"power": LEVEL16}, {"power": "0"}),
    "axis": ({"axis": AXIS}, {"axis": "y"}),
    "leaves": ({"distance": ["1", "2", "3", "4", "5", "6", "7"], "persistent": BOOL, "waterlogged": BOOL}, {"distance": "7", "persistent": "false", "waterlogged": "false"}),
    "sapling": ({"stage": ["0", "1"]}, {"stage": "0"}),
    "sign": ({"rotation": ROT16, "waterlogged": BOOL}, {"rotation": "0", "waterlogged": "false"}),
    "wall_sign": ({"facing": HFACING, "waterlogged": BOOL}, {"facing": "north", "waterlogged": "false"}),
    "hanging_sign": ({"attached": BOOL, "rotation": ROT16, "waterlogged": BOOL}, {"attached": "false", "rotation": "0", "waterlogged": "false"}),
    "wall_hanging_sign": ({"facing": HFACING, "waterlogged": BOOL}, {"facing": "north", "waterlogged": "false"}),
    "hfacing": ({"facing": HFACING}, {"facing": "north"}),
    "facing6": ({"facing": FACING6}, {"facing": "north"}),
    "facing6_up": ({"facing": FACING6}, {"facing": "up"}),
    "hfacing_wl": ({"facing": HFACING, "waterlogged": BOOL}, {"facing": "north", "waterlogged": "false"}),
    "waterlogged": ({"waterlogged": BOOL}, {"waterlogged": "false"}),
    "lit": ({"lit": BOOL}, {"lit": "false"}),
    "lit_true": ({"lit": BOOL}, {"lit": "true"}),
    "snowy": ({"snowy": BOOL}, {"snowy": "false"}),
    "level16": ({"level": LEVEL16}, {"level": "0"}),
    "age16": ({"age": LEVEL16}, {"age": "0"}),
    "half_plant": ({"half": ["upper", "lower"]}, {"half": "lower"}),
    "rotation": ({"rotation": ROT16}, {"rotation": "0"}),
    "lantern": ({"hanging": BOOL, "waterlogged": BOOL}, {"hanging": "false", "waterlogged": "false"}),
    "chain": ({"axis": AXIS, "waterlogged": BOOL}, {"axis": "y", "waterlogged": "false"}),
    "campfire": ({"facing": HFACING, "lit": BOOL, "signal_fire": BOOL, "waterlogged": BOOL}, {"facing": "north", "lit": "true", "signal_fire": "false", "waterlogged": "false"}),
    "bed": ({"facing": HFACING, "occupied": BOOL, "part": ["head", "foot"]}, {"facing": "north", "occupied": "false", "part": "foot"}),
    "candle": ({"candles": ["1", "2", "3", "4"], "lit": BOOL, "waterlogged": BOOL}, {"candles": "1", "lit": "false", "waterlogged": "false"}),
    "candle_cake": ({"lit": BOOL}, {"lit": "false"}),
    "furnace": ({"facing": HFACING, "lit": BOOL}, {"facing": "north", "lit": "false"}),
    "chest": ({"facing": HFACING, "type": ["single", "left", "right"], "waterlogged": BOOL}, {"facing": "north", "type": "single", "waterlogged": "false"}),
    "barrel": ({"facing": FACING6, "open": BOOL}, {"facing": "north", "open": "false"}),
    "lectern": ({"facing": HFACING, "has_book": BOOL, "powered": BOOL}, {"facing": "north", "has_book": "false", "powered": "false"}),
    "grindstone": ({"face": ["floor", "wall", "ceiling"], "facing": HFACING}, {"face": "wall", "facing": "north"}),
    "bell": ({"attachment": ["floor", "ceiling", "single_wall", "double_wall"], "facing": HFACING, "powered": BOOL}, {"attachment": "floor", "facing": "north", "powered": "false"}),
    "composter": ({"level": [str(i) for i in range(9)]}, {"level": "0"}),
    "brewing_stand": ({"has_bottle_0": BOOL, "has_bottle_1": BOOL, "has_bottle_2": BOOL}, {"has_bottle_0": "false", "has_bottle_1": "false", "has_bottle_2": "false"}),
    "jukebox": ({"has_record": BOOL}, {"has_record": "false"}),
    "note_block": ({"instrument": ["harp", "basedrum", "snare", "hat", "bass", "flute", "bell", "guitar", "chime", "xylophone", "iron_xylophone", "cow_bell", "didgeridoo", "bit", "banjo", "pling", "zombie", "skeleton", "creeper", "dragon", "wither_skeleton", "piglin", "custom_head"], "note": [str(i) for i in range(25)], "powered": BOOL}, {"instrument": "harp", "note": "0", "powered": "false"}),
    "ladder": ({"facing": HFACING, "waterlogged": BOOL}, {"facing": "north", "waterlogged": "false"}),
    "scaffolding": ({"bottom": BOOL, "distance": [str(i) for i in range(8)], "waterlogged": BOOL}, {"bottom": "false", "distance": "7", "waterlogged": "false"}),
    "vine": ({"east": BOOL, "north": BOOL, "south": BOOL, "up": BOOL, "west": BOOL}, {"east": "false", "north": "false", "south": "false", "up": "false", "west": "false"}),
    "lichen": ({"down": BOOL, "east": BOOL, "north": BOOL, "south": BOOL, "up": BOOL, "west": BOOL, "waterlogged": BOOL}, {"down": "false", "east": "false", "north": "false", "south": "false", "up": "false", "west": "false", "waterlogged": "false"}),
    "mushroom_block": ({"down": BOOL, "east": BOOL, "north": BOOL, "south": BOOL, "up": BOOL, "west": BOOL}, {"down": "true", "east": "true", "north": "true", "south": "true", "up": "true", "west": "true"}),
    "snow": ({"layers": [str(i) for i in range(1, 9)]}, {"layers": "1"}),
    "bamboo": ({"age": ["0", "1"], "leaves": ["none", "small", "large"], "stage": ["0", "1"]}, {"age": "0", "leaves": "none", "stage": "0"}),
    "farmland": ({"moisture": [str(i) for i in range(8)]}, {"moisture": "0"}),
    "redstone_wire": ({"east": ["up", "side", "none"], "north": ["up", "side", "none"], "power": LEVEL16, "south": ["up", "side", "none"], "west": ["up", "side", "none"]}, {"east": "none", "north": "none", "power": "0", "south": "none", "west": "none"}),
    "repeater": ({"delay": ["1", "2", "3", "4"], "facing": HFACING, "locked": BOOL, "powered": BOOL}, {"delay": "1", "facing": "north", "locked": "false", "powered": "false"}),
    "comparator": ({"facing": HFACING, "mode": ["compare", "subtract"], "powered": BOOL}, {"facing": "north", "mode": "compare", "powered": "false"}),
    "lever": ({"face": ["floor", "wall", "ceiling"], "facing": HFACING, "powered": BOOL}, {"face": "wall", "facing": "north", "powered": "false"}),
    "observer": ({"facing": FACING6, "powered": BOOL}, {"facing": "south", "powered": "false"}),
    "piston": ({"extended": BOOL, "facing": FACING6}, {"extended": "false", "facing": "north"}),
    "dispenser": ({"facing": FACING6, "triggered": BOOL}, {"facing": "north", "triggered": "false"}),
    "hopper": ({"enabled": BOOL, "facing": ["down", "north", "south", "west", "east"]}, {"enabled": "true", "facing": "down"}),
    "tnt": ({"unstable": BOOL}, {"unstable": "false"}),
    "daylight": ({"inverted": BOOL, "power": LEVEL16}, {"inverted": "false", "power": "0"}),
    "target": ({"power": LEVEL16}, {"power": "0"}),
    "tripwire_hook": ({"attached": BOOL, "facing": HFACING, "powered": BOOL}, {"attached": "false", "facing": "north", "powered": "false"}),
    "redstone_torch": ({"lit": BOOL}, {"lit": "true"}),
    "redstone_wall_torch": ({"facing": HFACING, "lit": BOOL}, {"facing": "north", "lit": "true"}),
    "rail": ({"shape": ["north_south", "east_west", "ascending_east", "ascending_west", "ascending_north", "ascending_south", "south_east", "south_west", "north_west", "north_east"], "waterlogged": BOOL}, {"shape": "north_south", "waterlogged": "false"}),
    "powered_rail": ({"powered": BOOL, "shape": ["north_south", "east_west", "ascending_east", "ascending_west", "ascending_north", "ascending_south"], "waterlogged": BOOL}, {"powered": "false", "shape": "north_south", "waterlogged": "false"}),
    "copper_bulb": ({"lit": BOOL, "powered": BOOL}, {"lit": "false", "powered": "false"}),
    "amethyst_cluster": ({"facing": FACING6, "waterlogged": BOOL}, {"facing": "up", "waterlogged": "false"}),
    "sculk_catalyst": ({"bloom": BOOL}, {"bloom": "false"}),
    "sculk_shrieker": ({"can_summon": BOOL, "shrieking": BOOL, "waterlogged": BOOL}, {"can_summon": "false", "shrieking": "false", "waterlogged": "false"}),
    "sculk_sensor": ({"power": LEVEL16, "sculk_sensor_phase": ["inactive", "active", "cooldown"], "waterlogged": BOOL}, {"power": "0", "sculk_sensor_phase": "inactive", "waterlogged": "false"}),
    "calibrated_sculk_sensor": ({"facing": HFACING, "power": LEVEL16, "sculk_sensor_phase": ["inactive", "active", "cooldown"], "waterlogged": BOOL}, {"facing": "north", "power": "0", "sculk_sensor_phase": "inactive", "waterlogged": "false"}),
    "skull": ({"powered": BOOL, "rotation": ROT16}, {"powered": "false", "rotation": "0"}),
    "wall_skull": ({"facing": HFACING, "powered": BOOL}, {"facing": "north", "powered": "false"}),
    "beehive": ({"facing": HFACING, "honey_level": [str(i) for i in range(6)]}, {"facing": "north", "honey_level": "0"}),
    "cake": ({"bites": [str(i) for i in range(7)]}, {"bites": "0"}),
    "respawn_anchor": ({"charges": ["0", "1", "2", "3", "4"]}, {"charges": "0"}),
    "chiseled_bookshelf": ({"facing": HFACING, **{f"slot_{i}_occupied": BOOL for i in range(6)}}, {"facing": "north", **{f"slot_{i}_occupied": "false" for i in range(6)}}),
    "pointed_dripstone": ({"thickness": ["tip_merge", "tip", "frustum", "middle", "base"], "vertical_direction": ["up", "down"], "waterlogged": BOOL}, {"thickness": "tip", "vertical_direction": "up", "waterlogged": "false"}),
    "sea_pickle": ({"pickles": ["1", "2", "3", "4"], "waterlogged": BOOL}, {"pickles": "1", "waterlogged": "true"}),
    "coral": ({"waterlogged": BOOL}, {"waterlogged": "true"}),
    "coral_wall_fan": ({"facing": HFACING, "waterlogged": BOOL}, {"facing": "north", "waterlogged": "true"}),
    "pink_petals": ({"facing": HFACING, "flower_amount": ["1", "2", "3", "4"]}, {"facing": "north", "flower_amount": "1"}),
    "turtle_egg": ({"eggs": ["1", "2", "3", "4"], "hatch": ["0", "1", "2"]}, {"eggs": "1", "hatch": "0"}),
    "mangrove_propagule": ({"age": ["0", "1", "2", "3", "4"], "hanging": BOOL, "stage": ["0", "1"], "waterlogged": BOOL}, {"age": "0", "hanging": "false", "stage": "0", "waterlogged": "false"}),
    "big_dripleaf": ({"facing": HFACING, "tilt": ["none", "unstable", "partial", "full"], "waterlogged": BOOL}, {"facing": "north", "tilt": "none", "waterlogged": "false"}),
    "small_dripleaf": ({"facing": HFACING, "half": ["upper", "lower"], "waterlogged": BOOL}, {"facing": "north", "half": "lower", "waterlogged": "false"}),
    "cave_vines": ({"age": [str(i) for i in range(26)], "berries": BOOL}, {"age": "0", "berries": "false"}),
    "cave_vines_plant": ({"berries": BOOL}, {"berries": "false"}),
    "weeping_vines": ({"age": [str(i) for i in range(26)]}, {"age": "0"}),
    "chorus_flower": ({"age": ["0", "1", "2", "3", "4", "5"]}, {"age": "0"}),
    "chorus_plant": ({"down": BOOL, "east": BOOL, "north": BOOL, "south": BOOL, "up": BOOL, "west": BOOL}, {"down": "false", "east": "false", "north": "false", "south": "false", "up": "false", "west": "false"}),
    "end_portal_frame": ({"eye": BOOL, "facing": HFACING}, {"eye": "false", "facing": "north"}),
    "crop7": ({"age": [str(i) for i in range(8)]}, {"age": "0"}),
    "crop3": ({"age": ["0", "1", "2", "3"]}, {"age": "0"}),
    "stem": ({"age": [str(i) for i in range(8)]}, {"age": "0"}),
    "attached_stem": ({"facing": HFACING}, {"facing": "north"}),
    "cocoa": ({"age": ["0", "1", "2"], "facing": HFACING}, {"age": "0", "facing": "north"}),
    "nether_wart": ({"age": ["0", "1", "2", "3"]}, {"age": "0"}),
    "sweet_berry": ({"age": ["0", "1", "2", "3"]}, {"age": "0"}),
    "kelp": ({"age": [str(i) for i in range(26)]}, {"age": "0"}),
    "tall_seagrass": ({"half": ["upper", "lower"]}, {"half": "lower"}),
    "fire": ({"age": LEVEL16, "east": BOOL, "north": BOOL, "south": BOOL, "up": BOOL, "west": BOOL}, {"age": "0", "east": "false", "north": "false", "south": "false", "up": "false", "west": "false"}),
    "light": ({"level": LEVEL16, "waterlogged": BOOL}, {"level": "15", "waterlogged": "false"}),
    "crafter": ({"crafting": BOOL, "orientation": ["down_east", "down_north", "down_south", "down_west", "up_east", "up_north", "up_south", "up_west", "west_up", "east_up", "north_up", "south_up"], "triggered": BOOL}, {"crafting": "false", "orientation": "north_up", "triggered": "false"}),
    "trial_spawner": ({"ominous": BOOL, "trial_spawner_state": ["inactive", "waiting_for_players", "active", "waiting_for_reward_ejection", "ejecting_reward", "cooldown"]}, {"ominous": "false", "trial_spawner_state": "inactive"}),
    "vault": ({"facing": HFACING, "ominous": BOOL, "vault_state": ["inactive", "active", "unlocking", "ejecting"]}, {"facing": "north", "ominous": "false", "vault_state": "inactive"}),
    "decorated_pot": ({"cracked": BOOL, "facing": HFACING, "waterlogged": BOOL}, {"cracked": "false", "facing": "north", "waterlogged": "false"}),
    "lightning_rod": ({"facing": FACING6, "powered": BOOL, "waterlogged": BOOL}, {"facing": "up", "powered": "false", "waterlogged": "false"}),
    "command_block": ({"conditional": BOOL, "facing": FACING6}, {"conditional": "false", "facing": "north"}),
    "structure_block": ({"mode": ["save", "load", "corner", "data"]}, {"mode": "load"}),
    "jigsaw": ({"orientation": ["down_east", "down_north", "down_south", "down_west", "up_east", "up_north", "up_south", "up_west", "west_up", "east_up", "north_up", "south_up"]}, {"orientation": "north_up"}),
    "hay": ({"axis": AXIS}, {"axis": "y"}),
    "sniffer_egg": ({"hatch": ["0", "1", "2"]}, {"hatch": "0"}),
    "frosted_ice": ({"age": ["0", "1", "2", "3"]}, {"age": "0"}),
    "water_cauldron": ({"level": ["1", "2", "3"]}, {"level": "1"}),
    "infested": ({}, {}),
    "bubble_column": ({"drag": BOOL}, {"drag": "true"}),
    "piston_head": ({"facing": FACING6, "short": BOOL, "type": ["normal", "sticky"]}, {"facing": "north", "short": "false", "type": "normal"}),
    "moving_piston": ({"facing": FACING6, "type": ["normal", "sticky"]}, {"facing": "north", "type": "normal"}),
    "nether_portal": ({"axis": ["x", "z"]}, {"axis": "x"}),
    "tripwire": ({"attached": BOOL, "disarmed": BOOL, "east": BOOL, "north": BOOL, "powered": BOOL, "south": BOOL, "west": BOOL}, {"attached": "false", "disarmed": "false", "east": "false", "north": "false", "powered": "false", "south": "false", "west": "false"}),
    "heavy_core": ({"waterlogged": BOOL}, {"waterlogged": "false"}),
}


class Gen:
    def __init__(self) -> None:
        self.blocks: Dict[str, Tuple[Dict[str, List[str]], Dict[str, str]]] = {}

    def add(self, name: str, tmpl: str = "none") -> None:
        if name in self.blocks:
            return
        props, defaults = P[tmpl]
        self.blocks[name] = (dict(props), dict(defaults))

    def many(self, names: List[str], tmpl: str = "none") -> None:
        for n in names:
            self.add(n, tmpl)

    def stone_family(self, base: str, stairs: bool = True, slab: bool = True, wall: bool = False, stem: str | None = None) -> None:
        """base block + optional stairs/slab/wall named after `stem` (default: singularised base)."""
        self.add(base)
        st = stem or singular(base)
        if stairs:
            self.add(f"{st}_stairs", "stairs")
        if slab:
            self.add(f"{st}_slab", "slab")
        if wall:
            self.add(f"{st}_wall", "wall")

    def dump(self) -> dict:
        out = []
        for name, (props, defaults) in sorted(self.blocks.items()):
            bid = f"minecraft:{name}"
            if props:
                default = bid + "[" + ",".join(f"{k}={defaults[k]}" for k in props) + "]"
            else:
                default = bid
            out.append({"id": bid, "properties": props, "default": default})
        return {"blocks": out, "mc_version": "1.21.1", "source": "scripts/gen_block_fallback.py"}


def singular(base: str) -> str:
    if base.endswith("_bricks"):
        return base[: -len("_bricks")] + "_brick"
    if base.endswith("_tiles"):
        return base[: -len("_tiles")] + "_tile"
    if base.endswith("_block"):
        return base[: -len("_block")]
    if base.endswith("_planks"):
        return base[: -len("_planks")]
    if base == "bricks":
        return "brick"
    return base


def build() -> Gen:
    g = Gen()
    g.add("air")
    g.add("cave_air")
    g.add("void_air")

    # ---------------- wood ----------------
    for w in WOODS:
        g.add(f"{w}_planks")
        g.add(f"{w}_stairs", "stairs")
        g.add(f"{w}_slab", "slab")
        g.add(f"{w}_fence", "fence")
        g.add(f"{w}_fence_gate", "fence_gate")
        g.add(f"{w}_door", "door")
        g.add(f"{w}_trapdoor", "trapdoor")
        g.add(f"{w}_button", "button")
        g.add(f"{w}_pressure_plate", "pressure_plate")
        g.add(f"{w}_sign", "sign")
        g.add(f"{w}_wall_sign", "wall_sign")
        g.add(f"{w}_hanging_sign", "hanging_sign")
        g.add(f"{w}_wall_hanging_sign", "wall_hanging_sign")
        if w in ("crimson", "warped"):
            g.add(f"{w}_stem", "axis")
            g.add(f"{w}_hyphae", "axis")
            g.add(f"stripped_{w}_stem", "axis")
            g.add(f"stripped_{w}_hyphae", "axis")
            g.add(f"{w}_fungus")
            g.add(f"{w}_roots")
            g.add(f"{w}_nylium")
        elif w == "bamboo":
            g.add("bamboo_block", "axis")
            g.add("stripped_bamboo_block", "axis")
            g.add("bamboo_mosaic")
            g.add("bamboo_mosaic_stairs", "stairs")
            g.add("bamboo_mosaic_slab", "slab")
            g.add("bamboo", "bamboo")
            g.add("bamboo_sapling")
        else:
            g.add(f"{w}_log", "axis")
            g.add(f"{w}_wood", "axis")
            g.add(f"stripped_{w}_log", "axis")
            g.add(f"stripped_{w}_wood", "axis")
            g.add(f"{w}_leaves", "leaves")
            if w == "mangrove":
                g.add("mangrove_propagule", "mangrove_propagule")
                g.add("mangrove_roots", "waterlogged")
                g.add("muddy_mangrove_roots", "axis")
            else:
                g.add(f"{w}_sapling", "sapling")
    g.many(["azalea_leaves", "flowering_azalea_leaves"], "leaves")
    g.many(["azalea", "flowering_azalea"])
    g.add("petrified_oak_slab", "slab")

    # ---------------- stone families ----------------
    g.stone_family("stone", wall=False)
    g.stone_family("cobblestone", wall=True)
    g.stone_family("mossy_cobblestone", wall=True)
    g.stone_family("stone_bricks", wall=True)
    g.stone_family("mossy_stone_bricks", wall=True)
    g.many(["cracked_stone_bricks", "chiseled_stone_bricks"])
    g.add("smooth_stone")
    g.add("smooth_stone_slab", "slab")
    for s in ("andesite", "diorite", "granite"):
        g.stone_family(s, wall=True)
        g.stone_family(f"polished_{s}")
    g.add("deepslate", "axis")
    g.stone_family("cobbled_deepslate", wall=True)
    g.stone_family("polished_deepslate", wall=True)
    g.stone_family("deepslate_bricks", wall=True)
    g.stone_family("deepslate_tiles", wall=True)
    g.many(["cracked_deepslate_bricks", "cracked_deepslate_tiles", "chiseled_deepslate", "reinforced_deepslate"])
    g.stone_family("tuff", wall=True)
    g.stone_family("polished_tuff", wall=True)
    g.stone_family("tuff_bricks", wall=True)
    g.many(["chiseled_tuff", "chiseled_tuff_bricks"])
    g.stone_family("blackstone", wall=True)
    g.stone_family("polished_blackstone", wall=True)
    g.add("polished_blackstone_button", "button")
    g.add("polished_blackstone_pressure_plate", "pressure_plate")
    g.stone_family("polished_blackstone_bricks", wall=True)
    g.many(["cracked_polished_blackstone_bricks", "chiseled_polished_blackstone", "gilded_blackstone"])
    g.add("basalt", "axis")
    g.add("polished_basalt", "axis")
    g.add("smooth_basalt")
    for s in ("sandstone", "red_sandstone"):
        g.stone_family(s, wall=True)
        g.stone_family(f"smooth_{s}")
        g.add(f"cut_{s}")
        g.add(f"cut_{s}_slab", "slab")
        g.add(f"chiseled_{s}")
    g.stone_family("bricks", wall=True)
    g.stone_family("mud_bricks", wall=True)
    g.many(["mud", "packed_mud"])
    g.stone_family("nether_bricks", wall=True)
    g.add("nether_brick_fence", "fence")
    g.stone_family("red_nether_bricks", wall=True)
    g.many(["cracked_nether_bricks", "chiseled_nether_bricks"])
    g.add("end_stone")
    g.stone_family("end_stone_bricks", wall=True)
    g.stone_family("prismarine", wall=True)
    g.stone_family("prismarine_bricks")
    g.stone_family("dark_prismarine")
    g.stone_family("purpur_block", stem="purpur")
    g.add("purpur_pillar", "axis")
    g.stone_family("quartz_block", stem="quartz")
    g.stone_family("smooth_quartz")
    g.many(["quartz_bricks", "chiseled_quartz_block"])
    g.add("quartz_pillar", "axis")
    g.many(["calcite", "dripstone_block", "obsidian", "crying_obsidian", "bedrock", "netherrack", "soul_sand", "soul_soil", "magma_block", "glowstone", "shroomlight", "sea_lantern", "nether_wart_block", "warped_wart_block", "bone_block"])
    g.add("bone_block", "axis")
    g.blocks["bone_block"] = (dict(P["axis"][0]), dict(P["axis"][1]))
    g.add("pointed_dripstone", "pointed_dripstone")
    g.add("respawn_anchor", "respawn_anchor")
    g.add("ancient_debris")
    g.add("lodestone")
    g.add("end_portal_frame", "end_portal_frame")
    g.add("dragon_egg")
    g.add("chorus_plant", "chorus_plant")
    g.add("chorus_flower", "chorus_flower")
    g.add("end_rod", "facing6_up")

    # ---------------- copper ----------------
    for wax in ("", "waxed_"):
        for ox in ("", "exposed_", "weathered_", "oxidized_"):
            g.add(f"{wax}{ox}copper_block" if ox == "" else f"{wax}{ox}copper")
            g.add(f"{wax}{ox}cut_copper")
            g.add(f"{wax}{ox}cut_copper_stairs", "stairs")
            g.add(f"{wax}{ox}cut_copper_slab", "slab")
            g.add(f"{wax}{ox}chiseled_copper")
            g.add(f"{wax}{ox}copper_grate", "waterlogged")
            g.add(f"{wax}{ox}copper_bulb", "copper_bulb")
            g.add(f"{wax}{ox}copper_door", "door")
            g.add(f"{wax}{ox}copper_trapdoor", "trapdoor")
    g.many(["raw_copper_block", "copper_ore", "deepslate_copper_ore"])
    g.add("lightning_rod", "lightning_rod")

    # ---------------- colours ----------------
    for c in COLORS:
        g.add(f"{c}_wool")
        g.add(f"{c}_carpet")
        g.add(f"{c}_concrete")
        g.add(f"{c}_concrete_powder")
        g.add(f"{c}_terracotta")
        g.add(f"{c}_glazed_terracotta", "hfacing")
        g.add(f"{c}_stained_glass")
        g.add(f"{c}_stained_glass_pane", "pane")
        g.add(f"{c}_bed", "bed")
        g.add(f"{c}_banner", "rotation")
        g.add(f"{c}_wall_banner", "hfacing")
        g.add(f"{c}_candle", "candle")
        g.add(f"{c}_candle_cake", "candle_cake")
        g.add(f"{c}_shulker_box", "facing6_up")
    g.many(["terracotta", "glass", "tinted_glass"])
    g.add("glass_pane", "pane")
    g.add("iron_bars", "pane")
    g.add("shulker_box", "facing6_up")
    g.add("candle", "candle")
    g.add("candle_cake", "candle_cake")
    g.add("cake", "cake")

    # ---------------- ores / metals ----------------
    for ore in ("coal", "iron", "gold", "redstone", "emerald", "lapis", "diamond"):
        g.add(f"{ore}_ore", "lit" if ore == "redstone" else "none")
        g.add(f"deepslate_{ore}_ore", "lit" if ore == "redstone" else "none")
    g.many(["nether_gold_ore", "nether_quartz_ore"])
    g.many(["coal_block", "iron_block", "gold_block", "redstone_block", "emerald_block", "lapis_block", "diamond_block", "netherite_block", "raw_iron_block", "raw_gold_block", "amethyst_block", "budding_amethyst"])
    g.add("amethyst_cluster", "amethyst_cluster")
    g.many(["large_amethyst_bud", "medium_amethyst_bud", "small_amethyst_bud"], "amethyst_cluster")
    g.add("iron_door", "door")
    g.add("iron_trapdoor", "trapdoor")
    g.add("heavy_weighted_pressure_plate", "weighted_plate")
    g.add("light_weighted_pressure_plate", "weighted_plate")
    g.add("chain", "chain")
    g.add("anvil", "hfacing")
    g.add("chipped_anvil", "hfacing")
    g.add("damaged_anvil", "hfacing")
    g.add("cauldron")
    g.add("water_cauldron", "water_cauldron")
    g.add("lava_cauldron")
    g.add("powder_snow_cauldron", "water_cauldron")
    g.add("bell", "bell")
    g.add("lantern", "lantern")
    g.add("soul_lantern", "lantern")
    g.add("heavy_core", "heavy_core")

    # ---------------- nature ----------------
    g.many(["dirt", "coarse_dirt", "rooted_dirt", "clay", "sand", "red_sand", "gravel", "moss_block", "moss_carpet", "snow_block", "powder_snow", "ice", "packed_ice", "blue_ice", "dirt_path", "suspicious_sand", "suspicious_gravel", "sponge", "wet_sponge", "dried_kelp_block", "hay_block", "melon", "pumpkin", "honey_block", "honeycomb_block", "slime_block", "cobweb", "sculk", "spawner", "barrier", "structure_void", "sea_pickle"])
    g.blocks["hay_block"] = (dict(P["axis"][0]), dict(P["axis"][1]))
    g.blocks["sea_pickle"] = (dict(P["sea_pickle"][0]), dict(P["sea_pickle"][1]))
    g.many(["grass_block", "podzol", "mycelium"], "snowy")
    g.add("farmland", "farmland")
    g.add("snow", "snow")
    g.add("frosted_ice", "frosted_ice")
    g.many(["water", "lava"], "level16")
    g.add("bubble_column", "bubble_column")
    g.many(["dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip", "white_tulip", "pink_tulip", "oxeye_daisy", "cornflower", "lily_of_the_valley", "wither_rose", "torchflower", "short_grass", "fern", "dead_bush", "seagrass", "brown_mushroom", "red_mushroom", "lily_pad", "nether_sprouts", "spore_blossom", "hanging_roots", "glow_lichen", "sculk_vein", "frogspawn", "moss_carpet"])
    g.blocks["glow_lichen"] = (dict(P["lichen"][0]), dict(P["lichen"][1]))
    g.blocks["sculk_vein"] = (dict(P["lichen"][0]), dict(P["lichen"][1]))
    g.blocks["hanging_roots"] = (dict(P["waterlogged"][0]), dict(P["waterlogged"][1]))
    g.blocks["seagrass"] = ({}, {})
    g.many(["sunflower", "lilac", "rose_bush", "peony", "tall_grass", "large_fern", "pitcher_plant"], "half_plant")
    g.add("tall_seagrass", "tall_seagrass")
    g.add("pink_petals", "pink_petals")
    g.add("vine", "vine")
    g.add("sugar_cane", "age16")
    g.add("cactus", "age16")
    g.many(["brown_mushroom_block", "red_mushroom_block", "mushroom_stem"], "mushroom_block")
    g.add("kelp", "kelp")
    g.add("kelp_plant")
    g.add("weeping_vines", "weeping_vines")
    g.add("weeping_vines_plant")
    g.add("twisting_vines", "weeping_vines")
    g.add("twisting_vines_plant")
    g.add("cave_vines", "cave_vines")
    g.add("cave_vines_plant", "cave_vines_plant")
    g.add("big_dripleaf", "big_dripleaf")
    g.add("big_dripleaf_stem", "hfacing_wl")
    g.add("small_dripleaf", "small_dripleaf")
    g.add("turtle_egg", "turtle_egg")
    g.add("sniffer_egg", "sniffer_egg")
    g.many(["wheat", "carrots", "potatoes"], "crop7")
    g.add("beetroots", "crop3")
    g.add("torchflower_crop", "crop3")
    g.add("pitcher_crop", "crop3")
    g.add("nether_wart", "nether_wart")
    g.add("sweet_berry_bush", "sweet_berry")
    g.add("cocoa", "cocoa")
    g.many(["melon_stem", "pumpkin_stem"], "stem")
    g.many(["attached_melon_stem", "attached_pumpkin_stem"], "attached_stem")
    g.many(["carved_pumpkin", "jack_o_lantern"], "hfacing")
    for coral in ("tube", "brain", "bubble", "fire", "horn"):
        for dead in ("", "dead_"):
            g.add(f"{dead}{coral}_coral_block")
            g.add(f"{dead}{coral}_coral", "coral")
            g.add(f"{dead}{coral}_coral_fan", "coral")
            g.add(f"{dead}{coral}_coral_wall_fan", "coral_wall_fan")
    g.many(["ochre_froglight", "verdant_froglight", "pearlescent_froglight"], "axis")
    g.many(["sculk_catalyst"], "sculk_catalyst")
    g.add("sculk_shrieker", "sculk_shrieker")
    g.add("sculk_sensor", "sculk_sensor")
    g.add("calibrated_sculk_sensor", "calibrated_sculk_sensor")
    g.many(["infested_stone", "infested_cobblestone", "infested_stone_bricks", "infested_mossy_stone_bricks", "infested_cracked_stone_bricks", "infested_chiseled_stone_bricks", "infested_deepslate"], "infested")
    g.blocks["infested_deepslate"] = (dict(P["axis"][0]), dict(P["axis"][1]))
    g.add("flower_pot")
    for p in ("torchflower", "oak_sapling", "spruce_sapling", "birch_sapling", "jungle_sapling", "acacia_sapling", "dark_oak_sapling", "cherry_sapling", "mangrove_propagule", "fern", "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip", "white_tulip", "pink_tulip", "oxeye_daisy", "cornflower", "lily_of_the_valley", "wither_rose", "red_mushroom", "brown_mushroom", "dead_bush", "cactus", "bamboo", "crimson_fungus", "warped_fungus", "crimson_roots", "warped_roots", "azalea_bush", "flowering_azalea_bush"):
        g.add(f"potted_{p}")

    # ---------------- lighting / props / functional ----------------
    g.many(["torch", "soul_torch"])
    g.many(["wall_torch", "soul_wall_torch"], "hfacing")
    g.add("redstone_torch", "redstone_torch")
    g.add("redstone_wall_torch", "redstone_wall_torch")
    g.add("redstone_lamp", "lit")
    g.add("campfire", "campfire")
    g.add("soul_campfire", "campfire")
    g.add("ladder", "ladder")
    g.add("scaffolding", "scaffolding")
    g.many(["bookshelf", "crafting_table", "smithing_table", "fletching_table", "cartography_table", "enchanting_table", "beacon", "conduit"])
    g.blocks["conduit"] = (dict(P["waterlogged"][0]), dict(P["waterlogged"][1]))
    g.add("chiseled_bookshelf", "chiseled_bookshelf")
    g.many(["furnace", "blast_furnace", "smoker"], "furnace")
    g.many(["chest", "trapped_chest"], "chest")
    g.add("ender_chest", "hfacing_wl")
    g.add("barrel", "barrel")
    g.add("lectern", "lectern")
    g.add("loom", "hfacing")
    g.add("stonecutter", "hfacing")
    g.add("grindstone", "grindstone")
    g.add("composter", "composter")
    g.add("brewing_stand", "brewing_stand")
    g.add("jukebox", "jukebox")
    g.add("note_block", "note_block")
    g.add("bee_nest", "beehive")
    g.add("beehive", "beehive")
    g.add("decorated_pot", "decorated_pot")
    g.add("crafter", "crafter")
    g.add("trial_spawner", "trial_spawner")
    g.add("vault", "vault")
    g.add("nether_portal", "nether_portal")
    g.add("end_portal")
    g.add("end_gateway")
    g.add("fire", "fire")
    g.add("soul_fire")
    g.add("light", "light")
    for skull in ("skeleton_skull", "wither_skeleton_skull", "zombie_head", "player_head", "creeper_head", "dragon_head", "piglin_head"):
        g.add(skull, "skull")
        wall = skull.replace("_skull", "_wall_skull").replace("_head", "_wall_head")
        g.add(wall, "wall_skull")

    # ---------------- redstone ----------------
    g.add("redstone_wire", "redstone_wire")
    g.add("repeater", "repeater")
    g.add("comparator", "comparator")
    g.add("lever", "lever")
    g.add("stone_button", "button")
    g.add("stone_pressure_plate", "pressure_plate")
    g.add("observer", "observer")
    g.many(["piston", "sticky_piston"], "piston")
    g.add("piston_head", "piston_head")
    g.add("moving_piston", "moving_piston")
    g.many(["dispenser", "dropper"], "dispenser")
    g.add("hopper", "hopper")
    g.add("tnt", "tnt")
    g.add("daylight_detector", "daylight")
    g.add("target", "target")
    g.add("tripwire_hook", "tripwire_hook")
    g.add("tripwire", "tripwire")
    g.add("rail", "rail")
    g.many(["powered_rail", "detector_rail", "activator_rail"], "powered_rail")
    g.many(["command_block", "chain_command_block", "repeating_command_block"], "command_block")
    g.add("structure_block", "structure_block")
    g.add("jigsaw", "jigsaw")
    return g


def main() -> None:
    g = build()
    data = g.dump()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {len(data['blocks'])} blocks to {OUT}")


if __name__ == "__main__":
    main()
