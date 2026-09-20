"""Resolver on every wood and stone family from the registry cache (T6).

`default_registry()` loads `copilot/data/cache/blocks_live.json` (the last mod dump) when present, else
the bundled fallback dump — either way the family tables below must resolve slab/stairs voxels to real,
validating block states for every building family the prompts can name.
"""
import numpy as np
import pytest

from copilot.engine.coretypes import Facing, FitResult, Half, Kind, RasterResult
from copilot.engine.registry import WOODS, default_registry
from copilot.engine.resolver import resolve
from copilot.engine.scene import Scene, apply_op

STONES = ["stone", "cobblestone", "mossy_cobblestone", "stone_bricks", "mossy_stone_bricks", "andesite", "polished_andesite",
          "diorite", "polished_diorite", "granite", "polished_granite", "cobbled_deepslate", "polished_deepslate", "deepslate_bricks",
          "deepslate_tiles", "tuff", "polished_tuff", "tuff_bricks", "blackstone", "polished_blackstone", "polished_blackstone_bricks",
          "sandstone", "smooth_sandstone", "red_sandstone", "smooth_red_sandstone", "bricks", "mud_bricks", "nether_bricks",
          "red_nether_bricks", "end_stone_bricks", "prismarine", "prismarine_bricks", "dark_prismarine", "purpur_block",
          "quartz_block", "smooth_quartz", "cut_copper"]


@pytest.fixture(scope="module")
def reg():
    return default_registry()


def _fitted_raster():
    """A 3-voxel row: slab(top) / stairs(east,bottom) / wall, all material 1."""
    shape = (3, 1, 3)
    r = RasterResult(origin=(0, 0, 0), material=np.zeros(shape, dtype=np.int16), material_names=["", "m"],
                     owner=np.full(shape, -1, dtype=np.int32), sdf=np.full(shape, np.inf, dtype=np.float32), props={})
    r.material[0:3, 0, 0:3] = 1
    f = FitResult.full_blocks(r)
    f.kind[0, 0, 0] = Kind.SLAB
    f.half[0, 0, 0] = Half.TOP
    f.kind[1, 0, 0] = Kind.STAIRS
    f.facing[1, 0, 0] = Facing.EAST
    f.half[1, 0, 0] = Half.BOTTOM
    f.kind[2, 0, 0] = Kind.WALL
    return r, f


def _resolve(reg, base, fit="walls"):
    sc, _ = apply_op(Scene(), "define_material", name="m", spec={"base": base, "fit": fit})
    r, f = _fitted_raster()
    return resolve(r, f, sc, reg)


@pytest.mark.parametrize("wood", WOODS)
def test_wood_family_slab_and_stairs(reg, wood):
    base = f"{wood}_planks"
    fam = reg.family(base)
    bm = _resolve(reg, base)
    assert bm[(0, 0, 0)] == f"{fam['slab']}[type=top]"
    assert bm[(1, 0, 0)] == f"{fam['stairs']}[facing=east,half=bottom]"
    assert bm[(1, 0, 1)] == f"minecraft:{base}"
    for state in bm.values():
        ok, msg = reg.validate_state(state)
        assert ok, (wood, state, msg)


@pytest.mark.parametrize("base", STONES)
def test_stone_family_slab_stairs_wall(reg, base):
    fam = reg.family(base)
    bm = _resolve(reg, base)
    assert bm[(0, 0, 0)] == f"{fam['slab']}[type=top]"
    assert bm[(1, 0, 0)] == f"{fam['stairs']}[facing=east,half=bottom]"
    if "wall" in fam:
        assert bm[(2, 0, 0)] == fam["wall"], base
    else:  # no wall form in this family: the voxel stays a full block, never an invalid id
        assert bm[(2, 0, 0)] == f"minecraft:{base}"
    for state in bm.values():
        ok, msg = reg.validate_state(state)
        assert ok, (base, state, msg)


def test_log_base_resolves_to_plank_family(reg):
    """A log-based material still gets plank stairs/slabs (the family stem is the wood)."""
    bm = _resolve(reg, "spruce_log", fit="stairs+slab")
    assert bm[(0, 0, 0)].startswith("minecraft:spruce_slab") and bm[(1, 0, 0)].startswith("minecraft:spruce_stairs")
    assert bm[(1, 0, 1)].startswith("minecraft:spruce_log")
