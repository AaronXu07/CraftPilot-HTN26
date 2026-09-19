import numpy as np
import pytest

from copilot.engine.coretypes import FACING_NAMES, Facing, FitResult, Half, Kind, RasterResult, parse_state
from copilot.engine.registry import default_registry
from copilot.engine.resolver import resolve
from copilot.engine.scene import Scene, apply_op


def make_raster(shape=(6, 5, 6), origin=(0, 0, 0), names=("", "wall", "roof")):
    X, Y, Z = shape
    material = np.zeros(shape, dtype=np.int16)
    owner = np.full(shape, -1, dtype=np.int32)
    sdf = np.full(shape, np.inf, dtype=np.float32)
    return RasterResult(origin=origin, material=material, material_names=list(names), owner=owner, sdf=sdf, props={})


def fit_from(raster):
    return FitResult.full_blocks(raster)


@pytest.fixture(scope="module")
def reg():
    return default_registry()


def scene_with(mats):
    sc = Scene()
    for n, spec in mats.items():
        sc, _ = apply_op(sc, "define_material", name=n, spec=spec)
    return sc


def test_full_blocks_validate_and_use_palette(reg):
    r = make_raster()
    r.material[1:5, 0:4, 1:5] = 1
    r.owner[1:5, 0:4, 1:5] = 0
    f = fit_from(r)
    sc = scene_with({"wall": {"base": "stone_bricks", "palette": [["stone_bricks", 0.6], ["cobblestone", 0.4]]}})
    w = []
    bm = resolve(r, f, sc, reg, seed=1, warnings=w)
    assert len(bm) == 4 * 4 * 4 and not w
    ids = {parse_state(s)[0] for s in bm.values()}
    assert ids <= {"minecraft:stone_bricks", "minecraft:cobblestone"} and len(ids) == 2
    for s in bm.values():
        assert reg.validate_state(s)[0]
    assert (1, 0, 1) in bm and (0, 0, 0) not in bm


def test_slab_and_stairs_substitution(reg):
    r = make_raster()
    r.material[0:3, 0, 0:3] = 1
    f = fit_from(r)
    f.kind[0, 0, 0] = Kind.SLAB
    f.half[0, 0, 0] = Half.TOP
    f.kind[1, 0, 0] = Kind.STAIRS
    f.facing[1, 0, 0] = Facing.EAST
    f.half[1, 0, 0] = Half.BOTTOM
    f.kind[2, 0, 0] = Kind.WALL
    sc = scene_with({"wall": {"base": "stone_bricks", "fit": "walls"}})
    bm = resolve(r, f, sc, reg)
    assert bm[(0, 0, 0)] == "minecraft:stone_brick_slab[type=top]"
    assert bm[(1, 0, 0)] == "minecraft:stone_brick_stairs[facing=east,half=bottom]"
    assert bm[(2, 0, 0)] == "minecraft:stone_brick_wall"
    assert bm[(1, 0, 1)] == "minecraft:stone_bricks"
    # fit=none keeps full blocks
    sc2 = scene_with({"wall": {"base": "stone_bricks", "fit": "none"}})
    bm2 = resolve(r, f, sc2, reg)
    assert bm2[(0, 0, 0)] == "minecraft:stone_bricks" and bm2[(1, 0, 0)] == "minecraft:stone_bricks"
    # fit=slab: stairs voxels fall back to full, slab voxels become slabs
    sc3 = scene_with({"wall": {"base": "stone_bricks", "fit": "slab"}})
    bm3 = resolve(r, f, sc3, reg)
    assert bm3[(0, 0, 0)].startswith("minecraft:stone_brick_slab") and bm3[(1, 0, 0)] == "minecraft:stone_bricks"


def test_family_without_form_falls_back(reg):
    r = make_raster()
    r.material[0:2, 0, 0] = 1
    f = fit_from(r)
    f.kind[0, 0, 0] = Kind.STAIRS
    f.kind[1, 0, 0] = Kind.SLAB
    # furnace has no stairs or slab -> full block
    sc = scene_with({"wall": {"base": "furnace"}})
    bm = resolve(r, f, sc, reg)
    assert bm[(0, 0, 0)] == "minecraft:furnace" and bm[(1, 0, 0)] == "minecraft:furnace"
    # cracked bricks have no stairs but the base (stone_bricks) does -> base family stairs
    sc = scene_with({"wall": {"base": "stone_bricks", "palette": [["cracked_stone_bricks", 1.0]]}})
    bm = resolve(r, f, sc, reg)
    assert bm[(0, 0, 0)].startswith("minecraft:stone_brick_stairs[")
    assert bm[(1, 0, 0)].startswith("minecraft:stone_brick_slab[")


def test_added_surface_voxels_inherit_neighbour_material(reg):
    r = make_raster()
    r.material[2, 0, 2] = 2  # a single roof voxel
    f = fit_from(r)
    f.kind[2, 1, 2] = Kind.SLAB  # fit added a slab above it in a voxel whose material is 0
    sc = scene_with({"roof": {"base": "deepslate_tiles"}, "wall": {"base": "stone_bricks"}})
    bm = resolve(r, f, sc, reg)
    assert bm[(2, 1, 2)] == "minecraft:deepslate_tile_slab[type=bottom]"


def test_props_merge_expand_and_validate(reg):
    r = make_raster()
    r.material[0, 0, 0] = 1
    r.props = {(3, 0, 3): "minecraft:lantern[hanging=true]", (4, 0, 4): "oak_door[facing=south]", (5, 0, 5): "bogus[foo=bar]", (1, 0, 1): "oak_leaves"}
    f = fit_from(r)
    sc = scene_with({"wall": {"base": "stone_bricks"}})
    w = []
    bm = resolve(r, f, sc, reg, warnings=w)
    assert bm[(3, 0, 3)] == "minecraft:lantern[hanging=true]"
    assert bm[(4, 0, 4)] == "minecraft:oak_door[facing=south]"
    assert bm[(4, 1, 4)] == "minecraft:oak_door[facing=south,half=upper]"
    assert bm[(1, 0, 1)] == "minecraft:oak_leaves[persistent=true]"
    assert (5, 0, 5) not in bm and any("bogus" in x for x in w)


def test_unknown_material_warns_and_uses_stone(reg):
    r = make_raster(names=("", "mystery"))
    r.material[0, 0, 0] = 1
    f = fit_from(r)
    w = []
    bm = resolve(r, f, Scene(), reg, warnings=w)
    assert bm[(0, 0, 0)] == "minecraft:stone" and w


def test_gradient_faces_and_origin_offsets(reg):
    r = make_raster(shape=(4, 8, 4), origin=(10, 5, -3))
    r.material[:, :, :] = 1
    f = fit_from(r)
    sc = scene_with({"wall": {"base": "stone_bricks", "gradient": {"axis": "y", "from": 5, "to": 7, "palette": ["cobblestone"]}, "faces": {"top": "stone_brick_slab"}}})
    bm = resolve(r, f, sc, reg)
    assert set(p[0] for p in bm) == {10, 11, 12, 13}
    assert bm[(10, 5, -3)] == "minecraft:cobblestone"
    assert bm[(10, 12, -3)] == "minecraft:stone_brick_slab"
    assert bm[(11, 10, -2)] == "minecraft:stone_bricks"


def test_empty(reg):
    r = make_raster()
    assert resolve(r, fit_from(r), Scene(), reg) == {}
