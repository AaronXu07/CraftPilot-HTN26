"""Mesh -> coloured voxel grid -> SemanticGrid of concrete blocks.

`voxelize_mesh` scales the mesh so its vertical extent is `height` blocks, rasterises it with trimesh
(surface + filled interior), and colours every voxel from the nearest mesh vertex. `to_grid` maps those
colours to the closest opaque full block in a curated palette and writes a `SemanticGrid`, so the object
gets the same litematic export and isometric preview as a building.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from craftpilot.blocks.state import BlockRef
from craftpilot.grid.enums import BShape, Role
from craftpilot.grid.semantic import SemanticGrid

# Opaque full blocks with their average texture colour (sRGB). Kept to blocks that read as "solid material"
# from a distance: stone families, concrete, terracotta, wool, wood, a few metals. Glass, leaves and
# anything with a distinctive pattern would break the statue's surface.
PALETTE: dict[str, tuple[int, int, int]] = {
    # greys / stone
    "stone": (125, 125, 125), "smooth_stone": (160, 160, 160), "stone_bricks": (122, 122, 122),
    "cobblestone": (127, 127, 127), "andesite": (136, 136, 137), "polished_andesite": (132, 135, 134),
    "diorite": (188, 188, 188), "polished_diorite": (192, 193, 194), "calcite": (223, 224, 220),
    "quartz_block": (236, 230, 223), "white_concrete": (207, 213, 214), "light_gray_concrete": (125, 125, 115),
    "gray_concrete": (55, 58, 62), "black_concrete": (8, 10, 15), "deepslate": (80, 80, 82),
    "polished_deepslate": (72, 72, 73), "deepslate_bricks": (70, 70, 71), "deepslate_tiles": (55, 55, 56),
    "cobbled_deepslate": (77, 77, 80), "blackstone": (42, 36, 41), "polished_blackstone": (53, 48, 56),
    "tuff": (108, 109, 102), "polished_tuff": (98, 102, 96), "iron_block": (220, 220, 220),
    "white_terracotta": (209, 178, 161), "light_gray_terracotta": (135, 107, 98), "gray_terracotta": (57, 42, 35),
    "black_terracotta": (37, 22, 16), "white_wool": (233, 236, 236), "light_gray_wool": (142, 142, 134),
    "gray_wool": (62, 68, 71), "black_wool": (20, 21, 25),
    # browns / wood / earth
    "oak_planks": (177, 144, 86), "spruce_planks": (114, 84, 48), "birch_planks": (196, 179, 123),
    "dark_oak_planks": (66, 43, 20), "jungle_planks": (160, 115, 80), "acacia_planks": (168, 90, 50),
    "mangrove_planks": (117, 54, 48), "cherry_planks": (226, 178, 172), "oak_log": (109, 85, 50),
    "spruce_log": (58, 37, 16), "stripped_oak_log": (180, 145, 85), "stripped_spruce_log": (116, 88, 51),
    "brown_concrete": (96, 60, 32), "brown_terracotta": (77, 51, 36), "brown_wool": (114, 72, 41),
    "terracotta": (152, 94, 68), "mud_bricks": (137, 104, 79), "packed_mud": (142, 106, 79), "dirt": (134, 96, 67),
    "granite": (149, 103, 86), "polished_granite": (154, 107, 89), "bricks": (150, 97, 83),
    # sand / yellow / orange
    "sandstone": (216, 203, 155), "smooth_sandstone": (223, 214, 170), "end_stone_bricks": (219, 222, 158),
    "yellow_concrete": (241, 175, 21), "yellow_terracotta": (186, 133, 35), "yellow_wool": (249, 198, 40),
    "orange_concrete": (224, 97, 1), "orange_terracotta": (162, 84, 38), "orange_wool": (241, 118, 20),
    "red_sandstone": (186, 99, 29), "gold_block": (246, 208, 62), "honeycomb_block": (229, 148, 30),
    "waxed_copper_block": (192, 107, 79), "waxed_exposed_copper": (161, 125, 104),
    # reds / pinks / purples
    "red_concrete": (142, 33, 33), "red_terracotta": (143, 61, 47), "red_wool": (161, 39, 35),
    "red_nether_bricks": (70, 7, 9), "nether_bricks": (44, 22, 26), "crimson_planks": (101, 49, 71),
    "pink_concrete": (214, 101, 143), "pink_terracotta": (162, 78, 79), "pink_wool": (238, 141, 172),
    "magenta_concrete": (169, 48, 159), "magenta_terracotta": (150, 88, 109), "magenta_wool": (190, 69, 180),
    "purple_concrete": (100, 32, 156), "purple_terracotta": (118, 70, 86), "purple_wool": (122, 42, 173),
    "purpur_block": (170, 126, 170),
    # greens
    "green_concrete": (73, 91, 36), "green_terracotta": (76, 83, 42), "green_wool": (85, 110, 27),
    "lime_concrete": (94, 169, 24), "lime_terracotta": (104, 118, 53), "lime_wool": (112, 185, 25),
    "moss_block": (89, 110, 45), "waxed_oxidized_copper": (82, 162, 132), "waxed_weathered_copper": (108, 153, 110),
    "prismarine": (99, 156, 151), "warped_planks": (43, 104, 99),
    # blues / cyans
    "blue_concrete": (45, 47, 143), "blue_terracotta": (74, 60, 91), "blue_wool": (53, 57, 157),
    "light_blue_concrete": (36, 137, 199), "light_blue_terracotta": (113, 108, 138), "light_blue_wool": (58, 175, 217),
    "cyan_concrete": (21, 119, 136), "cyan_terracotta": (87, 91, 91), "cyan_wool": (21, 137, 145),
    "lapis_block": (31, 67, 140), "dark_prismarine": (52, 92, 76),
}


# Full block -> (stairs, slab) variants that exist in 1.21.1 (checked against the mod's registry). Blocks
# without variants (concrete, wool, terracotta, metals, calcite, moss) stay full cubes on slopes.
STAIR_SLAB: dict[str, tuple[str | None, str | None]] = {
    "stone": ("stone_stairs", "stone_slab"), "smooth_stone": (None, "smooth_stone_slab"),
    "stone_bricks": ("stone_brick_stairs", "stone_brick_slab"), "cobblestone": ("cobblestone_stairs", "cobblestone_slab"),
    "andesite": ("andesite_stairs", "andesite_slab"), "polished_andesite": ("polished_andesite_stairs", "polished_andesite_slab"),
    "diorite": ("diorite_stairs", "diorite_slab"), "polished_diorite": ("polished_diorite_stairs", "polished_diorite_slab"),
    "granite": ("granite_stairs", "granite_slab"), "polished_granite": ("polished_granite_stairs", "polished_granite_slab"),
    "deepslate_bricks": ("deepslate_brick_stairs", "deepslate_brick_slab"), "deepslate_tiles": ("deepslate_tile_stairs", "deepslate_tile_slab"),
    "polished_deepslate": ("polished_deepslate_stairs", "polished_deepslate_slab"), "cobbled_deepslate": ("cobbled_deepslate_stairs", "cobbled_deepslate_slab"),
    "tuff": ("tuff_stairs", "tuff_slab"), "polished_tuff": ("polished_tuff_stairs", "polished_tuff_slab"),
    "blackstone": ("blackstone_stairs", "blackstone_slab"), "polished_blackstone": ("polished_blackstone_stairs", "polished_blackstone_slab"),
    "bricks": ("brick_stairs", "brick_slab"), "mud_bricks": ("mud_brick_stairs", "mud_brick_slab"),
    "sandstone": ("sandstone_stairs", "sandstone_slab"), "smooth_sandstone": ("smooth_sandstone_stairs", "smooth_sandstone_slab"),
    "red_sandstone": ("red_sandstone_stairs", "red_sandstone_slab"), "quartz_block": ("quartz_stairs", "quartz_slab"),
    "prismarine": ("prismarine_stairs", "prismarine_slab"), "dark_prismarine": ("dark_prismarine_stairs", "dark_prismarine_slab"),
    "end_stone_bricks": ("end_stone_brick_stairs", "end_stone_brick_slab"), "nether_bricks": ("nether_brick_stairs", "nether_brick_slab"),
    "red_nether_bricks": ("red_nether_brick_stairs", "red_nether_brick_slab"), "purpur_block": ("purpur_stairs", "purpur_slab"),
    "waxed_copper_block": ("waxed_cut_copper_stairs", "waxed_cut_copper_slab"),
    "waxed_exposed_copper": ("waxed_exposed_cut_copper_stairs", "waxed_exposed_cut_copper_slab"),
    "waxed_oxidized_copper": ("waxed_oxidized_cut_copper_stairs", "waxed_oxidized_cut_copper_slab"),
    "waxed_weathered_copper": ("waxed_weathered_cut_copper_stairs", "waxed_weathered_cut_copper_slab"),
    **{f"{w}_planks": (f"{w}_stairs", f"{w}_slab") for w in ("oak", "spruce", "birch", "dark_oak", "jungle", "acacia", "mangrove", "cherry", "crimson", "warped")},
}

# per-block shape codes from the 2x2x2 sub-voxel pattern (see classify_shapes)
FULL, SLAB_BOTTOM, SLAB_TOP, STAIR, STAIR_UPSIDE = 1, 2, 3, 4, 5
FACINGS = ("north", "east", "south", "west")  # index = facing code; north = -z, east = +x


@dataclass
class VoxelObject:
    occupancy: np.ndarray  # (W, H, D) bool, x east / y up / z south
    colors: np.ndarray  # (W, H, D, 3) uint8, meaningful where occupancy is True
    mesh_bounds: tuple[float, float, float]  # scaled size in blocks (w, h, d)
    shapes: np.ndarray | None = None  # (W, H, D) uint8 shape code (FULL … STAIR_UPSIDE), 0 = air
    facings: np.ndarray | None = None  # (W, H, D) int8 facing code for stairs, -1 otherwise

    @property
    def size(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.occupancy.shape)  # type: ignore[return-value]

    @property
    def block_count(self) -> int:
        return int(self.occupancy.sum())


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255) -> CIELAB, vectorised. Nearest-colour matching in Lab avoids the "grey matches to
    dark blue" failures of plain RGB distance."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def flatness(mesh: trimesh.Trimesh) -> float:
    """Thinnest extent over the largest: ~1 for a compact object, ~0.3 for a horse in profile or a low car."""
    ext = np.sort(mesh.bounds[1] - mesh.bounds[0])
    return float(ext[0] / max(ext[2], 1e-6))


def relief_like(mesh: trimesh.Trimesh, up_axis: int = 2, max_flat: float = 0.35, min_square: float = 0.75) -> bool:
    """True for the single-image failure mode where the reconstructor returns a pancake instead of a body:
    thin along the UP axis (TripoSR frame: z) with a near-square footprint. A rearing horse (thin sideways)
    or a low car (thin up, but 2:1 footprint) is not a relief."""
    ext = mesh.bounds[1] - mesh.bounds[0]
    up = float(ext[up_axis])
    others = sorted(float(v) for i, v in enumerate(ext) if i != up_axis)
    largest = max(others[1], 1e-6)
    return up / largest < max_flat and others[0] / largest > min_square


def load_mesh(path: Path | str) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"{path}: not a triangle mesh")
    return m


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is None or len(vc) != len(mesh.vertices):
        return np.full((len(mesh.vertices), 3), 128, dtype=np.uint8)
    return np.asarray(vc)[:, :3].astype(np.uint8)


def orient_y_up(mesh: trimesh.Trimesh, up: str = "z") -> trimesh.Trimesh:
    """Rotate so the model's up axis is +y. TripoSR exports are z-up (the default here); `up="y"` leaves
    a y-up mesh alone, `up="-y"` flips one that came out upside down."""
    m = mesh.copy()
    if up == "z":
        m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    elif up == "-z":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif up == "-y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    return m


def classify_shapes(occ2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2x-resolution occupancy (2W, 2H, 2D) -> per-block (occupancy, shape code, stair facing).

    Each block owns a 2x2x2 group of sub-voxels: 7-8 filled is a full block; a filled lower half alone a
    bottom slab (upper half alone a top slab); a filled half plus the two sub-voxels along one side of the
    other half is a stair whose full-height side faces that way. Anything else with >= 4 filled is a full
    block (>= 5 of 8), with fewer is air. This quarters the step size on slopes, the way builders do by hand."""
    W2, H2, D2 = occ2.shape
    W, H, D = W2 // 2, H2 // 2, D2 // 2
    sub = occ2[:2 * W, :2 * H, :2 * D].reshape(W, 2, H, 2, D, 2).transpose(0, 2, 4, 1, 3, 5)  # (W,H,D, dx,dy,dz)
    n = sub.sum(axis=(3, 4, 5))
    bottom = sub[..., :, 0, :].sum(axis=(3, 4))
    top = sub[..., :, 1, :].sum(axis=(3, 4))
    shape = np.zeros((W, H, D), dtype=np.uint8)
    facing = np.full((W, H, D), -1, dtype=np.int8)
    shape[n >= 7] = FULL
    slab_b = (bottom == 4) & (top == 0)
    slab_t = (top == 4) & (bottom == 0)
    shape[slab_b] = SLAB_BOTTOM
    shape[slab_t] = SLAB_TOP

    def side_pairs(half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For a (W,H,D,2,2) half layer with exactly two filled cells: which side they form (facing code)."""
        west = half[..., 0, :].sum(axis=3) == 2  # both at dx=0 -> full side faces -x
        east = half[..., 1, :].sum(axis=3) == 2
        north = half[..., :, 0].sum(axis=3) == 2  # dz=0 -> -z
        south = half[..., :, 1].sum(axis=3) == 2
        code = np.full(half.shape[:3], -1, dtype=np.int8)
        for c, m in ((0, north), (1, east), (2, south), (3, west)):
            code[m & (code < 0)] = c
        return code >= 0, code

    top_layer = sub[..., :, 1, :]
    bot_layer = sub[..., :, 0, :]
    ok_t, code_t = side_pairs(top_layer)
    stair = (bottom == 4) & (top == 2) & ok_t
    shape[stair] = STAIR
    facing[stair] = code_t[stair]
    ok_b, code_b = side_pairs(bot_layer)
    stair_u = (top == 4) & (bottom == 2) & ok_b & (shape == 0)
    shape[stair_u] = STAIR_UPSIDE
    facing[stair_u] = code_b[stair_u]
    # >= 5 of 8, not 4: the rasteriser marks every sub-voxel a surface passes through, so a vertical face
    # exactly on a block boundary would otherwise grow a full extra column of blocks
    rest = (shape == 0) & (n >= 5)
    shape[rest] = FULL
    return shape > 0, shape, facing


def voxelize_mesh(mesh: trimesh.Trimesh, height: int, max_width: int = 96, fill: bool = True, up: str = "z",
                  yaw_deg: float = 0.0, color_smooth_k: int = 24, fit: bool = True) -> VoxelObject:
    """Scale to `height` blocks tall (capped so width/depth stay <= max_width), voxelise at 1 block pitch,
    fill the interior, colour each voxel from the nearest vertex. `yaw_deg` rotates about +y first
    (0 keeps the reconstruction's own facing, which for TripoSR is the input image's view direction).
    With `fit`, the mesh is rasterised at half-block pitch and every block gets a shape (full / slab /
    stair) from its sub-voxels, so slopes read as smooth stairs instead of one-block steps."""
    m = orient_y_up(mesh, up)
    if yaw_deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 1, 0]))
    lo, hi = m.bounds
    extent = hi - lo
    scale = height / max(float(extent[1]), 1e-6)
    widest = max(float(extent[0]), float(extent[2])) * scale
    if widest > max_width:
        scale *= max_width / widest
    m.apply_translation(-lo)
    m.apply_scale(scale)
    # half-block offset: surfaces on integer planes would otherwise claim the voxels on both sides
    # (a 12-tall box became 13 layers); floor stays at y=0
    # whole-block rasterising: surfaces on integer planes would claim the voxels on both sides, so nudge by
    # half a block; half-block rasterising tests sub-voxel centres at .25/.75, so the mesh sits on the block
    # boundary itself (a hair above zero keeps the floor in the first layer)
    m.apply_translation([0.001, 0.001, 0.001] if fit else [0.501, 0.501, 0.501])
    size = m.bounds[1]
    shapes: np.ndarray | None = None
    facings: np.ndarray | None = None
    if fit:
        # rasterise at half-block pitch on a grid aligned to whole blocks, then classify each 2x2x2 group
        vg2 = m.voxelized(pitch=0.5)
        if fill:
            vg2 = vg2.fill()
        occ2 = np.asarray(vg2.matrix, dtype=bool)
        o2 = np.asarray(vg2.transform)[:3, 3]  # centre of sub-voxel (0,0,0), at k*0.5 + 0.25 for some k
        base = np.floor((o2 - 0.25) / 0.5 + 0.5).astype(int)  # sub-voxel index of the grid origin
        pad_lo = base % 2  # shift so sub-voxel pairs line up with block boundaries
        occ2 = np.pad(occ2, [(int(p), 0) for p in pad_lo])
        pad_hi = [(0, (2 - d % 2) % 2) for d in occ2.shape]
        occ2 = np.pad(occ2, pad_hi)
        occ, shapes, facings = classify_shapes(occ2)
        block_origin = (base - pad_lo) // 2  # block index of occ[0,0,0] in mesh space (block b spans [b, b+1))
        origin = block_origin.astype(np.float64) + 0.5  # block centre for colour lookup
    else:
        vg = m.voxelized(pitch=1.0)
        if fill:
            vg = vg.fill()
        occ = np.asarray(vg.matrix, dtype=bool)
        origin = np.asarray(vg.transform)[:3, 3]  # vg.points are voxel centres in mesh space
    idx = np.argwhere(occ)
    centres = idx.astype(np.float64) + origin
    tree = cKDTree(m.vertices)
    vcol = _vertex_colors(m).astype(np.float64)
    # average the k nearest vertex colours: reconstruction colour is noisy per vertex and a single sample
    # per block turns that noise into pink/cyan speckles on a grey statue
    k = max(1, min(color_smooth_k, len(m.vertices)))
    _, nearest = tree.query(centres, k=k)
    nearest = np.atleast_2d(nearest)
    colors = np.zeros(occ.shape + (3,), dtype=np.uint8)
    colors[idx[:, 0], idx[:, 1], idx[:, 2]] = np.clip(vcol[nearest].mean(axis=1), 0, 255).astype(np.uint8)
    # trim to the occupied box so the grid is tight
    def crop(a: np.ndarray, lo_i: np.ndarray, hi_i: np.ndarray) -> np.ndarray:
        return a[lo_i[0]:hi_i[0], lo_i[1]:hi_i[1], lo_i[2]:hi_i[2]]

    if idx.size:
        lo_i = idx.min(axis=0)
        hi_i = idx.max(axis=0) + 1
        occ, colors = crop(occ, lo_i, hi_i), crop(colors, lo_i, hi_i)
        if shapes is not None and facings is not None:
            shapes, facings = crop(shapes, lo_i, hi_i), crop(facings, lo_i, hi_i)
    # the rasteriser can still add a sparse boundary layer: drop the thinner end until the height is exact
    while occ.shape[1] > height and occ.shape[1] > 1:
        sl = (slice(None), slice(1, None), slice(None)) if occ[:, 0, :].sum() <= occ[:, -1, :].sum() else (slice(None), slice(None, -1), slice(None))
        occ, colors = occ[sl], colors[sl]
        if shapes is not None and facings is not None:
            shapes, facings = shapes[sl], facings[sl]
    return VoxelObject(occupancy=occ, colors=colors, mesh_bounds=(float(size[0]), float(size[1]), float(size[2])),
                       shapes=shapes, facings=facings)


def _surface_mask(occ: np.ndarray) -> np.ndarray:
    """Voxels with at least one exposed face (the only ones whose colour matters)."""
    p = np.pad(occ, 1, constant_values=False)
    inner = (
        p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1] & p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1] & p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:]
    )
    return occ & ~inner


def clamp_chroma(lab: np.ndarray, factor: float = 1.2, floor: float = 6.0, percentile: float = 90.0) -> np.ndarray:
    """Pull outlier chroma back to the object's own level: single-image reconstruction hallucinates
    magenta/cyan on the unseen side, and a grey statue must stay grey. The limit is a high percentile of
    the object's own chroma, so a red car whose tyres/windows/underside are black (the *median* voxel)
    keeps its red — only the rare hallucinated few percent get pulled in."""
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    limit = max(float(np.percentile(chroma, percentile)) * factor, floor)
    k = np.minimum(1.0, limit / np.maximum(chroma, 1e-6))
    out = lab.copy()
    out[:, 1] *= k
    out[:, 2] *= k
    return out


def match_blocks(colors: np.ndarray, palette: dict[str, tuple[int, int, int]] | None = None,
                 allowed: list[str] | None = None, max_types: int | None = 6) -> np.ndarray:
    """Nearest palette block (by CIELAB distance) for an (N, 3) array of sRGB colours -> array of block names.
    With `max_types`, a second pass keeps only the N most used blocks so the surface reads as one material
    with shading rather than a mosaic of near-identical greys."""
    pal = palette or PALETTE
    names = [n for n in pal if not allowed or n in allowed] or list(pal)
    lab_pal = _srgb_to_lab(np.array([pal[n] for n in names], dtype=np.float64))
    lab = clamp_chroma(_srgb_to_lab(colors)) if len(colors) > 8 else _srgb_to_lab(colors)
    chosen = _nearest(lab, lab_pal, names)
    if max_types and len(set(chosen.tolist())) > max_types:
        vals, counts = np.unique(chosen, return_counts=True)
        keep = [str(v) for v in vals[np.argsort(-counts)][:max_types]]
        lab_keep = _srgb_to_lab(np.array([pal[n] for n in keep], dtype=np.float64))
        chosen = _nearest(lab, lab_keep, keep)
    return chosen


def _nearest(lab: np.ndarray, lab_pal: np.ndarray, names: list[str]) -> np.ndarray:
    d = ((lab[:, None, :] - lab_pal[None, :, :]) ** 2).sum(-1)
    # a block may be duller than the sample but never much more saturated: a faint tint on grey stone must
    # not become pink terracotta
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    chroma_pal = np.hypot(lab_pal[:, 1], lab_pal[:, 2])
    excess = np.clip(chroma_pal[None, :] - chroma[:, None], 0, None)
    d = d + (3.0 * excess) ** 2
    return np.array(names)[d.argmin(axis=1)]


def despeckle(occ: np.ndarray, block: np.ndarray, passes: int = 2) -> np.ndarray:
    """Majority vote over the 6-neighbourhood: a surface block whose type matches none of its occupied
    neighbours takes their most common type. Removes the one-off odd blocks colour noise leaves behind."""
    out = block.copy()
    W, H, D = occ.shape
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for _ in range(passes):
        changed = 0
        for x, y, z in np.argwhere(occ):
            mine = out[x, y, z]
            votes: dict[int, int] = {}
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D and occ[nx, ny, nz]:
                    v = int(out[nx, ny, nz])
                    votes[v] = votes.get(v, 0) + 1
            if votes and mine not in votes:
                best = max(votes.items(), key=lambda kv: kv[1])
                if best[1] >= 2:
                    out[x, y, z] = best[0]
                    changed += 1
        if not changed:
            break
    return out


def to_grid(obj: VoxelObject, filler: str = "stone", allowed: list[str] | None = None, seed: int = 0,
            pad_xz: int = 1, max_types: int | None = 6) -> SemanticGrid:
    """Write the voxel object into a SemanticGrid: surface voxels get their matched block, the interior
    gets `filler` (invisible, but keeps the statue solid for placement and undo)."""
    W, H, D = obj.size
    grid = SemanticGrid(W + 2 * pad_xz, H, D + 2 * pad_xz, np.random.default_rng(seed))
    surf = _surface_mask(obj.occupancy)
    sidx = np.argwhere(surf)
    names = match_blocks(obj.colors[sidx[:, 0], sidx[:, 1], sidx[:, 2]], allowed=allowed, max_types=max_types) if sidx.size else np.array([])
    fill_i = grid.intern(BlockRef.make(f"minecraft:{filler}"))
    for x, y, z in np.argwhere(obj.occupancy & ~surf):
        grid.set(int(x) + pad_xz, int(y), int(z) + pad_xz, Role.WALL, BShape.FULL)
        grid.block[x + pad_xz, y, z + pad_xz] = fill_i
    cache: dict[str, int] = {}
    chosen = np.full(obj.size, -1, dtype=np.int32)
    for (x, y, z), name in zip(sidx, names):
        i = cache.get(name)
        if i is None:
            i = cache[name] = grid.intern(BlockRef.make(f"minecraft:{name}"))
        chosen[x, y, z] = i
    chosen = despeckle(surf, chosen)
    by_index = {i: n for n, i in cache.items()}
    shaped = 0
    ref_cache: dict[tuple, int] = {}
    for x, y, z in sidx:
        full_i = int(chosen[x, y, z])
        code = int(obj.shapes[x, y, z]) if obj.shapes is not None else FULL
        ref = shaped_ref(by_index[full_i], code, int(obj.facings[x, y, z]) if obj.facings is not None else -1)
        if ref is None:
            grid.set(int(x) + pad_xz, int(y), int(z) + pad_xz, Role.WALL, BShape.FULL)
            grid.block[x + pad_xz, y, z + pad_xz] = full_i
            continue
        key = (ref.block_id, ref.props)
        i = ref_cache.get(key)
        if i is None:
            i = ref_cache[key] = grid.intern(ref)
        bshape = {SLAB_BOTTOM: BShape.SLAB_BOTTOM, SLAB_TOP: BShape.SLAB_TOP, STAIR: BShape.STAIR, STAIR_UPSIDE: BShape.STAIR_UPSIDE}[code]
        grid.set(int(x) + pad_xz, int(y), int(z) + pad_xz, Role.WALL, bshape)
        grid.block[x + pad_xz, y, z + pad_xz] = i
        shaped += 1
    used = len({int(v) for v in chosen[surf]})
    grid.note(f"object: {obj.block_count} blocks, {used} block types, {W}x{H}x{D}" + (f", {shaped} stairs/slabs" if shaped else ""))
    return grid


def shaped_ref(full_name: str, code: int, facing_code: int) -> BlockRef | None:
    """The stair/slab BlockRef for a surface block, or None when it should stay a full cube (full shape, or
    the material has no stair/slab variant)."""
    if code in (0, FULL):
        return None
    stairs, slab = STAIR_SLAB.get(full_name, (None, None))
    if code in (SLAB_BOTTOM, SLAB_TOP):
        if slab is None:
            return None
        return BlockRef.make(f"minecraft:{slab}", type="bottom" if code == SLAB_BOTTOM else "top", waterlogged="false")
    if stairs is None or facing_code < 0:
        return None
    return BlockRef.make(f"minecraft:{stairs}", facing=FACINGS[facing_code], half="bottom" if code == STAIR else "top",
                         shape="straight", waterlogged="false")
