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


@dataclass
class VoxelObject:
    occupancy: np.ndarray  # (W, H, D) bool, x east / y up / z south
    colors: np.ndarray  # (W, H, D, 3) uint8, meaningful where occupancy is True
    mesh_bounds: tuple[float, float, float]  # scaled size in blocks (w, h, d)

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
    """Thinnest extent over the largest: ~1 for a compact object, < 0.3 means the reconstructor returned a
    relief/billboard instead of a body (a known single-image failure when the view is from above)."""
    ext = np.sort(mesh.bounds[1] - mesh.bounds[0])
    return float(ext[0] / max(ext[2], 1e-6))


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


def voxelize_mesh(mesh: trimesh.Trimesh, height: int, max_width: int = 96, fill: bool = True, up: str = "z",
                  yaw_deg: float = 0.0, color_smooth_k: int = 24) -> VoxelObject:
    """Scale to `height` blocks tall (capped so width/depth stay <= max_width), voxelise at 1 block pitch,
    fill the interior, colour each voxel from the nearest vertex. `yaw_deg` rotates about +y first
    (0 keeps the reconstruction's own facing, which for TripoSR is the input image's view direction)."""
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
    m.apply_translation([0.501, 0.501, 0.501])
    size = m.bounds[1]
    vg = m.voxelized(pitch=1.0)
    if fill:
        vg = vg.fill()
    occ = np.asarray(vg.matrix, dtype=bool)
    # vg.points are voxel centres in mesh space; map every voxel back to its centre for colour lookup
    origin = np.asarray(vg.transform)[:3, 3]
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
    if idx.size:
        lo_i = idx.min(axis=0)
        hi_i = idx.max(axis=0) + 1
        occ = occ[lo_i[0]:hi_i[0], lo_i[1]:hi_i[1], lo_i[2]:hi_i[2]]
        colors = colors[lo_i[0]:hi_i[0], lo_i[1]:hi_i[1], lo_i[2]:hi_i[2]]
    # the rasteriser can still add a sparse boundary layer: drop the thinner end until the height is exact
    while occ.shape[1] > height and occ.shape[1] > 1:
        if occ[:, 0, :].sum() <= occ[:, -1, :].sum():
            occ, colors = occ[:, 1:, :], colors[:, 1:, :]
        else:
            occ, colors = occ[:, :-1, :], colors[:, :-1, :]
    return VoxelObject(occupancy=occ, colors=colors, mesh_bounds=(float(size[0]), float(size[1]), float(size[2])))


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
    for x, y, z in sidx:
        grid.set(int(x) + pad_xz, int(y), int(z) + pad_xz, Role.WALL, BShape.FULL)
        grid.block[x + pad_xz, y, z + pad_xz] = chosen[x, y, z]
    used = len({int(v) for v in chosen[surf]})
    grid.note(f"object: {obj.block_count} blocks, {used} block types, {W}x{H}x{D}")
    return grid
