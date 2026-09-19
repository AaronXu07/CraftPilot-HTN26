"""Materials: palette, gradients, and texturing resolved over the semantic grid."""

from __future__ import annotations

import numpy as np

from craftpilot.blocks import catalog
from craftpilot.blocks.catalog import Family
from craftpilot.blocks.palette_data import WEATHERED_COMPANIONS, closest_family_with_shape
from craftpilot.blocks.state import BlockRef
from craftpilot.engine.noise import clustered
from craftpilot.grid.enums import OPPOSITE, DIR_NAME, DIR_VEC, HORIZONTAL, BShape, Dir, Flag, Role
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.program.model import BuildProgram, Gradient, PaletteSpec, RolePalette

# Grid role -> (palette role, fallback chain)
ROLE_PALETTE: dict[int, list[str]] = {
    Role.WALL: ["primary"],
    Role.FRAME: ["framing", "primary"],
    Role.BEAM: ["framing", "primary"],
    Role.FOUNDATION: ["foundation", "secondary", "primary"],
    Role.FLOOR: ["primary"],
    Role.ROOF: ["roof"],
    Role.ROOF_FILL: ["roof"],
    Role.ROOF_TRIM: ["roof"],
    Role.WINDOW: ["glass"],
    Role.DOOR: ["accent", "framing", "primary"],
    Role.SILL: ["trim", "roof"],
    Role.SHUTTER: ["accent", "trim", "framing", "primary"],
    Role.TRIM: ["trim", "roof"],
    Role.PILLAR: ["framing", "primary"],
    Role.RAILING: ["trim", "framing", "primary"],
    Role.CHIMNEY: ["secondary", "foundation", "primary"],
    Role.CHIMNEY_CAP: ["trim", "secondary", "primary"],
    Role.ACCENT: ["accent", "trim", "primary"],
    Role.PARAPET: ["primary"],
    Role.MERLON: ["primary"],
    Role.CORBEL: ["trim", "roof"],
    Role.STEP: ["foundation", "secondary", "primary"],
    Role.FLOOR_LIP: ["trim", "primary"],
    Role.STAIRCASE: ["primary"],
    Role.PARTITION: ["primary"],
    Role.DETAIL: ["primary"],
}

BANNER = "minecraft:red_wall_banner"

SHAPE_NAME: dict[int, str] = {
    BShape.FULL: "full",
    BShape.STAIR: "stairs",
    BShape.STAIR_UPSIDE: "stairs",
    BShape.SLAB_BOTTOM: "slab",
    BShape.SLAB_TOP: "slab",
    BShape.LOG: "log",
    BShape.STRIPPED_LOG: "stripped_log",
    BShape.PANE: "pane",
    BShape.FENCE: "fence",
    BShape.WALLBLOCK: "wall",
    BShape.TRAPDOOR: "trapdoor",
    BShape.DOOR_LOWER: "door",
    BShape.DOOR_UPPER: "door",
    BShape.PILLAR: "pillar",
}

DEFAULT_GLASS = RolePalette(families=[{"family": "glass", "weight": 1.0}])  # type: ignore[list-item]


def _role_palette(spec: PaletteSpec, chain: list[str], shape: str | None = None) -> RolePalette | None:
    for name in chain:
        rp = getattr(spec, name, None)
        if rp is not None and rp.families:
            if shape is None:
                return rp
            dom = max(rp.families, key=lambda f: f.weight).family
            fam = catalog.family(dom)
            if fam is not None and fam.has(shape):
                return rp
    if "glass" in chain:
        return DEFAULT_GLASS
    return spec.primary


_SHADE_CACHE: dict[str, str | None] = {}


def shade_family(name: str) -> str | None:
    """A darker relative of a family: same material, close hue, lightness 0.06 to 0.25 lower."""
    if name in _SHADE_CACHE:
        return _SHADE_CACHE[name]
    import colorsys
    src = catalog.family(name)
    if src is None:
        _SHADE_CACHE[name] = None
        return None
    h0, l0, s0 = colorsys.rgb_to_hls(*(c / 255.0 for c in src.rgb))
    kin = {"plaster": "painted", "terracotta": "painted"}
    best, best_d = None, 1e9
    for fam in catalog.FAMILIES.values():
        if fam.name == name or fam.loud or fam.tone == "glass" or fam.material == "organic":
            continue
        if kin.get(fam.material, fam.material) != kin.get(src.material, src.material):
            continue
        h, l, s_ = colorsys.rgb_to_hls(*(c / 255.0 for c in fam.rgb))
        drop = l0 - l
        if not (0.05 <= drop <= 0.22):
            continue
        dh = min(abs(h - h0), 1 - abs(h - h0))          # 0..0.5 of a turn
        if s0 > 0.25 and dh > 30 / 360:
            continue                                     # a coloured block must stay in its hue
        # Prefer a small drop, the same hue and saturation, and the same texture character.
        d = abs(drop - 0.12) * 3 + dh * 4 * max(s_, s0) + abs(s_ - s0) + abs(fam.noise - src.noise)
        if d < best_d:
            best, best_d = fam.name, d
    _SHADE_CACHE[name] = best
    return best


class _Resolver:
    def __init__(self, grid: SemanticGrid, spec: PaletteSpec):
        self.grid = grid
        self.spec = spec
        self.shading = 0.0
        rng = grid.rng
        shape = (grid.W, grid.H, grid.D)
        self.n_family = clustered(shape, 4.0, rng)
        self.n_tex = clustered(shape, 2.0, rng)
        self.u_tex = rng.random(shape).astype(np.float32)
        self.u_var = rng.random(shape).astype(np.float32)
        self._families: dict[str, list[tuple[Family, float]]] = {}
        self._jitter = clustered((grid.W, grid.D), 3.0, rng)

    def families_for(self, rp: RolePalette) -> list[tuple[Family, float]]:
        key = id(rp)
        cached = self._families.get(str(key))
        if cached is not None:
            return cached
        fams = []
        for fw in rp.families:
            f = catalog.family(fw.family)
            if f is not None and fw.weight > 0:
                fams.append((f, float(fw.weight)))
        if not fams:
            fams = [(catalog.FAMILIES["oak"], 1.0)]
        self._families[str(key)] = fams
        return fams

    def pick_family(self, rp: RolePalette, x: int, y: int, z: int, speckle: bool = False) -> Family:
        fams = self.families_for(rp)
        if len(fams) == 1:
            return fams[0][0]
        # Walls mix in soft patches; roofs mix as fine speckle, the way tile variation reads.
        u = (0.5 * float(self.n_tex[x, y, z]) + 0.5 * float(self.u_tex[x, y, z])) if speckle else float(self.n_family[x, y, z])
        weights = np.array([w for _, w in fams], dtype=np.float64)
        if rp.gradient == Gradient.horizontal:
            t = x / max(1, self.grid.W - 1) + 0.2 * (float(self._jitter[x, z]) - 0.5)
            n = len(fams)
            centers = np.linspace(0.0, 1.0, n)
            sigma = 0.6 / n
            weights = weights * np.exp(-((t - centers) / sigma) ** 2)
            if weights.sum() <= 0:
                weights = np.ones(n)
            cdf = np.cumsum(weights) / weights.sum()
            idx = int(np.searchsorted(cdf, u, side="right"))
            return fams[min(idx, len(fams) - 1)][0]
        if rp.gradient == Gradient.bands:
            t = float(self.grid.h_norm[x, y, z])
            n = len(fams)
            idx = min(n - 1, int(t * n * 0.9999))
            return fams[idx][0]
        if rp.gradient == Gradient.vertical:
            t = float(self.grid.h_norm[x, y, z]) + 0.25 * (float(self._jitter[x, z]) - 0.5)
            n = len(fams)
            centers = np.linspace(0.0, 1.0, n)
            sigma = 0.6 / n
            weights = weights * np.exp(-((t - centers) / sigma) ** 2)
            if weights.sum() <= 0:
                weights = np.ones(n)
        cdf = np.cumsum(weights) / weights.sum()
        idx = int(np.searchsorted(cdf, u, side="right"))
        return fams[min(idx, len(fams) - 1)][0]

    def pick_variant(self, fam: Family, rp: RolePalette, x: int, y: int, z: int) -> str:
        base = fam.shapes["full"]
        if rp.texture_rate <= 0:
            return base
        hn = float(self.grid.h_norm[x, y, z])
        if not fam.variants:
            # No cracked or mossy variant exists. Two kinds of kin mix in: a darker relative at the
            # texture rate (variation on an intact wall), and a weathering companion near the ground.
            u = 0.6 * float(self.n_tex[x, y, z]) + 0.4 * float(self.u_tex[x, y, z])
            comps = [c for c in WEATHERED_COMPANIONS.get(fam.name, []) if catalog.family(c)]
            p_weather = 0.5 * rp.texture_rate * rp.weathering * (1.5 - hn) if comps else 0.0
            if u < p_weather:
                return catalog.family(comps[0]).shapes["full"]
            dark = shade_family(fam.name)
            if dark and u < p_weather + 0.5 * rp.texture_rate:
                return catalog.family(dark).shapes["full"]
            return base
        p = rp.texture_rate * (1.0 + rp.weathering * (1.0 - hn))
        # Mix clustered and per-block randomness so wear comes in patches with speckle.
        u = 0.6 * float(self.n_tex[x, y, z]) + 0.4 * float(self.u_tex[x, y, z])
        if u >= p:
            return base
        weights = []
        for v in fam.variants:
            if "weathered" in v.tags:
                weights.append(0.3 + 2.0 * rp.weathering * (1.0 - hn))
            elif "log" in v.tags:
                weights.append(0.6)
            else:
                weights.append(0.5)
        w = np.array(weights)
        cdf = np.cumsum(w) / w.sum()
        i = int(np.searchsorted(cdf, float(self.u_var[x, y, z]), side="right"))
        return fam.variants[min(i, len(fam.variants) - 1)].block_id


_ROOF_COMPANION_CACHE: dict[str, str | None] = {}


def roof_companion(name: str) -> str | None:
    """The family with stairs and slabs closest in colour to this one (same material, small lightness
    difference), so roof variation reads like masonry texture rather than a two-tone patchwork."""
    if name in _ROOF_COMPANION_CACHE:
        return _ROOF_COMPANION_CACHE[name]
    import colorsys
    src = catalog.family(name)
    if src is None:
        _ROOF_COMPANION_CACHE[name] = None
        return None
    h0, l0, s0 = colorsys.rgb_to_hls(*(c / 255.0 for c in src.rgb))
    best, best_d = None, 1e9
    for fam in catalog.FAMILIES.values():
        if fam.name == name or fam.loud or fam.material != src.material or not (fam.has("stairs") and fam.has("slab")):
            continue
        h, l, s_ = colorsys.rgb_to_hls(*(c / 255.0 for c in fam.rgb))
        if abs(l - l0) > 0.16:
            continue
        dh = min(abs(h - h0), 1 - abs(h - h0))
        if max(s_, s0) > 0.25 and dh > 30 / 360:
            continue
        d = abs(l - l0) * 3 + dh * 4 * max(s_, s0) + abs(s_ - s0) + abs(fam.noise - src.noise) * 0.5
        if d < best_d:
            best, best_d = fam.name, d
    if best is None:
        for cand in WEATHERED_COMPANIONS.get(name, []) + [shade_family(name) or ""]:
            fam = catalog.family(cand)
            if fam is not None and fam.has("stairs") and fam.has("slab"):
                best = cand
                break
    _ROOF_COMPANION_CACHE[name] = best
    return best


def _occlusion(grid: SemanticGrid, x: int, y: int, z: int, normal: int) -> float:
    """How much overhang sits above this wall cell, in front of the wall plane: 1.0 right under it,
    fading to 0 three cells down."""
    vx, _, vz = DIR_VEC[normal]
    ox, oz = x + vx, z + vz
    for dist in (1, 2, 3):
        yy = y + dist
        if not grid.in_bounds(ox, yy, oz):
            break
        r = int(grid.role[ox, yy, oz])
        if r not in (Role.EMPTY, Role.INTERIOR, Role.FOLIAGE, Role.DETAIL, Role.LIGHT, Role.SHUTTER):
            return {1: 1.0, 2: 0.6, 3: 0.3}[dist]
    return 0.0


_BLOCK_TO_FAMILY: dict[str, Family] = {}
_WOOD_MATERIALS = {"wood"}


def _piece_family(spec: PaletteSpec, wall_fam: Family, want: str) -> Family | None:
    """Which family supplies a button, trapdoor or gate on this wall.

    Wood walls take the wood closest in colour. Stone, brick, plaster and terracotta walls take the
    palette's trim or framing wood (what builders do), except buttons, which come in stone."""
    from craftpilot.blocks.palette_data import info as _info

    wall_info = _info(wall_fam.name, wall_fam.tone)
    if wall_info.material in _WOOD_MATERIALS or want == "button" and wall_fam.has("button"):
        match = closest_family_with_shape(wall_fam.name, want) if wall_info.material in _WOOD_MATERIALS else wall_fam.name
        fam = catalog.family(match) if match else None
        if fam is not None and fam.has(want):
            return fam
    for role in ("trim", "framing", "accent"):
        rp = getattr(spec, role)
        if rp is None:
            continue
        for fw in rp.families:
            fam = catalog.family(fw.family)
            if fam is not None and fam.has(want) and _info(fam.name, fam.tone).material in _WOOD_MATERIALS:
                return fam
    match = closest_family_with_shape(wall_fam.name, want)
    return catalog.family(match) if match else None


def _family_of_block(block_id: str) -> Family | None:
    """Family whose full block or variant is this block id."""
    if not _BLOCK_TO_FAMILY:
        for fam in catalog.FAMILIES.values():
            for bid in list(fam.shapes.values()) + [v.block_id for v in fam.variants]:
                _BLOCK_TO_FAMILY.setdefault(bid, fam)
    return _BLOCK_TO_FAMILY.get(block_id)


def _connections(grid: SemanticGrid, x: int, y: int, z: int, kind: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for d in HORIZONTAL:
        vx, _, vz = DIR_VEC[d]
        nx, nz = x + vx, z + vz
        connected = False
        if grid.in_bounds(nx, y, nz):
            r = int(grid.role[nx, y, nz])
            s = int(grid.shape[nx, y, nz])
            if grid.is_solid(nx, y, nz) or kind == "pane" and r == Role.WINDOW and s == BShape.PANE or kind == "fence" and s == BShape.FENCE or kind == "wall" and s in (BShape.WALLBLOCK, BShape.FENCE):
                connected = True
        name = DIR_NAME[d]
        if kind == "wall":
            props[name] = "low" if connected else "none"
        else:
            props[name] = "true" if connected else "false"
    if kind == "wall":
        props["up"] = "true"
    props["waterlogged"] = "false"
    return props


def _block_for(res: _Resolver, grid: SemanticGrid, x: int, y: int, z: int) -> BlockRef | None:
    role = int(grid.role[x, y, z])
    shape = int(grid.shape[x, y, z])
    normal = int(grid.normal[x, y, z])
    flags = int(grid.flags[x, y, z])
    if role in (Role.EMPTY, Role.INTERIOR) or shape == BShape.NONE:
        return None
    if shape == BShape.LANTERN:
        hanging = "true" if normal == Dir.UP else "false"
        return BlockRef.make(catalog.LANTERN, hanging=hanging, waterlogged="false")
    if shape == BShape.BANNER:
        facing = DIR_NAME.get(normal if normal in HORIZONTAL else Dir.SOUTH, "south")
        fam_acc = _role_palette(res.spec, ["accent", "trim", "roof"])
        colour = "red"
        if fam_acc is not None and fam_acc.families:
            name = fam_acc.families[0].family
            for dye in ("white", "light_gray", "gray", "black", "brown", "red", "orange", "yellow", "lime", "green",
                        "cyan", "light_blue", "blue", "purple", "magenta", "pink"):
                if name.startswith(dye + "_"):
                    colour = dye
                    break
            tone = catalog.family(name).tone if catalog.family(name) else ""
            if colour == "red" and tone in ("dark_wood", "dark_stone"):
                colour = "black"
            elif colour == "red" and tone in ("white",):
                colour = "white"
            elif colour == "red" and tone in ("teal", "copper"):
                colour = "cyan"
        return BlockRef.make(f"minecraft:{colour}_wall_banner", facing=facing)
    if shape == BShape.BARS:
        return BlockRef.make("minecraft:iron_bars", **_connections(grid, x, y, z, "pane"))
    if shape == BShape.LICHEN:
        attach = DIR_NAME.get(OPPOSITE.get(normal, Dir.NORTH), "north")
        props = {d: "false" for d in ("down", "up", "north", "east", "south", "west")}
        props[attach] = "true"
        return BlockRef.make("minecraft:glow_lichen", waterlogged="false", **props)
    if shape in (BShape.BUTTON, BShape.TRAPDOOR, BShape.FENCE_GATE) and role in (Role.DETAIL, Role.SILL, Role.WINDOW):
        # The wall block this piece belongs to: behind a detail or sill, or the wall around a window.
        vx, _, vz = DIR_VEC[normal] if normal in HORIZONTAL else (0, 0, 0)
        wx, wz = (x - vx, z - vz) if role in (Role.DETAIL, Role.SILL) else (x, z)
        wall_fam = None
        if role == Role.WINDOW:
            # Look at a wall block beside the window for its family.
            ax, az = (1, 0) if normal in (Dir.NORTH, Dir.SOUTH) else (0, 1)
            for o in (-1, 1, -2, 2):
                px, pz = x + ax * o, z + az * o
                if grid.in_bounds(px, y, pz) and grid.role[px, y, pz] == Role.WALL and grid.block[px, y, pz] >= 0:
                    wall_fam = _family_of_block(grid.palette[int(grid.block[px, y, pz])].block_id)
                    if wall_fam:
                        break
        elif grid.in_bounds(wx, y, wz) and grid.block[wx, y, wz] >= 0:
            wall_fam = _family_of_block(grid.palette[int(grid.block[wx, y, wz])].block_id)
        if wall_fam is None:
            wall_fam = res.pick_family(_role_palette(res.spec, ["primary"]), wx, y, wz)
        want = {BShape.BUTTON: "button", BShape.TRAPDOOR: "trapdoor", BShape.FENCE_GATE: "fence_gate"}[shape]
        fam = _piece_family(res.spec, wall_fam, want)
        block_id = fam.shapes.get(want) if fam else None
        facing = DIR_NAME.get(normal if normal in HORIZONTAL else Dir.NORTH, "north")
        if shape == BShape.BUTTON:
            return BlockRef.make(block_id or "minecraft:stone_button", face="wall", facing=facing, powered="false")
        if shape == BShape.FENCE_GATE:
            return BlockRef.make(block_id or "minecraft:oak_fence_gate", facing=facing, in_wall="false", open="false",
                                 powered="false")
        if role == Role.SILL:
            return BlockRef.make(block_id or "minecraft:spruce_trapdoor", facing=facing, half="top", open="false",
                                 powered="false", waterlogged="false")
        if role == Role.WINDOW:
            # Boarded window: the panel stands in the opening flush with the outer face.
            inward = DIR_NAME.get(OPPOSITE.get(normal, Dir.NORTH), "north")
            return BlockRef.make(block_id or "minecraft:spruce_trapdoor", facing=inward, half="bottom", open="true",
                                 powered="false", waterlogged="false")
        return BlockRef.make(block_id or "minecraft:spruce_trapdoor", facing=facing, half="top", open="true",
                             powered="false", waterlogged="false")
    if shape == BShape.LEAVES:
        if flags & Flag.PROTRUDE:   # window box
            box = "minecraft:flowering_azalea_leaves" if (x + z) % 2 == 0 else "minecraft:azalea_leaves"
            return BlockRef.make(box, distance="7", persistent="true", waterlogged="false")
        return BlockRef.make(grid.leaf_block, distance="7", persistent="true", waterlogged="false")
    if shape == BShape.VINE:
        # The vine sits in the air cell; it attaches to the wall on the side opposite the wall's outward normal.
        attach = DIR_NAME.get(OPPOSITE.get(normal, Dir.NORTH), "north")
        props = {d: "false" for d in ("north", "east", "south", "west", "up")}
        props[attach] = "true"
        return BlockRef.make("minecraft:vine", **props)
    if shape == BShape.LADDER:
        facing = DIR_NAME.get(normal if normal in HORIZONTAL else Dir.SOUTH, "south")
        return BlockRef.make("minecraft:ladder", facing=facing, waterlogged="false")
    if shape == BShape.CAMPFIRE:
        return BlockRef.make(catalog.CAMPFIRE, lit="true", signal_fire="true", waterlogged="false",
                             facing="north")
    if shape == BShape.HAY:
        return BlockRef.make("minecraft:hay_block", axis="y")
    chain = ROLE_PALETTE.get(role, ["primary"])
    shape_name = SHAPE_NAME.get(shape, "full")
    need = shape_name if role in (Role.DOOR, Role.SHUTTER) else None
    rp = _role_palette(res.spec, chain, need)
    is_roof = role in (Role.ROOF, Role.ROOF_FILL, Role.ROOF_TRIM)
    fam = res.pick_family(rp, x, y, z, speckle=is_roof)
    # Roof texture: a single-family roof mixes in the closest-coloured kindred family at its texture rate.
    if is_roof and rp.texture_rate > 0 and len(rp.families) == 1:
        u = 0.4 * float(res.n_tex[x, y, z]) + 0.6 * float(res.u_tex[x, y, z])
        if u < rp.texture_rate:
            comp = roof_companion(fam.name)
            if comp:
                fam = catalog.family(comp)
    flags = int(grid.flags[x, y, z])

    if shape == BShape.FULL:
        if flags & Flag.NO_TEXTURE:
            return BlockRef.make(fam.shapes["full"])
        block = res.pick_variant(fam, rp, x, y, z)
        # Shading: a darker relative under eaves, balconies and jetties.
        if res.shading > 0 and role in (Role.WALL, Role.FOUNDATION) and (flags & Flag.PERIMETER) and normal in HORIZONTAL \
                and block == fam.shapes["full"]:
            strength = _occlusion(grid, x, y, z, normal)
            if strength > 0 and float(res.u_var[x, y, z]) < strength * res.shading:
                dark = shade_family(fam.name)
                if dark:
                    return BlockRef.make(catalog.family(dark).shapes["full"])
        return BlockRef.make(block)

    block_id = catalog.resolve_shape(fam, shape_name)
    if block_id is None:
        if shape_name == "door":
            block_id = catalog.DOOR_FALLBACK
        else:
            block_id = fam.shapes.get("full", catalog.AIR)
    facing = DIR_NAME.get(normal if normal in HORIZONTAL else Dir.NORTH, "north")

    if block_id.endswith("_stairs"):
        half = "top" if shape == BShape.STAIR_UPSIDE else "bottom"
        return BlockRef.make(block_id, facing=facing, half=half, shape="straight", waterlogged="false")
    if block_id.endswith("_slab"):
        t = "top" if shape == BShape.SLAB_TOP else "bottom"
        return BlockRef.make(block_id, type=t, waterlogged="false")
    if block_id.endswith(("_log", "_stem", "_pillar", "bamboo_block", "basalt", "bone_block")) or \
            block_id.endswith("_wood"):
        axis = "y"
        if role == Role.BEAM:
            axis = "x" if normal in (Dir.EAST, Dir.WEST) else "z"
        return BlockRef.make(block_id, axis=axis)
    if block_id.endswith("_trapdoor"):
        return BlockRef.make(block_id, facing=facing, half="bottom", open="true", powered="false",
                             waterlogged="false")
    if block_id.endswith("_door"):
        half = "upper" if shape == BShape.DOOR_UPPER else "lower"
        return BlockRef.make(block_id, facing=facing, half=half, hinge="left", open="false", powered="false")
    if block_id.endswith("_pane") or block_id == "minecraft:iron_bars":
        return BlockRef.make(block_id, **_connections(grid, x, y, z, "pane"))
    if block_id.endswith("_fence"):
        return BlockRef.make(block_id, **_connections(grid, x, y, z, "fence"))
    if block_id.endswith("_wall"):
        return BlockRef.make(block_id, **_connections(grid, x, y, z, "wall"))
    return BlockRef.make(block_id)


def materials(grid: SemanticGrid, program: BuildProgram) -> None:
    res = _Resolver(grid, program.palette)
    res.shading = program.depth.shading
    xs, ys, zs = np.nonzero((grid.role != Role.EMPTY) & (grid.role != Role.DETAIL))
    dx, dy, dz = np.nonzero(grid.role == Role.DETAIL)
    xs, ys, zs = np.concatenate([xs, dx]), np.concatenate([ys, dy]), np.concatenate([zs, dz])
    count = 0
    for x, y, z in zip(xs, ys, zs):
        ref = _block_for(res, grid, int(x), int(y), int(z))
        if ref is None:
            continue
        grid.block[x, y, z] = grid.intern(ref)
        count += 1
    grid.stage_done("materials", blocks=count, palette=len(grid.palette))
