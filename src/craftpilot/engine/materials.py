"""Materials: palette, gradients, and texturing resolved over the semantic grid."""

from __future__ import annotations

import numpy as np

from craftpilot.blocks import catalog
from craftpilot.blocks.catalog import Family
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
    Role.DOOR: ["framing", "primary"],
    Role.SILL: ["trim", "roof"],
    Role.SHUTTER: ["trim", "framing", "primary"],
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


def _role_palette(spec: PaletteSpec, chain: list[str]) -> RolePalette | None:
    for name in chain:
        rp = getattr(spec, name, None)
        if rp is not None and rp.families:
            return rp
    if "glass" in chain:
        return DEFAULT_GLASS
    return spec.primary


class _Resolver:
    def __init__(self, grid: SemanticGrid, spec: PaletteSpec):
        self.grid = grid
        self.spec = spec
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

    def pick_family(self, rp: RolePalette, x: int, y: int, z: int) -> Family:
        fams = self.families_for(rp)
        if len(fams) == 1:
            return fams[0][0]
        u = float(self.n_family[x, y, z])
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
        if not fam.variants or rp.texture_rate <= 0:
            return base
        hn = float(self.grid.h_norm[x, y, z])
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
    if shape == BShape.LEAVES:
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
        return BlockRef.make(catalog.CAMPFIRE, lit="true", signal_fire="false", waterlogged="false",
                             facing="north")
    chain = ROLE_PALETTE.get(role, ["primary"])
    rp = _role_palette(res.spec, chain)
    fam = res.pick_family(rp, x, y, z)
    shape_name = SHAPE_NAME.get(shape, "full")
    flags = int(grid.flags[x, y, z])

    if shape == BShape.FULL:
        if flags & Flag.NO_TEXTURE:
            return BlockRef.make(fam.shapes["full"])
        return BlockRef.make(res.pick_variant(fam, rp, x, y, z))

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
    xs, ys, zs = np.nonzero(grid.role != Role.EMPTY)
    count = 0
    for x, y, z in zip(xs, ys, zs):
        ref = _block_for(res, grid, int(x), int(y), int(z))
        if ref is None:
            continue
        grid.block[x, y, z] = grid.intern(ref)
        count += 1
    grid.stage_done("materials", blocks=count, palette=len(grid.palette))
