"""Test-only helpers: placeholder resolver turning raster+fit into a BlockMap without Track 4."""
from __future__ import annotations

import numpy as np

from copilot.engine.coretypes import BlockMap, Facing, FitResult, Kind, RasterResult
from copilot.engine.fit import fit_surface
from copilot.engine.raster import rasterize
from copilot.engine.scene import Scene

PLACEHOLDER_BASE = {"default": "stone", "stone": "stone", "roof": "deepslate", "wood": "oak_planks", "trim": "quartz_block"}


def to_blockmap(raster: RasterResult, fit: FitResult) -> BlockMap:
    """Placeholder states: stone / stone_slab / stone_stairs with the fitted facing/half."""
    bm: BlockMap = {}
    idx = np.argwhere(fit.kind > 0)
    for i, j, k in idx:
        x, y, z = raster.index_to_world(i, j, k)
        mat = raster.material_names[int(raster.material[i, j, k])] or "default"
        base = PLACEHOLDER_BASE.get(mat, "stone")
        kd = int(fit.kind[i, j, k])
        half = "top" if fit.half[i, j, k] else "bottom"
        if kd in (Kind.FULL, Kind.WALL):
            bm[(x, y, z)] = f"minecraft:{base}"
        elif kd == Kind.SLAB:
            bm[(x, y, z)] = f"minecraft:{base}_slab[type={half}]"
        else:
            bm[(x, y, z)] = f"minecraft:{base}_stairs[facing={Facing(int(fit.facing[i, j, k])).name_mc},half={half}]"
    bm.update(raster.props)
    return bm


def build(scene: Scene, fit_modes=None) -> BlockMap:
    """rasterize + fit + placeholder resolve."""
    r = rasterize(scene)
    f = fit_surface(r, fit_modes or {})
    return to_blockmap(r, f)


def occupied_world(raster: RasterResult) -> np.ndarray:
    """World coords [N,3] of occupied voxels."""
    idx = np.argwhere(raster.material > 0)
    return idx + np.asarray(raster.origin)
