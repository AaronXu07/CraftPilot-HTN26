"""Export the resolved grid to a Litematica schematic through litemapy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from litemapy import BlockState, Region, Schematic

from craftpilot.grid.semantic import SemanticGrid


def _trim_bounds(grid: SemanticGrid) -> tuple[int, int, int, int, int, int]:
    xs, ys, zs = np.nonzero(grid.block >= 0)
    if xs.size == 0:
        return 0, 0, 0, 1, 1, 1
    return int(xs.min()), int(ys.min()), int(zs.min()), int(xs.max()) + 1, int(ys.max()) + 1, int(zs.max()) + 1


def to_schematic(grid: SemanticGrid, name: str, author: str, description: str, mc_version: int) -> Schematic:
    x0, y0, z0, x1, y1, z1 = _trim_bounds(grid)
    # Keep the full width and depth so the schematic origin matches the requested bounds,
    # but trim empty height above the building.
    reg = Region(0, 0, 0, grid.W, y1, grid.D)
    states = [BlockState(ref.block_id, **ref.as_dict()) for ref in grid.palette]
    for x, y, z in zip(*np.nonzero(grid.block[:, :y1, :] >= 0)):
        reg[int(x), int(y), int(z)] = states[int(grid.block[x, y, z])]
    schem = Schematic(name=name, author=author, description=description, regions={"main": reg},
                      mc_version=mc_version)
    return schem


def save(grid: SemanticGrid, path: Path, name: str, author: str, description: str, mc_version: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    schem = to_schematic(grid, name, author, description, mc_version)
    schem.save(str(path))
    return path


def report_json(grid: SemanticGrid) -> str:
    return json.dumps(grid.report, default=str)
