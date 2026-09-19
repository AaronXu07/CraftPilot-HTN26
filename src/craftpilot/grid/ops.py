"""Whole-grid operations: rotation of the finished building."""

from __future__ import annotations

import numpy as np

from craftpilot.blocks.state import rotate_ref_cw
from craftpilot.grid.enums import CW
from craftpilot.grid.semantic import SemanticGrid

_FACING_ORDER = {"north": 0, "east": 1, "south": 2, "west": 3}


def quarter_turns_for_facing(facing: str, current: str = "south") -> int:
    """Clockwise quarter turns (viewed from above) that turn `current` into `facing`."""
    return (_FACING_ORDER[facing] - _FACING_ORDER[current]) % 4


def rotate_cw(grid: SemanticGrid, turns: int) -> SemanticGrid:
    """Rotate the resolved grid clockwise viewed from above (north -> east) by 90 degree steps."""
    turns %= 4
    if turns == 0:
        return grid
    W_before, D_before = grid.W, grid.D
    # In our frame (x east, z south) a clockwise turn maps (x, z) -> (W-1-z, x); rot90 with k=-1 on
    # axes (x, z) does that.
    def rot(a: np.ndarray) -> np.ndarray:
        if a.ndim == 3:
            return np.ascontiguousarray(np.rot90(a, k=-turns, axes=(2, 0)))
        return np.ascontiguousarray(np.rot90(a, k=-turns, axes=(1, 0)))

    for name in ("role", "shape", "part_id", "h_norm", "flags", "block", "reserved"):
        setattr(grid, name, rot(getattr(grid, name)))
    normal = rot(grid.normal)
    for _ in range(turns):
        remap = normal.copy()
        for src, dst in CW.items():
            remap[normal == src] = dst
        normal = remap
    grid.normal = normal
    for name in ("footprint", "exterior", "roof_surface", "roof_part"):
        setattr(grid, name, rot(getattr(grid, name)))
    grid.W, grid.D = grid.role.shape[0], grid.role.shape[2]
    grid.palette = [rotate_ref_cw(ref, turns) for ref in grid.palette]
    grid._palette_index = {ref: i for i, ref in enumerate(grid.palette)}
    front = grid.front
    for _ in range(turns):
        front = CW.get(front, front)
    grid.front = front
    if grid.door is not None:
        x, y, z = grid.door
        w, d = W_before, D_before
        for _ in range(turns):
            x, z = d - 1 - z, x      # one clockwise turn: (x, z) -> (D-1-z, x)
            w, d = d, w
        grid.door = (x, y, z)
    return grid
