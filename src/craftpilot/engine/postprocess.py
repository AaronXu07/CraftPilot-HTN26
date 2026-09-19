"""Postprocess on resolved blocks: stair shapes, attachable support, sanity checks."""

from __future__ import annotations

import numpy as np

from craftpilot.blocks.state import BlockRef
from craftpilot.grid.enums import CCW, DIR_VEC, OPPOSITE, BShape, Dir, Role
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.program.model import BuildProgram

_NAME_TO_DIR = {"north": Dir.NORTH, "east": Dir.EAST, "south": Dir.SOUTH, "west": Dir.WEST}


def _ref_at(grid: SemanticGrid, x: int, y: int, z: int) -> BlockRef | None:
    if not grid.in_bounds(x, y, z):
        return None
    i = int(grid.block[x, y, z])
    return grid.palette[i] if i >= 0 else None


def _stair_info(ref: BlockRef | None) -> tuple[int, str] | None:
    if ref is None or not ref.is_stairs():
        return None
    f = _NAME_TO_DIR.get(ref.prop("facing") or "north", Dir.NORTH)
    return f, ref.prop("half") or "bottom"


def _can_take_shape(grid: SemanticGrid, x: int, y: int, z: int, facing: int, half: str, d: int) -> bool:
    vx, _, vz = DIR_VEC[d]
    info = _stair_info(_ref_at(grid, x + vx, y, z + vz))
    return info is None or info[0] != facing or info[1] != half


def _stair_shape(grid: SemanticGrid, x: int, y: int, z: int, facing: int, half: str) -> str:
    """Port of StairsBlock.getStairsShape."""
    vx, _, vz = DIR_VEC[facing]
    back = _stair_info(_ref_at(grid, x + vx, y, z + vz))
    if back is not None and back[1] == half:
        bf = back[0]
        if (bf in (Dir.NORTH, Dir.SOUTH)) != (facing in (Dir.NORTH, Dir.SOUTH)) and \
                _can_take_shape(grid, x, y, z, facing, half, OPPOSITE[bf]):
            return "outer_left" if bf == CCW[facing] else "outer_right"
    front = _stair_info(_ref_at(grid, x - vx, y, z - vz))
    if front is not None and front[1] == half:
        ff = front[0]
        if (ff in (Dir.NORTH, Dir.SOUTH)) != (facing in (Dir.NORTH, Dir.SOUTH)) and \
                _can_take_shape(grid, x, y, z, facing, half, ff):
            return "inner_left" if ff == CCW[facing] else "inner_right"
    return "straight"


def _fix_stairs(grid: SemanticGrid) -> int:
    changed = 0
    updates = []
    for x, y, z in zip(*np.nonzero(grid.block >= 0)):
        ref = grid.palette[int(grid.block[x, y, z])]
        info = _stair_info(ref)
        if info is None:
            continue
        shape = _stair_shape(grid, int(x), int(y), int(z), info[0], info[1])
        if shape != (ref.prop("shape") or "straight"):
            updates.append((int(x), int(y), int(z), ref.with_props(shape=shape)))
    for x, y, z, ref in updates:
        grid.block[x, y, z] = grid.intern(ref)
        changed += 1
    return changed


def _fix_attachables(grid: SemanticGrid) -> int:
    removed = 0
    for x, y, z in zip(*np.nonzero(grid.role == Role.LIGHT)):
        x, y, z = int(x), int(y), int(z)
        below = grid.in_bounds(x, y - 1, z) and grid.block[x, y - 1, z] >= 0
        above = grid.in_bounds(x, y + 1, z) and grid.block[x, y + 1, z] >= 0
        ref = _ref_at(grid, x, y, z)
        if ref is None:
            continue
        if below:
            grid.block[x, y, z] = grid.intern(ref.with_props(hanging="false"))
        elif above:
            grid.block[x, y, z] = grid.intern(ref.with_props(hanging="true"))
        else:
            grid.block[x, y, z] = -1
            grid.role[x, y, z] = Role.EMPTY
            removed += 1
    return removed


def _fix_door_hinges(grid: SemanticGrid) -> None:
    # Give the door a hinge on the side that has a wall so it opens against it.
    for x, y, z in zip(*np.nonzero((grid.role == Role.DOOR) & (grid.shape == BShape.DOOR_LOWER))):
        x, y, z = int(x), int(y), int(z)
        ref = _ref_at(grid, x, y, z)
        if ref is None:
            continue
        facing = _NAME_TO_DIR.get(ref.prop("facing") or "north", Dir.NORTH)
        left = CCW[facing]
        lx, _, lz = DIR_VEC[left]
        hinge = "left" if grid.block[x + lx, y, z + lz] >= 0 else "right"
        for yy, sh in ((y, "lower"), (y + 1, "upper")):
            r = _ref_at(grid, x, yy, z)
            if r is not None:
                grid.block[x, yy, z] = grid.intern(r.with_props(hinge=hinge))


def postprocess(grid: SemanticGrid, program: BuildProgram) -> None:
    stairs = _fix_stairs(grid)
    removed = _fix_attachables(grid)
    _fix_door_hinges(grid)
    grid.stage_done("postprocess", stair_shapes=stairs, removed_attachables=removed, blocks=grid.block_count())
