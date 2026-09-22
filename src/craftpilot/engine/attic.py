"""Attic: storeys inside the roof, wherever the roof leaves enough head room.

The roof stage fills the volume above the top wall storey with attic air (Role.INTERIOR) under the
roof surface. This stage lays floor rows through that air, one storey height apart, for as long as
the cells with two clear rows above the floor still make a room. The first floor row is at eave_y,
so the top wall storey gets the ceiling the facade already assumes. Windows, dormers and stairs
read part.attic_heights afterwards.
"""

from __future__ import annotations

import numpy as np

from craftpilot.grid.enums import BShape, Dir, Role
from craftpilot.grid.semantic import LayoutPart, SemanticGrid
from craftpilot.program.model import BuildProgram, RoofType

MIN_CELLS = 12          # a room, not a crawl space
MIN_FRACTION = 0.3      # of the top wall storey's interior area
MIN_SPAN = 4            # narrowest usable bounding box side
HEAD_ROOM = 2           # clear rows above the floor row


def _air(grid: SemanticGrid, part: LayoutPart, y: int) -> np.ndarray:
    if not (0 <= y < grid.H):
        return np.zeros((grid.W, grid.D), dtype=bool)
    return (grid.role[:, y, :] == Role.INTERIOR) & (grid.part_id[:, y, :] == part.index)


def usable_mask(grid: SemanticGrid, part: LayoutPart, floor_y: int) -> np.ndarray:
    """Cells of an attic level a player can stand on: floor row present, HEAD_ROOM rows of air above."""
    m = part.floor_masks[-1] if part.floor_masks else part.mask
    ok = m & (grid.part_id[:, floor_y, :] == part.index) & np.isin(grid.role[:, floor_y, :], (Role.FLOOR, Role.INTERIOR))
    for dy in range(1, HEAD_ROOM + 1):
        ok &= _air(grid, part, floor_y + dy)
    return ok


def _room_enough(usable: np.ndarray, base_area: int) -> bool:
    n = int(usable.sum())
    if n < max(MIN_CELLS, int(MIN_FRACTION * base_area)):
        return False
    xs, zs = np.nonzero(usable)
    return int(xs.max() - xs.min() + 1) >= MIN_SPAN and int(zs.max() - zs.min() + 1) >= MIN_SPAN


def attic(grid: SemanticGrid, program: BuildProgram) -> None:
    info = []
    for part in grid.parts:
        part.attic_heights = []
        spec = part.spec
        if not spec.attic or not part.floor_masks or part.roof_surface is None:
            continue
        if spec.roof.type in (RoofType.flat, RoofType.parapet, RoofType.none):
            continue
        top = part.floor_masks[-1]
        base_area = int((top & _air(grid, part, part.floor_block_y(len(part.floor_heights) - 1) + 1)).sum())
        if base_area == 0:
            continue
        fh = max(3, spec.floor_height)
        floor_y = part.eave_y
        while floor_y + HEAD_ROOM < grid.H:
            usable = usable_mask(grid, part, floor_y)
            if not _room_enough(usable, base_area):
                break
            # The whole row of attic air becomes floor, so the storey below gets a complete ceiling
            # and the low space under the eaves is closed rather than a gap to fall through.
            for x, z in zip(*np.nonzero(top & _air(grid, part, floor_y))):
                grid.set(int(x), floor_y, int(z), Role.FLOOR, BShape.FULL, Dir.UP, part.index, 1.0)
            part.attic_heights.append(fh)
            floor_y += fh
        if part.attic_heights:
            info.append((spec.name, part.attic_floor_ys()))
    grid.stage_done("attic", parts=info)
