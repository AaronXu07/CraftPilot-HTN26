"""Where a generated grid lands in the world: in front of the player, centred, facing them."""

from __future__ import annotations

import math

import numpy as np

from craftpilot.grid.semantic import SemanticGrid

_LOOK = ["south", "west", "north", "east"]
_OPPOSITE = {"south": "north", "west": "east", "north": "south", "east": "west"}


def look_from_yaw(yaw: float) -> str:
    """Cardinal direction the player looks along. Yaw 0 = south, 90 = west (Minecraft convention)."""
    return _LOOK[int(((yaw % 360) + 45) // 90) % 4]


def facing_from_yaw(yaw: float) -> str:
    """The building faces the player, so its front is the opposite of the look direction."""
    return _OPPOSITE[look_from_yaw(yaw)]


def trimmed_bbox(grid: SemanticGrid) -> tuple[int, int, int, int, int, int]:
    """Inclusive (x0, y0, z0, x1, y1, z1) of the cells that hold a block."""
    xs, ys, zs = np.nonzero(grid.block >= 0)
    if xs.size == 0:
        return 0, 0, 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(zs.min()), int(xs.max()), int(ys.max()), int(zs.max())


def plan_origin(player: dict, bbox: tuple[int, int, int, int, int, int], gap: int = 2,
                sink: int = 0) -> tuple[int, int, int]:
    """World position of grid cell (0, 0, 0).

    The grid must already be rotated so its front faces the player. The near face of the building sits
    ``gap`` air blocks beyond the block the player stands in, laterally centred on the player, with grid
    y = 0 (the plinth row) at the player's feet minus ``sink``.
    """
    px, py, pz = (float(v) for v in player["pos"])
    bx, bz = math.floor(px), math.floor(pz)
    x0, _y0, z0, x1, _y1, z1 = bbox
    look = look_from_yaw(float(player.get("yaw", 0.0)))
    cx = round(px - (x0 + x1 + 1) / 2)
    cz = round(pz - (z0 + z1 + 1) / 2)
    if look == "south":
        ox, oz = cx, bz + 1 + gap - z0
    elif look == "north":
        ox, oz = cx, bz - 1 - gap - z1
    elif look == "east":
        ox, oz = bx + 1 + gap - x0, cz
    else:
        ox, oz = bx - 1 - gap - x1, cz
    return int(ox), math.floor(py) - int(sink), int(oz)
