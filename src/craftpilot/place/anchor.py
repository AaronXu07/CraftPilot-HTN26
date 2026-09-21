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


# ------------------------------------------------------------------ a base area marked by two points

Area = tuple[int, int, int, int, int, int]   # inclusive x0, y0, z0, x1, y1, z1, normalised


def normalise_area(points: list[list[int]] | list[tuple[int, int, int]]) -> Area:
    """Two marked blocks (any order) -> inclusive min/max corners."""
    if len(points) != 2 or any(len(p) != 3 for p in points):
        raise ValueError("area needs exactly two [x, y, z] points")
    (ax, ay, az), (bx, by, bz) = ((int(v) for v in p) for p in points)
    return (min(ax, bx), min(ay, by), min(az, bz), max(ax, bx), max(ay, by), max(az, bz))


def facing_from_area(area: Area, pos: list[float] | tuple[float, ...] | None, yaw: float = 0.0) -> str:
    """The building faces the side of the area the player stands on. Standing inside the area (or with
    no position) falls back to the look direction, as when aiming."""
    if pos is None or len(pos) != 3:
        return facing_from_yaw(yaw)
    x0, _, z0, x1, _, z1 = area
    px, pz = float(pos[0]), float(pos[2])
    dx = px - (x0 + x1 + 1) / 2
    dz = pz - (z0 + z1 + 1) / 2
    inside = x0 <= math.floor(px) <= x1 and z0 <= math.floor(pz) <= z1
    if inside or (dx == 0 and dz == 0):
        return facing_from_yaw(yaw)
    if abs(dx) > abs(dz):
        return "east" if dx > 0 else "west"
    return "south" if dz > 0 else "north"


def area_footprint(area: Area, facing: str) -> tuple[int, int]:
    """(width, depth) of the area in the engine's frame: width runs along the front, depth away from it.
    The engine builds facing south and is rotated afterwards, so an east or west front swaps the two."""
    x0, _, z0, x1, _, z1 = area
    w, d = x1 - x0 + 1, z1 - z0 + 1
    return (d, w) if facing in ("east", "west") else (w, d)


def area_origin(area: Area, world_w: int, world_d: int, sink: int = 0) -> tuple[int, int, int]:
    """World position of grid cell (0, 0, 0) for a grid already rotated to ``world_w`` x ``world_d``:
    the base row sits on top of the marked blocks (their lower y plus one, minus ``sink``), and a grid
    that does not match the area exactly is centred on it."""
    x0, y0, z0, x1, _, z1 = area
    ox = x0 + ((x1 - x0 + 1) - world_w) // 2
    oz = z0 + ((z1 - z0 + 1) - world_d) // 2
    return int(ox), int(y0) + 1 - int(sink), int(oz)
