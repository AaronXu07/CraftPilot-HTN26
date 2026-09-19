"""Enums stored in the semantic grid arrays."""

from __future__ import annotations

from enum import IntEnum


class Role(IntEnum):
    EMPTY = 0
    INTERIOR = 1        # air inside the building
    WALL = 2
    FRAME = 3
    FOUNDATION = 4
    FLOOR = 5
    ROOF = 6            # visible roof surface block
    ROOF_FILL = 7       # solid roof body under the surface
    ROOF_TRIM = 8       # brackets under the eave
    WINDOW = 9
    DOOR = 10
    SILL = 11
    SHUTTER = 12
    TRIM = 13
    PILLAR = 14
    RAILING = 15
    CHIMNEY = 16
    CHIMNEY_CAP = 17
    LIGHT = 18
    ACCENT = 19
    PARAPET = 20
    MERLON = 21
    CORBEL = 22
    BEAM = 23
    STEP = 24           # stairs leading to a door
    FLOOR_LIP = 25
    STAIRCASE = 26      # interior stairs between floors
    PARTITION = 27      # interior wall
    FOLIAGE = 28        # vines and leaves
    DETAIL = 29         # small protruding textures: buttons, trapdoors, lichen


class BShape(IntEnum):
    """Block shape within a family."""

    FULL = 0
    STAIR = 1
    STAIR_UPSIDE = 2
    SLAB_BOTTOM = 3
    SLAB_TOP = 4
    LOG = 5             # axis from normal
    PANE = 6
    FENCE = 7
    WALLBLOCK = 8
    TRAPDOOR = 9
    DOOR_LOWER = 10
    DOOR_UPPER = 11
    LANTERN = 12
    NONE = 13           # role present but rendered as air (slit windows)
    PILLAR = 14
    STRIPPED_LOG = 15
    CAMPFIRE = 16
    BANNER = 17         # wall banner, facing from normal
    LADDER = 18         # facing from normal (the side you climb from)
    BUTTON = 21         # wall button, facing = outward normal
    LICHEN = 22         # glow lichen on the wall face
    BARS = 23           # iron bars
    FENCE_GATE = 24     # facing from normal
    HAY = 25            # hay bale under a chimney campfire for tall smoke
    VINE = 19           # attached to the wall in direction OPPOSITE[normal]... normal = wall's outward side
    LEAVES = 20


class Dir(IntEnum):
    NONE = 0
    NORTH = 1   # -z
    EAST = 2    # +x
    SOUTH = 3   # +z
    WEST = 4    # -x
    UP = 5
    DOWN = 6


DIR_VEC: dict[int, tuple[int, int, int]] = {
    Dir.NONE: (0, 0, 0),
    Dir.NORTH: (0, 0, -1),
    Dir.EAST: (1, 0, 0),
    Dir.SOUTH: (0, 0, 1),
    Dir.WEST: (-1, 0, 0),
    Dir.UP: (0, 1, 0),
    Dir.DOWN: (0, -1, 0),
}

OPPOSITE: dict[int, int] = {
    Dir.NONE: Dir.NONE,
    Dir.NORTH: Dir.SOUTH,
    Dir.SOUTH: Dir.NORTH,
    Dir.EAST: Dir.WEST,
    Dir.WEST: Dir.EAST,
    Dir.UP: Dir.DOWN,
    Dir.DOWN: Dir.UP,
}

# Counter-clockwise when viewed from above, matching Minecraft's Direction.getCounterClockWise.
CCW: dict[int, int] = {Dir.NORTH: Dir.WEST, Dir.WEST: Dir.SOUTH, Dir.SOUTH: Dir.EAST, Dir.EAST: Dir.NORTH}
CW: dict[int, int] = {v: k for k, v in CCW.items()}

HORIZONTAL: tuple[int, ...] = (Dir.NORTH, Dir.EAST, Dir.SOUTH, Dir.WEST)

DIR_NAME: dict[int, str] = {
    Dir.NORTH: "north", Dir.EAST: "east", Dir.SOUTH: "south", Dir.WEST: "west",
    Dir.UP: "up", Dir.DOWN: "down", Dir.NONE: "north",
}

SIDE_TO_DIR: dict[str, int] = {"north": Dir.NORTH, "east": Dir.EAST, "south": Dir.SOUTH, "west": Dir.WEST}


class Flag(IntEnum):
    PROTRUDE = 1
    INSET = 2
    CORBEL = 4
    RESERVED = 8
    NO_TEXTURE = 16
    OVERHANG = 32
    PERIMETER = 64
    FIXED_SHAPE = 128   # stairs that must stay straight (slits, wall details)
