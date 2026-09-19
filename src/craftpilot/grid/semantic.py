"""The semantic voxel grid: roles and tags first, blocks last."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from craftpilot.blocks.state import BlockRef
from craftpilot.grid.enums import BShape, Dir, Role
from craftpilot.program.model import AnchorType, PartSpec


@dataclass
class LayoutPart:
    """A part after layout: absolute footprint inside the grid."""

    index: int
    spec: PartSpec
    mask: np.ndarray                   # (W, D) bool footprint
    x0: int
    z0: int
    x1: int                            # inclusive
    z1: int                            # inclusive
    parent: int | None = None
    is_attachment: bool = False
    jetty: int = 0                     # upper floors overhang the ground floor by this many blocks
    # Filled by massing.
    base_y: int = 0
    top_y: int = 0                     # last wall row
    floor_heights: list[int] = field(default_factory=list)
    floor_masks: list[np.ndarray] = field(default_factory=list)
    # Filled by roof.
    roof_surface: np.ndarray | None = None   # (W, D) float, -inf outside
    roof_mask: np.ndarray | None = None      # (W, D) bool, dilated footprint

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def depth(self) -> int:
        return self.z1 - self.z0 + 1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cz(self) -> float:
        return (self.z0 + self.z1) / 2

    @property
    def eave_y(self) -> int:
        return self.top_y + 1

    @property
    def wall_height(self) -> int:
        return self.top_y - self.base_y + 1

    def floor_block_y(self, k: int) -> int:
        return self.base_y + sum(self.floor_heights[:k]) - 1


@dataclass
class Anchor:
    type: AnchorType
    x: int
    y: int
    z: int
    normal: int
    part: int
    extent: int = 1
    used: bool = False


class SemanticGrid:
    def __init__(self, width: int, height: int, depth: int, rng: np.random.Generator):
        self.W, self.H, self.D = width, height, depth
        shape = (width, height, depth)
        self.role = np.zeros(shape, dtype=np.uint8)
        self.shape = np.zeros(shape, dtype=np.uint8)
        self.normal = np.zeros(shape, dtype=np.int8)
        self.part_id = np.full(shape, -1, dtype=np.int16)
        self.h_norm = np.zeros(shape, dtype=np.float32)
        self.flags = np.zeros(shape, dtype=np.uint16)
        self.block = np.full(shape, -1, dtype=np.int32)
        self.palette: list[BlockRef] = []
        self._palette_index: dict[BlockRef, int] = {}
        self.parts: list[LayoutPart] = []
        self.anchors: list[Anchor] = []
        self.reserved = np.zeros(shape, dtype=bool)
        self.footprint = np.zeros((width, depth), dtype=bool)   # union of all part masks
        self.exterior = np.zeros((width, depth), dtype=bool)    # outside footprint + outset
        self.roof_surface = np.full((width, depth), -np.inf, dtype=np.float32)
        self.roof_part = np.full((width, depth), -1, dtype=np.int16)
        self.front: int = Dir.SOUTH
        self.door: tuple[int, int, int] | None = None
        self.leaf_block: str = "minecraft:oak_leaves"
        self.report: dict[str, Any] = {"stages": [], "notes": []}
        self.rng = rng

    # ---- basic access -------------------------------------------------

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H and 0 <= z < self.D

    def set(self, x: int, y: int, z: int, role: int, shape: int = BShape.FULL, normal: int = Dir.NONE,
            part: int = -1, h_norm: float | None = None, flags: int = 0) -> bool:
        if not self.in_bounds(x, y, z):
            return False
        self.role[x, y, z] = role
        self.shape[x, y, z] = shape
        self.normal[x, y, z] = normal
        self.part_id[x, y, z] = part
        if h_norm is not None:
            self.h_norm[x, y, z] = h_norm
        self.flags[x, y, z] = flags
        return True

    def get_role(self, x: int, y: int, z: int) -> int:
        if not self.in_bounds(x, y, z):
            return Role.EMPTY
        return int(self.role[x, y, z])

    def is_air(self, x: int, y: int, z: int) -> bool:
        r = self.get_role(x, y, z)
        return r in (Role.EMPTY, Role.INTERIOR) or (r == Role.WINDOW and self.shape[x, y, z] == BShape.NONE)

    def is_solid(self, x: int, y: int, z: int) -> bool:
        """Full-cube-ish block: something panes and fences connect to."""
        if not self.in_bounds(x, y, z):
            return False
        r = int(self.role[x, y, z])
        if r in (Role.EMPTY, Role.INTERIOR):
            return False
        s = int(self.shape[x, y, z])
        return s in (BShape.FULL, BShape.LOG, BShape.PILLAR, BShape.STRIPPED_LOG, BShape.STAIR, BShape.STAIR_UPSIDE)

    def intern(self, ref: BlockRef) -> int:
        i = self._palette_index.get(ref)
        if i is None:
            i = len(self.palette)
            self.palette.append(ref)
            self._palette_index[ref] = i
        return i

    def note(self, text: str) -> None:
        self.report["notes"].append(text)

    def stage_done(self, name: str, **info: Any) -> None:
        self.report["stages"].append({"stage": name, **info})

    def part_by_name(self, name: str) -> LayoutPart | None:
        for p in self.parts:
            if p.spec.name == name:
                return p
        return None

    def block_count(self) -> int:
        return int((self.block >= 0).sum())
