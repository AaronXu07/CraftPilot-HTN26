"""Shared engine data types. See CONTRACTS.md §5."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
IVec3 = Tuple[int, int, int]
BlockMap = Dict[IVec3, str]  # world coords -> "minecraft:id[prop=val,...]" (never air)


class Facing(IntEnum):
    """Horizontal facing. For stairs this is the direction of the FULL (high) side."""

    NORTH = 0  # -z
    EAST = 1  # +x
    SOUTH = 2  # +z
    WEST = 3  # -x

    @property
    def vector(self) -> IVec3:
        return FACING_VECTORS[int(self)]

    @property
    def name_mc(self) -> str:
        return FACING_NAMES[int(self)]

    @staticmethod
    def from_name(name: str) -> "Facing":
        return Facing(FACING_NAMES.index(name.lower()))

    def opposite(self) -> "Facing":
        return Facing((int(self) + 2) % 4)

    def rotated(self, quarter_turns_cw: int) -> "Facing":
        """Rotate clockwise (viewed from above) by 90° steps. north->east->south->west."""
        return Facing((int(self) + quarter_turns_cw) % 4)


FACING_NAMES = ["north", "east", "south", "west"]
FACING_VECTORS: List[IVec3] = [(0, 0, -1), (1, 0, 0), (0, 0, 1), (-1, 0, 0)]


class Half(IntEnum):
    BOTTOM = 0
    TOP = 1


class Kind(IntEnum):
    """Fit result kinds."""

    AIR = 0
    FULL = 1
    SLAB = 2
    STAIRS = 3
    WALL = 4


@dataclass
class Bbox:
    lo: np.ndarray
    hi: np.ndarray

    @staticmethod
    def of(lo, hi) -> "Bbox":
        return Bbox(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))

    @staticmethod
    def empty() -> "Bbox":
        return Bbox(np.full(3, np.inf), np.full(3, -np.inf))

    def is_empty(self) -> bool:
        return bool(np.any(self.hi < self.lo))

    def union(self, other: "Bbox") -> "Bbox":
        if self.is_empty():
            return Bbox(other.lo.copy(), other.hi.copy())
        if other.is_empty():
            return Bbox(self.lo.copy(), self.hi.copy())
        return Bbox(np.minimum(self.lo, other.lo), np.maximum(self.hi, other.hi))

    def intersects(self, other: "Bbox") -> bool:
        if self.is_empty() or other.is_empty():
            return False
        return bool(np.all(self.lo <= other.hi) and np.all(other.lo <= self.hi))

    def expanded(self, pad: float) -> "Bbox":
        return Bbox(self.lo - pad, self.hi + pad)

    @property
    def size(self) -> np.ndarray:
        return np.maximum(self.hi - self.lo, 0.0)

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2.0

    def voxel_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Integer voxel index range [lo, hi) that covers this bbox (voxel v spans [v, v+1))."""
        return np.floor(self.lo).astype(int), np.ceil(self.hi).astype(int)

    def to_list(self):
        return [[float(v) for v in self.lo], [float(v) for v in self.hi]]

    def __repr__(self) -> str:
        return f"Bbox({self.lo.tolist()} .. {self.hi.tolist()})"


@dataclass
class RasterResult:
    origin: IVec3
    material: np.ndarray  # int16 [X,Y,Z], 0 = air
    material_names: List[str]  # index 0 = "" (air)
    owner: np.ndarray  # int32 [X,Y,Z], -1 = air
    sdf: np.ndarray  # float32 [X,Y,Z]
    props: Dict[IVec3, str] = field(default_factory=dict)
    scene_sdf: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None
    scene_hash: str = ""
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def shape(self) -> IVec3:
        return tuple(int(v) for v in self.material.shape)  # type: ignore[return-value]

    @property
    def occupied(self) -> np.ndarray:
        return self.material > 0

    def world_to_index(self, x: int, y: int, z: int) -> IVec3:
        return (int(x) - self.origin[0], int(y) - self.origin[1], int(z) - self.origin[2])

    def index_to_world(self, i: int, j: int, k: int) -> IVec3:
        return (int(i) + self.origin[0], int(j) + self.origin[1], int(k) + self.origin[2])

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        i, j, k = self.world_to_index(x, y, z)
        X, Y, Z = self.shape
        return 0 <= i < X and 0 <= j < Y and 0 <= k < Z

    def material_name_at(self, x: int, y: int, z: int) -> str:
        if not self.in_bounds(x, y, z):
            return ""
        i, j, k = self.world_to_index(x, y, z)
        return self.material_names[int(self.material[i, j, k])]

    def block_count(self) -> int:
        return int(np.count_nonzero(self.material)) + len(self.props)

    def bbox(self) -> Bbox:
        X, Y, Z = self.shape
        return Bbox.of(self.origin, (self.origin[0] + X, self.origin[1] + Y, self.origin[2] + Z))


@dataclass
class FitResult:
    kind: np.ndarray  # uint8 [X,Y,Z], see Kind
    facing: np.ndarray  # uint8 [X,Y,Z]
    half: np.ndarray  # uint8 [X,Y,Z]

    @staticmethod
    def full_blocks(raster: RasterResult) -> "FitResult":
        """A FitResult with every occupied voxel as a full block (fit disabled)."""
        kind = np.where(raster.material > 0, np.uint8(Kind.FULL), np.uint8(Kind.AIR)).astype(np.uint8)
        return FitResult(kind=kind, facing=np.zeros_like(kind), half=np.zeros_like(kind))

    def counts(self) -> Dict[str, int]:
        return {k.name.lower(): int(np.count_nonzero(self.kind == int(k))) for k in Kind if k != Kind.AIR}


@dataclass
class LintFinding:
    rule: str  # e.g. "R3"
    severity: str  # "warn" | "error"
    message: str
    objects: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        objs = f" [{', '.join(self.objects)}]" if self.objects else ""
        return f"{self.severity.upper()} {self.rule}: {self.message}{objs}"


def parse_state(state: str) -> Tuple[str, Dict[str, str]]:
    """'minecraft:oak_stairs[facing=north,half=top]' -> ('minecraft:oak_stairs', {...})."""
    state = state.strip()
    if "[" not in state:
        return normalize_id(state), {}
    base, rest = state.split("[", 1)
    rest = rest.rstrip("]")
    props: Dict[str, str] = {}
    if rest:
        for kv in rest.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                props[k.strip()] = v.strip()
    return normalize_id(base), props


def format_state(block_id: str, props: Optional[Dict[str, str]] = None) -> str:
    block_id = normalize_id(block_id)
    if not props:
        return block_id
    inner = ",".join(f"{k}={props[k]}" for k in sorted(props))
    return f"{block_id}[{inner}]"


def normalize_id(block_id: str) -> str:
    block_id = block_id.strip()
    if ":" not in block_id:
        return "minecraft:" + block_id
    return block_id


def short_id(block_id: str) -> str:
    return block_id.split(":", 1)[1] if block_id.startswith("minecraft:") else block_id
