"""Concrete block references and rotation of directional properties."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BlockRef:
    block_id: str
    props: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @staticmethod
    def make(block_id: str, **props: str) -> BlockRef:
        return BlockRef(block_id, tuple(sorted((k, str(v)) for k, v in props.items())))

    def prop(self, key: str) -> str | None:
        for k, v in self.props:
            if k == key:
                return v
        return None

    def with_props(self, **props: str) -> BlockRef:
        d = dict(self.props)
        d.update({k: str(v) for k, v in props.items()})
        return BlockRef(self.block_id, tuple(sorted(d.items())))

    def as_dict(self) -> dict[str, str]:
        return dict(self.props)

    def state_string(self) -> str:
        """The block state as the game's command syntax: ``minecraft:oak_stairs[facing=north,half=bottom]``."""
        if not self.props:
            return self.block_id
        return self.block_id + "[" + ",".join(f"{k}={v}" for k, v in self.props) + "]"

    def is_stairs(self) -> bool:
        return self.block_id.endswith("_stairs")


_ROT_CW = {"north": "east", "east": "south", "south": "west", "west": "north"}
_AXIS_ROT = {"x": "z", "z": "x", "y": "y"}


def rotate_ref_cw(ref: BlockRef, quarter_turns: int) -> BlockRef:
    """Rotate directional properties clockwise (viewed from above) by 90 degree steps."""
    quarter_turns %= 4
    if quarter_turns == 0:
        return ref
    d = dict(ref.props)
    for _ in range(quarter_turns):
        if "facing" in d and d["facing"] in _ROT_CW:
            d["facing"] = _ROT_CW[d["facing"]]
        if "axis" in d:
            d["axis"] = _AXIS_ROT[d["axis"]]
        # Connection booleans on panes, fences, walls.
        for a, b, c, e in (("north", "east", "south", "west"),):
            if all(k in d for k in (a, b, c, e)):
                d[a], d[b], d[c], d[e] = d[e], d[a], d[b], d[c]
    return BlockRef(ref.block_id, tuple(sorted(d.items())))
