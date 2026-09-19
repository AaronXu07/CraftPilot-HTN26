"""Block-map and scene diffs for incremental placement. See plan.md §9 and CONTRACTS.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .coretypes import Bbox, BlockMap, IVec3
from .scene import Scene


@dataclass
class DiffResult:
    added: Dict[IVec3, str] = field(default_factory=dict)
    removed: Set[IVec3] = field(default_factory=set)
    changed: Dict[IVec3, str] = field(default_factory=dict)  # new state at positions that changed

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def count(self) -> int:
        return len(self.added) + len(self.removed) + len(self.changed)


def diff_block_maps(old: BlockMap, new: BlockMap) -> DiffResult:
    """Blocks to add, remove and change to turn `old` into `new`."""
    d = DiffResult()
    for pos, state in new.items():
        prev = old.get(pos)
        if prev is None:
            d.added[pos] = state
        elif prev != state:
            d.changed[pos] = state
    for pos in old:
        if pos not in new:
            d.removed.add(pos)
    return d


def to_setblocks(diff: DiffResult) -> List[Tuple[int, int, int, str]]:
    """Flat setblock list (removed -> minecraft:air), removals first so supports are cleared before re-adds."""
    out: List[Tuple[int, int, int, str]] = [(p[0], p[1], p[2], "minecraft:air") for p in sorted(diff.removed)]
    out += [(p[0], p[1], p[2], s) for p, s in sorted(diff.changed.items())]
    out += [(p[0], p[1], p[2], s) for p, s in sorted(diff.added.items())]
    return out


def layer_chunks(blocks: List[Tuple[int, int, int, str]], chunk_size: int = 1500) -> List[List[Tuple[int, int, int, str]]]:
    """Split a setblock list into bottom-up chunks (sorted by y, then x, then z) for animated placement."""
    ordered = sorted(blocks, key=lambda b: (b[1], b[0], b[2]))
    return [ordered[i : i + chunk_size] for i in range(0, len(ordered), max(1, int(chunk_size)))]


def changed_bbox(scene_a: Scene, scene_b: Scene) -> Optional[Bbox]:
    """Union bbox (in both scenes) of objects that differ between two scenes; None if nothing differs."""
    a = {o.id: o for o in scene_a.objects}
    b = {o.id: o for o in scene_b.objects}
    bb = Bbox.empty()
    changed = False
    for oid in set(a) | set(b):
        oa, ob = a.get(oid), b.get(oid)
        if oa is not None and ob is not None and oa.to_dict() == ob.to_dict():
            continue
        changed = True
        if oa is not None:
            bb = bb.union(scene_a.object_bbox(oa))
        if ob is not None:
            bb = bb.union(scene_b.object_bbox(ob))
    if not changed:
        return None
    if scene_a.materials != scene_b.materials:
        bb = bb.union(scene_a.bbox()).union(scene_b.bbox())
    return bb


def diff_stats(diff: DiffResult) -> str:
    """One-line summary of a diff."""
    return f"+{len(diff.added)} -{len(diff.removed)} ~{len(diff.changed)} blocks"


def block_map_bbox(block_map: BlockMap) -> Optional[Bbox]:
    """Inclusive-exclusive bbox of a block map."""
    if not block_map:
        return None
    xs = [p[0] for p in block_map]
    ys = [p[1] for p in block_map]
    zs = [p[2] for p in block_map]
    return Bbox.of((min(xs), min(ys), min(zs)), (max(xs) + 1, max(ys) + 1, max(zs) + 1))
