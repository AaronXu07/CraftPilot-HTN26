"""Undo records for placed objects: the world positions a build wrote, so they can be set back to air.

Buildings never write one (their edits include graded terrain that cannot be undone from positions alone);
objects do, next to their other artefacts, so `craftpilot object-undo <placed.json>` removes them again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from craftpilot.place.bridge import FORCE_FLAGS, Block

AIR = "minecraft:air"


def write_record(path: Path, label: str, origin: tuple[int, int, int], blocks: list[Block]) -> Path:
    """Record the non-air positions of `blocks` (the object itself, not the cleared air around it)."""
    positions = [[x, y, z] for x, y, z, state in blocks if state != AIR]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"label": label, "origin": list(origin), "positions": positions}))
    return path


def undo(path: Path, bridge: Any, chunk_size: int = 4000) -> int:
    """Set every recorded position to air through `bridge` (HttpBridge or a fake). Returns the block count."""
    data = json.loads(Path(path).read_text())
    positions = data.get("positions", [])
    if not positions:
        return 0
    blocks: list[Block] = [(int(x), int(y), int(z), AIR) for x, y, z in positions]
    blocks.sort(key=lambda b: (-b[1], b[2], b[0]))  # top down, so nothing is left floating mid-way
    chunks = [(blocks[i:i + chunk_size], 0) for i in range(0, len(blocks), chunk_size)]
    bridge.setblocks(chunks, flags=FORCE_FLAGS, postprocess=False)
    return len(blocks)
