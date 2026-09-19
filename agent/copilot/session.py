"""Per-player session: scene document, op history with undo/redo, snapshots, brief, chat, world state.

See CONTRACTS.md §10. The session is the only mutable state in the agent; every scene change goes
through `Session.apply(op_name, **kwargs)` so undo/redo and the run log stay correct.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .engine.coretypes import BlockMap
from .engine.scene import READ_ONLY_OPS, Scene, SceneError, apply_op

MAX_CHAT_TURNS = 30
MAX_UNDO = 200


@dataclass
class HistoryEntry:
    op: str
    args: Dict[str, Any]
    message: str
    t: float = field(default_factory=time.time)
    scene_hash: str = ""


@dataclass
class WorldState:
    """What the agent has placed in the game for this session's scene."""

    anchor: Optional[Tuple[int, int, int]] = None  # world position of scene origin
    quarter_turns: int = 0  # clockwise 90° turns applied at placement (player facing snap)
    placed: BlockMap = field(default_factory=dict)  # last placed block map (world coords)
    pre_scan: Optional[BlockMap] = None  # world snapshot (incl. air) before the first placement
    pre_scan_bbox: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    placements: int = 0

    def is_placed(self) -> bool:
        return bool(self.placed)


class Session:
    def __init__(self, player: str = "player", scene: Optional[Scene] = None, run_dir: Optional[str] = None):
        self.player = player
        self.scene: Scene = scene or Scene(name="untitled")
        self.history: List[HistoryEntry] = []
        self._undo: List[Scene] = []
        self._redo: List[Scene] = []
        self.snapshots: Dict[str, Scene] = {}
        self.brief: Optional[Dict[str, Any]] = None
        self.chat: List[Dict[str, Any]] = []
        self.world = WorldState()
        self.cache: Dict[str, Any] = {}  # engine outputs keyed by scene hash (raster/fit/block_map/renders)
        self.live_preview: bool = False
        self.turn: int = 0
        self.created = time.time()
        self.run_dir = run_dir or os.path.join("runs", _safe(player) + "_" + time.strftime("%Y%m%d_%H%M%S"))
        self.stage: Optional[str] = None
        self.tool_calls_this_stage: int = 0

    # -- ops ------------------------------------------------------------------------------
    def apply(self, op_name: str, **kwargs) -> str:
        """Apply a scene op, record it, and return the one-line result message."""
        new_scene, msg = apply_op(self.scene, op_name, **kwargs)
        if op_name not in READ_ONLY_OPS:
            self._undo.append(self.scene)
            if len(self._undo) > MAX_UNDO:
                self._undo.pop(0)
            self._redo.clear()
            self.scene = new_scene
        self.history.append(HistoryEntry(op=op_name, args=_jsonable(kwargs), message=msg, scene_hash=self.scene.hash()))
        return msg

    def replace_scene(self, scene: Scene, message: str = "scene replaced") -> str:
        self._undo.append(self.scene)
        self._redo.clear()
        self.scene = scene
        self.history.append(HistoryEntry(op="replace_scene", args={}, message=message, scene_hash=scene.hash()))
        return message

    def undo(self, n: int = 1) -> str:
        n = max(1, int(n))
        done = 0
        for _ in range(n):
            if not self._undo:
                break
            self._redo.append(self.scene)
            self.scene = self._undo.pop()
            done += 1
        if done == 0:
            return "nothing to undo"
        self.history.append(HistoryEntry(op="undo", args={"n": done}, message=f"undid {done}", scene_hash=self.scene.hash()))
        return f"undid {done} op(s); scene now has {len(self.scene.objects)} objects"

    def redo(self, n: int = 1) -> str:
        n = max(1, int(n))
        done = 0
        for _ in range(n):
            if not self._redo:
                break
            self._undo.append(self.scene)
            self.scene = self._redo.pop()
            done += 1
        if done == 0:
            return "nothing to redo"
        self.history.append(HistoryEntry(op="redo", args={"n": done}, message=f"redid {done}", scene_hash=self.scene.hash()))
        return f"redid {done} op(s); scene now has {len(self.scene.objects)} objects"

    def snapshot(self, label: str) -> str:
        self.snapshots[str(label)] = self.scene.copy()
        return f"snapshot '{label}' saved ({len(self.scene.objects)} objects)"

    def restore(self, label: str) -> str:
        if label not in self.snapshots:
            raise SceneError(f"no snapshot named {label!r}; have: {', '.join(self.snapshots) or '(none)'}")
        self._undo.append(self.scene)
        self._redo.clear()
        self.scene = self.snapshots[label].copy()
        self.history.append(HistoryEntry(op="restore", args={"label": label}, message=f"restored {label}", scene_hash=self.scene.hash()))
        return f"restored snapshot '{label}' ({len(self.scene.objects)} objects)"

    def can_undo(self) -> int:
        return len(self._undo)

    def can_redo(self) -> int:
        return len(self._redo)

    # -- state ----------------------------------------------------------------------------
    def scene_hash(self) -> str:
        return self.scene.hash()

    def add_chat(self, role: str, content: Any) -> None:
        self.chat.append({"role": role, "content": content})
        if len(self.chat) > MAX_CHAT_TURNS * 2:
            self.chat = self.chat[-MAX_CHAT_TURNS * 2 :]

    def cached(self, key: str) -> Any:
        return self.cache.get((self.scene_hash(), key))

    def put_cache(self, key: str, value: Any) -> None:
        h = self.scene_hash()
        # keep the cache small: only the current scene + the previously placed one
        for k in list(self.cache):
            if k[0] != h and k[1] != "__pinned__":
                del self.cache[k]
        self.cache[(h, key)] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player": self.player,
            "scene": self.scene.to_dict(),
            "brief": self.brief,
            "history": [{"op": h.op, "args": h.args, "message": h.message, "t": h.t} for h in self.history[-100:]],
            "snapshots": list(self.snapshots),
            "world": {
                "anchor": self.world.anchor,
                "quarter_turns": self.world.quarter_turns,
                "placed_blocks": len(self.world.placed),
                "placements": self.world.placements,
            },
            "turn": self.turn,
        }

    def save(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.run_dir, "session.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path


class SessionStore:
    """One session per player."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get(self, player: str) -> Session:
        if player not in self._sessions:
            self._sessions[player] = Session(player)
        return self._sessions[player]

    def reset(self, player: str) -> Session:
        self._sessions[player] = Session(player)
        return self._sessions[player]

    def all(self) -> List[Session]:
        return list(self._sessions.values())


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:32] or "player"


def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(json.dumps(d, default=lambda o: getattr(o, "tolist", lambda: str(o))()))
    except Exception:  # noqa: BLE001
        return {k: str(v) for k, v in d.items()}
