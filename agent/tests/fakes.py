"""Test doubles for the pipeline: a fake tool dispatcher, registry, bridge and context.

The fake dispatch applies real scene ops through `Session.apply` and simulates render/lint/place so
the pipeline can be exercised end-to-end without Track 2a/3/4 modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from copilot.engine.scene import OPS
from copilot.session import Session


@dataclass
class FakeToolResult:
    text: str
    images: List[Any] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True


class FakeRegistry:
    def catalog_text(self) -> str:
        return "## Materials catalog (fake)\nstone_bricks, cracked_stone_bricks, cobblestone, dark_oak_planks, deepslate_tiles, lantern"

    def search(self, query: str, limit: int = 10) -> List[str]:
        return [f"minecraft:{query}_block"]


class FakeBridge:
    def __init__(self):
        self.world: Dict[tuple, str] = {}
        self.said: List[str] = []
        self.setblock_calls: List[List[tuple]] = []
        self.camera_calls: List[Dict[str, Any]] = []

    def health(self):
        return {"ok": True, "mod_version": "fake", "mc_version": "1.21.1", "world_loaded": True}

    def player(self):
        return {"name": "tester", "pos": [0.5, 64.0, 0.5], "yaw": 180.0, "pitch": 0.0, "facing": "north", "looking_at": None}

    def scan(self, lo, hi):
        return {k: v for k, v in self.world.items() if all(lo[i] <= k[i] <= hi[i] for i in range(3))}

    def setblocks(self, chunks, flags=3):
        n = 0
        for blocks, _delay in chunks:
            self.setblock_calls.append(list(blocks))
            for x, y, z, state in blocks:
                if state == "minecraft:air":
                    self.world.pop((x, y, z), None)
                else:
                    self.world[(x, y, z)] = state
                n += 1
        return n

    def say(self, text: str):
        self.said.append(text)

    def blocks(self):
        return [{"id": "minecraft:stone", "properties": {}, "default": "minecraft:stone"}]

    def camera(self, **kw):
        self.camera_calls.append(kw)
        return {"ok": True}


def fake_block_map(scene) -> Dict[tuple, str]:
    """Fill each visible `add` object's bbox with stone (a stand-in for the real rasteriser)."""
    out: Dict[tuple, str] = {}
    for o in scene.objects:
        if o.op != "add" or not o.visible:
            continue
        bb = scene.object_bbox(o)
        lo = np.floor(bb.lo).astype(int)
        hi = np.ceil(bb.hi).astype(int)
        hi = np.minimum(hi, lo + 40)
        state = "minecraft:stone" if o.shape["type"] != "block" else o.shape["state"]
        for x in range(lo[0], hi[0]):
            for y in range(lo[1], hi[1]):
                for z in range(lo[2], hi[2]):
                    out[(x, y, z)] = state
    for o in scene.objects:
        if o.op == "subtract":
            bb = scene.object_bbox(o)
            lo = np.floor(bb.lo).astype(int)
            hi = np.ceil(bb.hi).astype(int)
            for x in range(lo[0], hi[0]):
                for y in range(lo[1], hi[1]):
                    for z in range(lo[2], hi[2]):
                        out.pop((x, y, z), None)
    return out


class _ScriptScene:
    """`scene` object handed to run_script: every op goes through session.apply (so undo works)."""

    def __init__(self, session: Session):
        self._s = session

    def __getattr__(self, name):
        if name in OPS:
            return lambda *a, **kw: self._s.apply(name, **kw)
        raise AttributeError(name)


def make_dispatch(bridge: Optional[FakeBridge] = None) -> Callable[[Any, str, Dict[str, Any]], FakeToolResult]:
    def dispatch(ctx, name: str, args: Dict[str, Any]) -> FakeToolResult:
        session: Session = ctx.session
        br: FakeBridge = bridge or ctx.bridge
        if name in OPS:
            return FakeToolResult(text=session.apply(name, **args))
        if name == "undo":
            return FakeToolResult(text=session.undo(int(args.get("n", 1))))
        if name == "redo":
            return FakeToolResult(text=session.redo(int(args.get("n", 1))))
        if name == "snapshot":
            return FakeToolResult(text=session.snapshot(args.get("label", "snap")))
        if name == "restore":
            return FakeToolResult(text=session.restore(args.get("label", "snap")))
        if name == "render":
            views = list(args.get("views") or ["contact"])
            imgs = [Image.new("RGB", (64, 64), (120, 120, 120)) for _ in views]
            text = f"rendered {','.join(views)}: {len(session.scene.objects)} objects"
            if args.get("slice_y") is not None:
                text += f"\nplan at y={int(args['slice_y'])}:\n####\n#..#\n####"
            return FakeToolResult(text=text, images=imgs)
        if name == "lint":
            return FakeToolResult(text="lint: no findings")
        if name == "place":
            new = fake_block_map(session.scene)
            old = session.world.placed
            blocks = []
            if args.get("mode", "diff") == "full":
                blocks = [(x, y, z, s) for (x, y, z), s in new.items()]
            else:
                for k, s in new.items():
                    if old.get(k) != s:
                        blocks.append((k[0], k[1], k[2], s))
                for k in old:
                    if k not in new:
                        blocks.append((k[0], k[1], k[2], "minecraft:air"))
            if session.world.pre_scan is None:
                session.world.pre_scan = dict(br.world)
            br.setblocks([(blocks, 0)])
            session.world.placed = dict(new)
            session.world.placements += 1
            session.world.anchor = (0, 64, 0)
            added = sum(1 for b in blocks if b[3] != "minecraft:air")
            removed = len(blocks) - added
            return FakeToolResult(text=f"placed {len(new)} blocks ({added} changed, {removed} removed)", data={"blocks": len(new), "added": added, "removed": removed})
        if name == "undo_world":
            for k in list(session.world.placed):
                br.world.pop(k, None)
            br.world.update(session.world.pre_scan or {})
            session.world.placed = {}
            return FakeToolResult(text="world restored")
        if name == "say":
            br.say(str(args.get("text", "")))
            return FakeToolResult(text="said")
        if name == "set_brief":
            session.brief = args.get("brief", args)
            return FakeToolResult(text="brief set")
        if name == "export_schematic":
            return FakeToolResult(text=f"exported {args.get('name', 'build')}.litematic ({len(session.world.placed)} blocks)", data={"path": f"/tmp/{args.get('name', 'build')}.litematic"})
        if name == "materials_list":
            counts: Dict[str, int] = {}
            for s in fake_block_map(session.scene).values():
                counts[s] = counts.get(s, 0) + 1
            return FakeToolResult(text="\n".join(f"{k}: {v}" for k, v in counts.items()) or "(empty)")
        if name == "search_blocks":
            return FakeToolResult(text="stone_bricks, cracked_stone_bricks, mossy_stone_bricks")
        if name == "nearest_block":
            return FakeToolResult(text="minecraft:stone")
        if name == "get_player":
            p = br.player()
            return FakeToolResult(text=f"player {p['name']} at {p['pos']} facing {p['facing']}")
        if name == "run_script":
            g = {"scene": _ScriptScene(session)}
            exec(args.get("python", ""), g)  # noqa: S102 (test double)
            return FakeToolResult(text="script ran")
        if name == "finish":
            return FakeToolResult(text="ok")
        raise ValueError(f"unknown tool {name}")

    return dispatch


class FakeCtx:
    """Duck-typed ToolContext with a dispatch override."""

    def __init__(self, session: Optional[Session] = None, bridge: Optional[FakeBridge] = None, registry: Any = None, llm: Any = None, run_dir: Optional[str] = None):
        self.session = session or Session("tester", run_dir=run_dir)
        if run_dir:
            self.session.run_dir = run_dir
        self.bridge = bridge or FakeBridge()
        self.registry = registry or FakeRegistry()
        self.dispatch = make_dispatch(self.bridge)
        self.log = None
        self.llm = llm
        self.fast = None
        self.tools = None
        self.run_dir = run_dir
