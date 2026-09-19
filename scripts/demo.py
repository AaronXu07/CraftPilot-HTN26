#!/usr/bin/env python
"""Headless demo without an LLM: build a scripted castle with scene ops, rasterise, fit, resolve,
render a contact sheet to out/demo_contact.png and place it into the mock world (or the game with
--http). Prints stats. Exits with a clear message if Track 3/4 engine modules are missing.

    python scripts/demo.py            # mock world
    python scripts/demo.py --http     # place into the running game via the mod
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

from copilot.session import Session  # noqa: E402
from copilot.tools.dispatch import ToolContext, build_all, build_summary, dispatch  # noqa: E402


def build_castle(session: Session) -> None:
    a = session.apply
    a("define_material", name="stone_wall", spec={"base": "stone_bricks", "palette": [["stone_bricks", 0.7], ["cracked_stone_bricks", 0.2], ["mossy_stone_bricks", 0.1]], "gradient": {"axis": "y", "from": 0, "to": 4, "palette": [["cobblestone", 0.7], ["mossy_cobblestone", 0.3]]}, "fit": "stairs+slab"})
    a("define_material", name="slate_roof", spec={"base": "deepslate_tiles", "palette": [["deepslate_tiles", 0.8], ["cracked_deepslate_tiles", 0.2]], "fit": "stairs+slab"})
    a("define_material", name="timber", spec={"base": "spruce_planks", "fit": "stairs+slab"})
    # keep (L-shaped plan) and curtain walls
    a("add", id="keep", shape={"type": "extrude", "profile": [[-8, -8], [8, -8], [8, 2], [0, 2], [0, 8], [-8, 8]], "height": 18}, pos=[20, 0, 20], material="stone_wall", modifiers=[{"type": "shell", "thickness": 1}], group="keep_group")
    a("add", id="keep_roof", shape={"type": "pyramid", "base": [18, 12], "height": 7}, pos=[20, 18, 15], material="slate_roof", group="keep_group")
    a("add", id="wall_s", shape={"type": "box", "size": [40, 10, 2]}, pos=[20, 0, 40], material="stone_wall")
    a("add", id="wall_n", shape={"type": "box", "size": [40, 10, 2]}, pos=[20, 0, 0], material="stone_wall")
    a("add", id="wall_e", shape={"type": "box", "size": [2, 10, 40]}, pos=[40, 0, 20], material="stone_wall")
    a("add", id="wall_w", shape={"type": "box", "size": [2, 10, 40]}, pos=[0, 0, 20], material="stone_wall")
    # four round towers + cone roofs via a script-like loop
    for name, (x, z) in {"ne": (40, 0), "nw": (0, 0), "se": (40, 40), "sw": (0, 40)}.items():
        a("add", id=f"tower_{name}", shape={"type": "cylinder", "radius": 4.5, "height": 16}, pos=[x, 0, z], material="stone_wall", modifiers=[{"type": "shell", "thickness": 1}], tags=["tower"], group="towers")
        a("add", id=f"tower_{name}_roof", shape={"type": "cone", "radius": 5.5, "height": 8}, pos=[x, 16, z], material="slate_roof", tags=["roof"], group="towers")
    # battlements along the south wall, gate with an arch
    a("add", id="merlon_s", shape={"type": "box", "size": [1, 1, 2]}, pos=[2, 10, 40], material="stone_wall", modifiers=[{"type": "array", "count": 19, "offset": [2, 0, 0]}])
    a("add", id="gate_cut", shape={"type": "box", "size": [4, 5, 4]}, pos=[20, 0, 40], op="subtract")
    a("add", id="gate_arch", shape={"type": "cylinder", "radius": 2, "height": 4, "axis": "z"}, pos=[20, 3, 40], op="subtract")
    # windows on the keep's south face
    a("add", id="keep_window", shape={"type": "box", "size": [1, 3, 3]}, pos=[14, 6, 28], op="subtract", modifiers=[{"type": "array", "count": 3, "offset": [4, 0, 0]}, {"type": "array", "count": 2, "offset": [0, 6, 0]}])
    a("add", id="gate_lantern", shape={"type": "block", "state": "minecraft:lantern[hanging=true]"}, pos=[20, 6, 42])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="place into the game through the mod")
    ap.add_argument("--out", default=os.path.join(ROOT, "out"))
    args = ap.parse_args()
    if args.http:
        from copilot.bridge import HttpBridge

        bridge = HttpBridge()
    else:
        from mock_mod import MockBridge

        bridge = MockBridge()
    registry = None
    try:
        from copilot.engine.registry import Registry

        registry = Registry.load(bridge=bridge)
    except ImportError:
        print("engine.registry missing (Track 4): rendering with grey colours, resolver may be unavailable")
    except Exception as e:  # noqa: BLE001
        print(f"registry load failed: {e}")
    session = Session("demo")
    t0 = time.time()
    build_castle(session)
    print(session.scene.describe())
    ctx = ToolContext(session, bridge, registry, run_dir=os.path.join(args.out, "demo_run"))
    try:
        build = build_all(ctx)
    except Exception as e:  # noqa: BLE001
        print(f"\nengine build failed ({type(e).__name__}: {e}). Track 3 modules (raster/fit) are required for the demo.")
        return 1
    print("\n" + build_summary(ctx, build))
    print("timings:", build["timings"])
    os.makedirs(args.out, exist_ok=True)
    res = dispatch(ctx, "render", {"views": ["contact"]})
    if res.images:
        path = os.path.join(args.out, "demo_contact.png")
        res.images[0].save(path)
        print(f"contact sheet -> {path}")
    else:
        print(res.text)
    print(dispatch(ctx, "lint", {}).text)
    print(dispatch(ctx, "place", {"mode": "diff", "animate": not args.http}).text)
    print(dispatch(ctx, "set_shape", {"id": "tower_ne", "params": {"height": 24}}).text)
    print(dispatch(ctx, "stack", {"id": "tower_ne_roof", "on": "tower_ne"}).text)
    print(dispatch(ctx, "place", {"mode": "diff"}).text)
    print(dispatch(ctx, "materials_list", {}).text[:600])
    if not args.http:
        print(f"mock world now holds {bridge.count_non_air()} placed blocks; {len(bridge.calls)} setblocks calls")
    print(f"total {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
