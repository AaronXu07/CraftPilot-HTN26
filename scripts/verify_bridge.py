#!/usr/bin/env python3
"""Smoke test the Fabric bridge with the game open: health, player, and a round trip of baked block states.

Places a five-block probe two blocks in front of the player with the same flags the placer uses
(FORCE_STATE, no post-process), scans it back, checks every state string survived unchanged, then
restores the cells to air. Exits non-zero on any mismatch or rejected state.

    uv run python scripts/verify_bridge.py [--url http://127.0.0.1:7777] [--keep]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from craftpilot.place.anchor import look_from_yaw
from craftpilot.place.bridge import FORCE_FLAGS, BridgeError, HttpBridge
from craftpilot.place.placer import layer_chunks

PROBE = [
    # (dx, dy, dz, state): a stone base, a stair with a baked corner shape, a door pair with hinge, a hanging lantern.
    (0, 0, 0, "minecraft:stone"),
    (1, 0, 0, "minecraft:oak_stairs[facing=north,half=bottom,shape=inner_left,waterlogged=false]"),
    (2, 0, 0, "minecraft:oak_door[facing=north,half=lower,hinge=left,open=false,powered=false]"),
    (2, 1, 0, "minecraft:oak_door[facing=north,half=upper,hinge=left,open=false,powered=false]"),
    (0, 2, 0, "minecraft:stone"),
    (0, 1, 0, "minecraft:lantern[hanging=true,waterlogged=false]"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("CRAFTPILOT_MOD_URL", "http://127.0.0.1:7777"))
    ap.add_argument("--keep", action="store_true", help="leave the probe in the world")
    args = ap.parse_args()
    bridge = HttpBridge(args.url)
    ok = True

    try:
        health = bridge.health()
    except BridgeError as exc:
        print(f"FAIL health: {exc}")
        return 1
    print(f"health: mc {health.get('mc_version')} mod {health.get('mod_version')} world_loaded={health.get('world_loaded')}")
    catalog_version = os.environ.get("CRAFTPILOT_MC_VERSION", "26.2")
    if str(health.get("mc_version")) != catalog_version:
        print(f"WARN game is {health.get('mc_version')} but the block catalog is {catalog_version} "
              "(set CRAFTPILOT_MC_VERSION / run scripts/gen_catalog.py)")
    if not health.get("world_loaded"):
        print("FAIL no world loaded")
        return 1

    player = bridge.player()
    px, py, pz = (int(v // 1) for v in player["pos"])
    print(f"player: {player['name']} at ({px}, {py}, {pz}) looking {look_from_yaw(player['yaw'])}")

    # Probe sits 3 blocks east of the player so it never overlaps them.
    base = (px + 3, py, pz)
    blocks = [(base[0] + dx, base[1] + dy, base[2] + dz, s) for dx, dy, dz, s in PROBE]
    chunks = layer_chunks(blocks, max_blocks=4)
    resp = bridge.setblocks([(c, 0) for c in chunks], flags=FORCE_FLAGS, postprocess=False)
    print(f"setblocks: queued {resp.get('queued')} in {resp.get('chunks')} chunks, invalid {resp.get('invalid')}")
    if resp.get("invalid"):
        print(f"FAIL rejected states: {resp.get('invalid_samples')}")
        ok = False
    bridge.wait_idle(timeout_s=10)

    lo = (min(b[0] for b in blocks), min(b[1] for b in blocks), min(b[2] for b in blocks))
    hi = (max(b[0] for b in blocks), max(b[1] for b in blocks), max(b[2] for b in blocks))
    world = bridge.scan(lo, hi)
    for x, y, z, want in blocks:
        got = world.get((x, y, z))
        status = "ok  " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"{status} ({x},{y},{z}) want {want}\n                    got  {got}")

    if not args.keep:
        bridge.setblocks([([(x, y, z, "minecraft:air") for x, y, z, _ in reversed(blocks)], 0)],
                         flags=FORCE_FLAGS, postprocess=False)
        bridge.wait_idle(timeout_s=10)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
