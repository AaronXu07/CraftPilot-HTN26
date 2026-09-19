#!/usr/bin/env python
"""Facing calibration (plan.md §5.2): place one stair per facing and one slab per type near the
player, scan them back, and print the lookup table so nobody trusts memory.

Convention assumed by the engine (copilot.engine.coretypes.Facing): stairs `facing` = the direction
of the FULL (high) side; a stair placed while the player looks north has facing=north and its low
step nearest the player. Look at the placed row in-game and confirm each label.

    python scripts/calibrate_facing.py            # against the mod (or `python -m mock_mod`)
    python scripts/calibrate_facing.py --mock
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))

from copilot.bridge import HttpBridge  # noqa: E402
from copilot.engine.coretypes import FACING_NAMES, FACING_VECTORS, Facing, parse_state  # noqa: E402

BLOCK = "minecraft:stone_brick_stairs"
SLAB = "minecraft:stone_brick_slab"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("COPILOT_MOD_URL", "http://127.0.0.1:7777"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--keep", action="store_true", help="leave the calibration row in the world")
    args = ap.parse_args()
    if args.mock:
        from mock_mod import MockBridge

        b = MockBridge()
    else:
        b = HttpBridge(args.url)
    p = b.player()
    px, py, pz = (int(math.floor(v)) for v in p["pos"])
    print(f"player at ({px},{py},{pz}) facing {p['facing']}")
    # a row of blocks 3 blocks east of the player, running south
    placements = []
    labels = []
    z = pz
    for name in FACING_NAMES:
        for half in ("bottom", "top"):
            placements.append((px + 3, py, z, f"{BLOCK}[facing={name},half={half},shape=straight,waterlogged=false]"))
            labels.append(f"stairs facing={name} half={half}")
            z += 2
    for t in ("bottom", "top", "double"):
        placements.append((px + 3, py, z, f"{SLAB}[type={t},waterlogged=false]"))
        labels.append(f"slab type={t}")
        z += 2
    lo, hi = (px + 3, py, pz), (px + 3, py, z)
    before = b.scan(lo, hi)
    b.setblocks([(placements, 0)])
    time.sleep(0.5)
    after = b.scan(lo, hi)
    ok = True
    print(f"\n{'placed':<34} {'scanned back':<70} match")
    for (x, y, zz, state), label in zip(placements, labels):
        got = after.get((x, y, zz), "?")
        want_id, want_props = parse_state(state)
        got_id, got_props = parse_state(got)
        same = got_id == want_id and all(got_props.get(k) == v for k, v in want_props.items() if k != "shape")
        ok &= same
        print(f"{label:<34} {got:<70} {'ok' if same else 'MISMATCH'}")
    print("\nengine convention (Facing enum):")
    for f in Facing:
        print(f"  {f.name_mc:<6} = {f.vector}  -> stairs[facing={f.name_mc}]: full side toward {f.name_mc}, step toward {f.opposite().name_mc}")
    try:
        from copilot.engine.fit import FACING_CALIBRATION  # type: ignore

        print(f"fit.FACING_CALIBRATION = {FACING_CALIBRATION}")
    except Exception:  # noqa: BLE001
        print("fit.FACING_CALIBRATION not present (Track 3); the Facing enum above is the convention in use")
    print("\nNow LOOK at the row in-game (3 blocks east of you, running south) and confirm: the stair labelled facing=north must have its high side to the north.")
    if not args.keep:
        b.setblocks([([(x, y, zz, before.get((x, y, zz), "minecraft:air")) for (x, y, zz, _) in placements], 0)])
        print("calibration row restored")
    print("round-trip:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
