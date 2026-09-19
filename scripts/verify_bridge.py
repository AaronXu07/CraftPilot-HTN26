#!/usr/bin/env python
"""Hit all 7 mod endpoints and print a pass/fail table.

    python scripts/verify_bridge.py                # http://127.0.0.1:7777 (the Fabric mod or `python -m mock_mod`)
    python scripts/verify_bridge.py --url http://127.0.0.1:7777
    python scripts/verify_bridge.py --mock         # in-process MockBridge
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))

from copilot.bridge import BridgeError, HttpBridge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("COPILOT_MOD_URL", "http://127.0.0.1:7777"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-place", action="store_true", help="skip the setblocks test (does not modify the world)")
    args = ap.parse_args()
    if args.mock:
        from mock_mod import MockBridge

        b = MockBridge()
    else:
        b = HttpBridge(args.url)
    rows = []

    def check(name, fn):
        t0 = time.time()
        try:
            out = fn()
            rows.append((name, "PASS", f"{(time.time() - t0) * 1000:.0f} ms", str(out)[:70]))
            return out
        except (BridgeError, AssertionError, Exception) as e:  # noqa: BLE001
            rows.append((name, "FAIL", f"{(time.time() - t0) * 1000:.0f} ms", f"{type(e).__name__}: {e}"[:70]))
            return None

    h = check("GET /health", lambda: b.health())
    p = check("GET /player", lambda: b.player())
    check("GET /blocks", lambda: f"{len(b.blocks())} blocks")
    if p:
        px, py, pz = (int(math.floor(v)) for v in p["pos"])
        check("POST /scan", lambda: f"{len(b.scan((px - 2, py - 1, pz - 2), (px + 2, py + 1, pz + 2)))} positions")
        if not args.no_place:
            target = (px + 3, py, pz + 3)
            before = b.scan(target, target)[target]

            def place_and_verify():
                b.setblocks([([(target[0], target[1], target[2], "minecraft:glowstone")], 0)])
                time.sleep(0.3)
                after = b.scan(target, target)[target]
                assert after.startswith("minecraft:glowstone"), f"scan returned {after}"
                b.setblocks([([(target[0], target[1], target[2], before)], 0)])
                return f"placed+restored at {target}"

            check("POST /setblocks", place_and_verify)
    check("POST /say", lambda: b.say("[copilot] bridge verified"))
    check("POST /camera", lambda: b.camera(mode="return"))
    w = max(len(r[0]) for r in rows)
    print(f"{'endpoint':<{w}}  result  time     detail")
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]:<6}  {r[2]:<8} {r[3]}")
    fails = sum(1 for r in rows if r[1] == "FAIL")
    print(f"\n{len(rows) - fails}/{len(rows)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
