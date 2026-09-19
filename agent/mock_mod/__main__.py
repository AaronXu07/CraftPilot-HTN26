"""Serve the MockBridge over HTTP with the mod's 7 endpoints: `python -m mock_mod --port 7777`."""
from __future__ import annotations

import argparse

from mock_mod.server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="Fake Fabric-mod HTTP server (in-memory world)")
    ap.add_argument("--port", type=int, default=7777)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ground-y", type=int, default=63)
    args = ap.parse_args()
    import uvicorn

    uvicorn.run(create_app(ground_y=args.ground_y), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
