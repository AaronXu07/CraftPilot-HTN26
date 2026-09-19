"""FastAPI app exposing a MockBridge with the same JSON as the real mod (CONTRACTS.md §7)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from copilot.bridge import encode_scan
from mock_mod.bridge import MockBridge


def create_app(bridge: Optional[MockBridge] = None, ground_y: int = 63) -> FastAPI:
    """Build the app. Pass a MockBridge to share state with tests."""
    b = bridge or MockBridge(ground_y=ground_y)
    app = FastAPI(title="mock_mod")
    app.state.bridge = b

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return b.health()

    @app.get("/player")
    def player() -> Dict[str, Any]:
        return b.player()

    @app.post("/scan")
    async def scan(req: Request) -> Any:
        body = await req.json()
        lo, hi = body.get("min"), body.get("max")
        if not lo or not hi:
            return JSONResponse({"error": "min and max required"}, status_code=400)
        return encode_scan(b.scan(lo, hi))

    @app.post("/setblocks")
    async def setblocks(req: Request) -> Any:
        body = await req.json()
        chunks = body.get("chunks")
        if chunks is None:
            chunks = [{"blocks": body.get("blocks", []), "delay_ms": 0}]
        parsed = [([tuple(bl) for bl in c.get("blocks", [])], int(c.get("delay_ms", 0))) for c in chunks]
        n = b.setblocks(parsed, flags=int(body.get("flags", 3)))
        return {"queued": n, "chunks": len(parsed)}

    @app.post("/say")
    async def say(req: Request) -> Dict[str, Any]:
        body = await req.json()
        b.say(str(body.get("text", "")))
        return {"ok": True}

    @app.get("/blocks")
    def blocks() -> Dict[str, Any]:
        return {"blocks": b.blocks()}

    @app.post("/camera")
    async def camera(req: Request) -> Dict[str, Any]:
        body = await req.json()
        return b.camera(**body)

    return app
