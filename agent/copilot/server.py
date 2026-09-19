"""FastAPI agent server. The Fabric mod POSTs `/cp <text>` chats here (CONTRACTS.md §7).

    python -m copilot.server --port 8000 --bridge mock|http|auto

Endpoints: POST /chat {player, text} -> {reply, brief, images}; GET /health; GET /sessions;
POST /sessions/{player}/reset; GET /sessions/{player}/scene; GET /sessions/{player}/render.png.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .bridge import BridgeError, get_bridge
from .session import Session, SessionStore
from .tools.dispatch import ToolContext, build_all, dispatch


@dataclass
class ChatResult:
    """What the pipeline returns for one chat turn (mirrors copilot.pipeline.orchestrator.ChatResult)."""

    reply: str
    images: List[Any] = field(default_factory=list)
    brief: Optional[Dict[str, Any]] = None


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _load_registry(bridge) -> Any:
    """Registry.load(bridge=..., client_jar=...) or None if Track 4's module is missing."""
    try:
        from .engine.registry import Registry
    except ImportError:
        return None
    client_jar = None
    try:
        client_jar = bridge.health().get("client_jar")
    except BridgeError:
        pass
    try:
        return Registry.load(bridge=bridge, client_jar=client_jar)
    except Exception as e:  # noqa: BLE001
        print(f"[server] registry load failed: {e}; continuing without it")
        return None


def make_logger(session: Session):
    """JSONL logger writing runs/<session>/<turn>.jsonl."""

    def log(event: Dict[str, Any]) -> None:
        try:
            os.makedirs(session.run_dir, exist_ok=True)
            with open(os.path.join(session.run_dir, f"turn_{session.turn:03d}.jsonl"), "a") as f:
                f.write(json.dumps({"t": time.time(), **event}, default=str) + "\n")
        except OSError:
            pass

    return log


_DIRECT_RE = re.compile(r"^/op\s+([a-z_]+)\s*(.*)$", re.S)


def direct_ops_chat(ctx: ToolContext, text: str) -> ChatResult:
    """Fallback when no pipeline is installed: '/op <tool> {json args}' or plain 'describe'."""
    m = _DIRECT_RE.match(text.strip())
    if m:
        name, raw = m.group(1), m.group(2).strip()
        try:
            args = json.loads(raw) if raw else {}
            if not isinstance(args, dict):
                raise ValueError("args must be a JSON object")
        except ValueError as e:
            return ChatResult(f"ERROR: bad JSON args: {e}")
        res = dispatch(ctx, name, args)
        return ChatResult(res.text, images=res.images, brief=ctx.session.brief)
    if text.strip().lower() in ("describe", "outline", "scene"):
        return ChatResult(dispatch(ctx, "describe", {}).text)
    return ChatResult("No LLM pipeline is installed. Use '/op <tool> {json}' (e.g. /op add {\"id\":\"keep\",\"shape\":{\"type\":\"box\",\"size\":[10,8,10]},\"pos\":[0,0,0]}) or 'describe'.")


def handle_chat_for(ctx: ToolContext, text: str) -> ChatResult:
    """Route to the LLM pipeline if present, else to the direct-ops fallback."""
    try:
        from .pipeline.orchestrator import handle_chat  # type: ignore
    except ImportError:
        return direct_ops_chat(ctx, text)
    res = handle_chat(ctx, text)
    return ChatResult(reply=getattr(res, "reply", str(res)), images=list(getattr(res, "images", []) or []), brief=getattr(res, "brief", None))


def create_app(bridge=None, registry=None, store: Optional[SessionStore] = None, bridge_pref: Optional[str] = None, load_registry: bool = True) -> FastAPI:
    """Build the FastAPI app. Pass a bridge/registry to skip auto-detection (tests)."""
    _load_env()
    state: Dict[str, Any] = {"bridge": bridge, "registry": registry, "store": store or SessionStore(), "started": time.time()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["bridge"] is None:
            state["bridge"] = get_bridge(bridge_pref)
        if state["registry"] is None and load_registry:
            state["registry"] = _load_registry(state["bridge"])
        yield

    app = FastAPI(title="minecraft-copilot", lifespan=lifespan)
    app.state.copilot = state

    def ctx_for(player: str) -> ToolContext:
        session = state["store"].get(player)
        return ToolContext(session=session, bridge=state["bridge"], registry=state["registry"], log=make_logger(session), run_dir=session.run_dir)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        b = state["bridge"]
        bridge_ok, bridge_info = False, None
        if b is not None:
            try:
                bridge_info = b.health()
                bridge_ok = bool(bridge_info.get("ok"))
            except BridgeError as e:
                bridge_info = {"error": str(e)}
        return {
            "ok": True,
            "bridge": type(b).__name__ if b else None,
            "bridge_ok": bridge_ok,
            "bridge_info": bridge_info,
            "registry": (len(state["registry"]) if state["registry"] is not None else None),
            "sessions": len(state["store"].all()),
            "uptime_s": round(time.time() - state["started"], 1),
            "llm": bool(os.environ.get("AZURE_OPENAI_API_KEY")),
        }

    @app.post("/chat")
    async def chat(req: Request) -> Dict[str, Any]:
        body = await req.json()
        player = str(body.get("player") or "player")
        text = str(body.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "text required")
        ctx = ctx_for(player)
        ctx.session.turn += 1
        ctx.session.add_chat("user", text)
        ctx.emit({"event": "chat", "player": player, "text": text})
        t0 = time.time()
        try:
            res = handle_chat_for(ctx, text)
        except BridgeError as e:
            res = ChatResult(f"Bridge error: {e}")
        except Exception as e:  # noqa: BLE001
            ctx.emit({"event": "error", "error": repr(e)})
            res = ChatResult(f"Sorry, that failed: {type(e).__name__}: {e}")
        ctx.session.add_chat("assistant", res.reply)
        paths: List[str] = []
        for i, img in enumerate(res.images or []):
            try:
                os.makedirs(ctx.session.run_dir, exist_ok=True)
                p = os.path.join(ctx.session.run_dir, f"turn_{ctx.session.turn:03d}_{i}.png")
                img.save(p)
                paths.append(os.path.abspath(p))
            except Exception:  # noqa: BLE001
                pass
        ctx.emit({"event": "reply", "reply": res.reply[:1000], "ms": round((time.time() - t0) * 1000)})
        return {"reply": res.reply, "brief": res.brief if res.brief is not None else ctx.session.brief, "images": paths}

    @app.get("/sessions")
    def sessions() -> Dict[str, Any]:
        return {"sessions": [{"player": s.player, "objects": len(s.scene.objects), "turn": s.turn, "placed": len(s.world.placed)} for s in state["store"].all()]}

    @app.post("/sessions/{player}/reset")
    def reset(player: str) -> Dict[str, Any]:
        state["store"].reset(player)
        return {"ok": True, "player": player}

    @app.get("/sessions/{player}/scene")
    def scene(player: str) -> Any:
        s = state["store"].get(player)
        return JSONResponse({"scene": s.scene.to_dict(), "outline": s.scene.describe(), "brief": s.brief, "history": len(s.history)})

    @app.get("/sessions/{player}/render.png")
    def render_png(player: str, view: str = "contact") -> Response:
        import io

        ctx = ctx_for(player)
        res = dispatch(ctx, "render", {"views": [view]})
        if not res.images:
            raise HTTPException(404, res.text)
        buf = io.BytesIO()
        res.images[0].save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    return app


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Minecraft copilot agent server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("COPILOT_PORT", "8000")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--bridge", choices=["mock", "http", "auto"], default=None, help="world backend (default: env COPILOT_BRIDGE or auto)")
    args = ap.parse_args(argv)
    import uvicorn

    uvicorn.run(create_app(bridge_pref=args.bridge), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
