"""Local HTTP service the Fabric mod talks to.

The mod's ``/build`` chat command posts here; with ``place: true`` the service asks the mod where the
player stands, renders the building facing them, and streams the blocks back through the mod's bridge.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from craftpilot.config import SETTINGS
from craftpilot.place.anchor import facing_from_yaw
from craftpilot.program.model import Bounds, BuildProgram

__all__ = ["app", "facing_from_yaw"]

app = FastAPI(title="craftpilot", version="0.1.0")

_last_program: dict[str, BuildProgram] = {}
_last_text: dict[str, str] = {}
_last_yaw: dict[str, float] = {}


class PlaceFields(BaseModel):
    place: bool = False
    clear: bool = True
    gap: int | None = None
    sink: int | None = None


class BuildRequest(PlaceFields):
    text: str
    player: str = "player"
    yaw: float = 0.0
    bounds: Bounds | None = None
    seed: int | None = None
    preview: bool = False
    use_llm: bool | None = None  # None: use the LLM when it is configured


class EditRequest(PlaceFields):
    player: str = "player"
    text: str
    seed: int | None = None


class RegenerateRequest(PlaceFields):
    player: str = "player"
    seed: int | None = None


class SaveExemplarRequest(BaseModel):
    player: str = "player"
    name: str
    description: str = ""
    tags: list[str] = []


@app.get("/health")
def health() -> dict:
    from craftpilot.place.bridge import BridgeError, HttpBridge

    try:
        mod = HttpBridge().health()
    except BridgeError:
        mod = None
    return {"ok": True, "llm": SETTINGS.llm_configured, "schematics_dir": str(SETTINGS.schematics_dir),
            "mod_url": SETTINGS.mod_url, "mod": mod}


def _place_context(req: PlaceFields):
    from craftpilot.place.bridge import BridgeError, HttpBridge
    from craftpilot.place.placer import PlaceContext, PlaceOptions

    bridge = HttpBridge()
    try:
        player = bridge.player()
    except BridgeError as exc:
        raise HTTPException(status_code=503, detail=f"mod not reachable: {exc}") from exc
    opts = PlaceOptions(clear=req.clear)
    if req.gap is not None:
        opts.gap = req.gap
    if req.sink is not None:
        opts.sink = req.sink
    return PlaceContext(bridge=bridge, player=player, opts=opts)


def _summary(result: dict, label: str) -> str:
    w, h, d = result["bounds"]
    text = f"{label} #{result['seed']}: {result['blocks']} blocks, {w}x{h}x{d}, facing {result['facing']}"
    placement = result.get("placement")
    if placement:
        x, y, z = placement["origin"]
        text += f" - placing at ({x}, {y}, {z}) in {placement['chunks']} layers, ~{placement['estimated_seconds']}s"
    return text


def _run(player: str, program: BuildProgram, bounds: Bounds | None, seed: int | None, preview: bool,
         text: str, notes: list[str], source: str, yaw: float, req: PlaceFields) -> dict:
    from craftpilot.cli import run_build

    seed = seed if seed is not None else int(time.time()) % 1_000_000
    place_ctx = _place_context(req) if req.place else None
    if place_ctx is not None:
        yaw = float(place_ctx.player.get("yaw", yaw))
    try:
        result = run_build(program, bounds, seed, None, preview, player, text, notes, facing_from_yaw(yaw),
                           place_ctx)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["source"] = source
    result["summary"] = _summary(result, program.label)
    _last_program[player] = program
    _last_text[player] = text
    _last_yaw[player] = yaw
    return result


@app.post("/build")
def build(req: BuildRequest) -> dict:
    from craftpilot.llm.compose import compose

    use_llm = SETTINGS.llm_configured if req.use_llm is None else req.use_llm
    program, source, notes = compose(req.text, use_llm=use_llm, bounds_hint=req.bounds)
    return _run(req.player, program, req.bounds, req.seed, req.preview, req.text, notes, source, req.yaw, req)


@app.post("/edit")
def edit(req: EditRequest) -> dict:
    from craftpilot.llm.compose import edit as llm_edit

    prev = _last_program.get(req.player)
    if prev is None:
        raise HTTPException(status_code=404, detail="No previous build for this player")
    program, source, notes = llm_edit(prev, req.text)
    return _run(req.player, program, None, req.seed, False, f"{_last_text.get(req.player, '')} / {req.text}", notes,
                source, _last_yaw.get(req.player, 0.0), req)


@app.post("/regenerate")
def regenerate(req: RegenerateRequest) -> dict:
    prev = _last_program.get(req.player)
    if prev is None:
        raise HTTPException(status_code=404, detail="No previous build for this player")
    return _run(req.player, prev, None, req.seed, False, _last_text.get(req.player, ""), [], "previous",
                _last_yaw.get(req.player, 0.0), req)


@app.post("/cancel")
def cancel() -> dict:
    from craftpilot.place.bridge import BridgeError, HttpBridge

    try:
        return HttpBridge().cancel()
    except BridgeError as exc:
        raise HTTPException(status_code=503, detail=f"mod not reachable: {exc}") from exc


@app.post("/exemplars")
def save_exemplar(req: SaveExemplarRequest) -> dict:
    from craftpilot.program.exemplars import save

    prev = _last_program.get(req.player)
    if prev is None:
        raise HTTPException(status_code=404, detail="No previous build for this player")
    path = save(SETTINGS.exemplars_dir, req.name, req.description or _last_text.get(req.player, ""), req.tags, prev)
    return {"saved": str(path)}
