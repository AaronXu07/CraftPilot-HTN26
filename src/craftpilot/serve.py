"""Local HTTP service the Fabric mod talks to."""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from craftpilot.config import SETTINGS
from craftpilot.program.model import Bounds, BuildProgram

app = FastAPI(title="craftpilot", version="0.1.0")

_last_program: dict[str, BuildProgram] = {}
_last_text: dict[str, str] = {}
_last_yaw: dict[str, float] = {}


class BuildRequest(BaseModel):
    text: str
    player: str = "player"
    origin: tuple[int, int, int] = (0, 0, 0)
    yaw: float = 0.0
    bounds: Bounds | None = None
    seed: int | None = None
    preview: bool = False
    use_llm: bool = True


class EditRequest(BaseModel):
    player: str = "player"
    text: str
    seed: int | None = None


class RegenerateRequest(BaseModel):
    player: str = "player"
    seed: int | None = None


class SaveExemplarRequest(BaseModel):
    player: str = "player"
    name: str
    description: str = ""
    tags: list[str] = []


@app.get("/health")
def health() -> dict:
    return {"ok": True, "llm": SETTINGS.llm_configured, "schematics_dir": str(SETTINGS.schematics_dir)}


def facing_from_yaw(yaw: float) -> str:
    """The building faces the player: opposite of where the player looks. Yaw 0 = south, 90 = west."""
    look = ["south", "west", "north", "east"][int(((yaw % 360) + 45) // 90) % 4]
    return {"south": "north", "west": "east", "north": "south", "east": "west"}[look]


def _run(player: str, program: BuildProgram, bounds: Bounds | None, seed: int | None, preview: bool,
         text: str, notes: list[str], source: str, yaw: float = 0.0) -> dict:
    from craftpilot.cli import run_build

    seed = seed if seed is not None else int(time.time()) % 1_000_000
    try:
        result = run_build(program, bounds, seed, None, preview, player, text, notes, facing_from_yaw(yaw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["source"] = source
    _last_program[player] = program
    _last_text[player] = text
    return result


@app.post("/build")
def build(req: BuildRequest) -> dict:
    from craftpilot.llm.compose import compose

    program, source, notes = compose(req.text, use_llm=req.use_llm, bounds_hint=req.bounds)
    _last_yaw[req.player] = req.yaw
    return _run(req.player, program, req.bounds, req.seed, req.preview, req.text, notes, source, req.yaw)


@app.post("/edit")
def edit(req: EditRequest) -> dict:
    from craftpilot.llm.compose import edit as llm_edit

    prev = _last_program.get(req.player)
    if prev is None:
        raise HTTPException(status_code=404, detail="No previous build for this player")
    program, source, notes = llm_edit(prev, req.text)
    return _run(req.player, program, None, req.seed, False, f"{_last_text.get(req.player, '')} / {req.text}", notes,
                source, _last_yaw.get(req.player, 0.0))


@app.post("/regenerate")
def regenerate(req: RegenerateRequest) -> dict:
    prev = _last_program.get(req.player)
    if prev is None:
        raise HTTPException(status_code=404, detail="No previous build for this player")
    return _run(req.player, prev, None, req.seed, False, _last_text.get(req.player, ""), [], "previous",
                _last_yaw.get(req.player, 0.0))


class ObjectRequest(BaseModel):
    text: str
    player: str = "player"
    height: int | None = None
    use_llm: bool = True
    preview: bool = True
    yaw: float = 0.0
    place: bool = False  # also place it in the running game through the Fabric mod bridge


@app.post("/object")
def build_object_endpoint(req: ObjectRequest) -> dict:
    """Free-form object (statue, creature, vehicle, prop): image -> mesh -> voxels; see craftpilot.objects."""
    from craftpilot.objects.imagegen import ImageRejected
    from craftpilot.objects.pipeline import build_object
    from craftpilot.objects.recon import ReconUnavailable

    try:
        r = build_object(req.text, height=req.height, use_llm=req.use_llm, preview=req.preview, yaw_deg=req.yaw)
    except ReconUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImageRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    out = r.to_dict()
    out["source"] = "object"
    if req.place:
        from craftpilot.objects import place as P

        try:
            pl = P.place(r.grid, undo_dir=r.work_dir)
            out["placed"] = {"anchor": list(pl.anchor), "blocks": pl.blocks, "chunks": pl.chunks,
                             "estimated_seconds": pl.estimated_seconds, "undo_file": str(pl.undo_file)}
        except P.ModUnavailable as exc:
            out["placed"] = {"error": str(exc)}
    return out


@app.post("/exemplars")
def save_exemplar(req: SaveExemplarRequest) -> dict:
    from craftpilot.program.exemplars import save

    prev = _last_program.get(req.player)
    if prev is None:
        raise HTTPException(status_code=404, detail="No previous build for this player")
    path = save(SETTINGS.exemplars_dir, req.name, req.description or _last_text.get(req.player, ""), req.tags, prev)
    return {"saved": str(path)}
