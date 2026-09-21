"""Local HTTP service the Fabric mod talks to.

The mod's ``/build`` chat command posts here. Every request is routed first (craftpilot.route): generic
architecture goes to the procedural building generator, statues / creatures / vehicles / named landmarks go
to the image->3D objects path. Both answer in the same shapes, so the mod's preview -> G -> build flow is
identical: with ``ghost: true`` the service returns a hologram cloud, and the commit (``/regenerate
{place: true}``) asks the mod where the player locked the box, rotates the build toward them and streams
the blocks back through the mod's bridge, seated on the terrain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from craftpilot.config import SETTINGS
from craftpilot.place.anchor import (
    Area,
    area_footprint,
    area_origin,
    facing_from_area,
    facing_from_yaw,
    normalise_area,
)
from craftpilot.program.model import Bounds, BuildProgram
from craftpilot.route import Route

if TYPE_CHECKING:
    from craftpilot.objects.brief import ObjectBrief
    from craftpilot.objects.pipeline import ObjectResult

__all__ = ["app", "facing_from_yaw"]

app = FastAPI(title="craftpilot", version="0.2.0")

Kind = Literal["building", "object"]
KindArg = Literal["auto", "building", "object"]


@dataclass
class Pending:
    """What a player last asked for: the thing /edit, /regenerate (go, again, the G commit) and /exemplars act on."""

    kind: Kind
    text: str
    yaw: float = 0.0
    route: dict = field(default_factory=dict)
    program: BuildProgram | None = None       # building: the composed program
    brief: ObjectBrief | None = None          # object: the plan (editable without drawing)
    result: ObjectResult | None = None        # object: the cached draw, grid facing south; None = not drawn yet
    stale: bool = False                       # object: the brief changed after the cached draw
    area: Area | None = None                  # building: the base area marked with two points, if any
    facing: str | None = None                 # building: the front chosen for that area (kept so the build is the preview)


_pending: dict[str, Pending] = {}


def _remember(player: str, **fields: Any) -> Pending:
    prev = _pending.get(player)
    if prev is not None and "yaw" not in fields:
        fields["yaw"] = prev.yaw
    p = Pending(**fields)
    _pending[player] = p
    return p


def _pending_or_404(player: str) -> Pending:
    p = _pending.get(player)
    if p is None:
        raise HTTPException(status_code=404, detail="No previous build for this player")
    return p


class PlaceFields(BaseModel):
    place: bool = False
    clear: bool = True
    gap: int | None = None
    sink: int | None = None
    # Where the player stood when the command was typed. When given, the build is anchored there (where
    # the mod drew its placeholder outline) instead of wherever the player is once composing finishes.
    pos: list[float] | None = None
    yaw: float | None = None
    # Return the generated build as a voxel cloud for the mod's hologram instead of placing it.
    ghost: bool = False
    # Two marked blocks [[x, y, z], [x, y, z]]: the build's base area. Its width and depth come from the
    # area, the height from the model, and it lands on the marked blocks facing the side the player is on.
    area: list[list[int]] | None = None


class BuildRequest(PlaceFields):
    text: str
    player: str = "player"
    bounds: Bounds | None = None
    seed: int | None = None
    preview: bool = False
    use_llm: bool | None = None  # None: use the LLM when it is configured
    kind: KindArg = "auto"       # auto: the router decides; the mod's /build object|building force one
    height: int | None = None    # objects: largest dimension in blocks


class PlanRequest(BaseModel):
    text: str
    player: str = "player"
    bounds: Bounds | None = None
    use_llm: bool | None = None
    kind: KindArg = "auto"
    height: int | None = None
    area: list[list[int]] | None = None
    pos: list[float] | None = None
    yaw: float | None = None


class EditRequest(PlaceFields):
    player: str = "player"
    text: str
    seed: int | None = None
    plan: bool = False  # update the pending plan and describe it; do not build


class RegenerateRequest(PlaceFields):
    player: str = "player"
    seed: int | None = None


def _yaw(req: PlaceFields, fallback: float) -> float:
    return fallback if req.yaw is None else req.yaw


def _area_of(points: list[list[int]] | None) -> Area | None:
    if points is None:
        return None
    try:
        return normalise_area(points)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _gen_bounds(program: BuildProgram, area: Area, facing: str) -> Bounds:
    """The box to generate in for a marked area: its footprint (in the engine's south-facing frame) and
    the height the program asked for."""
    w, d = area_footprint(area, facing)
    return Bounds(width=w, height=program.bounds.height, depth=d)


class SaveExemplarRequest(BaseModel):
    player: str = "player"
    name: str
    description: str = ""
    tags: list[str] = []


@app.get("/health")
def health() -> dict:
    from craftpilot.objects import flow
    from craftpilot.place.bridge import BridgeError, HttpBridge

    try:
        mod = HttpBridge().health()
    except BridgeError:
        mod = None
    return {"ok": True, "llm": SETTINGS.llm_configured, "schematics_dir": str(SETTINGS.schematics_dir),
            "mod_url": SETTINGS.mod_url, "mod": mod, "objects": flow.unavailable_reason() or "ok"}


def _place_context(req: PlaceFields, area: Area | None = None):
    from craftpilot.place.bridge import BridgeError, HttpBridge
    from craftpilot.place.placer import PlaceContext, PlaceOptions

    bridge = HttpBridge()
    try:
        player = bridge.player()
    except BridgeError as exc:
        raise HTTPException(status_code=503, detail=f"mod not reachable: {exc}") from exc
    if req.pos is not None and len(req.pos) == 3:
        player["pos"] = [float(v) for v in req.pos]
    if req.yaw is not None:
        player["yaw"] = float(req.yaw)
    opts = PlaceOptions(clear=req.clear)
    if req.gap is not None:
        opts.gap = req.gap
    if req.sink is not None:
        opts.sink = req.sink
    return PlaceContext(bridge=bridge, player=player, opts=opts, area=area)


def _summary(result: dict, label: str) -> str:
    w, h, d = result["bounds"]
    text = f"{label} #{result['seed']}: {result['blocks']} blocks, {w}x{h}x{d}, facing {result['facing']}"
    placement = result.get("placement")
    if placement:
        x, y, z = placement["origin"]
        text += f" - placing at ({x}, {y}, {z}) in {placement['chunks']} layers, ~{placement['estimated_seconds']}s"
    return text


# ------------------------------------------------------------------------------------------------ routing

FALLBACK_WARNING = "WARNING: the model was not used; this is the offline exemplar fallback (check AZURE_OPENAI_* in .env)"


def _route(text: str, kind: str, use_llm: bool | None, bounds: Bounds | None) -> tuple[Route, str]:
    """Decide the path for a request. Returns (route, text) - a leading 'object:' / 'building:' in the text is
    an override too (for clients without the mod's literals). An auto route to objects on a machine that cannot
    run them is downgraded to a building with a note; an explicit object request answers 503 instead."""
    from craftpilot.objects import flow
    from craftpilot.route import classify

    override = None if kind == "auto" else kind
    lowered = text.lower().lstrip()
    for prefix, forced in (("object:", "object"), ("building:", "building")):
        if lowered.startswith(prefix):
            override = forced
            text = text.lstrip()[len(prefix):].strip()
            break
    route = classify(text, override=override, use_llm=use_llm, has_bounds=bounds is not None)
    if route.kind == "object":
        reason = flow.unavailable_reason()
        if reason and route.source == "override":
            raise HTTPException(status_code=503, detail=f"objects path unavailable: {reason}")
        if reason:
            from dataclasses import replace

            route = replace(route, kind="building", source="fallback",
                            reason=f"{route.reason}; objects path unavailable ({reason}), built as a building")
    return route, text


def _compose(text: str, use_llm: bool | None, bounds: Bounds | None,
             footprint: tuple[int, int] | None = None) -> tuple[BuildProgram, str, list[str]]:
    """compose() with the request's use_llm (None = try the model; compose says why when it cannot) and a loud
    first note when the offline fallback answered, so the player sees it in chat instead of a silent lookalike."""
    from craftpilot.llm.compose import compose

    program, source, notes = compose(text, use_llm=True if use_llm is None else use_llm, bounds_hint=bounds,
                                     footprint_hint=footprint)
    if source == "fallback" and use_llm is not False:
        notes = [FALLBACK_WARNING] + notes
    return program, source, notes


def _with_route(payload: dict, route: Route) -> dict:
    payload["route"] = route.to_dict()
    payload["notes"] = [route.note] + list(payload.get("notes", []))
    return payload


# ------------------------------------------------------------------------------------------------ buildings

def _run(player: str, program: BuildProgram, bounds: Bounds | None, seed: int | None, preview: bool,
         text: str, notes: list[str], source: str, yaw: float, req: PlaceFields, route: dict | None = None,
         area: Area | None = None, facing: str | None = None) -> dict:
    from craftpilot.cli import run_build

    seed = seed if seed is not None else int(time.time()) % 1_000_000
    place_ctx = _place_context(req, area) if req.place else None
    if place_ctx is not None:
        yaw = float(place_ctx.player.get("yaw", yaw))
    if area is not None:
        # The front was fixed when the area was previewed (or is fixed now, from where the player stands),
        # and the box is the area's footprint at the program's height.
        facing = facing or facing_from_area(area, req.pos or (place_ctx.player.get("pos") if place_ctx else None), yaw)
        bounds = _gen_bounds(program, area, facing)
        program.bounds = bounds
    else:
        facing = facing_from_yaw(yaw)
    try:
        result = run_build(program, bounds, seed, None, preview, player, text, notes, facing, place_ctx)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["kind"] = "building"
    result["source"] = source
    result["summary"] = _summary(result, program.label)
    _remember(player, kind="building", text=text, yaw=yaw, program=program, route=route or {}, area=area, facing=facing)
    return result


def _ghost_result(player: str, program: BuildProgram, text: str, seed: int | None, source: str,
                  notes: list[str], bounds: Bounds | None, route: dict | None = None,
                  area: Area | None = None, facing: str | None = None) -> dict:
    """Generate (facing south, unrotated) and return the blocks as a flat [x, y, z, rgb, ...] cloud.

    Nothing is placed. The mod shows the cloud, rotates it with the player, and commits with
    /regenerate {seed}: generation is deterministic, so the build is the ghost, block for block.
    With a marked ``area`` the cloud comes with an ``anchor`` (world origin and quarter turns) and the
    mod shows it fixed on the area instead of following the player."""
    from craftpilot.engine.pipeline import generate
    from craftpilot.grid.ops import quarter_turns_for_facing
    from craftpilot.preview.ghost import ghost_cloud
    from craftpilot.program.describe import describe
    from craftpilot.program.validate import clamp_bounds, repair

    program, r_notes = repair(program, SETTINGS.safety_limit)
    if area is not None:
        facing = facing or "south"
        bounds = _gen_bounds(program, area, facing)
    b, err = clamp_bounds(bounds or program.bounds, SETTINGS.safety_limit)
    if err:
        raise HTTPException(status_code=400, detail=err)
    if area is not None:
        program.bounds = b   # the commit regenerates from the program, so it must carry the same box
    seed = seed if seed is not None else int(time.time()) % 1_000_000
    t0 = time.time()
    grid = generate(program, b, seed)
    flat, height, n = ghost_cloud(grid)
    _remember(player, kind="building", text=text, program=program, route=route or {}, area=area, facing=facing)
    lines = describe(program, b)
    ghost: dict[str, Any] = {"width": grid.W, "height": height, "depth": grid.D, "blocks": flat, "area_used": area is not None}
    if area is not None:
        turns = quarter_turns_for_facing(facing)
        ww, dd = (grid.D, grid.W) if turns % 2 else (grid.W, grid.D)
        ox, oy, oz = area_origin(area, ww, dd, sink=SETTINGS.place_sink)
        ghost["anchor"] = {"origin": [ox, oy, oz], "turns": turns, "facing": facing,
                           "area": [list(area[:3]), list(area[3:])]}
    return {
        "kind": "building",
        "ghost": ghost,
        "seed": seed,
        "bounds": {"width": b.width, "height": b.height, "depth": b.depth},
        "blocks": n,
        "generate_seconds": round(time.time() - t0, 3),
        "plan": lines,
        "source": source,
        "notes": notes + r_notes,
        "summary": f"{program.label} #{seed}: {n} blocks, {b.width}x{b.height}x{b.depth} - preview ready",
    }


def _plan_result(player: str, program: BuildProgram, text: str, source: str, notes: list[str],
                 bounds: Bounds | None, route: dict | None = None, area: Area | None = None,
                 facing: str | None = None) -> dict:
    """Store the program as the player's pending plan and describe it; nothing is generated or placed."""
    from craftpilot.program.describe import describe
    from craftpilot.program.validate import clamp_bounds, repair

    program, r_notes = repair(program, SETTINGS.safety_limit)
    if area is not None:
        facing = facing or "south"
        bounds = _gen_bounds(program, area, facing)
    b, err = clamp_bounds(bounds or program.bounds, SETTINGS.safety_limit)
    if err:
        raise HTTPException(status_code=400, detail=err)
    program.bounds = b
    _remember(player, kind="building", text=text, program=program, route=route or {}, area=area, facing=facing)
    lines = describe(program, b)
    return {
        "kind": "building",
        "plan": lines,
        "bounds": {"width": b.width, "height": b.height, "depth": b.depth},
        "source": source,
        "notes": notes + r_notes,
        "summary": f"Plan: {lines[0]}",
    }


# ------------------------------------------------------------------------------------------------ objects

def _say(req: PlaceFields | None):
    """A best-effort chat line printer for the 15-35 s an object takes; silent (and cheap) without the mod."""
    if req is None or not (req.ghost or req.place):
        return None
    from craftpilot.place.bridge import BridgeError, HttpBridge

    bridge = HttpBridge()
    state = {"ok": True}

    def say(line: str) -> None:
        if not state["ok"]:
            return
        try:
            bridge.say(line)
        except BridgeError:
            state["ok"] = False  # one failed line, not one timeout per stage

    return say


def _use_llm(req_use_llm: bool | None) -> bool:
    return True if req_use_llm is None else req_use_llm


def _draw_or_http(text: str, brief: ObjectBrief | None, seed: int | None, use_llm: bool, height: int | None,
                  say, preview: bool = True) -> ObjectResult:
    from craftpilot.objects import flow

    try:
        return flow.draw(text, brief=brief, seed=seed if seed is not None else int(time.time()) % 1_000_000,
                         use_llm=use_llm, height=height, say=say, preview=preview)
    except flow.ObjectError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


def _object_ghost(player: str, text: str, req: PlaceFields, route: dict, use_llm: bool, brief: ObjectBrief | None,
                  seed: int | None, height: int | None, notes: list[str], source: str = "object") -> dict:
    """Draw (image -> mesh -> voxels), cache the south-facing grid and answer with the hologram."""
    from craftpilot.objects import flow

    result = _draw_or_http(text, brief, seed, use_llm, height, _say(req))
    _remember(player, kind="object", text=text, yaw=_yaw(req, _pending[player].yaw if player in _pending else 0.0),
              route=route, brief=result.brief, result=result)
    return flow.ghost_payload(result, notes + result.notes, source=f"{source}:{result.engine or 'object'}")


def _object_commit(player: str, p: Pending, req: PlaceFields, extra_notes: list[str]) -> dict:
    """The G commit: place the cached draw where the player locked the box (rotated toward them)."""
    from craftpilot.objects import flow
    from craftpilot.place.bridge import BridgeError

    result = p.result
    assert result is not None
    place_ctx = _place_context(req) if req.place else None
    yaw = float(place_ctx.player.get("yaw", _yaw(req, p.yaw))) if place_ctx is not None else _yaw(req, p.yaw)
    facing = facing_from_yaw(yaw) if place_ctx is not None else "south"
    notes = list(extra_notes)
    if p.stale:
        notes.append("your edit was not drawn yet - this is the previous preview; /build go draws the edited brief")
    out = flow.result_payload(result, facing, "previous", notes)
    if place_ctx is not None:
        try:
            placement = flow.place_object(result, place_ctx, facing)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except BridgeError as exc:
            raise HTTPException(status_code=503, detail=f"mod error: {exc}") from exc
        out["placement"] = placement
        out["notes"] += placement["warnings"]
        if placement.get("undo_file"):
            out["notes"].append(f"undo: craftpilot object-undo \"{placement['undo_file']}\"")
    out["summary"] = _summary(out, result.brief.label)
    p.yaw = yaw
    return out


def _object_plan(player: str, text: str, use_llm: bool, height: int | None, route: dict, notes: list[str],
                 brief: ObjectBrief | None = None, keep: Pending | None = None) -> dict:
    """The brief only (one small LLM call, no drawing): the object-path counterpart of a building plan."""
    from craftpilot.objects import flow
    from craftpilot.objects.brief import compose_brief

    plan_notes = list(notes)
    if brief is None:
        brief, meta = compose_brief(text, use_llm=use_llm, height=height)
        if meta.get("error"):
            plan_notes.append(f"brief: LLM failed ({meta['error']}); used the keyword fallback")
    h = brief.height
    p = _remember(player, kind="object", text=text, route=route, brief=brief,
                  result=keep.result if keep is not None else None, stale=keep is not None and keep.result is not None)
    lines = flow.describe_brief(brief)
    return {
        "kind": "object",
        "plan": lines,
        "bounds": {"width": h, "height": h, "depth": h},  # an honest upper bound: height is the largest dimension
        "source": f"object:{brief.source}",
        "notes": plan_notes,
        "summary": f"Plan: {lines[0]}",
        "stale_preview": p.stale,
    }


def _object_preview(player: str, text: str, req: BuildRequest, route: dict, use_llm: bool, notes: list[str]) -> dict:
    """`/build preview <text>` for an object: draw and report the artefacts; no hologram, no placement, and no
    `plan` key (the mod would arm plan mode). The grid is exported facing south like a building preview."""
    from craftpilot.objects import flow

    result = _draw_or_http(text, None, req.seed, use_llm, req.height, None, preview=True)
    _remember(player, kind="object", text=text, route=route, brief=result.brief, result=result)
    out = flow.result_payload(result, "south", f"object:{result.engine or 'object'}", notes + result.notes)
    out["notes"] += flow.describe_brief(result.brief, result.grid)
    out["summary"] = _summary(out, result.brief.label)
    return out


# ------------------------------------------------------------------------------------------------ endpoints

@app.post("/plan")
def plan(req: PlanRequest) -> dict:
    """Compose only. The plan waits for /edit {plan: true} refinements and /regenerate {place: true} to build."""
    route, text = _route(req.text, req.kind, req.use_llm, req.bounds)
    if route.kind == "object":
        return _with_route(_object_plan(req.player, text, _use_llm(req.use_llm), req.height, route.to_dict(), []),
                           route)
    area = _area_of(req.area)
    facing = facing_from_area(area, req.pos, req.yaw or 0.0) if area is not None else None
    program, source, notes = _compose(text, req.use_llm, req.bounds,
                                      area_footprint(area, facing) if area is not None else None)
    return _with_route(_plan_result(req.player, program, text, source, notes, req.bounds, route.to_dict(),
                                    area, facing), route)


@app.post("/build")
def build(req: BuildRequest) -> dict:
    route, text = _route(req.text, req.kind, req.use_llm, req.bounds)
    if route.kind == "object":
        use_llm = _use_llm(req.use_llm)
        if req.ghost:
            return _with_route(_object_ghost(req.player, text, req, route.to_dict(), use_llm, None, req.seed,
                                             req.height, []), route)
        if req.place:
            # a non-mod client asking to draw and place in one go: draw, then commit at the player's pose
            payload = _object_ghost(req.player, text, req, route.to_dict(), use_llm, None, req.seed, req.height, [])
            out = _object_commit(req.player, _pending[req.player], req, [])
            out["ghost_summary"] = payload["summary"]
            return _with_route(out, route)
        return _with_route(_object_preview(req.player, text, req, route.to_dict(), use_llm, []), route)
    area = _area_of(req.area)
    facing = facing_from_area(area, req.pos, _yaw(req, 0.0)) if area is not None else None
    program, source, notes = _compose(text, req.use_llm, req.bounds,
                                      area_footprint(area, facing) if area is not None else None)
    if req.ghost:
        return _with_route(_ghost_result(req.player, program, text, req.seed, source, notes, req.bounds,
                                         route.to_dict(), area, facing), route)
    return _with_route(_run(req.player, program, req.bounds, req.seed, req.preview, text, notes, source,
                            _yaw(req, 0.0), req, route.to_dict(), area, facing), route)


@app.post("/edit")
def edit(req: EditRequest) -> dict:
    p = _pending_or_404(req.player)
    text = f"{p.text} / {req.text}"
    if p.kind == "object":
        from craftpilot.objects.brief import edit_brief

        assert p.brief is not None
        brief, source, notes = edit_brief(p.brief, req.text, use_llm=True)
        if req.plan:
            return _object_plan(req.player, text, True, None, p.route, notes, brief=brief, keep=p)
        if req.ghost:
            return _object_ghost(req.player, text, req, p.route, True, brief, req.seed, None, notes,
                                 source="object:edit")
        _remember(req.player, kind="object", text=text, route=p.route, brief=brief, result=p.result, stale=True)
        return _object_plan(req.player, text, True, None, p.route, notes, brief=brief, keep=p)
    from craftpilot.llm.compose import edit as llm_edit

    assert p.program is not None
    program, source, notes = llm_edit(p.program, req.text)
    if req.plan:
        return _plan_result(req.player, program, text, source, notes, None, p.route, p.area, p.facing)
    if req.ghost:
        return _ghost_result(req.player, program, text, req.seed, source, notes, None, p.route, p.area, p.facing)
    return _run(req.player, program, None, req.seed, False, text, notes, source, _yaw(req, p.yaw), req, p.route,
                p.area, p.facing)


@app.post("/regenerate")
def regenerate(req: RegenerateRequest) -> dict:
    p = _pending_or_404(req.player)
    if p.kind == "object":
        if req.ghost:
            # /build go (nothing drawn yet) draws the plan; /build again (already drawn) draws a new take
            return _object_ghost(req.player, p.text, req, p.route, True, p.brief, req.seed, None, [],
                                 source="object:again" if p.result is not None else "object")
        if p.result is None:
            raise HTTPException(status_code=409, detail="nothing drawn yet for that plan; /build go first")
        notes = []
        if req.seed is not None and req.seed != p.result.seed:
            notes.append(f"preview #{req.seed} was replaced; building the current preview #{p.result.seed}")
        return _object_commit(req.player, p, req, notes)
    assert p.program is not None
    if req.ghost:
        return _ghost_result(req.player, p.program, p.text, req.seed, "previous", [], None, p.route, p.area, p.facing)
    return _run(req.player, p.program, None, req.seed, False, p.text, [], "previous", _yaw(req, p.yaw), req,
                p.route, p.area, p.facing)


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

    p = _pending_or_404(req.player)
    if p.kind != "building" or p.program is None:
        raise HTTPException(status_code=400, detail="exemplars are for buildings; the last build was an object")
    path = save(SETTINGS.exemplars_dir, req.name, req.description or p.text, req.tags, p.program)
    return {"saved": str(path)}
