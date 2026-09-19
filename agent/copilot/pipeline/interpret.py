"""Stage 0 — Interpret: player request → design brief (plan §7.2 step 0)."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..jobs import check_cancel
from ..llm import extract_json, single_call
from .common import get_tools
from .stages import load_prompt, system_prompt

DEFAULT_STAGES = ["blocking", "detailing", "materials", "decoration"]

SET_BRIEF_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_brief",
        "description": "Record the design brief for the requested build. Call exactly once.",
        "parameters": {
            "type": "object",
            "properties": {
                "brief": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "build_type": {"type": "string"},
                        "style": {"type": "string"},
                        "footprint": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "height": {"type": "number"},
                        "facing": {"type": "string", "enum": ["north", "south", "east", "west"]},
                        "key_features": {"type": "array", "items": {"type": "string"}},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                        "materials_intent": {"type": "string"},
                        "silhouette_plan": {"type": "string"},
                        "stages": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "build_type", "style", "footprint", "height", "facing", "key_features", "silhouette_plan"],
                }
            },
            "required": ["brief"],
        },
    },
}


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
    return (s[:24] or "build") + "_v1"


def normalize_brief(brief: Any, request: str = "") -> Dict[str, Any]:
    """Fill defaults and coerce types so every downstream consumer can rely on the keys."""
    b: Dict[str, Any] = dict(brief) if isinstance(brief, dict) else {}
    if "brief" in b and isinstance(b["brief"], dict):
        b = dict(b["brief"])
    build_type = str(b.get("build_type") or _guess_type(request) or "structure")
    b["build_type"] = build_type
    b["name"] = str(b.get("name") or _slug(build_type))
    b["style"] = str(b.get("style") or "")
    fp = b.get("footprint") or [20, 20]
    try:
        fp = [float(fp[0]), float(fp[1])]
    except (TypeError, ValueError, IndexError):
        fp = [20.0, 20.0]
    b["footprint"] = [max(3.0, min(200.0, v)) for v in fp]
    try:
        b["height"] = max(3.0, min(200.0, float(b.get("height") or 12)))
    except (TypeError, ValueError):
        b["height"] = 12.0
    facing = str(b.get("facing") or "south").lower()
    b["facing"] = facing if facing in ("north", "south", "east", "west") else "south"
    b["key_features"] = [str(x) for x in (b.get("key_features") or [])]
    b["constraints"] = [str(x) for x in (b.get("constraints") or [])]
    b["materials_intent"] = str(b.get("materials_intent") or "")
    plan = str(b.get("silhouette_plan") or "").strip()
    if not plan:
        plan = f"{build_type} with footprint {int(b['footprint'][0])}x{int(b['footprint'][1])} and height {int(b['height'])}, entrance facing {b['facing']}."
    b["silhouette_plan"] = plan
    stages = [str(s) for s in (b.get("stages") or DEFAULT_STAGES) if str(s) in DEFAULT_STAGES]
    if "blocking" not in stages:
        stages = ["blocking"] + stages
    b["stages"] = stages
    b["request"] = request
    return b


def _guess_type(request: str) -> str:
    words = ["castle", "villa", "pagoda", "lighthouse", "bridge", "cathedral", "temple", "treehouse", "tower", "greenhouse", "house", "hall", "church", "barn", "windmill", "keep", "fort", "cottage", "palace", "mansion", "shrine", "wall", "gate"]
    r = (request or "").lower()
    for w in words:
        if w in r:
            return w
    return ""


def brief_to_text(brief: Dict[str, Any]) -> str:
    """One line for the player, e.g. 'castle (medieval stone), 40x32, 26 tall: L-shaped keep, four towers…'."""
    feats = ", ".join(brief.get("key_features", [])[:4])
    fp = brief.get("footprint", [0, 0])
    s = f"{brief.get('build_type', 'build')}"
    if brief.get("style"):
        s += f" ({brief['style']})"
    s += f", {int(fp[0])}x{int(fp[1])}, {int(brief.get('height', 0))} tall, facing {brief.get('facing', 'south')}"
    if feats:
        s += f": {feats}"
    return s


def interpret(llm: Any, ctx: Any, request: str, temperature: float = 0.6) -> Dict[str, Any]:
    """Ask the model for a brief (via the set_brief tool, with JSON-in-text fallback); store it on the session."""
    tools = get_tools(ctx, names=["set_brief"]) or [SET_BRIEF_SCHEMA]
    tools = [t for t in tools if (t.get("function") or {}).get("name") == "set_brief"] or [SET_BRIEF_SCHEMA]
    sys_prompt = system_prompt(ctx) + "\n\n" + load_prompt("stage_interpret")
    user = f"Player request: {request.strip()}\n\nWrite the brief and call set_brief."
    brief: Optional[Dict[str, Any]] = None
    check_cancel(ctx)
    try:
        resp = single_call(llm, sys_prompt, user, temperature, tools, tool_choice={"type": "function", "function": {"name": "set_brief"}})
        for tc in resp.tool_calls:
            if tc.name == "set_brief":
                brief = tc.args.get("brief", tc.args) if isinstance(tc.args, dict) else None
                break
        if brief is None and resp.text:
            j = extract_json(resp.text)
            if isinstance(j, dict):
                brief = j
    except Exception as e:  # noqa: BLE001
        brief = {"error": str(e)}
    b = normalize_brief(brief, request)
    session = getattr(ctx, "session", None)
    if session is not None:
        session.brief = b
        if not session.scene.objects and b.get("name"):
            session.scene.name = b["name"]
            session.scene.meta["brief"] = {k: v for k, v in b.items() if k != "request"}
            session.scene.meta["style"] = b.get("style", "")
    return b
