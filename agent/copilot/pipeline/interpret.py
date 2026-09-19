"""Stage 0 — Interpret: player request → design brief (plan §7.2 step 0)."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from ..jobs import check_cancel
from ..llm import extract_json, single_call
from .budget import profile_row
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


_NUM = r"(\d+(?:\.\d+)?)"


def _tower_problems(plan: str) -> List[str]:
    """Towers/turrets/spires described wider than tall (P1), read from the silhouette plan's numbers.

    Understands `r=4 h=24`, `radius 4 … height 24`, `6x6 h=5`, `6x6x5`, `… 20 tall` within the clause
    that names the tower (up to the next `;` or `.`)."""
    out: List[str] = []
    for m in re.finditer(r"\b(towers?|turrets?|spires?|minarets?|pillars?)\b([^;.]*)", plan, flags=re.I):
        clause = m.group(2)
        width: Optional[float] = None
        height: Optional[float] = None
        r = re.search(r"\b(?:r|radius)\s*=?\s*" + _NUM, clause, flags=re.I)
        if r:
            width = 2 * float(r.group(1))
        box = re.search(r"\b" + _NUM + r"\s*[x×]\s*" + _NUM + r"(?:\s*[x×]\s*" + _NUM + r")?", clause, flags=re.I)
        if box:
            width = max(float(box.group(1)), float(box.group(2))) if width is None else width
            if box.group(3):
                height = float(box.group(3))
        h = re.search(r"\b(?:h|height)\s*=?\s*" + _NUM, clause, flags=re.I) or re.search(_NUM + r"\s*(?:tall|high)\b", clause, flags=re.I)
        if h:
            height = float(h.group(1))
        if width is not None and height is not None and width > height:
            out.append(f"{m.group(1).lower()} described {width:g} wide but only {height:g} tall — towers must be taller than wide (P1)")
    return out


def validate_brief(brief: Dict[str, Any]) -> List[str]:
    """Design-rule problems in a brief that the interpret stage must fix before building (T4 lever 1):
    towers wider than tall, a roof with no stated overhang, and no stated entrance."""
    plan = str(brief.get("silhouette_plan") or "")
    text = " ".join([plan] + [str(x) for x in brief.get("key_features", [])] + [str(x) for x in brief.get("constraints", [])])
    problems = _tower_problems(plan)
    if re.search(r"\broofs?\b", plan, flags=re.I) and not re.search(r"\b(overhang|eaves?|flat roofs?|parapet)", text, flags=re.I):
        problems.append("the plan has roofs but states no overhang — give every pitched/hip/cone roof an overhang ≥ 1 (P1), or say 'flat roof' with a parapet")
    if not re.search(r"\b(entrance|entry|door(way)?|gate(house|way)?|portal|portico|porch|archway|opening)\b", text, flags=re.I):
        problems.append(f"no entrance is stated — say where the door/gate is on the {brief.get('facing', 'south')} side and its size (P6)")
    return problems


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


def _retry_allowed(ctx: Any) -> bool:
    """Only spend a second interpret call when the plan schedule has room for it (≥ 20 s of the turn's plan)."""
    budget = getattr(ctx, "budget", None)
    try:
        return budget is None or float(budget.remaining) > 20.0
    except Exception:  # noqa: BLE001
        return True


def interpret(llm: Any, ctx: Any, request: str, temperature: float = 0.6) -> Dict[str, Any]:
    """Ask the model for a brief (via the set_brief tool, with JSON-in-text fallback); store it on the session."""
    tools = get_tools(ctx, names=["set_brief"]) or [SET_BRIEF_SCHEMA]
    tools = [t for t in tools if (t.get("function") or {}).get("name") == "set_brief"] or [SET_BRIEF_SCHEMA]
    sys_prompt = system_prompt(ctx) + "\n\n" + load_prompt("stage_interpret")
    user = f"Player request: {request.strip()}\n\nWrite the brief and call set_brief."
    brief: Optional[Dict[str, Any]] = None
    check_cancel(ctx)
    t0 = time.time()
    calls = 0
    problems: List[str] = []
    for attempt in range(2):  # one retry when the brief breaks a design rule (towers wider than tall, no overhang, no entrance)
        calls += 1
        try:
            resp = single_call(llm, sys_prompt, user, temperature, tools, tool_choice={"type": "function", "function": {"name": "set_brief"}}, ctx=ctx)
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
            break
        problems = validate_brief(normalize_brief(brief, request)) if isinstance(brief, dict) and "error" not in brief else []
        if not problems or attempt or not _retry_allowed(ctx):
            break
        check_cancel(ctx)
        user = (
            f"Player request: {request.strip()}\n\nYour previous brief was rejected:\n- " + "\n- ".join(problems)
            + f"\n\nPrevious silhouette_plan: {normalize_brief(brief, request)['silhouette_plan'][:800]}\n\nRewrite the brief so the plan fixes every point above and call set_brief again."
        )
    ms = int((time.time() - t0) * 1000)
    profile_row(ctx, "interpret", wall_ms=ms, llm_ms=ms, llm_calls=calls, note="; ".join(problems)[:120] if problems else "")
    b = normalize_brief(brief, request)
    if problems:
        # still failing after the retry (or no time for one): the builder gets the rule as a constraint
        b["constraints"] = list(b.get("constraints", [])) + [f"design rule: {p}" for p in problems]
    session = getattr(ctx, "session", None)
    if session is not None:
        session.brief = b
        if not session.scene.objects and b.get("name"):
            session.scene.name = b["name"]
            session.scene.meta["brief"] = {k: v for k, v in b.items() if k != "request"}
            session.scene.meta["style"] = b.get("style", "")
    return b
