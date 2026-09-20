"""Router for conversational edits (plan §7.5): request → stages + selection + direct ops."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..jobs import check_cancel
from ..llm import extract_json, single_call
from .budget import profile_row
from .common import fast_llm, get_tools, outline, tool_names
from .stages import STAGE_BY_NAME, load_prompt

INTENTS = ("build", "edit", "question", "meta")


@dataclass
class Route:
    intent: str = "edit"
    stages: List[str] = field(default_factory=list)
    selection: str = "all"
    direct_ops: List[Dict[str, Any]] = field(default_factory=list)
    needs_place: bool = True
    note: str = ""
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"intent": self.intent, "stages": self.stages, "selection": self.selection, "direct_ops": self.direct_ops, "needs_place": self.needs_place, "note": self.note}


def parse_route(text: Any, known_tools: Optional[List[str]] = None) -> Route:
    """Parse router JSON; unknown stage names and malformed ops are dropped; sensible defaults."""
    raw = "" if text is None else str(text)
    j = extract_json(raw)
    if not isinstance(j, dict):
        return Route(intent="edit", stages=["detailing"], selection="all", direct_ops=[], needs_place=True, note="unparsed", raw=raw)
    intent = str(j.get("intent") or "edit").lower()
    if intent not in INTENTS:
        intent = "edit"
    stages = [str(s).lower() for s in (j.get("stages") or []) if str(s).lower() in STAGE_BY_NAME]
    ops: List[Dict[str, Any]] = []
    for op in j.get("direct_ops") or []:
        if not isinstance(op, dict):
            continue
        name = op.get("name") or op.get("op") or op.get("tool")
        args = op.get("args") or op.get("arguments") or {k: v for k, v in op.items() if k not in ("name", "op", "tool")}
        if not name or not isinstance(args, dict):
            continue
        if known_tools and name not in known_tools:
            continue
        ops.append({"name": str(name), "args": args})
    if intent == "edit" and not stages and not ops:
        stages = ["detailing"]
    sel = j.get("selection")
    sel = str(sel) if sel else "all"
    needs_place = j.get("needs_place")
    needs_place = True if needs_place is None else bool(needs_place)
    if intent == "question":
        needs_place = False
    return Route(intent=intent, stages=stages, selection=sel, direct_ops=ops, needs_place=needs_place, note=str(j.get("note") or ""), raw=raw)


def route(llm: Any, ctx: Any, request: str) -> Route:
    """Cheap classification call (temperature 0) using the router prompt + current outline."""
    sys_prompt = load_prompt("router") + "\n\n## Current scene\n```\n" + outline(ctx) + "\n```"
    brief = getattr(getattr(ctx, "session", None), "brief", None)
    if brief:
        sys_prompt += "\n\n## Brief\n" + json.dumps({k: brief.get(k) for k in ("name", "build_type", "style", "facing", "key_features") if k in brief})
    check_cancel(ctx)
    t0 = time.time()
    try:
        resp = single_call(fast_llm(llm), sys_prompt, request.strip(), temperature=0.0, max_tokens=800, ctx=ctx)
        text = resp.text or ""
    except Exception as e:  # noqa: BLE001
        text = f"router error: {e}"
    ms = int((time.time() - t0) * 1000)
    profile_row(ctx, "route", wall_ms=ms, llm_ms=ms, llm_calls=1)
    known = tool_names(get_tools(ctx)) or None
    r = parse_route(text, known)
    log = getattr(ctx, "log", None)
    if log is not None and hasattr(log, "log"):
        try:
            log.log(stage="router", name="__router__", args={"request": request[:200]}, result_preview=json.dumps(r.to_dict())[:300], ms=0)
        except Exception:  # noqa: BLE001
            pass
    return r
