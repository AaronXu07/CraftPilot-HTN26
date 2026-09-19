"""Top-level chat handler: new builds run the staged pipeline, follow-ups go through the router.

`handle_chat(ctx, text, llm=None) -> ChatResult` never raises; failures become a readable reply.
"""
from __future__ import annotations

import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..jobs import JobCancelled, check_cancel
from ..llm import JsonlLogger, MockLLM, _emit, single_call
from .common import call_tool, env_flag, outline
from .critic import Critique, critique, fixes_text
from .interpret import DEFAULT_STAGES, brief_to_text, interpret
from .router import Route, route
from .stages import FIX_STAGE, STAGE_BY_NAME, StageResult, run_stage, stage_for_rule

CRITIC_PASS = 8
MAX_FINAL_FIX_ROUNDS = 2
FIX_ROUND_CALLS = 15

HELP_TEXT = (
    "Minecraft Copilot — talk to me like a designer:\n"
    "• `/cp build a castle with four round towers and a gatehouse facing me`\n"
    "• `/cp make the northeast tower 8 blocks taller and give it a copper roof`\n"
    "• `/cp swap the walls to deepslate with a mossy base`\n"
    "• `/cp undo` · `/cp redo` · `/cp status` · `/cp cancel` (stop the running job) · `/cp render` · `/cp place`\n"
    "• `/cp export [name]` (.litematic) · `/cp materials` (block counts)\n"
    "• `/cp preview on|off` (place each stage live) · `/cp reset` (forget the current build)"
)


@dataclass
class ChatResult:
    reply: str
    images: List[Any] = field(default_factory=list)
    brief: Optional[Dict[str, Any]] = None
    placed: bool = False
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"reply": self.reply, "brief": self.brief, "placed": self.placed, "images": len(self.images), "data": self.data}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _scripted_mock_llm() -> Any:
    """COPILOT_LLM=mock: the role-aware scripted builder (bench/mock_builder.py) when importable,
    so a demo without Azure credentials still produces a build; else an empty MockLLM."""
    try:
        from bench.mock_builder import ScriptedBuilderLLM  # noqa: WPS433

        return ScriptedBuilderLLM()
    except Exception:  # noqa: BLE001
        return MockLLM([])


def resolve_llm(ctx: Any, llm: Any = None) -> Any:
    """Explicit llm > ctx.llm > COPILOT_LLM=mock → MockLLM > AzureLLM()."""
    if llm is not None:
        return llm
    c = getattr(ctx, "llm", None)
    if c is not None:
        return c
    if os.environ.get("COPILOT_LLM", "").lower() == "mock":
        m = _scripted_mock_llm()
    else:
        from ..llm import AzureLLM

        m = AzureLLM()
    try:
        ctx.llm = m
    except Exception:  # noqa: BLE001
        pass
    return m


def _say(ctx: Any, text: str) -> None:
    text = " ".join(str(text).split())
    if not text:
        return
    call_tool(ctx, "say", {"text": text[:240]})


def _place(ctx: Any, mode: str = "diff") -> Any:
    return call_tool(ctx, "place", {"mode": mode, "animate": True})


def _one_line(s: str, n: int = 160) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _turn_logger(ctx: Any) -> Any:
    session = getattr(ctx, "session", None)
    current = getattr(ctx, "log", None)
    if current is not None and not isinstance(current, JsonlLogger):
        return current  # caller-provided logger
    run_dir = getattr(ctx, "run_dir", None) or getattr(session, "run_dir", None) or "runs/anon"
    turn = getattr(session, "turn", 0)
    logger = JsonlLogger(os.path.join(run_dir, f"{turn}.jsonl"))
    try:
        ctx.log = logger
    except Exception:  # noqa: BLE001
        pass
    return logger


def _fast(ctx: Any, fast: Optional[bool]) -> bool:
    if fast is not None:
        return bool(fast)
    if getattr(ctx, "fast", None) is not None:
        return bool(ctx.fast)
    return env_flag("COPILOT_FAST", False)


def _block_count(place_result: Any) -> Optional[int]:
    data = getattr(place_result, "data", None) or {}
    for k in ("blocks", "block_count", "total", "placed"):
        if isinstance(data.get(k), int):
            return data[k]
    m = re.search(r"(\d[\d,]*)\s+blocks?", getattr(place_result, "text", "") or "")
    if m:
        return int(m.group(1).replace(",", ""))
    return None


# ----------------------------------------------------------------------------------------------
# meta commands
# ----------------------------------------------------------------------------------------------
_META_RE = re.compile(
    r"^\s*/?(?:cp\s+)?(?P<cmd>undo|redo|export|materials?(?:\s+list)?|reset|clear|preview|help|status|describe|outline|render|place|snapshot|restore)\b\s*(?P<arg>.*)$",
    re.IGNORECASE,
)


def handle_meta(ctx: Any, text: str) -> Optional[ChatResult]:
    """Handle undo/redo/export/materials/reset/preview/help/status/render/place. None if not a meta command."""
    m = _META_RE.match(text)
    if not m:
        return None
    cmd = m.group("cmd").lower().split()[0]
    arg = (m.group("arg") or "").strip()
    session = ctx.session
    if cmd == "help":
        return ChatResult(reply=HELP_TEXT)
    if cmd in ("undo", "redo"):
        n = int(arg) if arg.isdigit() else 1
        msg = session.undo(n) if cmd == "undo" else session.redo(n)
        placed = False
        if session.world.is_placed():
            r = _place(ctx, "diff")
            placed = getattr(r, "ok", True) and "ERROR" not in (getattr(r, "text", "") or "")
        return ChatResult(reply=msg, placed=placed)
    if cmd == "export":
        name = arg or session.scene.name or "build"
        r = call_tool(ctx, "export_schematic", {"name": name})
        return ChatResult(reply=getattr(r, "text", "exported"), data=getattr(r, "data", {}) or {})
    if cmd.startswith("material"):
        r = call_tool(ctx, "materials_list", {})
        return ChatResult(reply=getattr(r, "text", ""))
    if cmd in ("reset", "clear"):
        from ..engine.scene import Scene
        from ..session import WorldState

        if session.world.is_placed():
            call_tool(ctx, "undo_world", {})
        session.snapshot("before_reset")
        session.replace_scene(Scene(name="untitled"), "reset")
        session.world = WorldState()
        session.brief = None
        return ChatResult(reply="Scene cleared and the world restored. Tell me what to build next.")
    if cmd == "preview":
        on = arg.lower() in ("on", "true", "1", "yes")
        session.live_preview = on
        return ChatResult(reply=f"Live preview {'on' if on else 'off'}: intermediate stages will {'now' if on else 'no longer'} be placed in the world.")
    if cmd in ("status", "describe", "outline"):
        head = ""
        if session.brief:
            head = "Brief: " + brief_to_text(session.brief) + "\n"
        return ChatResult(reply=head + outline(ctx, detail="full" if arg == "full" else "outline"))
    if cmd == "render":
        views = [v for v in re.split(r"[,\s]+", arg) if v] or ["iso", "front"]
        r = call_tool(ctx, "render", {"views": views})
        return ChatResult(reply=getattr(r, "text", "rendered"), images=list(getattr(r, "images", None) or []))
    if cmd == "place":
        mode = "full" if arg.lower() in ("full", "all") else "diff"
        r = _place(ctx, mode)
        return ChatResult(reply=getattr(r, "text", "placed"), placed=True)
    if cmd == "snapshot":
        return ChatResult(reply=session.snapshot(arg or f"snap_{len(session.snapshots) + 1}"))
    if cmd == "restore":
        try:
            msg = session.restore(arg)
        except Exception as e:  # noqa: BLE001
            return ChatResult(reply=str(e))
        placed = False
        if session.world.is_placed():
            _place(ctx, "diff")
            placed = True
        return ChatResult(reply=msg, placed=placed)
    return None


# ----------------------------------------------------------------------------------------------
# new build
# ----------------------------------------------------------------------------------------------
@dataclass
class BuildReport:
    brief: Dict[str, Any]
    stages: List[StageResult] = field(default_factory=list)
    critiques: List[Critique] = field(default_factory=list)
    final_score: Optional[int] = None
    tool_calls: int = 0
    seconds: float = 0.0
    place_text: str = ""
    blocks: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brief": self.brief,
            "stages": [{"stage": s.stage, "calls": s.tool_calls, "stopped": s.stopped_reason, "seconds": round(s.seconds, 1), "summary": s.text[:200]} for s in self.stages],
            "critiques": [c.to_dict() for c in self.critiques],
            "final_score": self.final_score,
            "tool_calls": self.tool_calls,
            "seconds": round(self.seconds, 1),
            "blocks": self.blocks,
        }


def run_build(ctx: Any, llm: Any, request: str, fast: bool = False, place: bool = True) -> "tuple[ChatResult, BuildReport]":
    """Interpret → (blocking → critic → fix)* … → final critique → place. Returns the reply + report."""
    session = ctx.session
    t0 = time.time()
    brief = interpret(llm, ctx, request)
    report = BuildReport(brief=brief)
    _say(ctx, "Plan: " + brief_to_text(brief) + ". Building…")
    stage_names = [s for s in brief.get("stages", DEFAULT_STAGES) if s in STAGE_BY_NAME] or list(DEFAULT_STAGES)
    carry: Optional[str] = None
    images: List[Any] = []
    for name in stage_names:
        check_cancel(ctx)
        stage = STAGE_BY_NAME[name]
        res = run_stage(llm, ctx, stage, request, brief, extra=carry)
        report.stages.append(res)
        carry = None
        images = res.images or images
        if not fast:
            crit = critique(llm, ctx, stage.name, brief, lint_text=res.lint_text)
            report.critiques.append(crit)
            if crit.fixes and not crit.passed(CRITIC_PASS):
                fix = run_stage(llm, ctx, stage, request, brief, extra=fixes_text(crit), max_calls=FIX_ROUND_CALLS, fix_round=True)
                report.stages.append(fix)
                images = fix.images or images
            elif crit.fixes:
                carry = fixes_text(crit)  # minor notes for the next stage
        _say(ctx, f"{stage.name}: {_one_line(res.text, 120)}")
        if getattr(session, "live_preview", False):
            _place(ctx, "diff")
    if not fast:
        for _ in range(MAX_FINAL_FIX_ROUNDS):
            crit = critique(llm, ctx, "final", brief)
            report.critiques.append(crit)
            report.final_score = crit.score
            if crit.passed(CRITIC_PASS) or not crit.fixes:
                break
            stage = stage_for_rule(crit.fixes[0].get("rule", "")) if crit.fixes else FIX_STAGE
            fix = run_stage(llm, ctx, stage, request, brief, extra=fixes_text(crit), max_calls=FIX_ROUND_CALLS, fix_round=True)
            report.stages.append(fix)
            images = fix.images or images
    placed = False
    if place:
        r = _place(ctx, "diff")
        report.place_text = getattr(r, "text", "") or ""
        report.blocks = _block_count(r)
        placed = getattr(r, "ok", True) and not report.place_text.startswith("ERROR")
    report.tool_calls = sum(s.tool_calls for s in report.stages)
    report.seconds = time.time() - t0
    n_obj = len(session.scene.objects)
    reply = f"Built {brief.get('name', 'the build')}: " + brief_to_text(brief) + f". {n_obj} objects"
    if report.blocks:
        reply += f", {report.blocks:,} blocks"
    if report.final_score is not None:
        reply += f"; critic {report.final_score}/10"
    last = next((s.text for s in reversed(report.stages) if s.text), "")
    if last:
        reply += ". " + _one_line(last, 140)
    reply += " Say what to change (e.g. \"make the towers taller\", \"copper roofs\") or `undo`."
    return ChatResult(reply=reply, images=images, brief=brief, placed=placed, data=report.to_dict()), report


# ----------------------------------------------------------------------------------------------
# edits and questions
# ----------------------------------------------------------------------------------------------
def run_edit(ctx: Any, llm: Any, request: str, r: Route, fast: bool = False) -> ChatResult:
    """Direct ops → scoped stage runs → render → diff place → one-line confirmation."""
    session = ctx.session
    changes: List[str] = []
    errors: List[str] = []
    for op in r.direct_ops:
        res = call_tool(ctx, op["name"], op["args"])
        txt = getattr(res, "text", "") or ""
        (errors if txt.startswith("ERROR") else changes).append(_one_line(txt, 120))
    images: List[Any] = []
    for name in r.stages:
        stage = STAGE_BY_NAME.get(name)
        if stage is None:
            continue
        extra = None
        if errors:
            extra = "Some direct ops failed; achieve the request another way:\n" + "\n".join(errors)
        res = run_stage(llm, ctx, stage, request, session.brief, selection=r.selection, extra=extra)
        images = res.images or images
        if res.text:
            changes.append(_one_line(res.text, 140))
        if not fast and stage.name in ("blocking", "detailing"):
            crit = critique(llm, ctx, stage.name, session.brief, lint_text=res.lint_text)
            if crit.fixes and not crit.passed(CRITIC_PASS):
                fix = run_stage(llm, ctx, stage, request, session.brief, selection=r.selection, extra=fixes_text(crit), max_calls=FIX_ROUND_CALLS, fix_round=True)
                images = fix.images or images
    if not images:
        rr = call_tool(ctx, "render", {"views": ["iso", "front"]})
        images = list(getattr(rr, "images", None) or [])
    placed = False
    if r.needs_place and (changes or r.stages):
        pr = _place(ctx, "diff")
        placed = not (getattr(pr, "text", "") or "").startswith("ERROR")
    if changes:
        reply = "Changed: " + "; ".join(changes[:4])
        if len(changes) > 4:
            reply += f" (+{len(changes) - 4} more)"
    elif errors:
        reply = "Could not apply: " + "; ".join(errors[:3])
    else:
        reply = "No change was needed."
    if placed:
        reply += " Updated in the world."
    return ChatResult(reply=reply, images=images, brief=session.brief, placed=placed, data={"route": r.to_dict(), "changes": changes, "errors": errors})


def answer_question(ctx: Any, llm: Any, request: str) -> ChatResult:
    """Answer a question about the current build from the outline (no tools, no changes)."""
    sys_prompt = (
        "You are the Minecraft build copilot. Answer the player's question about the current build in one or two "
        "sentences using the scene outline below (coordinates: x east, y up, z south; bbox a..b is min..max).\n\n"
        "## Scene\n```\n" + outline(ctx, detail="full") + "\n```"
    )
    brief = getattr(ctx.session, "brief", None)
    if brief:
        sys_prompt += "\n\n## Brief\n" + brief_to_text(brief)
    try:
        resp = single_call(llm, sys_prompt, request, temperature=0.2, max_tokens=400)
        text = resp.text or "I don't have an answer for that."
    except Exception as e:  # noqa: BLE001
        text = f"I couldn't answer that: {e}"
    return ChatResult(reply=text.strip(), brief=brief)


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------
_BUILD_RE = re.compile(r"^\s*(?:please\s+)?(?:build|make|create|construct|design|erect|put up|generate)\b", re.IGNORECASE)


def handle_chat(ctx: Any, text: str, llm: Any = None, fast: Optional[bool] = None) -> ChatResult:
    """Main entry: meta commands, new build (staged pipeline), edit (router), or question."""
    session = getattr(ctx, "session", None)
    if session is None:
        return ChatResult(reply="No session on the context.")
    text = (text or "").strip()
    if not text:
        return ChatResult(reply=HELP_TEXT)
    session.turn = getattr(session, "turn", 0) + 1
    session.add_chat("user", text)
    log = _turn_logger(ctx)
    fast_mode = _fast(ctx, fast)
    t0 = time.time()
    try:
        meta = handle_meta(ctx, text)
        if meta is not None:
            result = meta
        else:
            model = resolve_llm(ctx, llm)
            if not session.scene.objects:
                result, _ = run_build(ctx, model, text, fast=fast_mode)
            else:
                r = route(model, ctx, text)
                if r.intent == "build" or (r.intent == "edit" and not r.stages and not r.direct_ops and _BUILD_RE.match(text) and "new" in text.lower()):
                    from ..engine.scene import Scene
                    from ..session import WorldState

                    session.snapshot(f"before_{session.scene.name or 'build'}_{session.turn}")
                    session.replace_scene(Scene(name="untitled"), "new build")
                    session.world = WorldState()
                    session.brief = None
                    result, _ = run_build(ctx, model, text, fast=fast_mode)
                elif r.intent == "question":
                    result = answer_question(ctx, model, text)
                else:
                    result = run_edit(ctx, model, text, r, fast=fast_mode)
    except JobCancelled as e:
        # the job runner reports "[cp] stopped: <reason>" to the player; keep the turn's history consistent
        session.add_chat("assistant", f"[cp] stopped: {e.reason}")
        _emit(log, stage="turn", name="__cancelled__", args={"reason": e.reason}, result_preview="", ms=int((time.time() - t0) * 1000))
        try:
            session.save()
        except Exception:  # noqa: BLE001
            pass
        raise
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=3)
        _emit(log, stage="error", name="__exception__", args={}, result_preview=tb[-600:], ms=int((time.time() - t0) * 1000))
        result = ChatResult(reply=f"Sorry — that failed ({type(e).__name__}: {str(e)[:200]}). Try `undo`, `status`, or rephrase the request.")
    session.add_chat("assistant", result.reply)
    _emit(log, stage="turn", name="__reply__", args={"text": text[:200]}, result_preview=result.reply[:300], ms=int((time.time() - t0) * 1000))
    try:
        session.save()
    except Exception:  # noqa: BLE001
        pass
    return result
