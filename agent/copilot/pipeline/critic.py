"""Vision critic (plan §7.2, §8.1): render a contact sheet, ask for a score and ≤3 concrete fixes."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..jobs import check_cancel
from ..llm import IMAGES_PER_CALL, extract_json, image_parts, single_call, text_content
from .budget import profile_row
from .common import call_tool, outline
from .stages import brief_kind, load_prompt, prompt_variant

CRITIC_VIEWS = ["contact"]  # one image: iso + front + top (+ cutaway), sent as one 640 px JPEG (T2)


@dataclass
class Critique:
    score: Optional[int]
    fixes: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    raw: str = ""
    stage: str = ""
    used_vision: bool = False
    lint_text: str = ""

    @property
    def ok(self) -> bool:
        return self.score is not None

    def passed(self, threshold: int = 8) -> bool:
        return self.score is not None and self.score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "score": self.score, "fixes": self.fixes, "summary": self.summary, "used_vision": self.used_vision}


def _mid_y(ctx: Any) -> int:
    try:
        bb = ctx.session.scene.bbox()
        if bb.is_empty():
            return 1
        return int(max(bb.lo[1], (bb.lo[1] + bb.hi[1]) / 2.0 - 1))
    except Exception:  # noqa: BLE001
        return 1


_OP_CALL = re.compile(r"\b(add|set_shape|set_material|paint|move|remove|delete|reorder|define_material|run_script|scene\.\w+|mirror_copy|array|extrude|fit|shell|round|group|ungroup|place|block)\s*\(", re.I)


def _scene_ids(ctx: Any) -> Optional[List[str]]:
    try:
        return [o.id for o in ctx.session.scene.objects]
    except Exception:  # noqa: BLE001
        return None


def parse_critique(text: Any, stage: str = "", known_ids: Optional[Sequence[str]] = None) -> Critique:
    """Parse the critic's JSON; tolerate prose, fences and partial output.

    T4: at most 3 fixes; each must carry a concrete op call (`add(…)`, `set_shape(…)`, …) or it is
    dropped — prose alone gives the fix round nothing to execute. When `known_ids` is given, ids the
    critic invented are stripped from `objects`; a fix that then names nothing and does not `add`/`run_script`
    something new is dropped too (it would send the fix round hunting for a non-existent object)."""
    raw = "" if text is None else str(text)
    j = extract_json(raw)
    if not isinstance(j, dict):
        return Critique(score=None, fixes=[], summary=raw.strip()[:400], raw=raw, stage=stage)
    score: Optional[int]
    try:
        score = int(round(float(j.get("score"))))  # type: ignore[arg-type]
        score = max(1, min(10, score))
    except (TypeError, ValueError):
        score = None
    fixes: List[Dict[str, Any]] = []
    for f in (j.get("top_3_fixes") or j.get("fixes") or [])[:3]:
        if isinstance(f, str):
            fixes.append({"rule": "", "objects": [], "op_suggestion": f})
            continue
        if not isinstance(f, dict):
            continue
        objs = f.get("objects") or []
        if isinstance(objs, str):
            objs = [objs]
        fixes.append({"rule": str(f.get("rule", "")).upper(), "objects": [str(o) for o in objs], "op_suggestion": str(f.get("op_suggestion") or f.get("op") or f.get("fix") or "")})
    summary = str(j.get("summary") or "")
    return Critique(score=score, fixes=vet_fixes(fixes, known_ids), summary=summary, raw=raw, stage=stage)


def vet_fixes(fixes: List[Dict[str, Any]], known_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Keep only actionable fixes: a concrete op call, and object ids that exist (see parse_critique)."""
    known = set(known_ids) if known_ids is not None else None
    out: List[Dict[str, Any]] = []
    for f in fixes:
        op = str(f.get("op_suggestion") or "").strip()
        if not _OP_CALL.search(op):
            continue
        objs = list(f.get("objects") or [])
        if known is not None:
            objs = [o for o in objs if o in known or any(o == k.split("/")[-1] for k in known)]
            creates = bool(re.match(r"\s*(add|run_script|scene\.add|define_material|paint|place)\s*\(", op, re.I))
            if not objs and not creates:
                continue
        out.append({**f, "objects": objs})
    return out[:3]


def fixes_text(c: Critique) -> str:
    """Critic findings as a short numbered list for the next stage/fix round."""
    if not c.fixes and not c.summary:
        return ""
    lines = []
    if c.summary:
        lines.append(f"Critic ({c.stage or 'review'}, score {c.score if c.score is not None else '?'}/10): {c.summary}")
    for i, f in enumerate(c.fixes, 1):
        objs = ", ".join(f.get("objects") or []) or "-"
        lines.append(f"{i}. [{f.get('rule') or '-'}] objects: {objs} → {f.get('op_suggestion')}")
    return "\n".join(lines)


def critique(llm: Any, ctx: Any, stage: str, brief: Optional[Dict[str, Any]], lint_text: Optional[str] = None) -> Critique:
    """Render (contact sheet when vision is available) and ask the critic for a score + top-3 fixes."""
    supports_vision = bool(getattr(llm, "supports_vision", False))
    check_cancel(ctx)
    t0 = time.time()
    if lint_text is None:
        r = call_tool(ctx, "lint", {})
        lint_text = getattr(r, "text", "") or ""
    images: List[Any] = []
    render_text = ""
    if supports_vision:
        r = call_tool(ctx, "render", {"views": CRITIC_VIEWS})
        images = list(getattr(r, "images", None) or [])[:1]
        render_text = getattr(r, "text", "") or ""
        if not images:  # older/other renderers: ask for the views separately
            r = call_tool(ctx, "render", {"views": ["iso", "front"]})
            images = list(getattr(r, "images", None) or [])[:IMAGES_PER_CALL]
            render_text = getattr(r, "text", "") or render_text
    if not images:
        # text-only fallback: bbox/block summary plus an ASCII plan slice at mid height
        r = call_tool(ctx, "render", {"views": ["top"], "slice_y": _mid_y(ctx)})
        render_text = (getattr(r, "text", "") or "") or render_text
    sys_prompt = load_prompt(prompt_variant("critic", brief_kind(brief, ctx)))
    parts: List[Dict[str, Any]] = []
    head = [f"## Stage under review: {stage}"]
    if brief:
        head.append("## Brief\n```json\n" + json.dumps({k: v for k, v in brief.items() if k != "request"}, indent=1) + "\n```")
    head.append("## Scene outline\n```\n" + outline(ctx) + "\n```")
    head.append("## Lint findings\n" + (lint_text.strip() or "(none)"))
    if render_text:
        head.append("## Render summary\n" + render_text.strip()[:3000])
    if images:
        head.append("## Images\nContact sheet (isometric, front, top, cutaway). Judge what you see.")
    parts.append(text_content("\n\n".join(head)))
    parts.extend(image_parts(images))
    parts.append(text_content("Reply with the JSON object only."))
    t_llm = time.time()
    try:
        resp = single_call(llm, sys_prompt, parts, temperature=0.2, max_tokens=1200, ctx=ctx)
        text = resp.text or ""
    except Exception as e:  # noqa: BLE001
        text = f"critic error: {e}"
    llm_ms = int((time.time() - t_llm) * 1000)
    c = parse_critique(text, stage, known_ids=_scene_ids(ctx))
    c.used_vision = bool(images)
    c.lint_text = lint_text
    wall_ms = int((time.time() - t0) * 1000)
    profile_row(ctx, f"critic:{stage}", wall_ms=wall_ms, llm_ms=llm_ms, engine_ms=wall_ms - llm_ms, llm_calls=1, note=f"score {c.score}")
    log = getattr(ctx, "log", None)
    if log is not None and hasattr(log, "log"):
        try:
            log.log(stage=stage, name="__critic__", args={"vision": c.used_vision}, result_preview=json.dumps(c.to_dict())[:300], ms=0)
        except Exception:  # noqa: BLE001
            pass
    return c
