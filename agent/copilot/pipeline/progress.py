"""Chat progress lines (T3): `[cp·<tag> m:ss] <one line>` per pipeline event, never JSON.

`Progress` lives on `ctx.progress` for one chat turn. Stage summaries are derived from the scene delta
(objects/materials before vs after the stage) so they read like `added 7 solids` / `carved 14 openings`
rather than the model's prose. `verbose` (session.verbose, `/cp verbose on|off`) adds one line per tool
call; off by default keeps only the tagged summaries.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from .common import call_tool, env_flag

TAGS = {"interpret": "plan", "blocking": "block", "detailing": "detail", "materials": "materials", "decoration": "decor", "fix": "fix", "critic": "critic", "build": "build", "edit": "edit"}
MAX_LINE = 200

# words used for subtract objects by shape family ("carved 14 windows, 2 arches")
_OPENING_WORDS = (("arch", "arch"), ("door", "door"), ("gate", "gate"), ("window", "window"), ("win", "window"), ("slit", "slit"), ("cut", "opening"))
_PROP_WORDS = (("lantern", "lantern"), ("torch", "torch"), ("banner", "banner"), ("flag", "flag"), ("pot", "pot"), ("chest", "chest"), ("crenel", "battlement"), ("merlon", "battlement"), ("battlement", "battlement"), ("pilaster", "pilaster"), ("buttress", "buttress"), ("column", "column"))
_PLURALS = {"arch": "arches", "torch": "torches"}


def fmt_elapsed(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def one_line(text: Any, n: int = MAX_LINE) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _plural(n: int, one: str, many: Optional[str] = None) -> str:
    return f"{n} {one if n == 1 else (many or _PLURALS.get(one) or one + 's')}"


def _count(kinds: "Counter[str]", limit: int = 3) -> str:
    return ", ".join(_plural(n, w) for w, n in kinds.most_common(limit))


def _multiplicity(modifiers: Any) -> int:
    """How many copies an object stands for (array counts × mirror copies), for `carved 14 windows`."""
    n = 1
    for m in modifiers or []:
        if not isinstance(m, dict):
            continue
        if m.get("type") == "array":
            n *= max(1, int(m.get("count", 1) or 1))
        elif m.get("type") == "mirror" and m.get("keep_original", True):
            n *= 2
    return n


def _kind_word(obj_id: str, table: Tuple[Tuple[str, str], ...], default: str) -> str:
    low = obj_id.lower()
    for key, word in table:
        if key in low:
            return word
    return default


def scene_snapshot(scene: Any) -> Dict[str, Any]:
    """What a stage summary compares: object ids/ops/materials/modifier counts and material names."""
    objs: Dict[str, Tuple[str, Optional[str], int, str, int]] = {}
    for o in getattr(scene, "objects", []) or []:
        shape = getattr(o, "shape", None) or {}
        mat = getattr(o, "material", None)
        if isinstance(mat, dict):
            mat = str(mat.get("base") or mat.get("name") or "inline")
        mods = getattr(o, "modifiers", None) or []
        objs[o.id] = (getattr(o, "op", "add"), mat, len(mods), str(shape.get("type", "")), _multiplicity(mods))
    return {"objects": objs, "materials": set((getattr(scene, "materials", None) or {}).keys())}


def summarize_delta(stage: str, before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """Stage-specific one-liner from the scene delta; '' when nothing changed."""
    b, a = before["objects"], after["objects"]
    new_ids = [i for i in a if i not in b]
    removed = [i for i in b if i not in a]
    adds = [i for i in new_ids if a[i][0] == "add"]
    cuts = [i for i in new_ids if a[i][0] == "subtract"]
    remat = [i for i in a if i in b and a[i][1] != b[i][1]]
    remod = [i for i in a if i in b and a[i][2] != b[i][2]]
    reshaped = [i for i in a if i in b and a[i][3] != b[i][3]]
    new_mats = sorted(after["materials"] - before["materials"])
    parts: List[str] = []

    def kinds(ids: List[str], table: Tuple[Tuple[str, str], ...], default: str) -> "Counter[str]":
        c: Counter[str] = Counter()
        for i in ids:
            c[_kind_word(i, table, default)] += a[i][4]
        return c

    if stage == "blocking":
        if adds:
            parts.append(f"added {_plural(len(adds), 'solid')}")
        if cuts:
            parts.append(f"cut {_plural(len(cuts), 'void')}")
    elif stage == "materials":
        used = Counter(m for (_op, m, _n, _t, _k) in a.values() if m and m != "default")
        names = [m for m, _ in used.most_common(4)]
        if names:
            parts.append("/".join(names))
        if remat:
            parts.append(f"{len(remat)} objects repainted")
        elif new_mats:
            parts.append(f"{len(new_mats)} materials defined")
    elif stage == "decoration":
        if adds:
            parts.append("added " + _count(kinds(adds, _PROP_WORDS, "prop")))
        if cuts:
            parts.append(f"cut {_plural(len(cuts), 'opening')}")
    else:  # detailing / fix / edit
        if cuts:
            parts.append("carved " + _count(kinds(cuts, _OPENING_WORDS, "opening")))
        if adds:
            parts.append("added " + _count(kinds(adds, _PROP_WORDS, "detail")))
        if remod:
            parts.append(f"modifiers on {len(remod)}")
        if reshaped:
            parts.append(f"reshaped {len(reshaped)}")
        if remat:
            parts.append(f"repainted {len(remat)}")
    if removed:
        parts.append(f"removed {len(removed)}")
    return "; ".join(parts)


class Progress:
    """Per-turn chat progress: `say(tag, text)` prints `[cp·tag m:ss] text` through the bridge."""

    def __init__(self, ctx: Any, t0: Optional[float] = None, verbose: Optional[bool] = None, sink: Optional[Callable[[str], None]] = None):
        self.ctx = ctx
        self.t0 = t0 if t0 is not None else time.time()
        session = getattr(ctx, "session", None)
        self.verbose = bool(getattr(session, "verbose", False)) if verbose is None else verbose
        self.sink = sink
        self.lines: List[str] = []

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    def line(self, tag: str, text: Any, pct: Optional[int] = None) -> str:
        tag = TAGS.get(tag, tag)
        head = f"[cp·{tag} {pct}%]" if pct is not None else f"[cp·{tag} {fmt_elapsed(self.elapsed)}]"
        body = one_line(text)
        return (head + " " + body).strip() if body else head

    def say(self, tag: str, text: Any = "", pct: Optional[int] = None) -> str:
        msg = self.line(tag, text, pct)
        self.lines.append(msg)
        if self.sink is not None:
            self.sink(msg)
        else:
            call_tool(self.ctx, "say", {"text": msg})
        return msg

    def op(self, name: str, args: Dict[str, Any], result: Any) -> None:
        """Verbose-only line for one tool call (`add keep → added keep box 20x14x10`)."""
        if not self.verbose or name in ("finish", "say", "render", "lint", "describe", "select"):
            return
        session = getattr(self.ctx, "session", None)
        tag = TAGS.get(getattr(session, "stage", None) or "edit", "edit")
        target = args.get("id") or args.get("ids") or ""
        head = f"{name} {target}".strip()
        txt = one_line(result, 90)
        if txt.startswith("ERROR"):
            self.say(tag, f"{head} — {txt}")
        else:
            self.say(tag, f"{head}: {txt}" if txt else head)


def get_progress(ctx: Any) -> Optional[Progress]:
    p = getattr(ctx, "progress", None)
    return p if isinstance(p, Progress) else None


def new_progress(ctx: Any, t0: Optional[float] = None) -> Progress:
    p = Progress(ctx, t0=t0)
    try:
        ctx.progress = p
    except Exception:  # noqa: BLE001
        pass
    return p


def live_preview_enabled(session: Any) -> bool:
    """`/cp preview on|off` wins; otherwise env LIVE_PREVIEW (default true)."""
    v = getattr(session, "live_preview", None)
    if v is not None:
        return bool(v)
    return env_flag("LIVE_PREVIEW", True)
