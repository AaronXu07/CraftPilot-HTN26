"""Tool dispatch: LLM tool call -> engine/session/bridge. CONTRACTS.md §8.

`dispatch(ctx, name, args)` never raises for user-level errors; it returns `ToolResult("ERROR: ...")`
so the model gets immediate, actionable feedback. Engine modules (Track 3/4) are imported lazily.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..engine.scene import OPS, READ_ONLY_OPS, SceneError
from ..jobs import JobCancelled
from .schemas import AGENT_TOOL_NAMES, HISTORY_TOOL_NAMES, TOOLS_BY_NAME

MAX_RESULT_CHARS = 4000
SCRIPT_TIMEOUT_S = 10.0
DEFAULT_RENDER_SIZE = 1024
RASTER_CACHE_KEY = ("raster_cache", "__pinned__")


@dataclass
class ToolResult:
    text: str
    images: List[Any] = field(default_factory=list)  # PIL.Image.Image
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


@dataclass
class ToolContext:
    session: Any
    bridge: Any
    registry: Any = None
    log: Optional[Callable[[Dict[str, Any]], None]] = None
    run_dir: Optional[str] = None
    job: Any = None  # copilot.jobs.Job when running as a background chat job (cancel / time budget)
    budget: Any = None  # copilot.pipeline.budget.Budget for the current turn (stage deadlines, profile)

    def emit(self, event: Dict[str, Any]) -> None:
        if self.log:
            try:
                self.log(event)
            except Exception:  # noqa: BLE001
                pass


def truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 60] + f"\n... [truncated {len(text) - limit + 60} chars; ask for a narrower selection]"


# ----------------------------------------------------------------------------------------------
# Engine pipeline: rasterize -> fit -> resolve, cached by scene hash on the session
# ----------------------------------------------------------------------------------------------
def build_all(ctx: ToolContext) -> Dict[str, Any]:
    """Run rasterize -> fit_surface -> resolve for the session's scene (cached by scene hash).

    Returns {"raster": RasterResult, "fit": FitResult, "block_map": BlockMap, "warnings": [str], "timings": {...}}.
    Raises SceneError with a readable message when an engine module is missing or the scene is empty.
    """
    session = ctx.session
    cached = session.cached("build")
    if cached is not None:
        return cached
    # T2: read-only tool calls (render + lint) may run concurrently; build the scene once
    lock = getattr(session, "build_lock", None)
    if lock is None:
        return _build_all(ctx)
    with lock:
        cached = session.cached("build")
        if cached is not None:
            return cached
        return _build_all(ctx)


def _build_all(ctx: ToolContext) -> Dict[str, Any]:
    session = ctx.session
    scene = session.scene
    if not scene.objects:
        raise SceneError("the scene is empty; add objects first")
    try:
        from ..engine.raster import rasterize
    except ImportError as e:
        raise SceneError(f"engine.raster is not available yet: {e}")
    t0 = time.time()
    raster_cache = session.cache.setdefault(RASTER_CACHE_KEY, {})
    raster = rasterize(scene, pad=1, cache=raster_cache)
    t1 = time.time()
    warnings: List[str] = []
    try:
        from ..engine.fit import fit_surface
        from ..engine.materials import material_fit_modes

        fit = fit_surface(raster, material_fit_modes(scene))
    except ImportError as e:
        warnings.append(f"fit disabled ({e})")
        from ..engine.coretypes import FitResult

        fit = FitResult.full_blocks(raster)
    t2 = time.time()
    try:
        from ..engine.resolver import resolve

        block_map = resolve(raster, fit, scene, ctx.registry, seed=0, warnings=warnings)
    except ImportError as e:
        warnings.append(f"resolver unavailable ({e}); using stone")
        block_map = _fallback_block_map(raster)
    t3 = time.time()
    result = {
        "raster": raster,
        "fit": fit,
        "block_map": block_map,
        "warnings": warnings,
        "timings": {"raster": round(t1 - t0, 3), "fit": round(t2 - t1, 3), "resolve": round(t3 - t2, 3)},
    }
    session.put_cache("build", result)
    ctx.emit({"event": "build", "scene_hash": session.scene_hash(), "blocks": len(block_map), "timings": result["timings"]})
    return result


def _fallback_block_map(raster) -> Dict[Any, str]:
    import numpy as np

    out: Dict[Any, str] = {}
    idx = np.argwhere(raster.material > 0)
    for i, j, k in idx:
        out[raster.index_to_world(i, j, k)] = "minecraft:stone"
    out.update(raster.props)
    return out


def _color_fn(ctx: ToolContext) -> Callable[[str], Any]:
    reg = ctx.registry
    if reg is not None and hasattr(reg, "color"):
        return reg.color
    return lambda state: (150, 150, 150)


def build_summary(ctx: ToolContext, build: Dict[str, Any]) -> str:
    """Short text summary of a build: bbox, counts, top blocks, fit counts, warnings."""
    from collections import Counter

    bm = build["block_map"]
    if not bm:
        return "build is empty (0 blocks)"
    xs = [p[0] for p in bm]
    ys = [p[1] for p in bm]
    zs = [p[2] for p in bm]
    top = Counter(s.split("[")[0].replace("minecraft:", "") for s in bm.values()).most_common(5)
    parts = [
        f"{len(bm)} blocks, bbox x {min(xs)}..{max(xs)} y {min(ys)}..{max(ys)} z {min(zs)}..{max(zs)}",
        "top blocks: " + ", ".join(f"{k} {v}" for k, v in top),
    ]
    fit = build.get("fit")
    if fit is not None and hasattr(fit, "counts"):
        c = fit.counts()
        parts.append("fit: " + ", ".join(f"{k} {v}" for k, v in c.items() if v))
    if build.get("warnings"):
        parts.append("warnings: " + "; ".join(build["warnings"][:5]))
    return "\n".join(parts)


# ----------------------------------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------------------------------
def dispatch(ctx: ToolContext, name: str, args: Optional[Dict[str, Any]] = None) -> ToolResult:
    """Execute one tool call. Returns ToolResult; user errors become 'ERROR: ...' text."""
    args = dict(args or {})
    t0 = time.time()
    try:
        if name in OPS:
            res = _scene_op(ctx, name, args)
        elif name in HISTORY_TOOL_NAMES:
            res = _history(ctx, name, args)
        elif name in AGENT_TOOL_NAMES:
            res = _AGENT_TOOLS[name](ctx, args)
        else:
            res = ToolResult(f"ERROR: unknown tool {name!r}. Available: {', '.join(TOOLS_BY_NAME)}")
    except SceneError as e:
        res = ToolResult(f"ERROR: {e}")
    except JobCancelled:
        raise  # unwind the job; not a tool error
    except Exception as e:  # noqa: BLE001 - never crash the tool loop
        res = ToolResult(f"ERROR: {name} failed: {type(e).__name__}: {e}")
    res.text = truncate(res.text)
    ctx.emit({"event": "tool", "name": name, "args": _short(args), "result": res.text[:500], "images": len(res.images), "ms": round((time.time() - t0) * 1000)})
    try:
        ctx.session.tool_calls_this_stage += 1
    except Exception:  # noqa: BLE001
        pass
    return res


def _short(args: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in args.items():
        s = json.dumps(v, default=str)
        out[k] = v if len(s) < 300 else s[:300] + "..."
    return out


def _scene_op(ctx: ToolContext, name: str, args: Dict[str, Any]) -> ToolResult:
    kwargs = dict(args)
    if name in ("set_shape", "set_modifier"):
        params = kwargs.pop("params", None) or {}
        if not isinstance(params, dict):
            return ToolResult("ERROR: params must be an object")
        kwargs.update(params)
    msg = ctx.session.apply(name, **kwargs)
    if name not in READ_ONLY_OPS:
        warn = _scene_lint_warnings(ctx)
        if warn:
            msg += "\n" + warn
    return ToolResult(msg)


def _scene_lint_warnings(ctx: ToolContext) -> str:
    try:
        from ..engine.lint import lint_scene_only  # type: ignore
    except ImportError:
        return ""
    try:
        findings = lint_scene_only(ctx.session.scene)
    except Exception:  # noqa: BLE001
        return ""
    if not findings:
        return ""
    return "\n".join("lint " + str(f) for f in list(findings)[:5])


def _history(ctx: ToolContext, name: str, args: Dict[str, Any]) -> ToolResult:
    s = ctx.session
    if name == "undo":
        return ToolResult(s.undo(int(args.get("n", 1))))
    if name == "redo":
        return ToolResult(s.redo(int(args.get("n", 1))))
    if name == "snapshot":
        return ToolResult(s.snapshot(str(args.get("label", "snapshot"))))
    return ToolResult(s.restore(str(args.get("label", ""))))


# ----------------------------------------------------------------------------------------------
# agent tools
# ----------------------------------------------------------------------------------------------
def _run_script(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    """Run the script in a sandboxed subprocess and replay its ops through the session."""
    src = args.get("python") or args.get("code") or ""
    if not isinstance(src, str) or not src.strip():
        return ToolResult("ERROR: run_script needs `python` source text")
    payload = json.dumps({"scene": ctx.session.scene.to_dict(), "script": src})
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {"PYTHONPATH": root, "PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-s", "-B", os.path.join(root, "copilot", "tools", "script_runner.py")],
            input=payload,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_S,
            cwd=root,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(f"ERROR: script exceeded the {SCRIPT_TIMEOUT_S:.0f} s limit and was killed; no ops were applied. Avoid unbounded loops.")
    if proc.returncode != 0 or not proc.stdout.strip():
        err = (proc.stderr or "").strip().splitlines()
        return ToolResult("ERROR: script runner crashed: " + (err[-1] if err else f"exit {proc.returncode}"))
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except ValueError:
        return ToolResult("ERROR: script runner returned invalid output")
    ops = out.get("ops", [])
    applied = 0
    last = ""
    for op_name, kwargs, _msg in ops:
        try:
            last = ctx.session.apply(op_name, **kwargs)
            applied += 1
        except SceneError as e:  # the sandbox already validated; this is defensive
            return ToolResult(f"ERROR: replaying op {applied + 1} ({op_name}) failed: {e}. {applied} ops applied.")
    text = f"script performed {applied} ops"
    if last:
        text += f"; last: {last}"
    if out.get("error"):
        text += f"\nscript error after {applied} ops: {out['error']}"
    printed = (out.get("output") or "").strip()
    if printed:
        text += "\noutput:\n" + printed[:2000]
    if applied:
        warn = _scene_lint_warnings(ctx)
        if warn:
            text += "\n" + warn
    return ToolResult(text, data={"ops": applied, "error": out.get("error")})


def _render(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    build = build_all(ctx)
    bm = build["block_map"]
    views = args.get("views") or ["contact"]
    if isinstance(views, str):
        views = [views]
    cut = args.get("cutaway")
    cutaway = None
    if isinstance(cut, dict) and cut.get("axis"):
        cutaway = (str(cut["axis"]), float(cut.get("at", 0)))
    size = int(args.get("size", DEFAULT_RENDER_SIZE))
    images: List[Any] = []
    summary = build_summary(ctx, build)
    try:
        from ..engine import render as R
    except ImportError as e:
        return ToolResult(summary + f"\n(renderer unavailable: {e})")
    color_fn = _color_fn(ctx)
    cache_key = "render:" + json.dumps([views, cutaway, size])
    cached = ctx.session.cached(cache_key)
    if cached is not None:
        images = list(cached)
    else:
        for v in views:
            if v == "contact":
                img = R.contact_sheet(bm, color_fn, views=("iso", "front", "top"), cutaway=cutaway, size=size)
            else:
                img = R.render_blocks(bm, color_fn, view=v, size=size, cutaway=cutaway)
            images.append(img)
        ctx.session.put_cache(cache_key, list(images))
    text = summary
    if args.get("slice_y") is not None:
        try:
            text += "\nplan at y=%d:\n" % int(args["slice_y"]) + R.ascii_slice(bm, y=int(args["slice_y"]))
        except Exception as e:  # noqa: BLE001
            text += f"\n(slice failed: {e})"
    paths = _save_images(ctx, images, "render")
    if paths:
        text += "\nimages: " + ", ".join(paths)
    return ToolResult(text, images=images, data={"paths": paths, "views": views})


def _save_images(ctx: ToolContext, images: List[Any], prefix: str) -> List[str]:
    run_dir = ctx.run_dir or getattr(ctx.session, "run_dir", None)
    if not run_dir or not images:
        return []
    try:
        os.makedirs(run_dir, exist_ok=True)
        paths = []
        stamp = time.strftime("%H%M%S")
        for i, img in enumerate(images):
            p = os.path.join(run_dir, f"{prefix}_{getattr(ctx.session, 'turn', 0)}_{stamp}_{i}.png")
            img.save(p)
            paths.append(p)
        return paths
    except Exception:  # noqa: BLE001
        return []


def _lint(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    try:
        from ..engine.lint import lint
    except ImportError as e:
        return ToolResult(f"linter unavailable ({e})")
    build = build_all(ctx)
    findings = lint(ctx.session.scene, build["raster"], build["fit"], build["block_map"], ctx.registry)
    if not findings:
        return ToolResult("lint: no findings")
    lines = [str(f) for f in findings]
    return ToolResult(f"lint: {len(lines)} finding(s)\n" + "\n".join(lines), data={"count": len(lines)})


def _place(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    from ..placement import place_scene

    build = build_all(ctx)
    mode = str(args.get("mode", "diff"))
    if mode not in ("diff", "full"):
        return ToolResult("ERROR: mode must be diff or full")
    animate = bool(args.get("animate", True))
    on_progress = None
    progress = getattr(ctx, "progress", None)
    if args.get("report") and progress is not None and hasattr(progress, "say"):
        on_progress = lambda pct: progress.say("build", pct=int(pct))  # noqa: E731
    msg = place_scene(ctx.session, ctx.bridge, build["block_map"], mode=mode, animate=animate, on_progress=on_progress)
    return ToolResult(msg)


def _undo_world(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    from ..placement import undo_world

    return ToolResult(undo_world(ctx.session, ctx.bridge))


def _export_schematic(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    name = str(args.get("name") or ctx.session.scene.name or "build")
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "build"
    build = build_all(ctx)
    os.makedirs("out", exist_ok=True)
    path = os.path.join("out", f"{name}.litematic")
    try:
        from ..engine.schematic import export_litematic
    except ImportError as e:
        return ToolResult(f"ERROR: schematic export unavailable ({e})")
    export_litematic(build["block_map"], name, path)
    return ToolResult(f"exported {len(build['block_map'])} blocks to {os.path.abspath(path)}", data={"path": path})


def _materials_list(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    build = build_all(ctx)
    try:
        from ..engine.schematic import materials_list_text

        return ToolResult(materials_list_text(build["block_map"]))
    except ImportError:
        from collections import Counter

        c = Counter(s.split("[")[0] for s in build["block_map"].values())
        lines = [f"{k}: {v} ({v // 64} stacks + {v % 64})" for k, v in c.most_common()]
        return ToolResult("\n".join(lines))


def _get_player(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    p = ctx.bridge.player()
    pos = p.get("pos", [0, 0, 0])
    look = p.get("looking_at")
    text = f"{p.get('name', 'player')} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) facing {p.get('facing')} (yaw {p.get('yaw', 0):.0f}, pitch {p.get('pitch', 0):.0f})"
    if look:
        text += f"; looking at {look.get('block')} at {tuple(look.get('pos', []))}"
    return ToolResult(text, data=p)


def _say(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    text = " ".join(str(args.get("text", "")).split())
    if not text:
        return ToolResult("ERROR: say needs text")
    progress = getattr(ctx, "progress", None)
    if not text.startswith("[cp") and progress is not None and hasattr(progress, "line"):
        # a model-originated say(): give it the same `[cp·stage m:ss]` tag as the pipeline's lines
        text = progress.line(getattr(ctx.session, "stage", None) or "edit", text)
    ctx.bridge.say(text[:400])
    ctx.session.add_chat("assistant", text[:400])
    return ToolResult("said: " + text[:400])


def _search_blocks(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    q = str(args.get("query", "")).strip()
    limit = int(args.get("limit", 20))
    if ctx.registry is None:
        return ToolResult("ERROR: block registry not loaded")
    hits = ctx.registry.search(q, limit=limit)
    if not hits:
        return ToolResult(f"no blocks match {q!r}")
    lines = []
    for h in hits:
        if isinstance(h, str):
            props = ""
            try:
                bd = ctx.registry.blocks.get(h) or ctx.registry.blocks.get("minecraft:" + h)
                if bd is not None and getattr(bd, "properties", None):
                    props = " [" + ", ".join(f"{k}={'|'.join(v)}" for k, v in bd.properties.items()) + "]"
            except Exception:  # noqa: BLE001
                props = ""
            lines.append(h + props)
        else:
            lines.append(str(h))
    return ToolResult("\n".join(lines))


def _nearest_block(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    rgb = args.get("rgb")
    if not rgb or len(rgb) != 3:
        return ToolResult("ERROR: rgb must be [r, g, b]")
    if ctx.registry is None:
        return ToolResult("ERROR: block registry not loaded")
    res = ctx.registry.nearest(tuple(float(v) for v in rgb), category=args.get("category"))
    return ToolResult(f"nearest block to rgb{tuple(int(v) for v in rgb)}: {res}")


def _set_brief(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    brief = args.get("brief")
    if not isinstance(brief, dict):
        return ToolResult("ERROR: brief must be an object")
    ctx.session.brief = brief
    return ToolResult("brief stored: " + json.dumps(brief)[:1500], data={"brief": brief})


def _finish(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    summary = str(args.get("summary", "done")).strip()
    return ToolResult(summary, data={"finished": True})


_AGENT_TOOLS: Dict[str, Callable[[ToolContext, Dict[str, Any]], ToolResult]] = {
    "run_script": _run_script,
    "render": _render,
    "lint": _lint,
    "place": _place,
    "undo_world": _undo_world,
    "export_schematic": _export_schematic,
    "materials_list": _materials_list,
    "get_player": _get_player,
    "say": _say,
    "search_blocks": _search_blocks,
    "nearest_block": _nearest_block,
    "set_brief": _set_brief,
    "finish": _finish,
}
