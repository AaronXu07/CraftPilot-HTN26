"""Bench harness (plan §8.3): build every prompt headless, render, score with the vision model.

    cd agent && ../.venv/bin/python -m bench.run [--only NAME] [--quick] [--profile] [--jobs N] [--fast] [--mock] [--out DIR]
    cd agent && ../.venv/bin/python -m bench.run --profile --from bench/out/<dir>   # latency table of an old run
    cd agent && ../.venv/bin/python -m bench.run --rescore bench/out/<dir>          # re-judge rows whose scoring hit a 429
    cd agent && ../.venv/bin/python -m bench.run --compare DIR_A DIR_B              # per-prompt A/B score + latency table

Writes bench/out/<timestamp>/<name>.png, report.md and report.json. `--quick` runs the first
QUICK_N prompts (iterate on this; the full set costs real credits). `--profile` prints a per-stage
latency table (wall / LLM / engine seconds, calls, budget stops) after the run. With --mock a
scripted builder plays the LLM (smoke run; scores come from the scripted critic). Without --mock,
Azure OpenAI is used for building and for scoring (vision if the deployment supports it).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

QUICK_N = 5

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from copilot.llm import extract_json, image_content, single_call, text_content  # noqa: E402
from copilot.pipeline import handle_chat  # noqa: E402
from copilot.pipeline.common import call_tool  # noqa: E402
from copilot.pipeline.stages import load_prompt  # noqa: E402
from copilot.session import Session  # noqa: E402

SCORE_KEYS = ["silhouette", "detail", "materials", "fidelity"]


def make_bridge() -> Any:
    try:
        from mock_mod import MockBridge  # Track 2a

        return MockBridge()
    except Exception:  # noqa: BLE001
        from tests.fakes import FakeBridge

        return FakeBridge()


def make_registry(bridge: Any) -> Any:
    try:
        from copilot.engine.registry import Registry  # Track 4

        try:
            return Registry.load(bridge=None)
        except TypeError:
            return Registry.load()
    except Exception:  # noqa: BLE001
        from tests.fakes import FakeRegistry

        return FakeRegistry()


def make_ctx(name: str, bridge: Any, registry: Any, out_dir: str, mock: bool) -> Any:
    session = Session(f"bench_{name}", run_dir=os.path.join(out_dir, "runs", name))
    ctx: Any
    try:
        from copilot.tools.dispatch import ToolContext  # Track 2a (real engine; falls back to the test double)

        ctx = ToolContext(session, bridge, registry)
    except Exception:  # noqa: BLE001
        from tests.fakes import make_dispatch

        ctx = SimpleNamespace(session=session, bridge=bridge, registry=registry, dispatch=make_dispatch(bridge), log=None, llm=None, fast=None, tools=None, run_dir=session.run_dir)
    return ctx


def render_sheet(ctx: Any, path: str) -> Optional[Any]:
    r = call_tool(ctx, "render", {"views": ["contact"]})
    images = list(getattr(r, "images", None) or [])
    if not images:
        return None
    try:
        images[0].save(path)
    except Exception:  # noqa: BLE001
        return None
    return images[0]


def score_build(llm: Any, image: Any, item: Dict[str, Any], outline_text: str) -> Dict[str, Any]:
    """Ask the (vision) model for 1–10 scores on silhouette, detail, materials and fidelity."""
    rubric = load_prompt("critic").split("## Output")[0]
    sys_prompt = rubric + (
        "\n\n## Output — JSON only\n"
        '{"silhouette": 1-10, "detail": 1-10, "materials": 1-10, "fidelity": 1-10, "notes": "one sentence"}\n'
        "fidelity = how many of the must-have features are clearly present."
    )
    parts: List[Dict[str, Any]] = [text_content(f"Request: {item['prompt']}\nMust-have features: {', '.join(item.get('must_have', []))}\n\nScene outline:\n```\n{outline_text[:4000]}\n```")]
    if image is not None and getattr(llm, "supports_vision", False):
        parts.append(image_content(image))
    try:
        resp = single_call(llm, sys_prompt, parts, temperature=0.1, max_tokens=400)
        j = extract_json(resp.text) or {}
    except Exception as e:  # noqa: BLE001
        j = {"notes": f"scoring failed: {e}"}
    out: Dict[str, Any] = {}
    for k in SCORE_KEYS:
        try:
            out[k] = max(1, min(10, int(round(float(j.get(k))))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            out[k] = None
    out["notes"] = str(j.get("notes", ""))[:300]
    vals = [v for v in (out[k] for k in SCORE_KEYS) if v is not None]
    out["mean"] = round(sum(vals) / len(vals), 2) if vals else None
    return out


def run_one(item: Dict[str, Any], out_dir: str, llm: Any, fast: bool, mock: bool, profile_mode: bool = False) -> Dict[str, Any]:
    """Build, render and score one prompt; returns the report row."""
    name = item["name"]
    print(f"== {name}: {item['prompt'][:70]}…", flush=True)
    bridge = make_bridge()
    registry = make_registry(bridge)
    ctx = make_ctx(name, bridge, registry, out_dir, mock)
    t0 = time.time()
    try:
        res = handle_chat(ctx, item["prompt"], llm=llm, fast=fast)
    except Exception as e:  # noqa: BLE001 — one broken prompt must not kill the run
        res = SimpleNamespace(reply=f"ERROR: {type(e).__name__}: {e}", data={"error": str(e)})
    secs = time.time() - t0
    png = os.path.join(out_dir, f"{name}.png")
    image = render_sheet(ctx, png)
    outline_text = ctx.session.scene.describe()
    if mock:
        crit = (res.data or {}).get("final_score")
        scores = {k: crit for k in SCORE_KEYS}
        scores.update({"notes": "mock run (scripted critic)", "mean": crit})
    else:
        scores = score_build(llm, image, item, outline_text)
    log_records = getattr(getattr(ctx, "log", None), "records", []) or []
    tool_calls = sum(1 for r in log_records if not str(r.get("name", "")).startswith("__"))
    profile = (res.data or {}).get("profile") or profile_from_records(log_records, secs)
    row = {
        "name": name,
        "prompt": item["prompt"],
        "seconds": round(secs, 1),
        "tool_calls": tool_calls or (res.data or {}).get("tool_calls", 0),
        "objects": len(ctx.session.scene.objects),
        "blocks": (res.data or {}).get("blocks"),
        "final_critic": (res.data or {}).get("final_score"),
        "scores": scores,
        "reply": res.reply[:300],
        "png": png if image is not None else None,
        "profile": profile,
        "skipped": (res.data or {}).get("skipped", []),
    }
    print(f"   {name}: {secs:.0f}s, {row['tool_calls']} tool calls, {row['objects']} objects, scores={scores}", flush=True)
    if profile_mode:
        print("   " + profile_line(profile), flush=True)
    try:
        with open(os.path.join(out_dir, f"{name}.scene.json"), "w") as f:
            f.write(ctx.session.scene.to_json(indent=1))
    except Exception:  # noqa: BLE001
        pass
    return row


def run(items: List[Dict[str, Any]], out_dir: str, fast: bool, mock: bool, profile_mode: bool = False, jobs: int = 1) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    if mock:
        from bench.mock_builder import ScriptedBuilderLLM

        llm_factory = lambda: ScriptedBuilderLLM()  # noqa: E731
    else:
        from copilot.llm import AzureLLM

        shared = AzureLLM()
        llm_factory = lambda: shared  # noqa: E731
    if jobs > 1 and len(items) > 1:
        # prompts are independent (own session/bridge/registry); the Azure client is thread-safe and
        # the pipeline already runs on a thread pool under /chat — a full 20-prompt run in ~4 slots
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(lambda it: run_one(it, out_dir, llm_factory(), fast, mock, profile_mode), items))
    else:
        results = [run_one(item, out_dir, llm_factory(), fast, mock, profile_mode) for item in items]
    report: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "fast": fast, "mock": mock, "jobs": jobs, "mean": None, "results": results}
    _write_report(out_dir, report)
    print(f"\nmean score: {report['mean']}  →  {os.path.join(out_dir, 'report.md')}")
    if profile_mode:
        print()
        print(profile_table(report))
    return report


# ----------------------------------------------------------------------------------------------
# latency profile (T2)
# ----------------------------------------------------------------------------------------------
STAGE_COLS = ["interpret", "blocking", "detailing", "materials", "decoration", "critic", "fix", "place"]


def profile_from_records(records: List[Dict[str, Any]], wall_s: float) -> Dict[str, Any]:
    """Rebuild a turn profile from run-log records (for runs made before `__summary__` lines existed)."""
    rows: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for r in records:
        name = str(r.get("name", ""))
        if name == "__profile__" and isinstance(r.get("profile"), dict):
            p = r["profile"]
            key = p["stage"] + f"#{len(order)}"
            rows[key] = dict(p)
            order.append(key)
            continue
        stage = r.get("stage")
        if stage in (None, "turn", "error"):
            continue
        if name in ("__summary__", "__reply__"):
            continue
        key = str(stage)
        if key not in rows:
            rows[key] = {"stage": key, "wall_ms": 0, "llm_ms": 0, "engine_ms": 0, "llm_calls": 0, "tool_calls": 0, "stopped": "", "note": "", "_t0": r.get("t"), "_t1": r.get("t")}
            order.append(key)
        row = rows[key]
        if "_t0" in row:
            row["_t1"] = r.get("t", row["_t1"])
            row["wall_ms"] = int(((row["_t1"] or 0) - (row["_t0"] or 0)) * 1000)
        if name == "__llm__":
            row["llm_ms"] += int(r.get("ms", 0))
            row["llm_calls"] += 1
        elif name == "__critic__":
            row["note"] = "critic"
        elif not name.startswith("__"):
            row["engine_ms"] += int(r.get("ms", 0))
            row["tool_calls"] += 1
    if any("__profile__" == str(r.get("name")) for r in records):
        # profile rows are authoritative; drop the grouped-by-stage duplicates
        order = [k for k in order if "#" in k]
    stages = [{k: v for k, v in rows[key].items() if not k.startswith("_")} for key in order]
    return {
        "wall_ms": int(wall_s * 1000),
        "llm_ms": sum(r["llm_ms"] for r in stages),
        "engine_ms": sum(r["engine_ms"] for r in stages),
        "llm_calls": sum(r["llm_calls"] for r in stages),
        "tool_calls": sum(r["tool_calls"] for r in stages),
        "stages": stages,
    }


def _col_of(stage: str) -> str:
    if stage.startswith("critic"):
        return "critic"
    if stage.startswith("fix"):
        return "fix"
    return stage if stage in STAGE_COLS else "other"


def profile_line(profile: Dict[str, Any]) -> str:
    cols = _stage_seconds(profile)
    head = f"wall {profile['wall_ms'] / 1000:.1f}s  llm {profile['llm_ms'] / 1000:.1f}s/{profile['llm_calls']} calls  engine {profile['engine_ms'] / 1000:.1f}s/{profile['tool_calls']} tools"
    return head + "  |  " + "  ".join(f"{c} {cols[c]:.0f}" for c in STAGE_COLS + ["other"] if cols.get(c))


def _stage_seconds(profile: Dict[str, Any]) -> Dict[str, float]:
    cols: Dict[str, float] = {}
    for r in profile.get("stages", []):
        cols[_col_of(str(r["stage"]))] = cols.get(_col_of(str(r["stage"])), 0.0) + r.get("wall_ms", 0) / 1000.0
    return cols


def _stops(profile: Dict[str, Any]) -> str:
    out = []
    for r in profile.get("stages", []):
        st = r.get("stopped") or ""
        if st in ("budget", "max_calls", "skipped", "error"):
            out.append(f"{r['stage']}:{st}")
    return " ".join(out)


def profile_table(report: Dict[str, Any]) -> str:
    """Markdown latency table: one row per prompt, per-stage wall seconds, then median/max lines."""
    rows = [r for r in report["results"] if r.get("profile")]
    lines = ["| build | wall s | llm s | llm calls | engine s | tools | " + " | ".join(STAGE_COLS) + " | stops |", "|" + "---|" * (7 + len(STAGE_COLS))]
    walls: List[float] = []
    for r in rows:
        p = r["profile"]
        cols = _stage_seconds(p)
        wall = p["wall_ms"] / 1000.0
        walls.append(wall)
        lines.append(f"| {r['name']} | {wall:.0f} | {p['llm_ms'] / 1000:.0f} | {p['llm_calls']} | {p['engine_ms'] / 1000:.1f} | {p['tool_calls']} | " + " | ".join(f"{cols.get(c, 0):.0f}" for c in STAGE_COLS) + f" | {_stops(p)} |")
    if walls:
        lines.append("")
        lines.append(f"wall s: median {statistics.median(walls):.0f}, max {max(walls):.0f}, mean {statistics.mean(walls):.0f}  (target: median ≤ 150, max ≤ 300)")
    return "\n".join(lines)


def profile_existing(out_dir: str) -> str:
    """Latency table for a finished run directory (uses report.json + the per-prompt run logs)."""
    with open(os.path.join(out_dir, "report.json")) as f:
        report = json.load(f)
    for r in report["results"]:
        if r.get("profile"):
            continue
        recs: List[Dict[str, Any]] = []
        for path in sorted(glob.glob(os.path.join(out_dir, "runs", r["name"], "*.jsonl"))):
            with open(path) as f:
                recs += [json.loads(line) for line in f if line.strip()]
        r["profile"] = profile_from_records(recs, float(r.get("seconds", 0)))
    return profile_table(report)


def report_md(report: Dict[str, Any]) -> str:
    lines = [f"# Bench report {report['timestamp']}", "", f"mode: {'mock' if report['mock'] else 'azure'}{' fast' if report['fast'] else ''} · mean score **{report['mean']}**", ""]
    lines.append("| build | silhouette | detail | materials | fidelity | mean | tool calls | objects | blocks | wall s | notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in report["results"]:
        s = r["scores"]
        lines.append(f"| {r['name']} | {s.get('silhouette')} | {s.get('detail')} | {s.get('materials')} | {s.get('fidelity')} | {s.get('mean')} | {r['tool_calls']} | {r['objects']} | {r['blocks']} | {r['seconds']} | {s.get('notes', '')[:80]} |")
    lines.append("")
    for r in report["results"]:
        if r.get("png"):
            lines.append(f"## {r['name']}\n\n![{r['name']}]({os.path.basename(r['png'])})\n\n{r['reply']}\n")
    return "\n".join(lines)


def _write_report(out_dir: str, report: Dict[str, Any]) -> None:
    means = [r["scores"].get("mean") for r in report["results"] if r["scores"].get("mean") is not None]
    report["mean"] = round(sum(means) / len(means), 2) if means else None
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report_md(report))


def rescore(out_dir: str, prompts_path: str, force: bool = False) -> Dict[str, Any]:
    """Re-score rows whose scoring failed (429 on the judge call) from the saved `<name>.png` and
    `<name>.scene.json`; one judge call per row, the build itself is not re-run. Rewrites report.json/md."""
    from PIL import Image

    from copilot.engine.scene import Scene
    from copilot.llm import AzureLLM

    with open(os.path.join(out_dir, "report.json")) as f:
        report = json.load(f)
    with open(prompts_path) as f:
        items = {i["name"]: i for i in json.load(f)}
    llm = AzureLLM()
    for r in report["results"]:
        if r["scores"].get("mean") is not None and not force:
            continue
        item = items.get(r["name"]) or {"prompt": r["prompt"], "must_have": []}
        png = os.path.join(out_dir, f"{r['name']}.png")
        image = Image.open(png) if os.path.exists(png) else None
        try:
            with open(os.path.join(out_dir, f"{r['name']}.scene.json")) as f:
                outline_text = Scene.from_json(f.read()).describe()
        except Exception:  # noqa: BLE001
            outline_text = ""
        r["scores"] = score_build(llm, image, item, outline_text)
        print(f"   {r['name']}: scores={r['scores']}", flush=True)
    _write_report(out_dir, report)
    print(f"\nmean score: {report['mean']}  →  {os.path.join(out_dir, 'report.md')}")
    return report


def compare(a_dir: str, b_dir: str) -> str:
    """Per-prompt A/B table of two run directories: score, wall seconds and their deltas."""
    with open(os.path.join(a_dir, "report.json")) as f:
        a = json.load(f)
    with open(os.path.join(b_dir, "report.json")) as f:
        b = json.load(f)
    ra = {r["name"]: r for r in a["results"]}
    rb = {r["name"]: r for r in b["results"]}
    lines = ["| build | A score | B score | Δ | A s | B s | B skipped/stops |", "|---|---|---|---|---|---|---|"]
    deltas: List[float] = []
    for name in list(ra) + [n for n in rb if n not in ra]:
        x, y = ra.get(name), rb.get(name)
        sa = (x or {}).get("scores", {}).get("mean")
        sb = (y or {}).get("scores", {}).get("mean")
        d = round(sb - sa, 2) if sa is not None and sb is not None else None
        if d is not None:
            deltas.append(d)
        stops = _stops((y or {}).get("profile") or {}) if y else ""
        lines.append(f"| {name} | {sa} | {sb} | {'' if d is None else f'{d:+.2f}'} | {(x or {}).get('seconds', '')} | {(y or {}).get('seconds', '')} | {stops} |")

    def _med(rep: Dict[str, Any]) -> str:
        secs = [r["seconds"] for r in rep["results"]]
        return f"median {statistics.median(secs):.0f} s, max {max(secs):.0f} s" if secs else "—"

    lines.append("")
    lines.append(f"mean A {a.get('mean')} → B {b.get('mean')} ({len(deltas)} paired; Δ mean {sum(deltas) / len(deltas):+.2f})" if deltas else f"mean A {a.get('mean')} → B {b.get('mean')}")
    lines.append(f"latency A: {_med(a)}; B: {_med(b)}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run a single prompt by name")
    ap.add_argument("--quick", action="store_true", help=f"run only the first {QUICK_N} prompts")
    ap.add_argument("--profile", action="store_true", help="print a per-stage latency table")
    ap.add_argument("--from", dest="from_dir", help="with --profile: tabulate an existing run directory instead of running")
    ap.add_argument("--fast", action="store_true", help="skip critic rounds")
    ap.add_argument("--mock", action="store_true", help="use the scripted builder instead of Azure (smoke run)")
    ap.add_argument("--out", help="output directory (default bench/out/<timestamp>)")
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.json"))
    ap.add_argument("--every", type=float, default=0.0, help="repeat the run every N hours (plan §8.3: 4); 0 = once")
    ap.add_argument("--jobs", type=int, default=1, help="build N prompts concurrently (T4: a full run in ~4 slots)")
    ap.add_argument("--rescore", metavar="DIR", help="re-score the rows of an existing run whose judge call failed (no rebuild)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="print a per-prompt A/B table of two run directories")
    args = ap.parse_args(argv)
    if args.from_dir:
        print(profile_existing(args.from_dir))
        return 0
    if args.rescore:
        rescore(args.rescore, args.prompts)
        return 0
    if args.compare:
        print(compare(*args.compare))
        return 0
    with open(args.prompts) as f:
        items = json.load(f)
    if args.quick:
        items = items[:QUICK_N]
    if args.only:
        items = [i for i in items if i["name"] == args.only]
        if not items:
            print(f"no prompt named {args.only}", file=sys.stderr)
            return 2
    while True:
        out_dir = args.out or os.path.join(HERE, "out", time.strftime("%Y%m%d_%H%M%S"))
        run(items, out_dir, fast=args.fast, mock=args.mock, profile_mode=args.profile, jobs=max(1, args.jobs))
        if args.every <= 0:
            return 0
        print(f"next bench run in {args.every:g} h (ctrl-c to stop)")
        try:
            time.sleep(args.every * 3600)
        except KeyboardInterrupt:
            return 0
        args.out = None  # fresh timestamped directory each iteration


if __name__ == "__main__":
    sys.exit(main())
