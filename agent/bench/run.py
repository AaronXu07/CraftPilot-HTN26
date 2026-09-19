"""Bench harness (plan §8.3): build every prompt headless, render, score with the vision model.

    cd agent && ../.venv/bin/python -m bench.run [--only NAME] [--fast] [--mock] [--out DIR]

Writes bench/out/<timestamp>/<name>.png, report.md and report.json. With --mock a scripted builder
plays the LLM (smoke run; scores come from the scripted critic). Without --mock, Azure OpenAI is used
for building and for scoring (vision if the deployment supports it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

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


def run(items: List[Dict[str, Any]], out_dir: str, fast: bool, mock: bool) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    if mock:
        from bench.mock_builder import ScriptedBuilderLLM

        llm_factory = lambda: ScriptedBuilderLLM()  # noqa: E731
    else:
        from copilot.llm import AzureLLM

        shared = AzureLLM()
        llm_factory = lambda: shared  # noqa: E731
    results = []
    for item in items:
        name = item["name"]
        print(f"== {name}: {item['prompt'][:70]}…", flush=True)
        bridge = make_bridge()
        registry = make_registry(bridge)
        ctx = make_ctx(name, bridge, registry, out_dir, mock)
        llm = llm_factory()
        t0 = time.time()
        res = handle_chat(ctx, item["prompt"], llm=llm, fast=fast)
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
        }
        results.append(row)
        print(f"   {secs:.0f}s, {row['tool_calls']} tool calls, {row['objects']} objects, scores={scores}", flush=True)
        try:
            with open(os.path.join(out_dir, f"{name}.scene.json"), "w") as f:
                f.write(ctx.session.scene.to_json(indent=1))
        except Exception:  # noqa: BLE001
            pass
    means = [r["scores"].get("mean") for r in results if r["scores"].get("mean") is not None]
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "fast": fast, "mock": mock, "mean": round(sum(means) / len(means), 2) if means else None, "results": results}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report_md(report))
    print(f"\nmean score: {report['mean']}  →  {os.path.join(out_dir, 'report.md')}")
    return report


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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run a single prompt by name")
    ap.add_argument("--fast", action="store_true", help="skip critic rounds")
    ap.add_argument("--mock", action="store_true", help="use the scripted builder instead of Azure (smoke run)")
    ap.add_argument("--out", help="output directory (default bench/out/<timestamp>)")
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.json"))
    ap.add_argument("--every", type=float, default=0.0, help="repeat the run every N hours (plan §8.3: 4); 0 = once")
    args = ap.parse_args(argv)
    with open(args.prompts) as f:
        items = json.load(f)
    if args.only:
        items = [i for i in items if i["name"] == args.only]
        if not items:
            print(f"no prompt named {args.only}", file=sys.stderr)
            return 2
    while True:
        out_dir = args.out or os.path.join(HERE, "out", time.strftime("%Y%m%d_%H%M%S"))
        run(items, out_dir, fast=args.fast, mock=args.mock)
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
