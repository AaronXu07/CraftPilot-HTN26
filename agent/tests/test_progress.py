"""T3: tagged progress lines, live preview, animated final placement, verbose toggle, final reply."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from bench.mock_builder import ScriptedBuilderLLM
from copilot.engine.registry import Registry
from copilot.engine.scene import Scene
from copilot.pipeline import handle_chat
from copilot.pipeline.orchestrator import handle_meta
from copilot.pipeline.progress import Progress, fmt_elapsed, live_preview_enabled, scene_snapshot, summarize_delta
from copilot.placement import place_scene, wait_for_placement
from copilot.session import Session
from copilot.tools.dispatch import ToolContext, dispatch
from mock_mod import MockBridge
from tests.fakes import FakeCtx

TAG_RE = re.compile(r"^\[cp·(?P<tag>[a-z]+) (?P<t>\d+:\d\d|\d+%)\](?: (?P<body>.*))?$")

_REGISTRY: Any = None


def registry() -> Any:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry.load(bridge=None)
    return _REGISTRY


def real_ctx(tmp_path, bridge=None) -> ToolContext:
    session = Session("tester", run_dir=str(tmp_path))
    return ToolContext(session=session, bridge=bridge or MockBridge(), registry=registry(), run_dir=str(tmp_path))


def tags(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        m = TAG_RE.match(ln)
        out.append(m.group("tag") if m else "?")
    return out


# -- unit ---------------------------------------------------------------------------------------
def test_fmt_elapsed_and_line_format():
    assert fmt_elapsed(0) == "0:00" and fmt_elapsed(64.9) == "1:04" and fmt_elapsed(-3) == "0:00"
    said: List[str] = []
    p = Progress(ctx=FakeCtx(), t0=0.0, verbose=False, sink=said.append)
    p.t0 = __import__("time").time() - 31
    line = p.say("blocking", "added 7 solids")
    assert line.startswith("[cp·block 0:31] added 7 solids") and said == [line]
    assert p.say("build", pct=50) == "[cp·build 50%]"
    assert p.say("critic", "7/10 — fixing: blank east façade (rule P8)").startswith("[cp·critic 0:31] 7/10")
    long = p.say("detailing", "x " * 300)
    assert len(long) <= 220 and long.endswith("…")
    assert "\n" not in p.say("materials", "a\nb\n{\"json\": 1}")


def test_summarize_delta_per_stage():
    s = Scene(name="t")
    from copilot.engine.scene import apply_op

    before = scene_snapshot(s)

    s, _ = apply_op(s, "add", id="keep", shape={"type": "box", "size": [20, 14, 10]}, material="stone")
    s, _ = apply_op(s, "add", id="tower_ne", shape={"type": "cylinder", "radius": 4, "height": 24})
    s, _ = apply_op(s, "add", id="tower_nw", shape={"type": "cylinder", "radius": 4, "height": 24})
    assert summarize_delta("blocking", before, scene_snapshot(s)) == "added 3 solids"
    mid = scene_snapshot(s)
    s, _ = apply_op(s, "add", id="win_cut", shape={"type": "box", "size": [1, 2, 2]}, op="subtract", modifiers=[{"type": "array", "count": 7, "offset": [3, 0, 0]}, {"type": "mirror", "axis": "x"}])
    s, _ = apply_op(s, "add", id="gate_arch", shape={"type": "box", "size": [4, 6, 3]}, op="subtract")
    s, _ = apply_op(s, "add", id="merlons", shape={"type": "box", "size": [1, 1, 1]}, modifiers=[{"type": "array", "count": 12, "offset": [2, 0, 0]}])
    txt = summarize_delta("detailing", mid, scene_snapshot(s))
    assert txt == "carved 14 windows, 1 arch; added 12 battlements"
    mid = scene_snapshot(s)
    s, _ = apply_op(s, "define_material", name="stone_wall", spec={"base": "stone_bricks"})
    s, _ = apply_op(s, "define_material", name="slate_roof", spec={"base": "deepslate_tiles"})
    s, _ = apply_op(s, "set_material", ids="keep", material="stone_wall")
    s, _ = apply_op(s, "set_material", ids="tower_ne", material="slate_roof")
    txt = summarize_delta("materials", mid, scene_snapshot(s))
    assert txt.startswith("stone_wall/slate_roof") and "2 objects repainted" in txt
    mid = scene_snapshot(s)
    s, _ = apply_op(s, "add", id="lantern_l", shape={"type": "block", "state": "lantern"}, modifiers=[{"type": "mirror", "axis": "x"}])
    s, _ = apply_op(s, "add", id="torch_1", shape={"type": "block", "state": "torch"})
    assert summarize_delta("decoration", mid, scene_snapshot(s)) == "added 2 lanterns, 1 torch"
    assert summarize_delta("decoration", scene_snapshot(s), scene_snapshot(s)) == ""


def test_live_preview_default_and_toggle(monkeypatch):
    s = Session("p")
    monkeypatch.delenv("LIVE_PREVIEW", raising=False)
    assert live_preview_enabled(s) is True
    monkeypatch.setenv("LIVE_PREVIEW", "false")
    assert live_preview_enabled(s) is False
    ctx = FakeCtx(session=s)
    assert handle_meta(ctx, "preview on").reply.startswith("Live preview on") and live_preview_enabled(s) is True
    assert handle_meta(ctx, "/cp preview off").reply.startswith("Live preview off") and live_preview_enabled(s) is False
    assert handle_meta(ctx, "verbose on").reply.startswith("Verbose on") and s.verbose is True
    assert handle_meta(ctx, "verbose off").reply.startswith("Verbose off") and s.verbose is False


class _StatusBridge(MockBridge):
    """MockBridge with a placement queue that drains one poll at a time (like the mod's tick queue)."""

    def __init__(self):
        super().__init__()
        self.pending = 0
        self.polls = 0
        self.timeline: List[str] = []

    def setblocks(self, chunks, flags=3):
        n = super().setblocks(chunks, flags)
        self.pending += n
        self.timeline.append(f"set:{n}")
        return n

    def setblocks_status(self) -> Dict[str, Any]:
        self.polls += 1
        self.pending = max(0, self.pending - 400)
        return {"pending_blocks": self.pending, "pending_chunks": 1 if self.pending else 0}

    def say(self, text):
        super().say(text)
        self.timeline.append(text)


def test_place_scene_reports_progress_in_synced_batches(monkeypatch):
    monkeypatch.setattr("copilot.placement.WAIT_POLL_S", 0.0)
    b = _StatusBridge()
    s = Session("p")
    bm = {(x, y, z): "minecraft:stone" for x in range(10) for y in range(8) for z in range(10)}  # 800 blocks
    pct: List[int] = []
    msg = place_scene(s, b, bm, mode="diff", animate=True, chunk_size=100, on_progress=lambda p: (pct.append(p), b.say(f"[cp·build {p}%]")))
    assert pct == [25, 50, 75, 100]
    # 8 chunks of 100 → 4 batches of 2, each awaited (status polled) before the next `say`
    assert [t for t in b.timeline if t.startswith("set:")] == ["set:200"] * 4
    assert b.timeline == ["set:200", "[cp·build 25%]", "set:200", "[cp·build 50%]", "set:200", "[cp·build 75%]", "set:200", "[cp·build 100%]"]
    assert b.polls >= 4 and "800 set" in msg and len(s.world.placed) == 800
    # every chunk is bottom-up: chunk i's max y <= chunk i+1's min y
    ys = [[p[1] for p in c] for c in _chunks_of(b)]
    assert all(max(a) <= min(c) for a, c in zip(ys, ys[1:]))
    # a tiny placement (one chunk) still reports 100 %
    pct.clear()
    place_scene(s, b, {**bm, (0, 9, 0): "minecraft:glass"}, mode="diff", animate=True, on_progress=pct.append)
    assert pct == [100]


def _chunks_of(bridge: MockBridge) -> List[List[tuple]]:
    # MockBridge records counts only; reconstruct order from the world insertion order per call
    order = list(bridge.world.keys())
    out, i = [], 0
    for call in bridge.calls:
        for c in call["chunks"]:
            out.append(order[i : i + c["count"]])
            i += c["count"]
    return [c for c in out if c]


def test_wait_for_placement_without_status_sleeps_estimate(monkeypatch):
    slept: List[float] = []
    monkeypatch.setattr("copilot.placement.time.sleep", slept.append)
    b = MockBridge()  # no setblocks_status
    wait_for_placement(b, 0.3)
    wait_for_placement(b, 0.0)
    assert slept == [0.3]


# -- mock-bridge integration: the /say sequence and placement calls of a scripted build -------
def test_scripted_build_say_sequence_and_placements(tmp_path, monkeypatch):
    monkeypatch.delenv("LIVE_PREVIEW", raising=False)
    b = MockBridge()
    ctx = real_ctx(tmp_path, b)
    critic = [
        {"score": 5, "top_3_fixes": [{"rule": "P1", "objects": ["hall"], "op_suggestion": "set_shape(id='hall', height=10)"}], "summary": "squat"},  # blocking → fix
        {"score": 9, "top_3_fixes": [{"rule": "P8", "objects": ["hall"], "op_suggestion": "add(id='hall_win2', shape={'type':'box','size':[2,2,3]}, pos=[4,3,4], op='subtract')"}], "summary": "fine"},  # detailing: noted only
        {"score": 9, "top_3_fixes": [], "summary": "ok"},  # materials
        {"score": 9, "top_3_fixes": [], "summary": "ok"},  # decoration
        {"score": 8, "top_3_fixes": [], "summary": "good"},  # final
    ]
    llm = ScriptedBuilderLLM(critic_responses=critic)
    res = handle_chat(ctx, "build a small stone hall with a hip roof", llm=llm, fast=False)
    lines = list(b.chat)
    assert all(TAG_RE.match(ln) for ln in lines), lines
    assert tags(lines) == ["plan", "block", "critic", "fix", "detail", "critic", "materials", "critic", "decor", "critic", "critic", "build", "build"]
    assert lines[0].startswith("[cp·plan 0:0") and "hip roof" in lines[0]
    assert lines[1].endswith("] added 2 solids")
    assert lines[2].endswith("] 5/10 — fixing: set_shape hall (rule P1)")
    assert lines[4].endswith("] carved 3 windows, 1 door")
    assert lines[5].endswith("] 9/10 — noted: add hall_win2")
    assert lines[6].endswith("] castle_wall/roof_dark; 2 objects repainted")
    assert lines[8].endswith("] added 2 lanterns")
    assert lines[10].endswith("] 8/10 — good") and lines[-1] == "[cp·build 100%]"
    assert not any("{" in ln or "}" in ln for ln in lines)  # no JSON in chat
    # placements: live preview after blocking and after detailing, then the final animated build (batched)
    assert len(b.calls) >= 3
    assert b.calls[0]["count"] > 500 and all(c["delay_ms"] == 60 for c in b.calls[0]["chunks"])
    assert b.calls[1]["count"] < b.calls[0]["count"]
    n_final = sum(c["count"] for c in b.calls[2:])
    assert n_final > 0 and len(b.placed_blocks()) == len(ctx.session.world.placed)
    # final reply: 2–4 lines with what/dimensions/blocks/time and the next-step hint
    rl = res.reply.split("\n")
    assert 2 <= len(rl) <= 4 and rl[0].startswith("Built hall_v1: hall") and "hip roof" in rl[0]
    assert re.search(r"\d+×\d+ footprint, \d+ tall, 5 objects, [\d,]+ blocks, in \d:\d\d; critic 8/10", rl[1]), rl[1]
    assert rl[-1].startswith("Say `undo`, `export`, or an edit")
    assert res.placed and len(b.placed_blocks()) > 500


def test_preview_off_places_once_and_verbose_prints_ops(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_PREVIEW", "false")
    b = MockBridge()
    ctx = real_ctx(tmp_path, b)
    res = handle_chat(ctx, "build a small stone hall", llm=ScriptedBuilderLLM(), fast=True)
    t = tags(b.chat)
    assert res.placed and t[:5] == ["plan", "block", "detail", "materials", "decor"] and set(t[5:]) == {"build"} and b.chat[-1] == "[cp·build 100%]"
    n_batches = len(t) - 5
    assert 1 <= n_batches <= 4 and len(b.calls) == n_batches and sum(c["count"] for c in b.calls) == len(b.placed_blocks())  # one final placement, batched
    # verbose on: one line per op, tagged with the running stage
    ctx.session.verbose = True
    b.chat.clear()
    handle_chat(ctx, "verbose on", llm=ScriptedBuilderLLM())
    ctx2 = real_ctx(tmp_path, b)
    ctx2.session.verbose = True
    handle_chat(ctx2, "build a small stone hall", llm=ScriptedBuilderLLM(), fast=True)
    ops = [ln for ln in b.chat if TAG_RE.match(ln) and TAG_RE.match(ln).group("body") and TAG_RE.match(ln).group("body").split(" ")[0] in ("add", "define_material", "stack", "set_material")]
    assert len(ops) >= 8 and any(ln.startswith("[cp·block ") and "add hall:" in ln for ln in ops)
    assert not any(ln.startswith("[cp·block ") and "render" in ln for ln in b.chat)


def test_model_say_is_tagged(tmp_path):
    b = MockBridge()
    ctx = real_ctx(tmp_path, b)
    from copilot.pipeline.progress import new_progress

    new_progress(ctx)
    ctx.session.stage = "detailing"
    dispatch(ctx, "say", {"text": "carving the gate now\nline two"})
    assert b.chat == ["[cp·detail 0:00] carving the gate now line two"]
    dispatch(ctx, "say", {"text": "[cp·build 50%]"})
    assert b.chat[-1] == "[cp·build 50%]"
