"""T2 latency work: wall-clock budget, stage deadlines, op cap, parallel read-only tool calls,
per-call LLM timeouts, MODEL_FAST routing, small vision payloads and the bench latency table."""
from __future__ import annotations

import time

from PIL import Image

from bench.mock_builder import ScriptedBuilderLLM
from copilot import llm as L
from copilot.pipeline import handle_chat, route
from copilot.pipeline.budget import DECORATION_MIN_S, Budget, get_budget, new_budget
from copilot.pipeline.orchestrator import profile_line, run_build
from tests.fakes import FakeCtx


def _ctx(tmp_path, **kw):
    return FakeCtx(run_dir=str(tmp_path), **kw)


# -- Budget ------------------------------------------------------------------------------------
def test_budget_plan_is_a_rolling_schedule():
    b = Budget(total_s=300, plan_s=150)
    assert b.plan["blocking"] == 35 and b.plan["critic"] == 10 and sum(b.plan.values()) == 150
    assert Budget(plan_s=75).plan["detailing"] == 22.5  # the plan scales with COPILOT_PLAN_BUDGET_S
    # the first stage's deadline is the planned end; time saved rolls into the next stage
    b2 = Budget(total_s=300, plan_s=150)
    d_interp = b2.start_stage("interpret")
    assert abs(d_interp - (b2.t0 + 10)) < 0.05
    d_block = b2.start_stage("blocking")
    assert abs(d_block - (b2.t0 + 45)) < 0.05  # cumulative: interpret 10 + blocking 35
    assert not b2.behind() and b2.allow_critic() and b2.allow_fix()
    b2.start_critic()
    assert abs(b2.sched - (b2.t0 + 55)) < 0.05
    fd = b2.fix_deadline()
    assert fd <= b2.hard_deadline and fd > time.time()


def test_budget_late_stage_gets_half_allotment_and_skips_critic():
    b = Budget(total_s=300, plan_s=150)
    b.sched = time.time() - 60  # pretend the earlier stages ran 60 s late
    d = b.start_stage("detailing")
    assert abs(d - (time.time() + 22.5)) < 0.1  # floor: half of 45 s
    assert b.behind() and not b.allow_critic() and not b.allow_fix()
    b.sched = time.time() - 20  # 20 s late: a 30 s stage that finishes instantly catches up
    b.start_stage("materials")
    assert not b.behind() and b.allow_critic()


def test_budget_hard_cap_and_decoration_skip():
    b = Budget(total_s=30, plan_s=150)
    assert b.can_start("blocking") and not b.can_start("decoration")  # < DECORATION_MIN_S left
    assert DECORATION_MIN_S == 40
    assert b.start_stage("blocking") <= b.hard_deadline
    assert b.llm_timeout(120.0) <= 30.0
    b.hard_deadline = time.time() - 1
    assert b.over() and b.remaining == 0 and b.llm_timeout(120.0) == 8.0 and not b.can_start("blocking")


def test_new_budget_aligns_with_job_deadline(tmp_path):
    from copilot.jobs import Job

    ctx = _ctx(tmp_path)
    ctx.job = Job(id=1, player="p", text="x", budget_s=60.0, started=time.time())
    b = new_budget(ctx)
    assert get_budget(ctx) is b and abs(b.hard_deadline - ctx.job.deadline) < 0.01 and b.total_s <= 60.0


# -- tool loop ---------------------------------------------------------------------------------
def test_tool_loop_stops_after_current_call_when_over_deadline(tmp_path):
    ctx = _ctx(tmp_path)
    slow = ctx.dispatch

    def sleepy(c, name, args):
        time.sleep(0.05)
        return slow(c, name, args)

    script = [{"tool_calls": [{"name": "add", "args": {"id": f"b{i}", "shape": {"type": "box", "size": [2, 2, 2]}}}]} for i in range(6)] + [{"text": "done"}]
    m = L.MockLLM(script)
    res = L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, sleepy, deadline=time.time() + 0.12)
    assert res.stopped_reason == "budget"
    assert 1 <= res.tool_calls <= 4  # ended after the call during which the deadline passed
    assert "budget" in res.text
    assert len(ctx.session.scene.objects) == res.tool_calls  # every started call completed


def test_tool_loop_time_nudge_and_deadline_before_llm_call(tmp_path):
    ctx = _ctx(tmp_path)
    m = L.MockLLM([{"tool_calls": [{"name": "describe", "args": {}}]}, {"text": "late"}])
    res = L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch, deadline=time.time() + 5)
    # 5 s left of a 5 s loop -> the "about N s left" nudge is inserted before the first call
    assert any(x["role"] == "system" and "left in this stage" in x["content"] for x in m.requests[0]["messages"])
    assert res.stopped_reason == "text"
    m2 = L.MockLLM([{"text": "never"}])
    res2 = L.run_tool_loop(m2, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch, deadline=time.time() - 1)
    assert res2.stopped_reason == "budget" and m2.requests == []


def test_tool_loop_op_cap_redirects_to_run_script(tmp_path):
    ctx = _ctx(tmp_path)
    calls = [{"name": "add", "args": {"id": f"b{i}", "shape": {"type": "box", "size": [1, 1, 1]}, "pos": [i * 2, 0, 0]}} for i in range(14)]
    calls.append({"name": "describe", "args": {}})  # read-only: exempt
    calls.append({"name": "run_script", "args": {"python": "scene.add(id='s1', shape={'type':'box','size':[1,1,1]}, pos=[40,0,0])"}})  # exempt
    m = L.MockLLM([{"tool_calls": calls}, {"text": "ok"}])
    res = L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch, op_cap=12)
    assert res.ops == 14
    texts = [c[2] for c in res.calls]
    assert sum("op cap reached" in t for t in texts) == 2
    assert "run_script" in texts[12]
    ids = [o.id for o in ctx.session.scene.objects]
    assert ids == [f"b{i}" for i in range(12)] + ["s1"]  # 12 direct adds + the script's add
    assert not texts[-1].startswith("ERROR")


def test_tool_loop_runs_read_only_calls_concurrently(tmp_path):
    ctx = _ctx(tmp_path)
    inner = ctx.dispatch
    seen = []

    def slow(c, name, args):
        seen.append((name, time.time()))
        if name in ("render", "lint", "describe"):
            time.sleep(0.15)
        return inner(c, name, args)

    calls = [{"name": "add", "args": {"id": "k", "shape": {"type": "box", "size": [4, 4, 4]}}}, {"name": "render", "args": {"views": ["iso"]}}, {"name": "lint", "args": {}}, {"name": "describe", "args": {}}]
    m = L.MockLLM([{"tool_calls": calls}, {"text": "ok"}])
    t0 = time.time()
    res = L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, slow)
    wall = time.time() - t0
    assert res.call_names() == ["add", "render", "lint", "describe"]  # results stay in order
    assert wall < 0.35, wall  # three 0.15 s read-only calls overlapped (sequential would be >= 0.45)
    starts = {n: t for n, t in seen}
    assert starts["add"] <= starts["render"]  # the mutating call ran first
    assert res.tool_ms >= 3 * 150 - 20  # per-call time is still accounted per call
    tool_msgs = [x for x in m.requests[1]["messages"] if x["role"] == "tool"]
    assert len(tool_msgs) == 4 and "rendered" in tool_msgs[1]["content"] and "lint" in tool_msgs[2]["content"]


def test_llm_call_passes_remaining_budget_as_timeout(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.budget = Budget(total_s=50, plan_s=150)
    m = L.MockLLM([{"text": "hi"}])
    m.timeout = 120.0
    L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch)
    assert 8.0 <= m.requests[0]["timeout"] <= 50.0
    # single_call with ctx caps too; without ctx nothing is passed
    m2 = L.MockLLM([{"text": "a"}, {"text": "b"}])
    L.single_call(m2, "s", "u", ctx=ctx)
    L.single_call(m2, "s", "u")
    assert m2.requests[0]["timeout"] is not None and m2.requests[1]["timeout"] is None


# -- pipeline ----------------------------------------------------------------------------------
def test_run_build_records_profile_and_skips_decoration_when_short(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.budget = Budget(total_s=39, plan_s=150)  # under DECORATION_MIN_S from the start
    llm = ScriptedBuilderLLM()
    res, report = run_build(ctx, llm, "build a small stone hall", fast=True)
    assert "decoration" not in llm.roles and res.data["skipped"] == ["decoration"]
    assert "Skipped decoration" in res.reply
    stages = [r.stage for r in ctx.budget.rows]
    assert stages.count("place") == 3  # live preview after blocking and detailing (T3) + the final placement
    stages = [s for s in stages if s != "place"]
    assert stages[:4] == ["interpret", "blocking", "detailing", "materials"]
    assert stages[4:] == ["decoration"]
    assert next(r for r in ctx.budget.rows if r.stage == "decoration").stopped == "skipped"
    row = ctx.budget.rows[1]
    assert row.llm_calls >= 1 and row.tool_calls >= 1 and row.wall_ms >= 0 and row.stopped == "finish"


def test_handle_chat_emits_turn_summary_and_profile(tmp_path):
    ctx = _ctx(tmp_path)
    llm = ScriptedBuilderLLM()
    res = handle_chat(ctx, "build a small stone hall", llm=llm)
    prof = res.data["profile"]
    names = [r["stage"] for r in prof["stages"]]
    assert names[0] == "interpret" and "critic:blocking" in names and "critic:final" in names and names[-1] == "place"
    assert prof["llm_calls"] == llm.calls
    recs = ctx.log.records
    summ = [r for r in recs if r.get("name") == "__summary__"]
    assert len(summ) == 1 and summ[0]["profile"]["llm_calls"] == llm.calls
    line = profile_line(prof)
    assert line.startswith("wall ") and "blocking" in line
    assert sum(1 for r in recs if r.get("name") == "__profile__") == len(prof["stages"])


def test_behind_schedule_skips_critic_rounds(tmp_path):
    ctx = _ctx(tmp_path)
    inner = ctx.dispatch

    def slow(c, name, args):
        time.sleep(0.03)
        return inner(c, name, args)

    ctx.dispatch = slow
    ctx.budget = Budget(total_s=300, plan_s=0.6)  # ~0.14 s per stage: every stage runs late
    llm = ScriptedBuilderLLM()
    res, report = run_build(ctx, llm, "build a small stone hall", fast=False)
    assert "critic" not in llm.roles  # never on schedule -> no critic call
    assert report.final_score is None
    assert all(s.stopped_reason in ("budget", "finish") for s in report.stages)
    assert any(s.stopped_reason == "budget" for s in report.stages)


# -- MODEL_FAST -------------------------------------------------------------------------------
class _FastAware(ScriptedBuilderLLM):
    def __init__(self):
        super().__init__()
        self.deployment = "big"
        self.fast_llm = ScriptedBuilderLLM()
        self.fast_llm.deployment = "mini"

    def fast(self):
        return self.fast_llm


def test_router_and_fix_rounds_use_fast_model(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.session.apply("add", id="keep", shape={"type": "box", "size": [8, 6, 8]})
    llm = _FastAware()
    r = route(llm, ctx, "make it taller")
    assert r.intent == "edit" and llm.fast_llm.roles == ["router"] and llm.roles == []
    ctx2 = _ctx(tmp_path)
    llm2 = _FastAware()
    llm2.critic_responses = [{"score": 4, "top_3_fixes": [{"rule": "P8", "objects": ["hall"], "op_suggestion": "add windows"}], "summary": "flat"}]
    run_build(ctx2, llm2, "build a small stone hall", fast=False)
    assert "blocking" in llm2.fast_llm.roles  # the fix round after the failing critic ran on the fast model
    assert "critic" in llm2.roles and "blocking" in llm2.roles  # the stage itself and the critic stayed on the main model


def test_azure_fast_view_uses_model_fast(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_FAST_DEPLOYMENT", raising=False)
    monkeypatch.setenv("MODEL", "gpt-big")
    monkeypatch.setenv("MODEL_FAST", "gpt-mini")
    monkeypatch.setenv("AZURE_OPENAI_API", "chat")
    a = L.AzureLLM()
    assert a.deployment == "gpt-big" and a.fast_deployment == "gpt-mini"
    f = a.fast()
    assert f is not a and f.deployment == "gpt-mini" and f.is_fast and f.fast() is f
    assert f.client is a.client and f.total_usage is a.total_usage
    monkeypatch.delenv("MODEL_FAST")
    b = L.AzureLLM()
    assert b.fast() is b and b.fast_deployment == "gpt-big"


# -- vision payloads ---------------------------------------------------------------------------
def test_image_payloads_are_small_jpegs_and_capped_at_two():
    big = Image.new("RGB", (2048, 1024), (200, 30, 30))
    part = L.image_content(big)
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    import base64
    import io

    raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
    im = Image.open(io.BytesIO(raw))
    assert im.size == (640, 320) and im.format == "JPEG"
    assert len(raw) < 60_000
    parts = L.image_parts([Image.new("RGB", (10, 10)) for _ in range(5)])
    assert len(parts) == 2
    png = L.image_content(Image.new("RGBA", (10, 10)), fmt="png")
    assert png["image_url"]["url"].startswith("data:image/png;base64,")


def test_critic_sends_at_most_one_sheet(tmp_path):
    from copilot.pipeline import critique

    ctx = _ctx(tmp_path)
    ctx.session.apply("add", id="keep", shape={"type": "box", "size": [8, 6, 8]})
    m = L.MockLLM([{"text": '{"score": 8}'}], supports_vision=True)
    critique(m, ctx, "blocking", {"name": "x"})
    parts = m.requests[0]["messages"][1]["content"]
    assert sum(p.get("type") == "image_url" for p in parts) == 1
    assert m.requests[0]["timeout"] is None  # no budget on this ctx -> no cap


# -- bench table -------------------------------------------------------------------------------
def test_bench_profile_table_and_legacy_records():
    from bench.run import profile_from_records, profile_table

    t = 1000.0
    legacy = [
        {"t": t, "stage": "blocking", "name": "__llm__", "ms": 4000},
        {"t": t + 4, "stage": "blocking", "name": "add", "ms": 30},
        {"t": t + 4.1, "stage": "blocking", "name": "__llm__", "ms": 3000},
        {"t": t + 7.1, "stage": "blocking", "name": "finish", "ms": 0},
        {"t": t + 8, "stage": "detailing", "name": "__llm__", "ms": 5000},
        {"t": t + 13, "stage": "detailing", "name": "finish", "ms": 0},
        {"t": t + 14, "stage": "turn", "name": "__reply__", "ms": 14000},
    ]
    p = profile_from_records(legacy, 14.0)
    assert p["llm_calls"] == 3 and p["llm_ms"] == 12000 and [r["stage"] for r in p["stages"]] == ["blocking", "detailing"]
    assert p["stages"][0]["wall_ms"] == 7100 and p["stages"][0]["tool_calls"] == 2
    report = {"results": [{"name": "a", "profile": p}, {"name": "b", "profile": {"wall_ms": 200000, "llm_ms": 1, "engine_ms": 1, "llm_calls": 1, "tool_calls": 1, "stages": [{"stage": "decoration", "wall_ms": 1000, "stopped": "budget"}, {"stage": "critic:final", "wall_ms": 2000}, {"stage": "fix:detailing", "wall_ms": 3000}]}}]}
    table = profile_table(report)
    assert "| a | 14 | 12 | 3 |" in table and "decoration:budget" in table
    assert "median 107, max 200" in table
    lines = table.splitlines()
    assert lines[0].count("|") == lines[1].count("|") == lines[2].count("|")
