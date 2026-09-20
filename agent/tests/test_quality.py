"""T4 — build-quality levers: brief validation with one retry, vetted critic fixes, op gists in chat,
second final critic round only below 7, R9 materials lint (see test_lint), presets."""
from __future__ import annotations

import json

from bench.mock_builder import ScriptedBuilderLLM
from copilot import llm as L
from copilot.engine.materials import PRESETS
from copilot.pipeline import handle_chat
from copilot.pipeline.critic import parse_critique, vet_fixes
from copilot.pipeline.interpret import interpret, normalize_brief, validate_brief
from copilot.pipeline.orchestrator import op_gist
from tests.fakes import FakeCtx


def _ctx(tmp_path, **kw):
    return FakeCtx(run_dir=str(tmp_path), **kw)


GOOD = "square keep 12x12 h=18 with a hip roof (overhang 1); round towers r=3 h=20 at the corners; entrance: 2x3 arched gate centred on the south wall"


# -- lever 1: brief validation ------------------------------------------------------------------
def test_validate_brief_rules():
    assert validate_brief(normalize_brief({"silhouette_plan": GOOD})) == []
    p = validate_brief(normalize_brief({"silhouette_plan": "square tower 10x10 h=6 with a cone roof (overhang 1); door on the south"}))
    assert len(p) == 1 and "10 wide but only 6 tall" in p[0]
    p = validate_brief(normalize_brief({"silhouette_plan": "round tower radius 5, height 8, flat roof with parapet; entrance on the south"}))
    assert len(p) == 1 and "tower" in p[0]
    p = validate_brief(normalize_brief({"silhouette_plan": "hall 20x10 h=8 with a gable roof; towers r=2 h=12"}))
    assert [x[:20] for x in p] == ["the plan has roofs b", "no entrance is state"]
    # 'eaves' / 'flat roof … parapet' count as an overhang statement; the entrance may be in key_features
    assert validate_brief(normalize_brief({"silhouette_plan": "pagoda tiers with up-curved eaves; spire r=1 h=6", "key_features": ["door on the south"]})) == []
    assert validate_brief(normalize_brief({"silhouette_plan": "villa 20x14 h=8, flat roof with a parapet, front door 2x3 on the south"})) == []
    # the prompt's own examples pass
    import re

    from copilot.pipeline.stages import load_prompt

    text = load_prompt("stage_interpret")
    plans = [m.replace("\n  ", " ") for m in re.findall(r'"silhouette_plan": "([^"]+)"', text)] + [m.replace("\n  ", " ") for m in re.findall(r': "((?:Tapered|Five square)[^"]+)"', text)]
    assert len(plans) == 3 and all(validate_brief(normalize_brief({"silhouette_plan": p})) == [] for p in plans)


def test_interpret_retries_once_on_rejected_brief(tmp_path):
    ctx = _ctx(tmp_path)
    bad = {"build_type": "castle", "footprint": [30, 30], "height": 20, "silhouette_plan": "square towers 8x8 h=6 with cone roofs"}
    good = {**bad, "silhouette_plan": GOOD}
    m = L.MockLLM([{"tool_calls": [{"name": "set_brief", "args": {"brief": bad}}]}, {"tool_calls": [{"name": "set_brief", "args": {"brief": good}}]}])
    b = interpret(m, ctx, "build a castle")
    assert b["silhouette_plan"] == GOOD and len(m.requests) == 2
    retry_user = [x for x in m.requests[1]["messages"] if x["role"] == "user"][-1]["content"]
    assert "rejected" in retry_user and "8 wide but only 6 tall" in retry_user and "no overhang" in retry_user and "no entrance" in retry_user
    assert not any(c.startswith("design rule") for c in b["constraints"])
    # still bad after the retry: the rule travels with the brief as a constraint, no third call
    ctx2 = _ctx(tmp_path)
    m2 = L.MockLLM([{"tool_calls": [{"name": "set_brief", "args": {"brief": bad}}]}] * 3)
    b2 = interpret(m2, ctx2, "build a castle")
    assert len(m2.requests) == 2 and sum(c.startswith("design rule") for c in b2["constraints"]) == 3
    # a good brief costs one call
    ctx3 = _ctx(tmp_path)
    m3 = L.MockLLM([{"tool_calls": [{"name": "set_brief", "args": {"brief": good}}]}])
    interpret(m3, ctx3, "build a castle")
    assert len(m3.requests) == 1


# -- lever 5: critic fixes must be actionable -----------------------------------------------------
def test_vet_fixes_drops_prose_and_unknown_ids():
    fixes = [
        {"rule": "P8", "objects": ["wall_n"], "op_suggestion": "add windows to the north wall"},  # prose → dropped
        {"rule": "P1", "objects": ["ghost"], "op_suggestion": "set_shape(id='ghost', height=9)"},  # unknown id → dropped
        {"rule": "P2", "objects": ["wall_n", "ghost"], "op_suggestion": "add(id='wall_n_pil', shape={'type':'box','size':[1,8,1]}, pos=[0,0,0])"},
        {"rule": "F", "objects": [], "op_suggestion": "run_script(code=\"scene.add(...)\")"},  # creates something → kept
        {"rule": "P7", "objects": [], "op_suggestion": "scene.add(id='lamp', shape={'type':'block','block':'lantern'}, pos=[0,3,7])"},
    ]
    out = vet_fixes(fixes, known_ids=["wall_n", "keep"])
    assert [f["rule"] for f in out] == ["P2", "F", "P7"] and out[0]["objects"] == ["wall_n"]
    # without known ids only the op-call rule applies; the cap is 3
    assert [f["rule"] for f in vet_fixes(fixes)] == ["P1", "P2", "F"]
    c = parse_critique('{"score": 5, "top_3_fixes": [{"rule": "P8", "objects": ["x"], "op_suggestion": "make it nicer"}], "summary": "meh"}', "final", known_ids=["x"])
    assert c.score == 5 and c.fixes == []


def test_op_gist_has_no_json():
    assert op_gist("add(id='hall_win', shape={'type':'box','size':[2,2,3]}, pos=[0,3,4], op='subtract')") == "add hall_win"
    assert op_gist("scene.set_shape(id=\"roof\", height=9)") == "set_shape roof"
    assert op_gist("run_script(code='...')") == "run_script"
    assert op_gist("", "the {east} façade is blank") == "the east façade is blank"


def test_second_final_round_only_below_seven(tmp_path):
    fix = {"rule": "P7", "objects": [], "op_suggestion": "add(id='lamp', shape={'type':'block','block':'lantern'}, pos=[0,3,7])"}
    stage_ok = {"score": 9, "top_3_fixes": [], "summary": "ok"}
    # first final round 7 with a fix → one fix round, then no second critic
    critic = [stage_ok] * 4 + [{"score": 7, "top_3_fixes": [fix], "summary": "dim"}, {"score": 9, "top_3_fixes": [], "summary": "good"}]
    llm = ScriptedBuilderLLM(critic_responses=critic)
    r = handle_chat(_ctx(tmp_path), "build a stone hall", llm=llm, fast=False)
    assert llm.roles.count("critic") == 5 and "critic 7/10" in r.reply
    # first final round 6 → fix → second final round runs
    critic = [stage_ok] * 4 + [{"score": 6, "top_3_fixes": [fix], "summary": "dim"}, {"score": 8, "top_3_fixes": [], "summary": "good"}]
    llm = ScriptedBuilderLLM(critic_responses=critic)
    r = handle_chat(_ctx(tmp_path), "build a stone hall", llm=llm, fast=False)
    assert llm.roles.count("critic") == 6 and "critic 8/10" in r.reply


# -- lever 3: presets ----------------------------------------------------------------------------
def test_t4_presets_present_with_grounding():
    walls = ["limestone_pale", "tudor_plaster", "dark_slate_wall", "red_brick_victorian", "weathered_wood"]
    others = ["turf_roof", "spruce_shingle_roof", "stone_trim_light", "quartz_trim", "sandstone_trim"]
    assert all(n in PRESETS for n in walls + others)
    assert all(PRESETS[n].get("gradient") for n in walls) and all(len(PRESETS[n]["palette"]) >= 2 for n in walls + others)
    assert all(PRESETS[n]["fit"] == "slab" for n in ("stone_trim_light", "quartz_trim", "sandstone_trim"))


def _fake_report(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    rep = {"timestamp": "t", "fast": False, "mock": False, "mean": None, "results": rows}
    (d / "report.json").write_text(json.dumps(rep))
    return d


def test_bench_compare_pairs_scores_and_latency(tmp_path):
    from bench.run import compare

    a = _fake_report(tmp_path / "a", [
        {"name": "castle", "prompt": "p", "seconds": 200.0, "scores": {"mean": 6.0}, "profile": {}},
        {"name": "villa", "prompt": "p", "seconds": 100.0, "scores": {"mean": None}, "profile": {}},
    ])
    b = _fake_report(tmp_path / "b", [
        {"name": "castle", "prompt": "p", "seconds": 150.0, "scores": {"mean": 7.5}, "profile": {"stages": [{"stage": "detailing", "stopped_reason": "budget"}]}},
        {"name": "villa", "prompt": "p", "seconds": 120.0, "scores": {"mean": 7.0}, "profile": {}},
    ])
    out = compare(str(a), str(b))
    assert "| castle | 6.0 | 7.5 | +1.50 | 200.0 | 150.0 |" in out
    assert "| villa | None | 7.0 |  |" in out  # unpaired rows are listed but not averaged
    assert "1 paired; Δ mean +1.50" in out
    assert "median 150 s, max 200 s" in out and "median 135 s, max 150 s" in out


def test_bench_rescore_only_failed_rows(tmp_path, monkeypatch):
    import bench.run as br

    d = _fake_report(tmp_path / "r", [
        {"name": "castle", "prompt": "build a castle", "seconds": 1.0, "tool_calls": 0, "objects": 0, "blocks": 0, "scores": {"mean": 6.0, "notes": "ok"}, "png": None, "reply": ""},
        {"name": "villa", "prompt": "build a villa", "seconds": 1.0, "tool_calls": 0, "objects": 0, "blocks": 0, "scores": {"mean": None, "notes": "scoring failed: 429"}, "png": None, "reply": ""},
    ])
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps([{"name": "villa", "prompt": "build a villa", "must_have": ["pool"]}]))
    scored = []

    class _LLM:
        supports_vision = False

    monkeypatch.setattr("copilot.llm.AzureLLM", lambda: _LLM())
    monkeypatch.setattr(br, "score_build", lambda llm, image, item, outline: scored.append(item["name"]) or {"silhouette": 8, "detail": 8, "materials": 8, "fidelity": 8, "notes": "n", "mean": 8.0})
    rep = br.rescore(str(d), str(prompts))
    assert scored == ["villa"]  # the scored row is left alone
    assert rep["mean"] == 7.0
    assert json.loads((d / "report.json").read_text())["results"][1]["scores"]["mean"] == 8.0
    assert (d / "report.md").exists()
