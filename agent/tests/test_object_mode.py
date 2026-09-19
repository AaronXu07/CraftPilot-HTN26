"""Object mode: statues, creatures, vehicles and props are briefed as a parts list, built with the
skeleton-first prompts, judged by the sculpture rubric and linted without the façade rules."""
from __future__ import annotations

import re

import numpy as np

from copilot import llm as L
from copilot.engine.lint import is_object_scene, lint
from copilot.engine.scene import Scene, op_add, op_set_shape
from copilot.pipeline.critic import critique
from copilot.pipeline.interpret import OBJECT_STAGES, guess_kind, interpret, normalize_brief, validate_brief
from copilot.pipeline.stages import STAGE_BY_NAME, brief_kind, prompt_variant, run_stage, stage_for_rule, system_prompt
from tests.fakes import FakeCtx

DRAGON_PLAN = (
    "Plinth box 14x3x22 at origin. Body ellipsoid radii [3,3,7] centred (0,9,0), long axis z, head faces +z. "
    "Neck line (0,10,6)->(0,15,11) t=3; head ellipsoid [2,2,3.5] centred (0,15.5,13). Four legs lines t=2 from (±2,8,±4) down to y=3."
)


def test_guess_kind_objects_vs_buildings():
    assert guess_kind("build a dragon statue on a plinth") == "object"
    assert guess_kind("build a red sports car with a spoiler") == "object"
    assert guess_kind("a giant sword stuck in the ground") == "object"
    assert guess_kind("build a castle with four towers") == "building"
    assert guess_kind("a statue of a knight in front of the castle gate") == "building"  # a building word wins
    assert guess_kind("make me a treehouse") == "building"
    assert guess_kind("") == "building"


def test_normalize_brief_sets_kind_and_object_stages():
    b = normalize_brief({"silhouette_plan": DRAGON_PLAN}, "build a dragon statue")
    assert b["kind"] == "object" and b["stages"] == OBJECT_STAGES and b["build_type"] == "object"
    # the model's own kind wins over the keyword guess, and buildings keep all four stages
    b2 = normalize_brief({"kind": "building", "build_type": "hall", "silhouette_plan": "hall 20x10 h=8, door on the south"}, "build a dragon hall")
    assert b2["kind"] == "building" and b2["stages"] == ["blocking", "detailing", "materials", "decoration"]
    # an object brief that lists decoration anyway loses it
    b3 = normalize_brief({"kind": "object", "stages": ["blocking", "detailing", "materials", "decoration"]}, "build a car")
    assert b3["stages"] == OBJECT_STAGES
    assert normalize_brief({"kind": "spaceship"}, "build a house")["kind"] == "building"  # unknown kind → guess


def test_validate_brief_object_rules_replace_building_rules():
    ok = normalize_brief({"kind": "object", "silhouette_plan": DRAGON_PLAN}, "dragon")
    assert validate_brief(ok) == []  # no entrance / overhang / tower complaints for an object
    vague = normalize_brief({"kind": "object", "silhouette_plan": "a majestic dragon with spread wings"}, "dragon")
    problems = validate_brief(vague)
    assert len(problems) == 2 and "numbers" in problems[0] and "faces" in problems[1]
    car = normalize_brief({"kind": "object", "silhouette_plan": "body ellipsoid [3,1.6,8] at (0,2.6,0); four wheels r=1.5 at (±3,1.5,±5); nose points south"}, "car")
    assert validate_brief(car) == []


def test_interpret_object_skips_building_retry(tmp_path):
    """A statue brief with no entrance must not burn a second interpret call (seen in the object bench)."""
    ctx = FakeCtx(run_dir=str(tmp_path))
    brief = {"name": "dragon_v1", "kind": "object", "build_type": "statue", "style": "stone", "footprint": [14, 22], "height": 20, "facing": "south", "key_features": ["wings"], "silhouette_plan": DRAGON_PLAN}
    m = L.MockLLM([{"tool_calls": [{"name": "set_brief", "args": {"brief": brief}}]}, {"tool_calls": [{"name": "set_brief", "args": {"brief": brief}}]}])
    b = interpret(m, ctx, "build a dragon statue")
    assert b["kind"] == "object" and len(m.requests) == 1 and not b["constraints"]
    assert ctx.session.scene.meta["brief"]["kind"] == "object"
    assert "## Objects (kind = \"object\")" in m.requests[0]["messages"][0]["content"]


def test_prompt_variant_and_brief_kind(tmp_path):
    assert prompt_variant("stage_blocking", "object") == "stage_blocking_object"
    assert prompt_variant("critic", "object") == "critic_object"
    assert prompt_variant("stage_blocking", "building") == "stage_blocking"
    assert prompt_variant("stage_decoration", "object") == "stage_decoration"  # no object variant → shared prompt
    ctx = FakeCtx(run_dir=str(tmp_path))
    assert brief_kind({"kind": "object"}, ctx) == "object" and brief_kind(None, ctx) is None
    ctx.session.scene.meta["brief"] = {"kind": "object"}
    assert brief_kind({}, ctx) == "object"  # falls back to the scene's brief (edits on a later turn)
    sp = system_prompt(ctx, STAGE_BY_NAME["blocking"], kind="object")
    assert "Blocking an OBJECT" in sp and "# Stage 1 — Blocking (massing)" not in sp
    assert "Materials catalog (fake)" in sp


def test_run_stage_uses_object_prompt_for_object_brief(tmp_path):
    ctx = FakeCtx(run_dir=str(tmp_path))
    m = L.MockLLM([{"tool_calls": [{"name": "finish", "args": {"summary": "skeleton"}}]}])
    run_stage(m, ctx, STAGE_BY_NAME["detailing"], "build a dragon", {"kind": "object", "silhouette_plan": DRAGON_PLAN})
    assert "Detailing an OBJECT" in m.requests[0]["messages"][0]["content"]


def test_critic_uses_object_rubric_and_routes_s_rules(tmp_path):
    ctx = FakeCtx(run_dir=str(tmp_path))
    m = L.MockLLM([{"text": '{"score": 6, "top_3_fixes": [], "summary": "stiff"}'}])
    critique(m, ctx, "blocking", {"kind": "object", "silhouette_plan": DRAGON_PLAN})
    assert "Critic — OBJECT" in m.requests[0]["messages"][0]["content"]
    assert stage_for_rule("S1").name == "blocking" and stage_for_rule("S3").name == "blocking"
    assert stage_for_rule("S6").name == "materials" and stage_for_rule("P8").name == "detailing"


def test_object_prompts_examples_use_real_signatures():
    """The worked example in the object blocking prompt must run on the engine (a prompt that teaches
    a broken snippet is worse than none)."""
    from copilot.pipeline.stages import load_prompt

    script = re.search(r'run_script\(python="""(.*?)"""\)', load_prompt("stage_blocking_object"), re.S).group(1)
    sc = Scene()

    class Ed:  # minimal editor: the ops the example uses
        def define_material(self, name, spec):
            sc.materials[name] = spec

        def add(self, **kw):
            nonlocal sc
            sc, _ = op_add(sc, **kw)

        def mirror_copy(self, ids, axis, plane, new_suffix="_m"):
            nonlocal sc
            from copilot.engine.scene import op_mirror_copy

            sc, _ = op_mirror_copy(sc, ids, axis, plane, new_suffix)

    exec(script, {"scene": Ed()})  # noqa: S102 — the prompt's own example
    ids = {o.id for o in sc.objects}
    assert {"plinth", "body", "neck", "head", "leg_fr", "leg_fr_l", "horn_r_l"} <= ids
    neck = sc.object_bbox(sc.get("neck"))
    assert neck.lo[1] > 8 and neck.hi[2] > 12, "line endpoints are world coordinates"


def test_line_and_sweep_points_are_world_coordinates():
    sc = Scene()
    sc, _ = op_add(sc, "beam", {"type": "line", "from": [2, 8, 4], "to": [2.5, 3, 4], "thickness": 2})
    bb = sc.object_bbox(sc.get("beam"))
    assert sc.get("beam").transform.anchor == "origin"
    assert np.allclose(bb.lo, [1, 2, 3]) and np.allclose(bb.hi, [3.5, 9, 5])
    sc, _ = op_add(sc, "pipe", {"type": "sweep", "radius": 1, "path": [[0, 10, 0], [10, 10, 0]]})
    assert sc.object_bbox(sc.get("pipe")).lo[1] >= 8.9
    # an explicit non-default anchor is respected; changing a box into a line switches to origin
    sc, _ = op_add(sc, "beam2", {"type": "line", "from": [0, 5, 0], "to": [0, 9, 0], "thickness": 2}, anchor="center")
    assert sc.get("beam2").transform.anchor == "center" and sc.object_bbox(sc.get("beam2")).lo[1] == -3
    sc, _ = op_add(sc, "b", {"type": "box", "size": [2, 2, 2]}, pos=[0, 0, 0])
    sc, _ = op_set_shape(sc, "b", params={"type": "line", "from": [0, 6, 0], "to": [0, 9, 0], "thickness": 2})
    assert sc.get("b").transform.anchor == "origin" and sc.object_bbox(sc.get("b")).lo[1] == 5
    lines = {ln.split()[0]: ln for ln in sc.describe().splitlines() if ln.strip()}
    assert "anchor=" not in lines["beam"] and "anchor=" not in lines["b"]  # origin is natural for lines: not shown
    assert "anchor=center" in lines["beam2"]


def test_lint_skips_facade_rules_for_object_scenes():
    from copilot.engine import raster as R
    from copilot.engine.fit import fit_surface

    sc = Scene()
    sc.materials["m"] = {"base": "stone"}
    sc, _ = op_add(sc, "body", {"type": "box", "size": [12, 12, 12]}, material="m")  # a blank 12x12 face on every side
    sc, _ = op_add(sc, "horn", {"type": "cone", "radius": 1, "height": 4}, pos=[0, 12, 0], material="m")  # a "roof" with a bad slope
    rr = R.rasterize(sc)
    fr = fit_surface(rr)
    building_rules = {f.rule for f in lint(sc, rr, fr, None)}
    assert {"R1", "R2", "R3"} <= building_rules and "R9" in building_rules
    sc.meta["brief"] = {"kind": "object"}
    assert is_object_scene(sc)
    object_rules = {f.rule for f in lint(sc, rr, fr, None)}
    assert not ({"R1", "R2", "R3"} & object_rules)
    assert all("gradient" not in f.message for f in lint(sc, rr, fr, None) if f.rule == "R9")


def test_object_build_critiques_blocking_and_final_only(tmp_path):
    """Objects: one critique after blocking, one final round (a second final round never happens), no
    critiques after detailing/materials — about 50 s less critic/fix overhead per build on the bench."""
    from bench.mock_builder import ScriptedBuilderLLM
    from copilot.pipeline import handle_chat

    ctx = FakeCtx(run_dir=str(tmp_path))
    low = {"score": 5, "top_3_fixes": [{"rule": "S2", "objects": ["hall"], "op_suggestion": "set_shape(id='hall', params={'height': 10})"}], "summary": "squat"}
    llm = ScriptedBuilderLLM(critic_responses=[low, low, low, low, low, low])
    r = handle_chat(ctx, "build a dragon statue on a plinth", llm=llm, fast=False)
    assert ctx.session.brief["kind"] == "object" and ctx.session.brief["stages"] == OBJECT_STAGES
    assert llm.roles.count("critic") == 2  # blocking + final (no detailing/materials critique, no 2nd final round)
    names = [s["stage"] for s in r.data["stages"]]
    assert names == ["blocking", "blocking", "detailing", "materials", "blocking"]  # stage, its fix, …, one final fix
    # buildings keep the full critic schedule
    ctx2 = FakeCtx(run_dir=str(tmp_path / "b"))
    llm2 = ScriptedBuilderLLM(critic_responses=[low] * 8)
    handle_chat(ctx2, "build a stone hall", llm=llm2, fast=False)
    assert ctx2.session.brief["kind"] == "building" and llm2.roles.count("critic") == 6
