"""Pipeline tests with a scripted LLM and a fake dispatcher (no game, no Azure)."""
from __future__ import annotations

import json

from bench.mock_builder import ScriptedBuilderLLM
from copilot import llm as L
from copilot.pipeline import (
    STAGE_BY_NAME,
    Critique,
    brief_to_text,
    critique,
    fixes_text,
    handle_chat,
    interpret,
    normalize_brief,
    route,
    run_stage,
    system_prompt,
)
from copilot.pipeline.critic import parse_critique
from copilot.pipeline.orchestrator import handle_meta
from copilot.pipeline.router import parse_route
from tests.fakes import FakeCtx


def _ctx(tmp_path, **kw):
    return FakeCtx(run_dir=str(tmp_path), **kw)


# -- interpret -------------------------------------------------------------------------------
def test_interpret_via_set_brief_tool(tmp_path):
    ctx = _ctx(tmp_path)
    brief = {"name": "castle_v1", "build_type": "castle", "style": "medieval", "footprint": [40, 32], "height": 26, "facing": "south", "key_features": ["four towers"], "silhouette_plan": "keep + towers"}
    m = L.MockLLM([{"tool_calls": [{"name": "set_brief", "args": {"brief": brief}}]}])
    b = interpret(m, ctx, "build a castle with four towers")
    assert b["build_type"] == "castle" and b["footprint"] == [40.0, 32.0] and b["stages"][0] == "blocking"
    assert ctx.session.brief is b and ctx.session.scene.name == "castle_v1"
    req = m.requests[0]
    assert req["tools"] == ["set_brief"] and req["temperature"] == 0.6
    assert req["tool_choice"]["function"]["name"] == "set_brief"
    assert "# Stage 0 — Interpret" in req["messages"][0]["content"]


def test_interpret_falls_back_to_json_text_and_defaults(tmp_path):
    ctx = _ctx(tmp_path)
    m = L.MockLLM([{"text": 'Here: ```json\n{"build_type": "lighthouse", "height": 30, "facing": "nowhere"}\n```'}])
    b = interpret(m, ctx, "build a lighthouse")
    assert b["build_type"] == "lighthouse" and b["height"] == 30 and b["facing"] == "south" and b["footprint"] == [20.0, 20.0]
    assert "lighthouse" in b["silhouette_plan"]
    b2 = normalize_brief("garbage", "make me a treehouse")
    assert b2["build_type"] == "treehouse" and b2["name"].endswith("_v1")
    assert "treehouse" in brief_to_text(b2)


# -- critic ----------------------------------------------------------------------------------
def test_critic_parser_handles_malformed_and_fenced_json():
    c = parse_critique("I think it's fine but no json", "blocking")
    assert c.score is None and c.fixes == [] and c.summary.startswith("I think")
    c2 = parse_critique('```json\n{"score": "6.4", "top_3_fixes": [{"rule": "p8", "objects": "wall_n", "op_suggestion": "add(...)"}, "just a string"], "summary": "flat",}\n```', "detailing")
    assert c2.score == 6 and c2.fixes[0] == {"rule": "P8", "objects": ["wall_n"], "op_suggestion": "add(...)"}
    assert c2.fixes[1]["op_suggestion"] == "just a string" and c2.summary == "flat"
    assert "[P8]" in fixes_text(c2) and "score 6/10" in fixes_text(c2)
    assert parse_critique('{"score": 42}').score == 10


def test_critique_uses_contact_sheet_when_vision(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.session.apply("add", id="keep", shape={"type": "box", "size": [8, 6, 8]}, material="stone_wall")
    m = L.MockLLM([{"text": '{"score": 5, "top_3_fixes": [{"rule": "P8", "objects": ["keep"], "op_suggestion": "add windows"}], "summary": "blank"}'}], supports_vision=True)
    c = critique(m, ctx, "blocking", {"name": "x"})
    assert c.score == 5 and c.used_vision and c.fixes[0]["objects"] == ["keep"]
    assert m.requests[0]["has_images"] is True
    parts = m.requests[0]["messages"][1]["content"]
    assert "keep" in parts[0]["text"] and "Lint findings" in parts[0]["text"]
    m2 = L.MockLLM([{"text": '{"score": 8}'}], supports_vision=False)
    c2 = critique(m2, ctx, "final", None)
    assert c2.score == 8 and not c2.used_vision and m2.requests[0]["has_images"] is False
    assert "plan at y=" in m2.requests[0]["messages"][1]["content"][0]["text"]


# -- router ----------------------------------------------------------------------------------
def test_router_parser():
    r = parse_route('{"intent":"edit","stages":["materials","bogus"],"selection":"tag:tower","direct_ops":[{"name":"set_shape","args":{"id":"t","height":30}},{"op":"move","ids":"r","delta":[0,8,0]},"junk"],"needs_place":true}')
    assert r.stages == ["materials"] and r.selection == "tag:tower" and r.needs_place
    assert r.direct_ops == [{"name": "set_shape", "args": {"id": "t", "height": 30}}, {"name": "move", "args": {"ids": "r", "delta": [0, 8, 0]}}]
    r2 = parse_route("nonsense")
    assert r2.intent == "edit" and r2.stages == ["detailing"] and r2.selection == "all"
    r3 = parse_route('{"intent": "question", "needs_place": true}')
    assert r3.intent == "question" and r3.needs_place is False and r3.stages == []
    r4 = parse_route('{"intent":"edit","direct_ops":[{"name":"evil","args":{}}]}', known_tools=["move"])
    assert r4.direct_ops == [] and r4.stages == ["detailing"]


def test_route_call_uses_outline_and_temperature_zero(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.session.apply("add", id="tower_ne", shape={"type": "cylinder", "radius": 4, "height": 20}, pos=[10, 0, 0])
    m = L.MockLLM([{"text": '{"intent":"edit","stages":[],"selection":"tower_ne","direct_ops":[{"name":"set_shape","args":{"id":"tower_ne","height":28}}]}'}])
    r = route(m, ctx, "make the tower taller")
    assert r.direct_ops[0]["args"]["height"] == 28 and m.requests[0]["temperature"] == 0.0
    assert "tower_ne" in m.requests[0]["messages"][0]["content"] and "# Router" in m.requests[0]["messages"][0]["content"]


# -- stage runner ----------------------------------------------------------------------------
def test_run_stage_composes_prompt_and_renders_and_lints(tmp_path):
    ctx = _ctx(tmp_path)
    m = L.MockLLM([{"tool_calls": [{"name": "add", "args": {"id": "keep", "shape": {"type": "box", "size": [10, 8, 10]}, "material": "stone_wall"}}]}, {"tool_calls": [{"name": "finish", "args": {"summary": "keep placed"}}]}])
    res = run_stage(m, ctx, STAGE_BY_NAME["blocking"], "build a keep", {"name": "keep_v1", "silhouette_plan": "one keep"}, extra="1. [P1] make it taller")
    assert res.text == "keep placed" and res.call_names() == ["add", "finish"] and res.stopped_reason == "finish"
    assert res.lint_text.startswith("lint") and len(res.images) == 2  # iso + front
    sysmsg = m.requests[0]["messages"][0]["content"]
    assert "# Stage 1 — Blocking" in sysmsg and "Materials catalog (fake)" in sysmsg and "Encouraged tools" in sysmsg
    user = m.requests[0]["messages"][1]["content"]
    assert "build a keep" in user and "one keep" in user and "Critic findings" in user and "## Current scene" in user
    assert "place" not in m.requests[0]["tools"]  # stage tools exclude placement (when schemas exist)
    assert ctx.session.stage is None


# -- end to end ------------------------------------------------------------------------------
def test_end_to_end_build_edit_diff_place_undo(tmp_path):
    ctx = _ctx(tmp_path)
    routes = [
        {"intent": "edit", "stages": [], "selection": "hall", "direct_ops": [{"name": "set_shape", "args": {"id": "hall", "size": [16, 12, 12]}}, {"name": "move", "args": {"ids": "hall_roof", "delta": [0, 4, 0]}}], "needs_place": True},
        {"intent": "question", "stages": [], "selection": "hall", "direct_ops": [], "needs_place": False},
    ]
    llm = ScriptedBuilderLLM(routes=routes)
    # 1. new build (fast: no critics)
    r = handle_chat(ctx, "build a small stone hall with a hip roof", llm=llm, fast=True)
    assert r.reply.startswith("Built") and r.placed and r.brief["build_type"] == "hall"
    ids = [o.id for o in ctx.session.scene.objects]
    assert ids == ["hall", "hall_roof", "hall_door_cut", "hall_win_cut", "door_lantern"]
    assert ctx.session.scene.materials["castle_wall"]["fit"] == "stairs+slab"
    assert ctx.bridge.said[0].startswith("Plan:") and any(s.startswith("blocking:") for s in ctx.bridge.said)
    assert llm.roles.count("critic") == 0 and llm.roles[0] == "interpret"
    full_blocks = len(ctx.bridge.setblock_calls[0])
    assert full_blocks > 100 and len(ctx.bridge.world) == full_blocks
    assert len(r.images) >= 1
    # 2. edit via direct ops → diff placement sends only the delta
    r2 = handle_chat(ctx, "make the hall 4 blocks taller", llm=llm, fast=True)
    assert r2.reply.startswith("Changed:") and "hall is now box 16x12x12" in r2.reply and r2.placed
    assert ctx.session.scene.get("hall").shape["size"][1] == 12
    diff_blocks = len(ctx.bridge.setblock_calls[-1])
    assert 0 < diff_blocks < full_blocks
    assert llm.roles[-1] == "router"
    # 3. question: no change, no placement
    n_calls = len(ctx.bridge.setblock_calls)
    r3 = handle_chat(ctx, "how tall is the hall?", llm=llm, fast=True)
    assert "8 blocks tall" in r3.reply and not r3.placed and len(ctx.bridge.setblock_calls) == n_calls
    # 4. undo (meta) → scene reverts and the world is diff-placed back
    r4 = handle_chat(ctx, "undo", llm=llm)
    assert r4.reply.startswith("undid") and r4.placed
    assert ctx.session.scene.get("hall").shape["size"][1] == 12  # undo reverted only the roof move
    handle_chat(ctx, "undo", llm=llm)
    assert ctx.session.scene.get("hall").shape["size"][1] == 8
    assert len(ctx.bridge.world) == full_blocks
    # chat history + run log
    assert ctx.session.chat[0]["role"] == "user" and len(ctx.session.chat) == 10
    import os

    assert os.path.exists(os.path.join(str(tmp_path), "1.jsonl"))


def test_build_with_critic_fix_rounds(tmp_path):
    ctx = _ctx(tmp_path)
    critic = [
        {"score": 5, "top_3_fixes": [{"rule": "P1", "objects": ["hall"], "op_suggestion": "set_shape(id='hall', height=10)"}], "summary": "squat"},  # blocking → fix round
        {"score": 9, "top_3_fixes": [], "summary": "ok"},  # detailing
        {"score": 9, "top_3_fixes": [], "summary": "ok"},  # materials
        {"score": 9, "top_3_fixes": [], "summary": "ok"},  # decoration
        {"score": 6, "top_3_fixes": [{"rule": "P7", "objects": [], "op_suggestion": "add lanterns"}], "summary": "dark"},  # final 1 → fix
        {"score": 8, "top_3_fixes": [], "summary": "good"},  # final 2
    ]
    llm = ScriptedBuilderLLM(critic_responses=critic)
    r = handle_chat(ctx, "build a stone hall", llm=llm, fast=False)
    assert r.placed and "critic 8/10" in r.reply
    assert llm.roles.count("critic") == 6
    stages = r.data["stages"]
    names = [s["stage"] for s in stages]
    assert names == ["blocking", "blocking", "detailing", "materials", "decoration", "decoration"]
    fix_msgs = [q for q in llm.requests if any("FIX round" in str(m.get("content")) for m in q["messages"] if m["role"] == "user")]
    assert len(fix_msgs) >= 2
    assert any("Critic findings" in str(m.get("content")) and "squat" in str(m.get("content")) for q in llm.requests for m in q["messages"] if m["role"] == "user")


def test_meta_commands(tmp_path):
    ctx = _ctx(tmp_path)
    assert "Copilot" in handle_meta(ctx, "help").reply
    assert handle_meta(ctx, "build a house") is None
    ctx.session.apply("add", id="a", shape={"type": "box", "size": [2, 2, 2]})
    assert "box 2x2x2" in handle_meta(ctx, "/cp status").reply
    assert handle_meta(ctx, "preview on").reply.startswith("Live preview on") and ctx.session.live_preview
    r = handle_meta(ctx, "place")
    assert r.placed and ctx.session.world.is_placed()
    assert "exported castle" in handle_meta(ctx, "export castle").reply
    assert "minecraft:stone" in handle_meta(ctx, "materials").reply
    assert handle_meta(ctx, "render iso top").images and handle_meta(ctx, "snapshot v1").reply.startswith("snapshot")
    r = handle_meta(ctx, "reset")
    assert "cleared" in r.reply and not ctx.session.scene.objects and not ctx.session.world.is_placed()


def test_handle_chat_never_raises(tmp_path):
    ctx = _ctx(tmp_path)

    class Boom:
        supports_vision = False

        def chat(self, *a, **k):
            raise RuntimeError("kaboom")

    r = handle_chat(ctx, "build a castle", llm=Boom(), fast=True)
    assert r.reply.startswith("Built") or r.reply.startswith("Sorry")  # interpret degrades to defaults; stages error out gracefully
    ctx2 = _ctx(tmp_path)
    ctx2.session.apply("add", id="a", shape={"type": "box", "size": [2, 2, 2]})
    r2 = handle_chat(ctx2, "make it taller", llm=Boom(), fast=True)
    assert isinstance(r2.reply, str) and r2.reply


def test_system_prompt_without_registry(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.registry = None
    sp = system_prompt(ctx)
    assert "Materials catalog (abridged)" in sp and "{{materials_catalog}}" not in sp
