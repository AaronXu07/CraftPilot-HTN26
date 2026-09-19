"""Router mapping for 15 edit phrasings (T6).

The router is an LLM call; here a MockLLM answers each phrasing with the JSON the router prompt asks
for (in the varied shapes real replies take: fenced, prose-prefixed, `op`/`tool` keys, string bools)
and we check that `route()` maps it to the right stages / selection / direct ops — and that every
direct op actually applies to the scene from the outline.
"""
import json

import pytest

from copilot import llm as L
from copilot.pipeline import route
from tests.fakes import FakeCtx

# (phrase, raw router reply, expected route fields, scene check after applying direct ops)
CASES = [
    ("make the northeast tower 8 blocks taller",
     '{"intent":"edit","stages":[],"selection":"tower_ne","direct_ops":[{"name":"set_shape","args":{"id":"tower_ne","height":28}},{"name":"move","args":{"ids":"tower_ne_roof","delta":[0,8,0]}}],"needs_place":true}',
     dict(intent="edit", stages=[], selection="tower_ne", ops=["set_shape", "move"], place=True),
     lambda sc: sc.get("tower_ne").shape["height"] == 28 and sc.get("tower_ne_roof").transform.pos[1] == 28),
    ("give it a copper roof",
     '```json\n{"intent":"edit","stages":["materials"],"selection":"name:*_roof","direct_ops":[],"needs_place":true}\n```',
     dict(intent="edit", stages=["materials"], selection="name:*_roof", ops=[], place=True), None),
    ("move the whole thing 10 blocks east",
     'Sure. {"intent":"edit","stages":[],"selection":"all","direct_ops":[{"op":"move","ids":"all","delta":[10,0,0]}],"needs_place":"true"}',
     dict(intent="edit", stages=[], selection="all", ops=["move"], place=True),
     lambda sc: all(o.transform.pos[0] >= 10 for o in sc.objects)),
    ("shift the keep 4 south",
     '{"intent":"edit","stages":[],"selection":"keep","direct_ops":[{"tool":"move","arguments":{"ids":"keep","delta":[0,0,4]}}],"needs_place":true}',
     dict(intent="edit", stages=[], selection="keep", ops=["move"], place=True),
     lambda sc: sc.get("keep").transform.pos[2] == 4),
    ("delete the gatehouse",
     '{"intent":"edit","stages":[],"selection":"gatehouse","direct_ops":[{"name":"delete","args":{"ids":"gatehouse"}}],"needs_place":true}',
     dict(intent="edit", stages=[], selection="gatehouse", ops=["delete"], place=True),
     lambda sc: not sc.has("gatehouse")),
    ("make the keep hexagonal",
     '{"intent":"edit","stages":["blocking"],"selection":"keep","direct_ops":[{"name":"set_shape","args":{"id":"keep","type":"prism","sides":6,"radius":9,"height":16}}],"needs_place":true}',
     dict(intent="edit", stages=["blocking"], selection="keep", ops=["set_shape"], place=True),
     lambda sc: sc.get("keep").shape["type"] == "prism" and sc.get("keep").shape["sides"] == 6),
    ("add arrow slits to the north wall",
     '{"intent":"edit","stages":["detailing"],"selection":"wall_n","direct_ops":[],"needs_place":true}',
     dict(intent="edit", stages=["detailing"], selection="wall_n", ops=[], place=True), None),
    ("swap the walls to deepslate with a mossy base",
     '{"intent":"edit","stages":["materials"],"selection":"material:castle_wall","direct_ops":[],"needs_place":true}',
     dict(intent="edit", stages=["materials"], selection="material:castle_wall", ops=[], place=True), None),
    ("add lanterns along the walls",
     '{"intent":"edit","stages":["decoration"],"selection":"name:wall_*","direct_ops":[],"needs_place":true}',
     dict(intent="edit", stages=["decoration"], selection="name:wall_*", ops=[], place=True), None),
    ("make it more detailed",
     '{"intent":"edit","stages":["detailing","decoration"],"selection":"all","direct_ops":[],"needs_place":true}',
     dict(intent="edit", stages=["detailing", "decoration"], selection="all", ops=[], place=True), None),
    ("widen the keep to 24 by 24",
     '{"intent":"edit","stages":[],"selection":"keep","direct_ops":[{"name":"set_shape","args":{"id":"keep","size":[24,12,24]}}],"needs_place":true}',
     dict(intent="edit", stages=[], selection="keep", ops=["set_shape"], place=True),
     lambda sc: sc.get("keep").shape["size"] == [24.0, 12.0, 24.0]),
    ("paint the roofs dark",
     '{"intent":"edit","stages":["materials"],"selection":"name:*_roof","direct_ops":[{"name":"set_material","args":{"ids":"name:*_roof","material":"deepslate_tiles"}}],"needs_place":true}',
     dict(intent="edit", stages=["materials"], selection="name:*_roof", ops=["set_material"], place=True),
     lambda sc: sc.get("tower_ne_roof").material == "deepslate_tiles"),
    ("mirror the northeast tower to the west side",
     '{"intent":"edit","stages":[],"selection":"name:tower_ne*","direct_ops":[{"name":"mirror_copy","args":{"ids":["tower_ne","tower_ne_roof"],"axis":"x","plane":0,"new_suffix":"_w"}}],"needs_place":true}',
     dict(intent="edit", stages=[], selection="name:tower_ne*", ops=["mirror_copy"], place=True),
     lambda sc: sc.has("tower_ne_w") and sc.has("tower_ne_roof_w")),
    ("how tall is the keep?",
     '{"intent":"question","stages":[],"selection":"keep","direct_ops":[],"needs_place":true}',
     dict(intent="question", stages=[], selection="keep", ops=[], place=False), None),
    ("build a lighthouse next to it",
     '{"intent":"build","stages":["blocking","detailing","materials"],"selection":"all","direct_ops":[],"needs_place":true}',
     dict(intent="build", stages=["blocking", "detailing", "materials"], selection="all", ops=[], place=True), None),
]


def _ctx(tmp_path):
    ctx = FakeCtx(run_dir=str(tmp_path))
    s = ctx.session
    s.apply("add", id="keep", shape={"type": "box", "size": [18, 12, 18]}, pos=[0, 0, 0], material="castle_wall")
    s.apply("add", id="tower_ne", shape={"type": "cylinder", "radius": 4, "height": 20}, pos=[12, 0, -12], material="castle_wall")
    s.apply("add", id="tower_ne_roof", shape={"type": "cone", "radius": 5, "height": 6}, pos=[12, 20, -12], material="roof_slate")
    s.apply("add", id="wall_n", shape={"type": "box", "size": [30, 8, 2]}, pos=[0, 0, -14], material="castle_wall")
    s.apply("add", id="gatehouse", shape={"type": "box", "size": [8, 10, 6]}, pos=[0, 0, 14], material="castle_wall")
    return ctx


@pytest.mark.parametrize("phrase,reply,expect,check", CASES, ids=[c[0] for c in CASES])
def test_router_phrasing(tmp_path, phrase, reply, expect, check):
    ctx = _ctx(tmp_path)
    m = L.MockLLM([{"text": reply}])
    r = route(m, ctx, phrase)
    assert r.intent == expect["intent"] and r.stages == expect["stages"] and r.selection == expect["selection"]
    assert [op["name"] for op in r.direct_ops] == expect["ops"] and r.needs_place is expect["place"]
    req = m.requests[0]
    assert req["messages"][-1]["content"].strip() == phrase and req["temperature"] == 0.0
    assert "keep" in req["messages"][0]["content"] and "tower_ne" in req["messages"][0]["content"]
    for op in r.direct_ops:
        ctx.session.apply(op["name"], **op["args"])
    if check is not None:
        assert check(ctx.session.scene), json.dumps(r.to_dict())


def test_unknown_direct_ops_are_dropped_but_stage_kept(tmp_path):
    ctx = _ctx(tmp_path)
    m = L.MockLLM([{"text": '{"intent":"edit","stages":["detailing"],"selection":"keep","direct_ops":[{"name":"teleport","args":{"id":"keep"}}]}'}])
    r = route(m, ctx, "teleport the keep")
    assert r.direct_ops == [] and r.stages == ["detailing"]


def test_router_llm_failure_falls_back_to_detailing(tmp_path):
    ctx = _ctx(tmp_path)

    class Boom:
        supports_vision = False

        def chat(self, *a, **k):
            raise RuntimeError("azure down")

    r = route(Boom(), ctx, "make it prettier")
    assert r.intent == "edit" and r.stages == ["detailing"] and r.selection == "all" and r.note == "unparsed"
