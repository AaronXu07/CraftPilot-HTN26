"""The building-vs-object router: word rules (no network) and the mocked LLM tie-break."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from craftpilot import route as R
from craftpilot.route import LLM_THRESHOLD, Route, classify, rule_route

# (text, kind, confident) - confident = the rules alone decide, the LLM is never asked
CASES = [
    ("build the eiffel tower", "object", True),
    ("build the Eiffel Tower", "object", True),
    ("big ben", "object", True),
    ("statue of liberty", "object", True),
    ("a golden gate bridge", "object", True),
    ("the taj mahal", "object", True),
    ("hogwarts castle", "object", True),
    ("the white house", "object", True),
    ("build a big suspension bridge", "object", True),
    ("a statue of a dragon", "object", True),
    ("a dragon statue", "object", True),
    ("a 40 block tall dragon statue", "object", True),
    ("a pirate ship", "object", True),
    ("make me a pirate ship", "object", True),
    ("a medieval bridge over the river", "object", True),
    ("build a stone arch bridge 40 blocks long", "object", True),
    ("a windmill", "object", True),
    ("an aqueduct 60 blocks long", "object", True),
    ("build a t-rex", "object", True),
    ("a giant pikachu", "object", True),
    ("build a tower with these dimensions 20x40x20", "building", True),
    ("build a tower with these dimensions and with this structure", "building", True),
    ("a tower 20 blocks tall with arrow slits", "building", True),
    ("cottage with a red roof", "building", True),
    ("a stone lighthouse", "building", True),
    ("small stone cottage", "building", True),
    ("small cottage", "building", True),
    ("a cozy cottage", "building", True),
    ("a white house", "building", True),
    ("a dragon-themed castle", "building", True),
    ("a wizard tower", "building", True),
    ("a japanese pagoda", "building", True),
    ("build a skyscraper 20 floors", "building", True),
    ("a tall office building with 5 floors", "building", True),
    ("something with two floors and a red roof", "building", True),
    ("a red timber barn with a silo", "building", True),
    ("a haunted mansion", "building", True),
    ("build a modern villa with a pool and two storeys", "building", True),
    # unsure: the rules lean one way but the LLM gets to confirm
    ("a house shaped like a pineapple", "object", False),
    ("a pineapple house", "building", False),
    ("a dog house", "building", False),
    ("build me something cool", "building", False),
    ("build a wooden sailing ship on the ground with a stern castle", "object", False),
]


@pytest.mark.parametrize("text,kind,confident", CASES)
def test_rule_table(text, kind, confident):
    r = rule_route(text)
    assert r.kind == kind, (text, r)
    assert (r.confidence >= LLM_THRESHOLD) == confident, (text, r)
    assert classify(text, use_llm=False).source == "rule"


def test_landmark_beats_the_generic_tower_word():
    r = rule_route("build the eiffel tower")
    assert r.source == "rule" and r.confidence == R.LANDMARK_CONFIDENCE
    assert r.matched == ["landmark:eiffel tower"]  # no head:tower=building evidence at all


def test_theme_words_score_nothing():
    r = rule_route("a dragon-themed castle")
    assert "word:dragon=object(modifier)" in r.matched and r.scores[1] == 0 and r.kind == "building"


def test_bounds_and_exemplars_count_as_building_evidence():
    r = rule_route("a big tower", has_bounds=True)
    assert any(m.startswith("cue:a bounding box") for m in r.matched) and r.kind == "building"
    r = rule_route("a red timber barn with a silo")
    assert "cue:like the barn exemplar" in r.matched


def test_capitalised_names_are_object_evidence():
    r = rule_route("build the Krusty Krab")  # not in the landmark list, but typed as a name
    assert r.kind == "object" and any(m.startswith("proper:") for m in r.matched)
    assert rule_route("build the krusty krab").kind == "building"  # lowercase: no name signal, default wins


def test_override_wins_and_never_calls_the_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(R, "_llm_available", lambda: True)
    monkeypatch.setattr(R, "_llm_route", lambda text, rule: calls.append(text) or rule)
    r = classify("a dragon", override="building")
    assert (r.kind, r.source, r.confidence) == ("building", "override", 1.0)
    r = classify("a house", override="object")
    assert (r.kind, r.source) == ("object", "override")
    assert calls == []


def test_route_note_and_dict():
    r = Route("object", "llm", 0.9, "silhouette house", ["llm"], (3, 5))
    assert r.note == "routed to objects (LLM): silhouette house"
    d = r.to_dict()
    assert d["scores"] == [3, 5] and d["note"] == r.note and d["kind"] == "object"


# --- the LLM layer, with structured_call faked ---------------------------------------------------------------

@pytest.fixture
def llm(monkeypatch):
    """A fake structured_call recording its kwargs; `answers` is consumed in order (str -> output, Exception -> raise)."""
    import craftpilot.llm.azure as azure

    state = {"calls": [], "answers": []}

    def fake(deployment, instructions, messages, fmt, effort, cache=True, tag="compose", **kw):
        state["calls"].append({"deployment": deployment, "messages": messages, "fmt": fmt, "effort": effort,
                               "tag": tag, **kw})
        answer = state["answers"].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer, {"cached": False, "seconds": 0.1}

    monkeypatch.setattr(azure, "structured_call", fake)
    monkeypatch.setattr(R, "_llm_available", lambda: True)
    monkeypatch.setattr(R, "route_deployment", lambda: "gpt-5.4-mini")
    return state


def test_llm_confirms_an_unsure_rule(llm):
    from craftpilot.config import SETTINGS

    llm["answers"] = [json.dumps({"kind": "object", "confidence": 0.9, "reason": "silhouette-defined house"})]
    r = classify("a house shaped like a pineapple")
    assert (r.kind, r.source, r.confidence) == ("object", "llm", 0.9)
    assert r.reason == "silhouette-defined house" and "llm" in r.matched
    call = llm["calls"][0]
    assert call["tag"] == "route" and call["effort"] == "low" and call["deployment"] == "gpt-5.4-mini"
    assert call["timeout"] == SETTINGS.route_timeout and call["max_retries"] == 0 and call["fmt"]["name"] == "build_route"
    assert call["messages"] == [{"role": "user", "content": "a house shaped like a pineapple"}]


def test_llm_can_overturn_the_rules(llm):
    llm["answers"] = [json.dumps({"kind": "object", "confidence": 0.8, "reason": "the famous pineapple house"})]
    assert classify("a pineapple house").kind == "object"


def test_llm_failure_falls_back_to_the_rules(llm):
    llm["answers"] = [TimeoutError("slow")]
    r = classify("a house shaped like a pineapple")
    assert r.source == "fallback" and r.kind == rule_route("a house shaped like a pineapple").kind
    assert "TimeoutError" in r.reason


@pytest.mark.parametrize("bad", ["not json", json.dumps({"kind": "castle", "confidence": 1, "reason": ""}), ""])
def test_llm_garbage_falls_back_to_the_rules(llm, bad):
    llm["answers"] = [bad]
    assert classify("a dog house").source == "fallback"


def test_confident_rules_never_ask(llm):
    llm["answers"] = [json.dumps({"kind": "object", "confidence": 1, "reason": "x"})]
    assert classify("a cozy cottage").source == "rule"
    assert classify("the eiffel tower").source == "rule"
    assert classify("a dog house", use_llm=False).source == "rule"
    assert llm["calls"] == []


def test_router_stays_light():
    code = ("import sys, craftpilot.route; "
            "assert 'trimesh' not in sys.modules and 'openai' not in sys.modules; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"
