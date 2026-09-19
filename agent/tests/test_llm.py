"""Tests for copilot.llm: MockLLM, the tool loop, JSON helpers, logging."""
from __future__ import annotations

import json
import os

from PIL import Image

from copilot import llm as L
from copilot.session import Session
from tests.fakes import FakeCtx, FakeToolResult


def _ctx(tmp_path=None, **kw):
    return FakeCtx(run_dir=str(tmp_path) if tmp_path else None, **kw)


def test_mock_llm_returns_script_in_order_and_records_requests():
    m = L.MockLLM([{"tool_calls": [{"name": "add", "args": {"id": "a"}}]}, {"text": "done"}])
    r1 = m.chat([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "add"}}], 0.2)
    assert r1.tool_calls[0].name == "add" and r1.tool_calls[0].args == {"id": "a"}
    r2 = m.chat([], None, 0.2)
    assert r2.text == "done" and not r2.tool_calls
    r3 = m.chat([], None, 0.2)
    assert "exhausted" in r3.text
    assert len(m.requests) == 3 and m.requests[0]["tools"] == ["add"] and m.requests[0]["temperature"] == 0.2


def test_tool_loop_executes_parallel_calls_in_order(tmp_path):
    ctx = _ctx(tmp_path)
    m = L.MockLLM(
        [
            {"tool_calls": [{"name": "add", "args": {"id": "a", "shape": {"type": "box", "size": [2, 2, 2]}}}, {"name": "add", "args": {"id": "b", "shape": {"type": "box", "size": [2, 2, 2]}, "pos": [4, 0, 0]}}]},
            {"text": "all done"},
        ]
    )
    res = L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch)
    assert res.stopped_reason == "text" and res.text == "all done"
    assert res.call_names() == ["add", "add"]
    assert [o.id for o in ctx.session.scene.objects] == ["a", "b"]
    msgs = m.requests[1]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[-3]["role"] == "assistant" and len(msgs[-3]["tool_calls"]) == 2
    assert msgs[-2]["role"] == "tool" and msgs[-1]["role"] == "tool"
    assert msgs[-2]["tool_call_id"] == msgs[-3]["tool_calls"][0]["id"]
    assert "added a" in msgs[-2]["content"]


def test_tool_loop_attaches_images_when_vision(tmp_path):
    ctx = _ctx(tmp_path)
    script = [{"tool_calls": [{"name": "render", "args": {"views": ["iso", "front"]}}]}, {"text": "looks good"}]
    m = L.MockLLM(list(script), supports_vision=True)
    res = L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch)
    assert res.images_shown == 2
    assert m.requests[1]["has_images"] is True
    last_user = [x for x in m.requests[1]["messages"] if x["role"] == "user"][-1]
    parts = last_user["content"]
    assert isinstance(parts, list) and sum(p.get("type") == "image_url" for p in parts) == 2
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # without vision: no image parts
    ctx2 = _ctx(tmp_path)
    m2 = L.MockLLM(list(script), supports_vision=False)
    res2 = L.run_tool_loop(m2, ctx2, "sys", [{"role": "user", "content": "go"}], None, ctx2.dispatch)
    assert res2.images_shown == 0 and m2.requests[1]["has_images"] is False


def test_tool_loop_stops_on_finish(tmp_path):
    ctx = _ctx(tmp_path)
    m = L.MockLLM([{"tool_calls": [{"name": "describe", "args": {}}, {"name": "finish", "args": {"summary": "blocked out the keep"}}]}, {"text": "should not be reached"}])
    res = L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch)
    assert res.stopped_reason == "finish" and res.text == "blocked out the keep"
    assert m.calls == 1 and res.call_names() == ["describe", "finish"]


def test_tool_loop_respects_max_calls_and_nudges(tmp_path):
    ctx = _ctx(tmp_path)
    steps = [{"tool_calls": [{"name": "describe", "args": {}}]} for _ in range(20)]
    m = L.MockLLM(steps)
    res = L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch, max_calls=7)
    assert res.stopped_reason == "max_calls" and res.tool_calls == 7
    # nudge appears once, after remaining <= 5 (i.e. after the 2nd call)
    nudges = [x for x in m.requests[-1]["messages"] if x["role"] == "system" and "tool calls left" in str(x["content"])]
    assert len(nudges) == 1
    assert not any("tool calls left" in str(x["content"]) for x in m.requests[1]["messages"] if x["role"] == "system")
    assert any("tool calls left" in str(x["content"]) for x in m.requests[2]["messages"] if x["role"] == "system")


def test_tool_loop_logs_jsonl_and_survives_tool_errors(tmp_path):
    ctx = _ctx(tmp_path)
    path = os.path.join(str(tmp_path), "0.jsonl")
    log = L.JsonlLogger(path)
    m = L.MockLLM([{"tool_calls": [{"name": "move", "args": {"ids": "ghost", "delta": [1, 0, 0]}}]}, {"tool_calls": [{"name": "nope_tool", "args": {}}]}, {"text": "ok"}])
    res = L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch, log=log)
    assert res.stopped_reason == "text"
    assert res.calls[0][2].startswith("ERROR in move") and "unknown object" in res.calls[0][2]
    assert res.calls[1][2].startswith("ERROR in nope_tool")
    lines = [json.loads(l) for l in open(path)]
    names = [r["name"] for r in lines]
    assert "move" in names and "nope_tool" in names and "__llm__" in names
    rec = next(r for r in lines if r["name"] == "move")
    assert set(rec) >= {"t", "stage", "name", "args", "result_preview", "ms"}


def test_default_logger_writes_under_run_dir(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.session.turn = 3
    m = L.MockLLM([{"text": "hi"}])
    L.run_tool_loop(m, ctx, "sys", [{"role": "user", "content": "go"}], None, ctx.dispatch)
    assert os.path.exists(os.path.join(str(tmp_path), "3.jsonl"))


def test_extract_json_and_parse_args():
    assert L.extract_json('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert L.extract_json('Sure! {"score": 7, "top_3_fixes": []} — done') == {"score": 7, "top_3_fixes": []}
    assert L.extract_json("no json here") is None
    assert L.parse_args('{"id": "x", "pos": [1, 2') == {"id": "x", "pos": [1, 2]}
    assert L.parse_args("") == {} and L.parse_args(None) == {}
    assert L.parse_args({"k": 1}) == {"k": 1}


def test_image_content_downscales():
    part = L.image_content(Image.new("RGB", (2048, 1024)), max_px=512)
    assert part["type"] == "image_url" and part["image_url"]["detail"] == "high"
    import base64, io

    data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
    im = Image.open(io.BytesIO(data))
    assert max(im.size) == 512


def test_strip_images_replaces_parts_with_note():
    msgs = [{"role": "user", "content": [L.text_content("a"), L.image_content(Image.new("RGB", (4, 4)))]}]
    out = L.strip_images(msgs)
    assert all(p["type"] == "text" for p in out[0]["content"]) and "omitted" in out[0]["content"][-1]["text"]


def test_azure_llm_requires_config(monkeypatch):
    for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_VISION_DEPLOYMENT", "AZURE_OPENAI_API"):
        monkeypatch.delenv(k, raising=False)
    import pytest

    with pytest.raises(RuntimeError):
        L.AzureLLM()
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    a = L.AzureLLM()
    assert a.supports_vision and a.vision_deployment == "gpt-4o"
