"""Tests for copilot.llm: MockLLM, the tool loop, JSON helpers, logging."""
from __future__ import annotations

import json
import os

from PIL import Image

from copilot import llm as L
from tests.fakes import FakeCtx


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
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")  # T2: JPEG payloads
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
    import base64
    import io

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


def test_azure_llm_times_out_with_one_retry(monkeypatch):
    """Each Azure call is bounded: 120 s timeout, one retry, no SDK-internal retries (T1)."""
    import types

    import httpx
    import openai
    import pytest

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    monkeypatch.delenv("AZURE_OPENAI_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_TIMEOUT_S", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API", "chat")
    a = L.AzureLLM()
    assert a.timeout == 120.0 and a.max_attempts == 2
    assert a.client.max_retries == 0

    calls = {"n": 0}
    sleeps = []
    monkeypatch.setattr(L.time, "sleep", lambda s: sleeps.append(s))

    def create(**kw):
        calls["n"] += 1
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://x"))

    a._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        a.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2 and len(sleeps) == 1  # one retry, no sleep after the last attempt

    # a transient failure followed by success still returns normally
    calls["n"] = 0

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://x"))
        msg = types.SimpleNamespace(content="ok", tool_calls=None)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)

    a._client.chat.completions.create = flaky
    assert a.chat([{"role": "user", "content": "hi"}]).text == "ok" and calls["n"] == 2


def test_azure_llm_retries_429_a_few_times_and_passes_timeout(monkeypatch):
    """T2: 429s get up to AZURE_OPENAI_RATE_LIMIT_ATTEMPTS (4) short attempts; the per-call timeout
    reaches the SDK and bounds the backoff sleep."""
    import types

    import httpx
    import openai
    import pytest

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    monkeypatch.delenv("AZURE_OPENAI_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_RATE_LIMIT_ATTEMPTS", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API", "chat")
    a = L.AzureLLM()
    assert a.rate_limit_attempts == 4
    sleeps = []
    monkeypatch.setattr(L.time, "sleep", lambda s: sleeps.append(s))
    seen = []

    def create(**kw):
        seen.append(kw)
        resp = httpx.Response(429, request=httpx.Request("POST", "https://x"), headers={"retry-after": "9"})
        raise openai.RateLimitError("rate limited", response=resp, body=None)

    a._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError, match="after 4 attempts"):
        a.chat([{"role": "user", "content": "hi"}], timeout=5.0)
    assert len(seen) == 4 and all(kw["timeout"] == 5.0 for kw in seen)
    assert sleeps == [5.0, 5.0, 5.0]  # retry-after 9 s capped by the 5 s call timeout
    assert "timeout" not in {k for kw in seen for k in kw} - {"timeout", "model", "messages", "temperature", "max_tokens"}
    # no timeout given -> nothing forwarded, plain timeouts still get one retry
    seen.clear()
    sleeps.clear()

    def timeout_(**kw):
        seen.append(kw)
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://x"))

    a._client.chat.completions.create = timeout_
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        a.chat([{"role": "user", "content": "hi"}])
    assert len(seen) == 2 and "timeout" not in seen[0]


def _responses_llm(monkeypatch, create, deployment="gpt-5.4-mini", **env):
    import types

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.services.ai.azure.com/openai/v1/responses")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", deployment)
    monkeypatch.delenv("AZURE_OPENAI_REASONING_HEADROOM_TOKENS", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    a = L.AzureLLM()
    assert a.api == "responses"
    a._client = types.SimpleNamespace(responses=types.SimpleNamespace(create=create))
    return a


def _incomplete(output_tokens):
    import types

    return types.SimpleNamespace(
        status="incomplete",
        incomplete_details=types.SimpleNamespace(reason="max_output_tokens"),
        output=[],
        usage=types.SimpleNamespace(input_tokens=100, output_tokens=output_tokens, total_tokens=100 + output_tokens),
    )


def _completed_call():
    import types

    call = types.SimpleNamespace(type="function_call", call_id="c1", name="add", arguments='{"id": "keep"}')
    return types.SimpleNamespace(status="completed", incomplete_details=None, output=[call], usage=None)


def test_responses_reasoning_models_get_headroom_above_max_tokens(monkeypatch):
    """Reasoning tokens count against max_output_tokens: a 4000-token visible budget must not cap the
    model's thinking, or medium/high effort silently returns nothing (seen in the object bench)."""
    seen = []

    def create(**kw):
        seen.append(kw)
        return _completed_call()

    a = _responses_llm(monkeypatch, create)
    r = a.chat([{"role": "user", "content": "hi"}], max_tokens=4000)
    assert r.tool_calls and r.tool_calls[0].name == "add"
    assert seen[0]["max_output_tokens"] == 4000 + 16000
    # non-reasoning deployments keep the plain budget
    seen.clear()
    b = _responses_llm(monkeypatch, create, deployment="gpt-4o")
    b.chat([{"role": "user", "content": "hi"}], max_tokens=4000)
    assert seen[0]["max_output_tokens"] == 4000
    # the headroom is tunable
    seen.clear()
    c = _responses_llm(monkeypatch, create, AZURE_OPENAI_REASONING_HEADROOM_TOKENS="500")
    c.chat([{"role": "user", "content": "hi"}], max_tokens=4000)
    assert seen[0]["max_output_tokens"] == 4500


def test_responses_truncated_reasoning_retries_once_then_errors(monkeypatch):
    """An `incomplete` response with no tool calls and no text is a truncation, not a normal stop: retry
    once with double the budget, then fail loudly so the stage records an error instead of 'ops 0'."""
    import pytest

    seen = []

    def create(**kw):
        seen.append(kw)
        return _incomplete(kw["max_output_tokens"]) if len(seen) == 1 else _completed_call()

    a = _responses_llm(monkeypatch, create)
    r = a.chat([{"role": "user", "content": "hi"}], max_tokens=4000)
    assert r.tool_calls and len(seen) == 2
    assert seen[1]["max_output_tokens"] == 2 * seen[0]["max_output_tokens"]

    seen.clear()

    def always_truncated(**kw):
        seen.append(kw)
        return _incomplete(kw["max_output_tokens"])

    b = _responses_llm(monkeypatch, always_truncated)
    with pytest.raises(RuntimeError, match="max_output_tokens"):
        b.chat([{"role": "user", "content": "hi"}], max_tokens=4000)
    assert len(seen) == 2

    # an incomplete response that still carries a tool call or text is used as-is
    import types

    partial = _incomplete(20000)
    partial.output = [types.SimpleNamespace(type="message", content=[types.SimpleNamespace(type="output_text", text="done")])]
    c = _responses_llm(monkeypatch, lambda **kw: partial)
    assert c.chat([{"role": "user", "content": "hi"}]).text == "done"


def test_foundry_deepseek_kimi_grok_get_reasoning_headroom_but_keep_temperature(monkeypatch):
    """The Foundry-hosted open models reason too (Kimi returns `reasoning` items) but accept temperature."""
    seen = []

    def create(**kw):
        seen.append(kw)
        return _completed_call()

    for dep in ("DeepSeek-V4-Pro", "Kimi-K2.7-Code", "grok-4.6"):
        seen.clear()
        a = _responses_llm(monkeypatch, create, deployment=dep)
        a.chat([{"role": "user", "content": "hi"}], max_tokens=4000)
        assert seen[0]["max_output_tokens"] == 20000, dep
        assert "temperature" in seen[0] and "reasoning" not in seen[0], dep
