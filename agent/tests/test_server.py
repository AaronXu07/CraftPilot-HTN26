import threading
import time

import pytest
from fastapi.testclient import TestClient

import copilot.server as server
from copilot.jobs import JobCancelled, check_cancel
from copilot.session import SessionStore
from mock_mod import MockBridge


@pytest.fixture
def bridge():
    return MockBridge()


@pytest.fixture
def client(bridge):
    app = server.create_app(bridge=bridge, registry=None, store=SessionStore(), load_registry=False)
    with TestClient(app) as c:
        yield c


def wait_status(client, job_id, statuses=("done", "failed", "cancelled", "timeout"), timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in statuses:
            return j
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {statuses}: {client.get(f'/jobs/{job_id}').json()}")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] and j["bridge"] == "MockBridge" and j["bridge_ok"] is True and j["sessions"] == 0


def test_chat_direct_ops_fallback(client, monkeypatch):
    monkeypatch.setattr(server, "handle_chat_for", lambda ctx, text: server.direct_ops_chat(ctx, text))
    r = client.post("/chat", json={"player": "steve", "text": '/op add {"id":"keep","shape":{"type":"box","size":[10,8,10]},"pos":[0,0,0],"material":"stone"}'})
    assert r.status_code == 200 and r.json()["status"] == "done" and r.json()["reply"].startswith("added keep")
    r = client.post("/chat", json={"player": "steve", "text": "describe"})
    assert "keep" in r.json()["reply"]
    r = client.post("/chat", json={"player": "steve", "text": "/op move {bad json"})
    assert r.json()["reply"].startswith("ERROR")
    r = client.post("/chat", json={"player": "steve", "text": "build me a castle"})
    assert "No LLM pipeline" in r.json()["reply"]
    assert client.post("/chat", json={"player": "steve", "text": ""}).status_code == 400
    s = client.get("/sessions").json()["sessions"]
    assert s[0]["player"] == "steve" and s[0]["objects"] == 1 and s[0]["turn"] == 4
    sc = client.get("/sessions/steve/scene").json()
    assert sc["scene"]["objects"][0]["id"] == "keep" and "keep" in sc["outline"]
    assert client.post("/sessions/steve/reset").json()["ok"]
    assert client.get("/sessions").json()["sessions"][0]["objects"] == 0


def test_short_turn_is_answered_inline(client, bridge, monkeypatch):
    monkeypatch.setattr(server, "handle_chat_for", lambda ctx, text: server.ChatResult("built it", [], {"build_type": "hut"}))
    r = client.post("/chat", json={"player": "p", "text": "build a hut"})
    j = r.json()
    assert j["job_id"] == 1 and j["status"] == "done" and j["reply"] == "built it" and j["brief"] == {"build_type": "hut"}
    assert bridge.chat == []  # answered inline, nothing pushed through /say
    assert client.get("/jobs/1").json()["reply"] == "built it"
    assert client.get("/jobs/99").status_code == 404


def test_pipeline_exception_is_reported(client, bridge, monkeypatch):
    def boom(ctx, text):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(server, "handle_chat_for", boom)
    r = client.post("/chat", json={"player": "p", "text": "x"})
    j = r.json()
    assert r.status_code == 200 and j["status"] == "failed" and "kaboom" in j["reply"]
    assert "kaboom" in client.get("/jobs/1").json()["error"]


class SlowPipeline:
    """Fake pipeline: blocks until released, polling check_cancel like the real tool loop does."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()

    def __call__(self, ctx, text):
        self.started.set()
        ctx.session.stage = "blocking"
        while not self.release.wait(0.02):
            check_cancel(ctx)
        ctx.session.stage = None
        return server.ChatResult(f"done: {text}")


def test_long_turn_returns_job_id_and_pushes_reply_via_say(client, bridge, monkeypatch):
    pipe = SlowPipeline()
    monkeypatch.setattr(server, "CHAT_INLINE_WAIT_S", 0.05)
    monkeypatch.setattr(server, "handle_chat_for", pipe)
    t0 = time.time()
    r = client.post("/chat", json={"player": "p", "text": "build a castle"})
    assert time.time() - t0 < 1.0
    j = r.json()
    assert j["status"] in ("queued", "running") and j["reply"] is None and j["job_id"] == 1
    assert pipe.started.wait(1.0)
    st = client.get("/jobs/1").json()
    assert st["status"] == "running" and st["stage"] == "blocking" and st["elapsed_s"] >= 0 and "running" in st["line"]
    # a second request for the same player while busy is refused with a pointer to the job
    r2 = client.post("/chat", json={"player": "p", "text": "add a moat"}).json()
    assert r2["busy"] and r2["job_id"] == 1 and "still working" in r2["reply"]
    # `status` while busy reports the job instead of the scene
    r3 = client.post("/chat", json={"player": "p", "text": "status"}).json()
    assert r3["job_id"] == 1 and "job 1: running" in r3["reply"]
    # another player is not blocked
    monkeypatch.setattr(server, "handle_chat_for", lambda ctx, text: server.ChatResult("hi"))
    assert client.post("/chat", json={"player": "q", "text": "hello"}).json()["reply"] == "hi"
    pipe.release.set()
    fin = wait_status(client, 1)
    assert fin["status"] == "done" and fin["reply"] == "done: build a castle" and fin["stage"] is None
    assert bridge.chat == ["done: build a castle"]  # pushed to the player's chat
    assert client.get("/jobs").json()["jobs"][0]["status"] == "done"


def test_cancel_endpoint_stops_job(client, bridge, monkeypatch):
    pipe = SlowPipeline()
    monkeypatch.setattr(server, "CHAT_INLINE_WAIT_S", 0.05)
    monkeypatch.setattr(server, "handle_chat_for", pipe)
    r = client.post("/chat", json={"player": "p", "text": "build a castle"}).json()
    assert pipe.started.wait(1.0)
    c = client.post(f"/jobs/{r['job_id']}/cancel").json()
    assert "cancelling" in c["line"]
    fin = wait_status(client, r["job_id"])
    assert fin["status"] == "cancelled" and fin["reply"] == "[cp] stopped: cancelled"
    assert bridge.chat == ["[cp] stopped: cancelled"]
    # cancelling a finished job is a no-op
    assert client.post(f"/jobs/{r['job_id']}/cancel").json()["status"] == "cancelled"
    assert client.post("/jobs/42/cancel").status_code == 404
    # the player can start a new turn afterwards
    monkeypatch.setattr(server, "handle_chat_for", lambda ctx, text: server.ChatResult("ok"))
    assert client.post("/chat", json={"player": "p", "text": "undo"}).json()["reply"] == "ok"


def test_cancel_via_chat_text(client, bridge, monkeypatch):
    pipe = SlowPipeline()
    monkeypatch.setattr(server, "CHAT_INLINE_WAIT_S", 0.05)
    monkeypatch.setattr(server, "handle_chat_for", pipe)
    r = client.post("/chat", json={"player": "p", "text": "build a castle"}).json()
    assert pipe.started.wait(1.0)
    c = client.post("/chat", json={"player": "p", "text": "/cp cancel"}).json()
    assert c["cancelling"] and c["job_id"] == r["job_id"]
    assert wait_status(client, r["job_id"])["status"] == "cancelled"


def test_hard_budget_times_out_a_stuck_job(client, bridge, monkeypatch):
    """A job that ignores cancellation is still closed by the watchdog and the player is told."""
    hang = threading.Event()

    def stuck(ctx, text):
        hang.wait(5.0)
        return server.ChatResult("too late")

    monkeypatch.setattr(server, "CHAT_INLINE_WAIT_S", 0.05)
    monkeypatch.setattr(server, "handle_chat_for", stuck)
    monkeypatch.setenv("COPILOT_HARD_BUDGET_S", "0.3")
    r = client.post("/chat", json={"player": "p", "text": "build a castle"}).json()
    assert r["reply"] is None
    fin = wait_status(client, r["job_id"], timeout=3.0)
    assert fin["status"] == "timeout" and fin["reply"] == "[cp] stopped: over time budget"
    assert bridge.chat == ["[cp] stopped: over time budget"]
    hang.set()
    time.sleep(0.1)
    assert client.get(f"/jobs/{r['job_id']}").json()["reply"] == "[cp] stopped: over time budget"  # late result dropped


def test_cooperative_budget_raises_in_pipeline(client, bridge, monkeypatch):
    """The pipeline sees the budget through check_cancel and unwinds with JobCancelled → timeout."""
    seen = {}

    def pipe(ctx, text):
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                check_cancel(ctx)
            except JobCancelled as e:
                seen["reason"] = e.reason
                raise
            time.sleep(0.01)
        return server.ChatResult("finished")

    monkeypatch.setattr(server, "CHAT_INLINE_WAIT_S", 0.05)
    monkeypatch.setattr(server, "handle_chat_for", pipe)
    monkeypatch.setenv("COPILOT_HARD_BUDGET_S", "0.3")
    r = client.post("/chat", json={"player": "p", "text": "build"}).json()
    fin = wait_status(client, r["job_id"], timeout=3.0)
    assert fin["status"] == "timeout" and seen["reason"] == "over time budget"
