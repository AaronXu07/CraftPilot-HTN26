import pytest
from fastapi.testclient import TestClient

import copilot.server as server
from copilot.session import SessionStore
from mock_mod import MockBridge


@pytest.fixture
def client():
    app = server.create_app(bridge=MockBridge(), registry=None, store=SessionStore(), load_registry=False)
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] and j["bridge"] == "MockBridge" and j["bridge_ok"] is True and j["sessions"] == 0


def test_chat_direct_ops_fallback(client, monkeypatch):
    monkeypatch.setattr(server, "handle_chat_for", lambda ctx, text: server.direct_ops_chat(ctx, text))
    r = client.post("/chat", json={"player": "steve", "text": '/op add {"id":"keep","shape":{"type":"box","size":[10,8,10]},"pos":[0,0,0],"material":"stone"}'})
    assert r.status_code == 200 and r.json()["reply"].startswith("added keep")
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


def test_chat_uses_pipeline_when_present(client, monkeypatch):
    class FakeRes:
        reply = "built it"
        images = []
        brief = {"build_type": "hut"}

    monkeypatch.setattr(server, "handle_chat_for", lambda ctx, text: server.ChatResult("built it", [], {"build_type": "hut"}))
    r = client.post("/chat", json={"player": "p", "text": "build a hut"})
    assert r.json() == {"reply": "built it", "brief": {"build_type": "hut"}, "images": []}


def test_pipeline_exception_is_reported(client, monkeypatch):
    def boom(ctx, text):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(server, "handle_chat_for", boom)
    r = client.post("/chat", json={"player": "p", "text": "x"})
    assert r.status_code == 200 and "kaboom" in r.json()["reply"]
