"""T5 hardening: dispatcher never crashes and names bad fields, bridge retries + 'mod not connected',
session undo/redo/undo_world/reset with nothing to undo, sandbox getattr."""
import httpx
import pytest
from fastapi.testclient import TestClient

import copilot.bridge as bridge_mod
import copilot.server as server
from copilot.bridge import BridgeError, HttpBridge
from copilot.engine.scene import Scene, SceneError, apply_op, bad_args_message, op_arg_names
from copilot.pipeline.orchestrator import handle_meta
from copilot.placement import undo_world
from copilot.session import Session, SessionStore
from copilot.tools.dispatch import ToolContext, dispatch
from mock_mod import MockBridge

BOX = {"type": "box", "size": [2, 2, 2]}


def _ctx(events=None):
    s = Session("t")
    return ToolContext(session=s, bridge=MockBridge(), registry=None, log=(events.append if events is not None else None), run_dir=None)


# -- dispatcher --------------------------------------------------------------------------------
def test_dispatch_names_unknown_and_missing_fields():
    ctx = _ctx()
    r = dispatch(ctx, "add", {"id": "a", "shape": BOX, "bogus": 1}).text
    assert r.startswith("ERROR: add: unknown argument 'bogus'") and "valid arguments: id, shape" in r
    r = dispatch(ctx, "add", {"shape": BOX}).text
    assert r.startswith("ERROR: add: missing required argument(s) id")
    r = dispatch(ctx, "move", {"ids": "a", "by": [1, 0, 0]}).text
    assert "unknown argument 'by'" in r and "delta" in r
    assert "id" in op_arg_names("add") and op_arg_names("nope") == []
    assert bad_args_message("move", TypeError("something odd")).startswith("move: bad arguments")
    with pytest.raises(SceneError, match="unknown argument 'bogus'"):
        apply_op(Scene(), "add", id="a", shape=BOX, bogus=1)
    # the same message reaches scripts
    r = dispatch(ctx, "run_script", {"python": "scene.add(id='b', shape={'type':'box','size':[1,1,1]}, bogus=2)"}).text
    assert "add: unknown argument 'bogus'" in r


def test_dispatch_tool_exception_becomes_error_and_logs_traceback(monkeypatch):
    events = []
    ctx = _ctx(events)
    import copilot.tools.dispatch as d

    def boom(ctx, args):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(d._AGENT_TOOLS, "render", boom)
    r = dispatch(ctx, "render", {"views": ["iso"]})
    assert r.text == "ERROR: render failed: RuntimeError: kaboom"
    errs = [e for e in events if e.get("event") == "tool_error"]
    assert len(errs) == 1 and "kaboom" in errs[0]["traceback"] and errs[0]["name"] == "render"
    # a later call still works: the job is alive
    assert dispatch(ctx, "add", {"id": "a", "shape": BOX}).text.startswith("added a")


def test_duplicate_id_message_explains_partial_script_rerun():
    ctx = _ctx()
    dispatch(ctx, "add", {"id": "a", "shape": BOX})
    r = dispatch(ctx, "add", {"id": "a", "shape": BOX}).text
    assert "already exists" in r and "partial error" in r


def test_script_sandbox_getattr_public_only():
    ctx = _ctx()
    code = (
        "print(hasattr(scene, 'add'), hasattr(scene, '_x'))\n"
        "print(getattr(scene, 'nope', 'dflt'))\n"
        "try:\n    getattr(scene, '__class__')\nexcept AttributeError as e:\n    print('blocked')\n"
    )
    r = dispatch(ctx, "run_script", {"python": code}).text
    assert "True False" in r and "dflt" in r and "blocked" in r


# -- bridge ------------------------------------------------------------------------------------
def test_http_bridge_retries_connection_errors_with_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(bridge_mod.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"ok": True, "world_loaded": True})

    b = HttpBridge("http://mod.test", retries=2, backoff_s=0.5)
    b._client = httpx.Client(base_url="http://mod.test", transport=httpx.MockTransport(handler))
    assert b.health()["ok"] is True
    assert calls["n"] == 3 and sleeps == [0.5, 1.0]

    calls["n"] = -10  # never recovers
    with pytest.raises(BridgeError, match="mod not connected"):
        b.health()

    # read timeouts are not retried: the mod already received the request
    def slow(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("read timed out", request=request)

    calls["n"] = 0
    b._client = httpx.Client(base_url="http://mod.test", transport=httpx.MockTransport(slow))
    with pytest.raises(BridgeError, match="unreachable"):
        b.say("hi")
    assert calls["n"] == 1


class _DownBridge(MockBridge):
    def __init__(self, reason="down"):
        super().__init__()
        self.reason = reason

    def health(self):
        if self.reason == "down":
            raise BridgeError("mod not connected at http://127.0.0.1:8765 (/health: refused)")
        return {"ok": True, "world_loaded": False}


@pytest.mark.parametrize("reason", ["down", "noworld"])
def test_chat_reports_mod_not_connected_before_starting_a_job(reason):
    app = server.create_app(bridge=_DownBridge(reason), registry=None, store=SessionStore(), load_registry=False)
    with TestClient(app) as c:
        r = c.post("/chat", json={"player": "steve", "text": "build a hut"}).json()
        assert r["status"] == "rejected" and r["reply"].startswith("[cp] mod not connected") and r["mod_down"]
        if reason == "noworld":
            assert "world" in r["reply"]
        assert c.get("/jobs").json()["jobs"] == []


# -- session -----------------------------------------------------------------------------------
def test_undo_redo_undo_world_reset_with_nothing_to_undo():
    s = Session("t")
    assert s.undo() == "nothing to undo" and s.redo() == "nothing to redo"
    assert s.undo(5) == "nothing to undo"
    b = MockBridge()
    assert undo_world(s, b).startswith("nothing to undo in the world")
    ctx = ToolContext(session=s, bridge=b, registry=None, log=None, run_dir=None)
    assert dispatch(ctx, "undo", {}).text == "nothing to undo"
    assert dispatch(ctx, "undo_world", {}).text.startswith("nothing to undo in the world")
    # reset on an empty session is fine, and clears world-diff state after a placement
    assert "cleared" in handle_meta(ctx, "reset").reply
    dispatch(ctx, "add", {"id": "a", "shape": BOX, "material": "stone"})
    dispatch(ctx, "place", {"mode": "diff"})
    assert s.world.is_placed() and s.world.pre_scan is not None
    assert "cleared" in handle_meta(ctx, "reset").reply
    assert not s.world.is_placed() and s.world.pre_scan is None and s.world.anchor is None
    assert s.scene.objects == [] and s.brief is None
    assert s.undo() != "nothing to undo"  # the pre-reset scene is still one undo away
