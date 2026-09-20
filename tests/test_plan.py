"""Plan mode: describing a program, /plan, refining with /edit {plan}, and building the pending plan."""

from __future__ import annotations

from fastapi.testclient import TestClient

from craftpilot.config import SETTINGS
from craftpilot.place.bridge import FakeBridge
from craftpilot.program.describe import describe
from craftpilot.program.exemplars import load_all


def test_describe_every_exemplar_is_short_and_mentions_the_essentials():
    for ex in load_all(SETTINGS.exemplars_dir):
        lines = describe(ex.program)
        assert 5 <= len(lines) <= 14, ex.name
        assert ex.program.label in lines[0] and "wide" in lines[0]
        root = ex.program.root()
        root_line = next(line for line in lines if line.startswith(root.name + ":"))
        assert f"{root.floors} floor" in root_line and root.roof.type.value in root_line
        assert any(line.startswith("materials: ") and "walls " in line for line in lines)
        assert all(len(line) < 260 for line in lines)


def test_plan_then_edit_then_go(monkeypatch, tmp_path):
    from craftpilot import serve
    from craftpilot.place import bridge as bridge_mod

    live = FakeBridge(pos=(500.5, 70.0, 500.5), yaw=0.0)
    monkeypatch.setattr(bridge_mod, "HttpBridge", lambda *a, **k: live)
    monkeypatch.setattr(SETTINGS, "schematics_dir", tmp_path)
    client = TestClient(serve.app)

    r = client.post("/plan", json={"text": "a stone lighthouse", "player": "p", "use_llm": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["plan"][0].startswith("a stone lighthouse - ") and "placement" not in body and "schematic" not in body
    assert body["bounds"]["width"] > 0 and body["summary"].startswith("Plan:")
    assert live.calls == []                                   # nothing placed, nothing asked of the mod

    def fake_edit(previous, text):
        updated = previous.model_copy(deep=True)
        updated.label = previous.label + " (edited)"
        return updated, "llm", []

    monkeypatch.setattr("craftpilot.llm.compose.edit", fake_edit)
    r = client.post("/edit", json={"text": "make it taller", "player": "p", "plan": True})
    assert r.status_code == 200, r.text
    assert r.json()["plan"][0].startswith("a stone lighthouse (edited) - ") and "placement" not in r.json()

    r = client.post("/regenerate", json={"player": "p", "place": True, "seed": 4,
                                         "pos": [0.5, 64.0, 0.5], "yaw": 180.0})
    assert r.status_code == 200, r.text
    placement = r.json()["placement"]
    assert placement["origin"][1] == 64 and len(live.calls) == 1
    assert r.json()["bounds"] == [body["bounds"]["width"], body["bounds"]["height"], body["bounds"]["depth"]]


def test_plan_with_bounds_hint_and_unknown_player_edit():
    from craftpilot import serve

    client = TestClient(serve.app)
    r = client.post("/plan", json={"text": "small cottage", "player": "q", "use_llm": False,
                                   "bounds": {"width": 15, "height": 18, "depth": 13}})
    assert r.status_code == 200 and r.json()["bounds"] == {"width": 15, "height": 18, "depth": 13}
    r = client.post("/edit", json={"text": "x", "player": "nobody", "plan": True})
    assert r.status_code == 404


def test_ghost_matches_the_build_it_commits(monkeypatch, tmp_path):
    from craftpilot import serve
    from craftpilot.place import bridge as bridge_mod
    from craftpilot.place.placer import AIR

    live = FakeBridge(pos=(0.5, 64.0, 0.5), yaw=180.0)
    monkeypatch.setattr(bridge_mod, "HttpBridge", lambda *a, **k: live)
    monkeypatch.setattr(SETTINGS, "schematics_dir", tmp_path)
    client = TestClient(serve.app)
    bounds = {"width": 13, "height": 16, "depth": 11}

    r = client.post("/build", json={"text": "small stone cottage", "player": "g", "use_llm": False, "seed": 5,
                                    "bounds": bounds, "ghost": True, "place": True})
    assert r.status_code == 200, r.text
    body = r.json()
    ghost = body["ghost"]
    assert "placement" not in body and live.calls == [] and live.outlines == []
    assert body["seed"] == 5 and body["bounds"] == bounds and body["plan"]
    assert (ghost["width"], ghost["depth"]) == (13, 11) and 0 < ghost["height"] <= 16
    flat = ghost["blocks"]
    assert len(flat) == 4 * body["blocks"] and body["blocks"] > 100
    for i in range(0, len(flat), 4):
        x, y, z, rgb = flat[i:i + 4]
        assert 0 <= x < 13 and 0 <= y < ghost["height"] and 0 <= z < 11 and 0 <= rgb <= 0xFFFFFF

    r = client.post("/regenerate", json={"player": "g", "place": True, "seed": 5,
                                         "pos": [0.5, 64.0, 0.5], "yaw": 180.0})
    assert r.status_code == 200, r.text
    placed = {pos for pos, state in live.world.items() if state != AIR}
    assert len(placed) == body["blocks"]                      # same grid, block for block
    ox, oy, oz = r.json()["placement"]["origin"]
    # Facing south is the engine's native orientation, so the ghost maps onto the world by translation.
    ghost_cells = {(flat[i] + ox, flat[i + 1] + oy, flat[i + 2] + oz) for i in range(0, len(flat), 4)}
    assert ghost_cells == placed


def test_ghost_from_plan_and_from_edit(monkeypatch):
    from craftpilot import serve

    client = TestClient(serve.app)
    r = client.post("/plan", json={"text": "a stone lighthouse", "player": "h", "use_llm": False})
    assert r.status_code == 200
    r = client.post("/regenerate", json={"player": "h", "ghost": True, "seed": 2})
    assert r.status_code == 200 and r.json()["ghost"]["blocks"] and r.json()["seed"] == 2

    monkeypatch.setattr("craftpilot.llm.compose.edit", lambda prev, text: (prev, "llm", []))
    r = client.post("/edit", json={"text": "taller", "player": "h", "ghost": True})
    assert r.status_code == 200 and r.json()["ghost"]["blocks"] and "placement" not in r.json()
    assert client.post("/regenerate", json={"player": "nobody", "ghost": True}).status_code == 404


def test_preview_colours_do_not_load_the_objects_toolchain():
    """block_color (the hologram and PNG previews) reads main's catalog colours; it must never import trimesh."""
    import subprocess
    import sys

    code = ("import sys; from craftpilot.preview.render import block_color; "
            "assert block_color('minecraft:deepslate_tile_slab') != block_color('minecraft:oak_planks'); "
            "assert 'trimesh' not in sys.modules, 'trimesh imported by preview'; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "ok"


def test_prepare_warms_the_composer_and_routes(monkeypatch):
    from craftpilot import serve

    monkeypatch.setattr(serve, "_pending", {})
    client = TestClient(serve.app)
    r = client.post("/prepare", json={"text": "a small stone cottage", "player": "p", "use_llm": False})
    assert r.status_code == 200 and r.json()["ok"] and r.json()["kind"] == "building"
    assert "p" not in serve._pending  # warming composes; it does not become the player's pending build
    r = client.post("/prepare", json={"text": "a dragon statue", "player": "p", "use_llm": False})
    assert r.status_code == 200 and r.json()["kind"] == "object"  # objects are never drawn speculatively


def test_commit_reuses_the_hologram_grid(monkeypatch, tmp_path):
    """The G commit places the very grid the hologram showed instead of generating it a second time."""
    import craftpilot.engine.pipeline as pipeline_mod
    import craftpilot.place.bridge as bridge_mod
    from craftpilot import serve
    from craftpilot.place.bridge import FakeBridge

    live = FakeBridge(pos=(0.5, 64.0, 0.5), yaw=180.0)
    monkeypatch.setattr(bridge_mod, "HttpBridge", lambda *a, **k: live)
    monkeypatch.setattr(SETTINGS, "schematics_dir", tmp_path)
    monkeypatch.setattr(serve, "_pending", {})
    real = pipeline_mod.generate
    calls = []
    monkeypatch.setattr(pipeline_mod, "generate", lambda *a, **k: calls.append(a) or real(*a, **k))
    client = TestClient(serve.app)
    ghost = client.post("/build", json={"text": "small cottage", "player": "p", "use_llm": False, "ghost": True,
                                        "seed": 5, "pos": [0.5, 64.0, 0.5], "yaw": 180.0}).json()
    assert len(calls) == 1
    built = client.post("/regenerate", json={"player": "p", "place": True, "seed": 5, "pos": [0.5, 64.0, 0.5],
                                             "yaw": 180.0}).json()
    assert len(calls) == 1 and built["blocks"] == ghost["blocks"]
    assert serve._pending["p"].grid is None  # consumed
    again = client.post("/regenerate", json={"player": "p", "place": True, "seed": 6, "pos": [0.5, 64.0, 0.5],
                                             "yaw": 180.0}).json()
    assert len(calls) == 2 and again["seed"] == 6  # a different seed renders again


def test_structured_call_runs_one_model_call_per_prompt(monkeypatch, tmp_path):
    """Single flight: a /prepare warm-up and the real /build for the same text share one model call."""
    import threading
    import time
    import types

    import craftpilot.llm.azure as azure

    monkeypatch.setattr(azure, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(azure, "LOG_DIR", tmp_path / "logs")
    created = []

    class FakeResponses:
        def create(self, **kw):
            created.append(kw)
            time.sleep(0.3)
            return types.SimpleNamespace(output_text='{"ok": 1}', id="r1", usage=None, status="completed")

    monkeypatch.setattr(azure, "client", lambda timeout=None, max_retries=1: types.SimpleNamespace(responses=FakeResponses()))
    fmt = {"type": "json_schema", "name": "t", "schema": {"type": "object"}}
    results = []

    def go():
        results.append(azure.structured_call("dep", "sys", [{"role": "user", "content": "x"}], fmt, "low", tag="t"))

    threads = [threading.Thread(target=go) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(created) == 1 and len(results) == 3
    assert all(out == '{"ok": 1}' for out, _ in results)
    assert sum(1 for _, meta in results if meta.get("waited")) == 2
    out, meta = azure.structured_call("dep", "sys", [{"role": "user", "content": "x"}], fmt, "low", tag="t")
    assert meta["cached"] and len(created) == 1


def test_compose_escalates_effort_when_the_answer_does_not_validate(monkeypatch):
    import craftpilot.llm.azure as azure
    from craftpilot.llm import compose as compose_mod

    efforts = []

    def fake(deployment, instructions, messages, fmt, effort, cache=True, tag="compose", **kw):
        efforts.append(effort)
        if effort == "low":
            return "{not json", {"cached": False, "seconds": 1.0}
        ex = load_all(SETTINGS.exemplars_dir)[0]
        return ex.program.model_dump_json(), {"cached": False, "seconds": 2.0}

    monkeypatch.setattr(azure, "structured_call", fake)
    monkeypatch.setattr(SETTINGS, "compose_effort", "low")
    monkeypatch.setattr(SETTINGS, "azure_endpoint", "https://x.openai.azure.com")
    monkeypatch.setattr(SETTINGS, "azure_api_key", "k")
    monkeypatch.setattr(SETTINGS, "compose_deployment", "dep")
    monkeypatch.setattr(SETTINGS, "edit_deployment", "dep")
    program, source, notes = compose_mod.compose("a cottage", use_llm=True)
    assert efforts == ["low", "medium"] and source == "llm"
    assert any("did not validate" in n for n in notes) and any("effort medium" in n for n in notes)
