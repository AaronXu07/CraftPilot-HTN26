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
