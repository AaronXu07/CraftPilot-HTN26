"""Objects through the Fabric mod's two-lock flow: ghost -> G commit, plan -> edit -> go -> again, preview,
error mapping, and rotation parity with the hologram. build_object is faked; nothing draws."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import craftpilot.objects.pipeline as pipeline
import craftpilot.place.bridge as bridge_mod
from craftpilot import serve
from craftpilot.blocks.state import BlockRef
from craftpilot.config import SETTINGS
from craftpilot.grid.enums import BShape, Role
from craftpilot.grid.ops import quarter_turns_for_facing, rotate_cw
from craftpilot.grid.semantic import SemanticGrid
from craftpilot.objects.brief import ObjectBrief
from craftpilot.objects.pipeline import ObjectResult
from craftpilot.place.anchor import plan_origin
from craftpilot.place.bridge import FakeBridge

STAIR = "minecraft:stone_stairs"


def make_grid(front: str = "+z") -> SemanticGrid:
    """A 5x3x3 object: a column at (2, *, 1), a nose at (4, 1, 1) on the +x side, a stair at (0, 0, 1) facing east."""
    g = SemanticGrid(5, 3, 3, np.random.default_rng(0))
    stone = g.intern(BlockRef.make("minecraft:stone"))
    stair = g.intern(BlockRef.make(STAIR, facing="east", half="bottom"))
    for y in range(3):
        g.set(2, y, 1, Role.WALL, BShape.FULL)
        g.block[2, y, 1] = stone
    g.set(4, 1, 1, Role.WALL, BShape.FULL)
    g.block[4, 1, 1] = stone
    g.set(0, 0, 1, Role.WALL, BShape.STAIR)
    g.block[0, 0, 1] = stair
    g.report["object_front"] = front
    return g


def fake_result(work: Path, front: str = "+z", plinth: bool = False, seed: int = 7) -> ObjectResult:
    brief = ObjectBrief(subject="a test thing", plinth=plinth, height=3, palette="auto", style="plain",
                        label="test_thing", source="fallback", request="a test thing")
    work.mkdir(parents=True, exist_ok=True)
    (work / "test_thing.litematic").write_bytes(b"")
    g = make_grid(front)
    return ObjectResult(brief=brief, grid=g, work_dir=work, image=work / "reference.png", mesh=work / "mesh.ply",
                        preview=None, litematic=work / "test_thing.litematic", blocks=5, size=(g.W, g.H, g.D),
                        timings={"brief_s": 0.1, "image_s": 1.0, "mesh_s": 2.0, "voxel_s": 0.1}, seed=seed,
                        engine="fake", stamp="t")


def cells(grid: SemanticGrid) -> set[tuple[int, int, int]]:
    return {(int(x), int(y), int(z)) for x, y, z in zip(*np.nonzero(grid.block >= 0))}


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A fake mod, a fake build_object that counts its draws, and a fresh service state."""
    live = FakeBridge(pos=(0.5, 64.0, 0.5), yaw=180.0)
    monkeypatch.setattr(bridge_mod, "HttpBridge", lambda *a, **k: live)
    monkeypatch.setattr(SETTINGS, "schematics_dir", tmp_path)
    monkeypatch.setattr(serve, "_pending", {})
    draws = {"n": 0, "front": "+z", "plinth": False, "briefs": []}

    def fake_build(text, out_dir=None, height=None, use_llm=True, preview=True, schematic=True, resolution=256,
                   seed=0, yaw_deg=0.0, max_types=None, max_recon_tries=2, on_stage=None, engine=None, brief=None):
        draws["n"] += 1
        draws["briefs"].append(brief)
        if on_stage:
            on_stage("image", "drawing")
            on_stage("mesh", "meshing")
        r = fake_result(tmp_path / f"objects/test_thing_{draws['n']}", draws["front"], draws["plinth"], seed=seed)
        if brief is not None:
            r.brief = brief
        if height is not None:
            r.brief.height = height
        return r

    monkeypatch.setattr(pipeline, "build_object", fake_build)
    return {"bridge": live, "draws": draws, "client": TestClient(serve.app), "tmp": tmp_path}


def post(client, path, **body):
    body.setdefault("player", "p")
    body.setdefault("use_llm", False)
    return client.post(path, json=body)


def test_ghost_then_commit_places_the_cached_draw(world):
    c, live, draws = world["client"], world["bridge"], world["draws"]
    r = post(c, "/build", text="a statue of a dragon", place=True, ghost=True, pos=[0.5, 64.0, 0.5], yaw=180.0,
             seed=7)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "object" and body["route"]["kind"] == "object"
    assert body["notes"][0].startswith("routed to objects")
    assert body["ghost"]["width"] == 5 and body["ghost"]["depth"] == 3 and body["ghost"]["height"] == 3
    assert body["bounds"] == {"width": 5, "height": 3, "depth": 3} and body["blocks"] == 5
    assert body["plan"][0].startswith("test thing - 5 wide x 3 tall x 3 deep")
    assert body["seed"] == 7 and draws["n"] == 1
    assert any(s.startswith("object · image") for s in live.said)  # progress lines reached the chat
    assert live.calls == []  # nothing placed yet

    r = post(c, "/regenerate", place=True, seed=7, pos=[0.5, 64.0, 0.5], yaw=180.0)
    assert r.status_code == 200, r.text
    out = r.json()
    assert draws["n"] == 1  # committed from the cache, no second draw
    assert out["kind"] == "object" and out["facing"] == "south" and out["source"] == "previous"
    placement = out["placement"]
    origin = plan_origin({"pos": [0.5, 64.0, 0.5], "yaw": 180.0}, (0, 0, 0, 4, 2, 2), gap=SETTINGS.place_gap)
    assert placement["origin"] == list(origin)
    placed = {(x, y, z) for (x, y, z), s in live.world.items() if s != "minecraft:air"}
    expected = {(x + origin[0], y + origin[1], z + origin[2]) for x, y, z in cells(make_grid())}
    assert placed == expected
    assert [o["phase"] for o in live.outlines] == ["placing"]
    assert out["summary"].startswith("test_thing #7: 5 blocks, 5x3x3, facing south - placing at")
    undo = Path(placement["undo_file"])
    assert undo.exists() and sorted(map(tuple, json.loads(undo.read_text())["positions"])) == sorted(placed)
    assert any(n.startswith("undo: craftpilot object-undo") for n in out["notes"])


def test_commit_rotates_a_copy_toward_the_player(world):
    c, live = world["client"], world["bridge"]
    post(c, "/build", text="a statue of a dragon", place=True, ghost=True, pos=[0.5, 64.0, 0.5], yaw=270.0)
    cached = serve._pending["p"].result.grid
    before = cells(cached)
    r = post(c, "/regenerate", place=True, seed=7, pos=[0.5, 64.0, 0.5], yaw=270.0)
    assert r.status_code == 200 and r.json()["facing"] == "west"  # looking east (yaw 270): the front faces west, at the player
    assert cells(cached) == before  # the cache is still south-facing
    turned = copy.deepcopy(make_grid())
    rotate_cw(turned, quarter_turns_for_facing("west"))
    origin = plan_origin({"pos": [0.5, 64.0, 0.5], "yaw": 270.0}, (0, 0, 0, turned.W - 1, 2, turned.D - 1),
                         gap=SETTINGS.place_gap)
    placed = {(x, y, z) for (x, y, z), s in live.world.items() if s != "minecraft:air"}
    assert placed == {(x + origin[0], y + origin[1], z + origin[2]) for x, y, z in cells(turned)}
    stairs = [s for s in live.world.values() if s.startswith(STAIR)]
    assert stairs and all("facing=south" in s for s in stairs)  # east + one CW turn


def test_plan_edit_go_again(world, monkeypatch):
    c, draws = world["client"], world["draws"]
    import craftpilot.objects.brief as brief_mod

    monkeypatch.setattr(brief_mod, "edit_brief", lambda prev, change, use_llm=True: (
        ObjectBrief(subject=prev.subject + ", " + change, plinth=prev.plinth, height=prev.height, palette="colorful",
                    style=prev.style, label=prev.label, source="fallback", request=prev.request), "fallback", ["edited"]))
    r = post(c, "/plan", text="a dragon statue")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "object" and "ghost" not in body and draws["n"] == 0
    assert body["plan"][0].startswith("dragon statue - about 40 blocks")
    assert body["bounds"] == {"width": 40, "height": 40, "depth": 40}
    r = post(c, "/edit", text="make it red", plan=True, place=False)
    assert r.status_code == 200 and "make it red" in r.json()["plan"][1] and draws["n"] == 0
    r = post(c, "/regenerate", ghost=True)  # /build go
    assert r.status_code == 200 and "ghost" in r.json() and draws["n"] == 1
    assert draws["briefs"][0] is not None and draws["briefs"][0].palette == "colorful"  # drawn from the edited brief
    r = post(c, "/regenerate", ghost=True, seed=11)  # /build again 11
    assert r.status_code == 200 and r.json()["seed"] == 11 and draws["n"] == 2
    r = post(c, "/regenerate", place=True, seed=11, pos=[0.5, 64.0, 0.5], yaw=180.0)
    assert r.status_code == 200 and draws["n"] == 2 and "placement" in r.json()


def test_edit_after_a_preview_keeps_the_hologram_commit_honest(world, monkeypatch):
    c, draws = world["client"], world["draws"]
    import craftpilot.objects.brief as brief_mod

    monkeypatch.setattr(brief_mod, "edit_brief", lambda prev, change, use_llm=True: (prev, "fallback", []))
    post(c, "/plan", text="a dragon statue")
    post(c, "/regenerate", ghost=True)  # go -> hologram armed
    r = post(c, "/edit", text="make it red", plan=True, place=False)  # plan mode is still on in the mod
    assert r.status_code == 200 and r.json()["stale_preview"] is True
    r = post(c, "/regenerate", place=True, pos=[0.5, 64.0, 0.5], yaw=180.0)  # G on the old hologram
    assert r.status_code == 200 and draws["n"] == 1
    assert any("edit was not drawn yet" in n for n in r.json()["notes"])


def test_preview_returns_artefacts_without_a_plan_or_placement(world):
    c, live = world["client"], world["bridge"]
    r = post(c, "/build", text="a red sports car", place=False)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "object" and "plan" not in body and "placement" not in body and "ghost" not in body
    assert body["facing"] == "south" and body["schematic"].endswith("test_thing.litematic")
    assert live.calls == [] and live.said == []


def test_error_mapping(world, monkeypatch):
    c = world["client"]
    from craftpilot.objects.imagegen import ImageRejected
    from craftpilot.objects.recon import ReconUnavailable

    def boom(exc):
        def fake(*a, **k):
            raise exc
        return fake

    monkeypatch.setattr(pipeline, "build_object", boom(ReconUnavailable("worker died")))
    r = post(c, "/build", text="a marble lion statue", place=True, ghost=True)
    assert r.status_code == 503 and "3D worker" in r.json()["detail"]
    monkeypatch.setattr(pipeline, "build_object", boom(ImageRejected("no", kind="blocklist")))
    r = post(c, "/build", text="a pikachu", place=True, ghost=True)
    assert r.status_code == 422 and "protected" in r.json()["detail"]
    monkeypatch.setattr(pipeline, "build_object", boom(RuntimeError("cuda")))
    r = post(c, "/build", text="a dragon", place=True, ghost=True)
    assert r.status_code == 500 and "object pipeline failed" in r.json()["detail"]
    assert "p" not in serve._pending  # a failed draw leaves no half state behind


def test_unavailable_objects_path_downgrades_auto_routes_only(world, monkeypatch):
    c = world["client"]
    import craftpilot.objects.flow as flow

    monkeypatch.setattr(flow, "unavailable_reason", lambda: "3D worker not installed")
    r = post(c, "/plan", text="a dragon statue")
    assert r.status_code == 200 and r.json()["kind"] == "building"
    assert r.json()["route"]["source"] == "fallback" and "unavailable" in r.json()["notes"][0]
    r = post(c, "/plan", text="a dragon statue", kind="object")
    assert r.status_code == 503 and "3D worker" in r.json()["detail"]


def test_overrides_and_text_prefix(world):
    c = world["client"]
    r = post(c, "/plan", text="a dragon statue", kind="building")
    assert r.status_code == 200 and r.json()["kind"] == "building" and r.json()["route"]["source"] == "override"
    r = post(c, "/plan", text="object: a small cottage")
    assert r.status_code == 200 and r.json()["kind"] == "object" and r.json()["route"]["source"] == "override"
    r = post(c, "/plan", text="x", kind="banana")
    assert r.status_code == 422


def test_exemplars_refuse_objects_and_buildings_still_work(world):
    c = world["client"]
    post(c, "/plan", text="a dragon statue")
    r = post(c, "/exemplars", name="x")
    assert r.status_code == 400
    r = post(c, "/plan", text="a small stone cottage")
    assert r.status_code == 200 and r.json()["kind"] == "building" and r.json()["notes"][0].startswith("routed to buildings")


def test_undo_record_removes_the_object(world):
    c, live = world["client"], world["bridge"]
    from craftpilot.place.undo import undo

    post(c, "/build", text="a statue of a dragon", place=True, ghost=True, pos=[0.5, 64.0, 0.5], yaw=180.0)
    out = post(c, "/regenerate", place=True, seed=7, pos=[0.5, 64.0, 0.5], yaw=180.0).json()
    n = undo(Path(out["placement"]["undo_file"]), live)
    assert n == 5 and all(s == "minecraft:air" for s in live.world.values())


def test_face_south_matches_the_old_presentation_rotation():
    """The new in-place normalisation reproduces plan-1's centred grid_blocks() rotation for every front."""
    from craftpilot.objects import place as P
    from craftpilot.objects.orient import face_south

    for front in ("+x", "+z", "-x", "-z"):
        g = make_grid(front)
        ref = P.grid_blocks(g)  # (x, y, z, state) centred, rotated by PRESENTATION_TURNS
        face_south(g)
        assert g.report["object_front"] == "+z"
        new = [(x, y, z, g.palette[g.block[x, y, z]].state_string()) for x, y, z in sorted(cells(g))]
        # same shape up to translation: compare offsets from the min corner, and the same stair facings
        def norm(bs):
            mx, mz = min(b[0] for b in bs), min(b[2] for b in bs)
            return sorted((x - mx, y, z - mz, s.split("[")[0], "facing=" + s.split("facing=")[1].split(",")[0] if "facing=" in s else "") for x, y, z, s in bs)
        assert norm(new) == norm(ref), front
        # TripoSR's "+x" is the object's back (camera axis), so the +x nose ends at the north face (z = 0) and the
        # -x stair, the presented side, ends at the south face (z = D-1) - toward the player once placed
        if front == "+x":
            nose = [(x, y, z) for x, y, z in cells(g) if y == 1 and not any((x, yy, z) in cells(g) for yy in (0, 2))]
            assert nose and all(z == 0 for _, _, z in nose)
            stair = [(x, y, z) for x, y, z in cells(g) if g.palette[g.block[x, y, z]].block_id == STAIR]
            assert stair and all(z == g.D - 1 for _, _, z in stair)
