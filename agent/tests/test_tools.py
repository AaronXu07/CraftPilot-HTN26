import pytest

from copilot.engine.scene import OPS
from copilot.session import Session
from copilot.tools.dispatch import ToolContext, ToolResult, build_all, dispatch, truncate
from copilot.tools.schemas import TOOL_SCHEMAS, TOOLS_BY_NAME, scene_op_names, tool_subset
from mock_mod import MockBridge


@pytest.fixture
def ctx():
    return ToolContext(session=Session("tester"), bridge=MockBridge(), registry=None)


def test_schema_coverage_and_format():
    assert set(OPS) <= set(TOOLS_BY_NAME)
    assert set(scene_op_names()) == set(OPS)
    for t in TOOL_SCHEMAS:
        f = t["function"]
        assert t["type"] == "function" and f["name"] and len(f["description"]) > 20
        assert f["parameters"]["type"] == "object"
        for r in f["parameters"].get("required", []):
            assert r in f["parameters"]["properties"], (f["name"], r)
    assert [t["function"]["name"] for t in tool_subset(["add", "render"])] == ["add", "render"]
    for name in ("run_script", "render", "lint", "place", "undo_world", "export_schematic", "materials_list", "get_player", "say", "search_blocks", "nearest_block", "set_brief", "finish", "undo", "redo", "snapshot", "restore"):
        assert name in TOOLS_BY_NAME


def test_scene_ops_dispatch(ctx):
    r = dispatch(ctx, "add", {"id": "keep", "shape": {"type": "box", "size": [10, 8, 10]}, "pos": [0, 0, 0], "material": "stone"})
    assert isinstance(r, ToolResult) and r.text.startswith("added keep")
    r = dispatch(ctx, "set_shape", {"id": "keep", "params": {"type": "prism", "sides": 6}})
    assert "prism n6" in r.text
    r = dispatch(ctx, "add_modifier", {"id": "keep", "modifier": {"type": "shell", "thickness": 1}})
    assert "shell(1)" in r.text
    r = dispatch(ctx, "set_modifier", {"id": "keep", "index": 0, "params": {"thickness": 2}})
    assert "shell(2)" in r.text
    assert "keep" in dispatch(ctx, "describe", {}).text
    assert "keep" in dispatch(ctx, "select", {"query": "shape:prism"}).text
    assert len(ctx.session.history) >= 5


def test_error_paths_return_text(ctx):
    assert dispatch(ctx, "move", {"ids": "nope", "delta": [1, 0, 0]}).text.startswith("ERROR:")
    assert dispatch(ctx, "add", {"id": "Bad Id", "shape": {"type": "box"}, "pos": [0, 0, 0]}).text.startswith("ERROR:")
    assert dispatch(ctx, "no_such_tool", {}).text.startswith("ERROR: unknown tool")
    assert dispatch(ctx, "set_shape", {"id": "x", "params": "oops"}).text.startswith("ERROR:")
    assert dispatch(ctx, "render", {}).text.startswith("ERROR:")  # empty scene


def test_history_tools(ctx):
    dispatch(ctx, "add", {"id": "a", "shape": {"type": "box", "size": [1, 1, 1]}, "pos": [0, 0, 0]})
    dispatch(ctx, "snapshot", {"label": "one"})
    dispatch(ctx, "add", {"id": "b", "shape": {"type": "box", "size": [1, 1, 1]}, "pos": [3, 0, 0]})
    assert "undid 1" in dispatch(ctx, "undo", {}).text and len(ctx.session.scene.objects) == 1
    assert "redid 1" in dispatch(ctx, "redo", {}).text and len(ctx.session.scene.objects) == 2
    assert "restored" in dispatch(ctx, "restore", {"label": "one"}).text and len(ctx.session.scene.objects) == 1
    assert dispatch(ctx, "restore", {"label": "zzz"}).text.startswith("ERROR:")


def test_run_script_replays_ops_and_undo(ctx):
    src = (
        "for i in range(4):\n"
        "    scene.add(id=f'tower_{i}', shape={'type':'cylinder','radius':3,'height':10}, pos=[i*10,0,0], material='stone')\n"
        "print('built', len(scene.ids()))\n"
    )
    r = dispatch(ctx, "run_script", {"python": src})
    assert r.data["ops"] == 4 and "script performed 4 ops" in r.text and "built 4" in r.text
    assert [o.id for o in ctx.session.scene.objects] == ["tower_0", "tower_1", "tower_2", "tower_3"]
    assert [h.op for h in ctx.session.history[-4:]] == ["add"] * 4
    assert "undid 1" in dispatch(ctx, "undo", {}).text
    assert len(ctx.session.scene.objects) == 3


def test_run_script_errors_and_sandbox(ctx):
    r = dispatch(ctx, "run_script", {"python": "scene.add(id='a', shape={'type':'box','size':[1,1,1]}, pos=[0,0,0])\nscene.move('zzz', [1,0,0])"})
    assert r.data["ops"] == 1 and "script error after 1 ops" in r.text and "unknown object" in r.text
    assert dispatch(ctx, "run_script", {"python": "open('/etc/passwd')"}).text.find("open") > 0
    assert dispatch(ctx, "run_script", {"python": "open('/etc/passwd')"}).data["ops"] == 0
    r = dispatch(ctx, "run_script", {"python": "import os\nos.system('echo hi')"})
    assert r.data["ops"] == 0 and "import" in r.text.lower()
    assert dispatch(ctx, "run_script", {"python": ""}).text.startswith("ERROR:")


def test_run_script_timeout(ctx, monkeypatch):
    import copilot.tools.dispatch as d

    monkeypatch.setattr(d, "SCRIPT_TIMEOUT_S", 1.5)
    r = dispatch(ctx, "run_script", {"python": "while True: pass"})
    assert r.text.startswith("ERROR:") and "limit" in r.text
    assert len(ctx.session.scene.objects) == 0


def test_agent_tools_without_engine(ctx):
    assert "Steve" in dispatch(ctx, "get_player", {}).text
    assert dispatch(ctx, "say", {"text": "hi"}).text == "said: hi" and ctx.bridge.chat == ["hi"]
    assert dispatch(ctx, "set_brief", {"brief": {"build_type": "castle"}}).data["brief"]["build_type"] == "castle"
    assert ctx.session.brief == {"build_type": "castle"}
    r = dispatch(ctx, "finish", {"summary": "all done"})
    assert r.data["finished"] and r.text == "all done"
    assert dispatch(ctx, "search_blocks", {"query": "stairs"}).text.startswith("ERROR")  # no registry
    assert dispatch(ctx, "undo_world", {}).text.startswith("nothing to undo")


def test_build_all_with_stubbed_engine(ctx, monkeypatch):
    """build_all caches by scene hash; render/place/lint/materials_list work through a fake build."""
    import copilot.tools.dispatch as d

    calls = {"n": 0}

    def fake_build(c):
        cached = c.session.cached("build")
        if cached is not None:
            return cached
        calls["n"] += 1
        bm = {(x, 0, z): "minecraft:stone" for x in range(3) for z in range(3)}
        bm[(1, 1, 1)] = "minecraft:oak_stairs[facing=north,half=bottom,shape=straight,waterlogged=false]"
        res = {"raster": None, "fit": None, "block_map": bm, "warnings": [], "timings": {}}
        c.session.put_cache("build", res)
        return res

    monkeypatch.setattr(d, "build_all", fake_build)
    dispatch(ctx, "add", {"id": "keep", "shape": {"type": "box", "size": [3, 1, 3]}, "pos": [1, 0, 1]})
    r = dispatch(ctx, "place", {"animate": False})
    assert "placed diff: 10 set" in r.text and ctx.bridge.calls[-1]["count"] == 10
    assert "up to date" in dispatch(ctx, "place", {"animate": False}).text
    ml = dispatch(ctx, "materials_list", {}).text
    assert "stone" in ml
    assert calls["n"] == 1
    dispatch(ctx, "move", {"ids": "keep", "delta": [1, 0, 0]})
    dispatch(ctx, "materials_list", {})
    assert calls["n"] == 2  # scene changed -> rebuilt
    assert "restored" in dispatch(ctx, "undo_world", {}).text
    assert ctx.bridge.count_non_air() == 0


def test_truncate():
    assert truncate("x" * 10) == "x" * 10
    t = truncate("y" * 10000)
    assert len(t) < 4200 and "truncated" in t
