from copilot.engine.diff import changed_bbox, diff_block_maps, diff_stats, layer_chunks, to_setblocks
from copilot.engine.scene import Scene, SceneEditor
from tests.engine_helpers import build


def base_scene():
    e = SceneEditor(Scene("d"))
    e.add("keep", {"type": "box", "size": [10, 8, 10]}, pos=[0, 0, 0], material="stone")
    e.add("tower", {"type": "cylinder", "radius": 2.5, "height": 12}, pos=[20, 0, 0], material="stone")
    return e


def test_diff_basic():
    old = {(0, 0, 0): "a", (1, 0, 0): "b", (2, 0, 0): "c"}
    new = {(0, 0, 0): "a", (1, 0, 0): "B", (3, 0, 0): "d"}
    d = diff_block_maps(old, new)
    assert d.added == {(3, 0, 0): "d"} and d.removed == {(2, 0, 0)} and d.changed == {(1, 0, 0): "B"}
    sb = to_setblocks(d)
    assert sb[0] == (2, 0, 0, "minecraft:air") and (1, 0, 0, "B") in sb and (3, 0, 0, "d") in sb
    assert diff_stats(d) == "+1 -1 ~1 blocks"
    assert diff_block_maps(new, new).is_empty()


def test_layer_chunks_bottom_up():
    blocks = [(0, y, 0, "s") for y in (5, 1, 3, 0, 2, 4)]
    chunks = layer_chunks(blocks, chunk_size=4)
    assert [b[1] for b in chunks[0]] == [0, 1, 2, 3] and [b[1] for b in chunks[1]] == [4, 5]


def test_edit_re_emits_only_object_bbox():
    e = base_scene()
    s1 = e.scene
    bm1 = build(s1)
    e.set_shape("tower", height=20)
    s2 = e.scene
    bm2 = build(s2)
    d = diff_block_maps(bm1, bm2)
    assert not d.is_empty()
    bb = changed_bbox(s1, s2)
    assert bb is not None
    for p in list(d.added) + list(d.removed) + list(d.changed):
        assert bb.lo[0] - 1 <= p[0] < bb.hi[0] + 1 and bb.lo[2] - 1 <= p[2] < bb.hi[2] + 1
        assert p[0] >= 15  # nothing near the keep changed
    assert changed_bbox(s2, s2) is None
