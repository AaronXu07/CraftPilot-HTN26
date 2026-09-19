import hashlib
import io

import numpy as np

from copilot.engine.render import apply_cutaway, ascii_slice, block_boxes, contact_sheet, default_color_fn, render_blocks
from copilot.engine.scene import Scene, SceneEditor
from tests.engine_helpers import build


def scenes():
    a = SceneEditor(Scene("tower"))
    a.add("t", {"type": "cylinder", "radius": 4.5, "height": 12}, pos=[0, 0, 0], material="stone", modifiers=[{"type": "shell", "thickness": 1}])
    a.add("r", {"type": "cone", "radius": 5.5, "height": 6}, pos=[0, 12, 0], material="roof")
    b = SceneEditor(Scene("hall"))
    b.add("h", {"type": "box", "size": [16, 6, 10]}, pos=[0, 0, 0], material="stone")
    b.add("roof", {"type": "wedge", "size": [8, 4, 12], "slope_axis": "x"}, pos=[-4, 6, 0], material="roof")
    b.add("roof2", {"type": "wedge", "size": [8, 4, 12], "slope_axis": "-x"}, pos=[4, 6, 0], material="roof")
    b.add("door", {"type": "box", "size": [2, 3, 3]}, pos=[0, 0, 5], op="subtract")
    c = SceneEditor(Scene("dome"))
    c.add("d", {"type": "sphere", "radius": 6, "half": True}, pos=[0, 0, 0], material="stone")
    c.add("lantern", {"type": "block", "state": "minecraft:lantern"}, pos=[0, 6, 0])
    return [a.scene, b.scene, c.scene]


def png_hash(img) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return hashlib.sha1(buf.getvalue()).hexdigest()


def test_deterministic_hashes_and_sizes():
    for sc in scenes():
        bm = build(sc)
        h1 = png_hash(render_blocks(bm, default_color_fn, view="iso", size=512))
        h2 = png_hash(render_blocks(bm, default_color_fn, view="iso", size=512))
        assert h1 == h2
        im = render_blocks(bm, default_color_fn, view="front", size=300)
        assert im.size == (300, 300)
        # something was drawn (not all background)
        arr = np.asarray(im)
        assert (arr != np.array([235, 238, 242])).any()


def test_all_views_and_cutaway():
    bm = build(scenes()[1])
    for v in ("iso", "front", "back", "left", "right", "top"):
        im = render_blocks(bm, default_color_fn, view=v, size=256)
        assert im.size == (256, 256)
    cut = apply_cutaway(bm, ("z", 0))
    assert cut and all(p[2] <= 0 for p in cut)
    cut2 = apply_cutaway(bm, ("-z", 0))
    assert all(p[2] >= 0 for p in cut2)
    im = render_blocks(bm, default_color_fn, view="iso", size=256, cutaway=("y", 5))
    assert im.size == (256, 256)


def test_contact_sheet():
    bm = build(scenes()[0])
    sheet = contact_sheet(bm, default_color_fn, size=512)
    assert sheet.size == (512, 512)
    sheet2 = contact_sheet(bm, default_color_fn, size=512, cutaway=("x", 0))
    assert png_hash(sheet2) == png_hash(contact_sheet(bm, default_color_fn, size=512, cutaway=("x", 0)))


def test_block_boxes_geometry():
    assert len(block_boxes("minecraft:stone")) == 1
    slab = block_boxes("minecraft:stone_slab[type=top]")
    assert np.allclose(slab[0][0], [0, 0.5, 0])
    st = block_boxes("minecraft:oak_stairs[facing=east,half=bottom]")
    assert len(st) == 2 and np.allclose(st[1][0], [0.5, 0.5, 0]) and np.allclose(st[1][1], [1, 1, 1])
    st = block_boxes("minecraft:oak_stairs[facing=north,half=top]")
    assert np.allclose(st[1][0], [0, 0, 0]) and np.allclose(st[1][1], [1, 0.5, 0.5])


def test_ascii_slice():
    bm = build(scenes()[1])
    txt = ascii_slice(bm, y=2)
    lines = txt.splitlines()
    assert lines[0].startswith("slice y=2")
    assert any("#" in ln for ln in lines[1:-1])
    assert "legend" in lines[-1]
    vert = ascii_slice(bm, axis="x", at=0)
    assert vert.splitlines()[0].startswith("slice x=0")
    assert ascii_slice({}) == "(empty)"


def test_empty_render():
    im = render_blocks({}, default_color_fn, size=64)
    assert im.size == (64, 64)
