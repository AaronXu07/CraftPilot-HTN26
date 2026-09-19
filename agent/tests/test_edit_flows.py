"""Edit flows (T6): set_shape re-emits only the changed bbox, moving a group, mirror_copy symmetry."""
import numpy as np

from copilot.engine.coretypes import Facing, Kind
from copilot.engine.diff import changed_bbox, diff_block_maps, to_setblocks
from copilot.engine.fit import fit_surface
from copilot.engine.raster import rasterize
from copilot.engine.scene import Scene, SceneEditor, apply_op
from tests.engine_helpers import build


def _scene():
    e = SceneEditor(Scene("edit"))
    e.add("keep", {"type": "box", "size": [10, 8, 10]}, pos=[0, 0, 0], material="stone")
    e.add("tower", {"type": "cylinder", "radius": 2.5, "height": 12}, pos=[20, 0, 0], material="stone")
    e.add("shed", {"type": "box", "size": [4, 3, 4]}, pos=[-20, 0, 0], material="wood")
    return e


def _inside(p, bb, pad=0):
    return all(bb.lo[i] - pad <= p[i] < bb.hi[i] + pad for i in range(3))


def test_set_shape_diff_only_touches_changed_bbox():
    e = _scene()
    s1 = e.scene
    bm1 = build(s1)
    e.set_shape("tower", height=20)
    s2 = e.scene
    bm2 = build(s2)
    d = diff_block_maps(bm1, bm2)
    sb = to_setblocks(d)
    assert len(sb) > 0
    bb = changed_bbox(s1, s2)
    assert bb is not None
    keep_bb = s2.object_bbox("keep")
    shed_bb = s2.object_bbox("shed")
    for x, y, z, _ in sb:
        assert _inside((x, y, z), bb, pad=1), (x, y, z)
        assert not _inside((x, y, z), keep_bb) and not _inside((x, y, z), shed_bb)
    # only additions (the tower grew); nothing removed or re-sent below the old top
    assert not d.removed and not d.changed
    assert min(y for _, y, _, _ in sb) >= 12
    # an identical edit is a no-op diff
    assert diff_block_maps(bm2, build(s2)).is_empty()


def test_move_group_shifts_members_only():
    e = _scene()
    e.group(["keep", "tower"], "castle")
    s1 = e.scene
    bm1 = build(s1)
    shed_before = {p for p, s in bm1.items() if p[0] < -10}
    e.move("castle", [5, 0, 3])
    s2 = e.scene
    assert s2.get("keep").group == "castle" and s2.group_members("castle") == ["keep", "tower"]
    bm2 = build(s2)
    shifted = {(x + 5, y, z + 3): s for (x, y, z), s in bm1.items() if (x, y, z) not in shed_before}
    shed_after = {p for p in bm2 if p[0] < -10}
    assert shed_after == shed_before
    moved_after = {p: s for p, s in bm2.items() if p not in shed_after}
    assert moved_after == shifted
    # the diff is bounded by the union of old and new castle bboxes
    bb = changed_bbox(s1, s2)
    for x, y, z, _ in to_setblocks(diff_block_maps(bm1, bm2)):
        assert _inside((x, y, z), bb, pad=1)


def test_ungroup_then_move_by_group_fails():
    e = _scene()
    e.group(["keep", "tower"], "castle")
    e.ungroup("castle")
    try:
        e.move("castle", [1, 0, 0])
    except Exception as ex:  # noqa: BLE001
        assert "unknown object or group" in str(ex)
    else:
        raise AssertionError("moving a deleted group should fail")


def _reflect_x(coords):
    return {(-1 - x, y, z) for x, y, z in coords}


def test_mirror_copy_box_and_wedge_symmetric_across_x():
    e = SceneEditor(Scene("mir"))
    e.add("wing", {"type": "box", "size": [4, 6, 8]}, pos=[-8, 0, 0], material="stone")
    e.add("roof", {"type": "wedge", "size": [4, 3, 8], "slope_axis": "x"}, pos=[-8, 6, 0], material="roof")
    e.mirror_copy(["wing", "roof"], axis="x", plane=0)
    s = e.scene
    assert [o.id for o in s.objects] == ["wing", "wing_m", "roof", "roof_m"]
    for oid in ("wing", "roof"):
        a, b = s.object_bbox(oid), s.object_bbox(oid + "_m")
        assert np.allclose(b.lo[0], -a.hi[0]) and np.allclose(b.hi[0], -a.lo[0])
        assert np.allclose(a.lo[1:], b.lo[1:]) and np.allclose(a.hi[1:], b.hi[1:])
    # voxel-level symmetry: the mirrored half's occupied set is the exact reflection of the original
    orig = Scene("o")
    orig, _ = apply_op(orig, "add", id="wing", shape={"type": "box", "size": [4, 6, 8]}, pos=[-8, 0, 0], material="stone")
    orig, _ = apply_op(orig, "add", id="roof", shape={"type": "wedge", "size": [4, 3, 8], "slope_axis": "x"}, pos=[-8, 6, 0], material="roof")
    copy, _ = apply_op(s, "delete", ids=["wing", "roof"])
    bm_o, bm_c = build(orig), build(copy)
    assert _reflect_x(bm_o) == set(bm_c)
    # stairs on the slope face the opposite way on the mirrored wedge
    ro = rasterize(orig)
    fo = fit_surface(ro, {})
    rc = rasterize(copy)
    fc = fit_surface(rc, {})
    fo_dirs = {Facing(int(fo.facing[tuple(t)])) for t in np.argwhere(fo.kind == Kind.STAIRS)}
    fc_dirs = {Facing(int(fc.facing[tuple(t)])) for t in np.argwhere(fc.kind == Kind.STAIRS)}
    assert fo_dirs and fc_dirs
    assert {Facing.EAST if d == Facing.WEST else Facing.WEST if d == Facing.EAST else d for d in fo_dirs} == fc_dirs


def test_mirror_copy_across_z_with_offset_plane_and_suffix():
    e = SceneEditor(Scene("mz"))
    e.add("t", {"type": "cylinder", "radius": 3, "height": 9}, pos=[0, 0, 4], material="stone")
    e.mirror_copy("t", axis="z", plane=10, new_suffix="_south")
    s = e.scene
    a, b = s.object_bbox("t"), s.object_bbox("t_south")
    assert np.allclose(b.lo[2], 20 - a.hi[2]) and np.allclose(b.hi[2], 20 - a.lo[2])
    bm = build(s)
    pts = set(bm)
    assert {(x, y, 19 - z) for x, y, z in pts} == pts
