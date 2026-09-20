import numpy as np

from copilot.engine.coretypes import Facing, Half, Kind
from copilot.engine.fit import FACING_CALIBRATION, exposed_faces, fit_surface, gravity_check
from copilot.engine.raster import rasterize
from copilot.engine.scene import Scene, SceneEditor


def fitted(scene, modes=None):
    r = rasterize(scene)
    f = fit_surface(r, modes or {})
    return r, f


def test_cone_stairs_face_the_axis():
    """Stairs on a cone slope: the full (high) side faces the axis, like hand-built roof stairs."""
    e = SceneEditor(Scene("cone"))
    e.add("cone", {"type": "cone", "radius": 6, "height": 5}, pos=[0, 0, 0], material="roof")
    r, f = fitted(e.scene)
    st = np.argwhere(f.kind == Kind.STAIRS)
    assert len(st) >= 24
    for i, j, k in st:
        x, y, z = r.index_to_world(i, j, k)
        v = Facing(int(f.facing[i, j, k])).vector
        toward_axis = (0 - (x + 0.5)) * v[0] + (0 - (z + 0.5)) * v[2]
        assert toward_axis > 0, (x, y, z, Facing(int(f.facing[i, j, k])).name_mc)
        assert f.half[i, j, k] == Half.BOTTOM
    assert FACING_CALIBRATION["missing_top_pair_on_east"] == "west"


def test_dome_uses_slabs_at_crown():
    e = SceneEditor(Scene("dome"))
    e.add("dome", {"type": "sphere", "radius": 6, "half": True}, pos=[0, 0, 0], material="stone")
    r, f = fitted(e.scene)
    top_y = max(r.index_to_world(*t)[1] for t in np.argwhere(f.kind > 0))
    crown = [t for t in np.argwhere(f.kind == Kind.SLAB) if r.index_to_world(*t)[1] == top_y]
    assert len(crown) >= 4
    assert all(f.half[tuple(t)] == Half.BOTTOM for t in crown)
    assert (f.kind == Kind.STAIRS).sum() > 0


def test_arch_gets_stairs_along_curve():
    e = SceneEditor(Scene("arch"))
    e.add("wall", {"type": "box", "size": [16, 12, 3]}, pos=[0, 0, 0], material="stone")
    e.add("arch", {"type": "cylinder", "radius": 5, "height": 5, "axis": "z"}, pos=[0, 0, 0], op="subtract")
    r, f = fitted(e.scene)
    st = np.argwhere(f.kind == Kind.STAIRS)
    assert len(st) >= 12
    ys = sorted(set(int(r.index_to_world(*t)[1]) for t in st))
    assert min(ys) <= 2 and max(ys) >= 7  # bottom and shoulder of the arch
    tops = [t for t in st if f.half[tuple(t)] == Half.TOP]
    assert tops, "upper arch curve should use top-half stairs"
    for t in tops:
        x = r.index_to_world(*t)[0]
        fc = Facing(int(f.facing[tuple(t)]))
        # the full side is away from the opening (west of centre -> facing west)
        assert (fc == Facing.WEST) == (x < 0)


def test_wedge_slope_stairs_face_uphill():
    e = SceneEditor(Scene("w"))
    e.add("roof", {"type": "wedge", "size": [10, 5, 8], "slope_axis": "x"}, pos=[0, 0, 0], material="roof")
    r, f = fitted(e.scene)
    st = np.argwhere(f.kind == Kind.STAIRS)
    assert len(st) > 0
    assert all(Facing(int(f.facing[tuple(t)])) == Facing.EAST for t in st)


def test_fit_modes():
    e = SceneEditor(Scene("m"))
    e.add("cone", {"type": "cone", "radius": 6, "height": 5}, pos=[0, 0, 0], material="roof")
    r0 = rasterize(e.scene)
    n_occ = int((r0.material > 0).sum())
    r, f = fitted(e.scene, {"roof": "none"})
    assert f.counts()["stairs"] == 0 and f.counts()["slab"] == 0 and f.counts()["full"] == n_occ
    assert np.array_equal(r.material > 0, f.kind > 0)
    _, f_slab = fitted(e.scene, {"roof": "slab"})
    assert f_slab.counts()["stairs"] == 0
    _, f_st = fitted(e.scene, {"roof": "stairs"})
    assert f_st.counts()["slab"] == 0 and f_st.counts()["stairs"] > 0


def test_raster_material_synced_after_fit():
    e = SceneEditor(Scene("s"))
    e.add("dome", {"type": "sphere", "radius": 5, "half": True}, pos=[0, 0, 0], material="stone")
    r, f = fitted(e.scene)
    assert np.array_equal(r.material > 0, f.kind > 0)
    assert np.all(r.owner[f.kind > 0] >= 0)


def test_prop_support_kept():
    e = SceneEditor(Scene("p"))
    e.add("post", {"type": "cylinder", "radius": 0.6, "height": 3}, pos=[0, 0, 0], material="stone")
    e.add("lantern", {"type": "block", "state": "minecraft:lantern"}, pos=[0, 3, 0])
    r, f = fitted(e.scene)
    i = r.world_to_index(0, 2, 0)
    assert f.kind[i] > 0


def test_exposed_faces_and_gravity():
    occ = np.zeros((3, 3, 3), dtype=bool)
    occ[1, 1, 1] = True
    ex = exposed_faces(occ)
    assert ex.shape == (6, 3, 3, 3) and ex[:, 1, 1, 1].all()
    occ[1, 0, 1] = True
    ex = exposed_faces(occ)
    assert not ex[2, 1, 1, 1] and not ex[3, 1, 0, 1]
    kind = np.zeros((5, 5, 5), dtype=np.uint8)
    kind[2, 3, 2] = 1
    assert gravity_check(kind, origin=(10, 0, 10)) == [(12, 3, 12)]
    kind[2, 2, 2] = 1
    assert gravity_check(kind) == [(2, 2, 2)]  # the pair still hangs in the air
    kind[2, 0:4, 2] = 1
    assert gravity_check(kind) == []
