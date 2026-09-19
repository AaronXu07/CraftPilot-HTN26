"""Fitting goldens (T6): cone / dome / wedge / arch / torus keep their symmetry and facing rules."""
import numpy as np
import pytest

from copilot.engine.coretypes import Facing, Half, Kind
from copilot.engine.fit import fit_surface
from copilot.engine.raster import rasterize
from copilot.engine.scene import Scene, SceneEditor

OPP = {Facing.EAST: Facing.WEST, Facing.WEST: Facing.EAST, Facing.NORTH: Facing.SOUTH, Facing.SOUTH: Facing.NORTH}


def _fit(shape, pos=(0, 0, 0), op="add", extra=None):
    e = SceneEditor(Scene("g"))
    if extra:
        e.add("base", extra, pos=[0, 0, 0], material="stone")
    e.add("s", shape, pos=list(pos), material="stone", op=op)
    r = rasterize(e.scene)
    return r, fit_surface(r, {})


def _table(r, f):
    """{world (x,y,z): (kind, facing, half)} for every fitted voxel."""
    out = {}
    for t in np.argwhere(f.kind > 0):
        t = tuple(t)
        out[r.index_to_world(*t)] = (int(f.kind[t]), int(f.facing[t]), int(f.half[t]))
    return out


def _assert_mirror_symmetric(tab, axis):
    """Reflect across the x=0 (axis 0) or z=0 (axis 2) plane: kinds/halves match, facings flip."""
    for (x, y, z), (kind, facing, half) in tab.items():
        m = (-1 - x, y, z) if axis == 0 else (x, y, -1 - z)
        assert m in tab, ("missing mirror voxel", (x, y, z), m)
        mk, mf, mh = tab[m]
        assert mk == kind and mh == half, ((x, y, z), (kind, half), (mk, mh))
        if kind == Kind.STAIRS:
            fa, fm = Facing(facing), Facing(mf)
            along = fa in ((Facing.EAST, Facing.WEST) if axis == 0 else (Facing.NORTH, Facing.SOUTH))
            assert fm == (OPP[fa] if along else fa), ((x, y, z), fa, fm)


@pytest.mark.parametrize("shape", [
    {"type": "cone", "radius": 6, "height": 5},
    {"type": "sphere", "radius": 6, "half": True},
    {"type": "torus", "major": 6, "minor": 2, "axis": "y"},
])
def test_round_shapes_are_mirror_symmetric_in_x_and_z(shape):
    r, f = _fit(shape)
    tab = _table(r, f)
    assert (f.kind == Kind.STAIRS).sum() > 0 or shape["type"] == "torus"
    _assert_mirror_symmetric(tab, 0)
    _assert_mirror_symmetric(tab, 2)


def test_torus_golden():
    r, f = _fit({"type": "torus", "major": 6, "minor": 2, "axis": "y"})
    tab = _table(r, f)
    xs = [p[0] for p in tab]
    ys = [p[1] for p in tab]
    assert min(xs) == -8 and max(xs) == 7 and min(ys) == 0 and max(ys) == 3  # sits on y=0, spans ~2*(major+minor)
    assert all(abs(x + 0.5) > 3 or abs(z + 0.5) > 3 for x, _, z in tab)  # hole in the middle
    # the upper and lower rims are fitted (slabs/stairs), never a full-block staircase only
    assert sum(1 for k, _, _ in tab.values() if k in (Kind.SLAB, Kind.STAIRS)) >= 16
    top = max(ys)
    assert all(tab[p][2] == Half.BOTTOM for p in tab if p[1] == top and tab[p][0] == Kind.SLAB)
    assert all(tab[p][2] == Half.TOP for p in tab if p[1] == 0 and tab[p][0] == Kind.SLAB)


def test_wedge_golden_each_slope_axis():
    for axis, expect in (("x", Facing.EAST), ("z", Facing.SOUTH)):
        r, f = _fit({"type": "wedge", "size": [10, 5, 8], "slope_axis": axis})
        st = np.argwhere(f.kind == Kind.STAIRS)
        assert len(st) > 0
        assert {Facing(int(f.facing[tuple(t)])) for t in st} == {expect}, axis
        assert all(f.half[tuple(t)] == Half.BOTTOM for t in st)


def test_arch_golden_symmetric_and_top_half_at_crown():
    e = SceneEditor(Scene("a"))
    e.add("wall", {"type": "box", "size": [16, 12, 3]}, pos=[0, 0, 0], material="stone")
    e.add("arch", {"type": "cylinder", "radius": 5, "height": 5, "axis": "z"}, pos=[0, 0, 0], op="subtract")
    r = rasterize(e.scene)
    f = fit_surface(r, {})
    tab = _table(r, f)
    _assert_mirror_symmetric(tab, 0)
    crown = [p for p, (k, _, h) in tab.items() if k == Kind.STAIRS and h == Half.TOP]
    assert crown and max(p[1] for p in crown) >= 3
    assert all(p[2] in (-2, -1, 0, 1) for p in crown)  # inside the wall thickness only


def test_dome_golden_crown_and_shoulders():
    r, f = _fit({"type": "sphere", "radius": 6, "half": True})
    tab = _table(r, f)
    top = max(p[1] for p in tab)
    crown = {p: v for p, v in tab.items() if p[1] == top}
    assert crown and any(v[0] == Kind.SLAB for v in crown.values())
    assert all(v[2] == Half.BOTTOM for v in crown.values())  # nothing upside-down on the crown ring
    stairs = {p: v for p, v in tab.items() if v[0] == Kind.STAIRS}
    assert len(stairs) >= 8
    for (x, y, z), (_, facing, half) in stairs.items():
        v = Facing(facing).vector
        assert (0 - (x + 0.5)) * v[0] + (0 - (z + 0.5)) * v[2] > 0, "stairs' full side faces the dome axis"
