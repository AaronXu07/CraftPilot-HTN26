"""SDF primitives (Inigo Quilez formulas). See CONTRACTS.md §3 and shapes.py for conventions.

Every solid evaluates `sdf(pts)` on local-space points `[N,3]` and returns signed distances
(negative inside). `local_bbox()` matches `shapes.shape_local_bbox`. `shrunk(r)` returns the same
primitive shrunk inward by `r` (used by the `round` modifier: `shrunk(r).sdf(p) - r`), or None when
the primitive cannot be shrunk meaningfully.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .shapes import ShapeError, shape_local_bbox, validate_shape

Vec = np.ndarray


def _len(v: Vec) -> Vec:
    """Row-wise Euclidean length."""
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def _box_dist(q: Vec) -> Vec:
    """Signed distance for an axis-aligned box given q = |p - c| - half (any dimension)."""
    outside = _len(np.maximum(q, 0.0))
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside


class Solid:
    """Base class: a validated shape dict plus an SDF in local space."""

    def __init__(self, shape: Dict[str, Any]):
        self.shape = validate_shape(shape)
        self._lo, self._hi = shape_local_bbox(self.shape)

    @property
    def type(self) -> str:
        return self.shape["type"]

    def local_bbox(self) -> Tuple[Vec, Vec]:
        """Local-space AABB (lo, hi), identical to shapes.shape_local_bbox."""
        return self._lo.copy(), self._hi.copy()

    def sdf(self, pts: Vec) -> Vec:  # pragma: no cover - abstract
        """Signed distance at local points [N,3]."""
        raise NotImplementedError

    def shrunk(self, r: float) -> Optional["Solid"]:
        """Return this primitive shrunk inward by r (same local frame), or None if unsupported."""
        return None

    def __repr__(self) -> str:
        return f"Solid({self.shape})"


class _Offset(Solid):
    """Wrap a solid so that it is evaluated shifted by `offset` (used to keep shrunk solids centred)."""

    def __init__(self, inner: Solid, offset):
        self.inner = inner
        self.offset = np.asarray(offset, dtype=np.float64)
        self.shape = inner.shape
        self._lo = inner._lo + self.offset
        self._hi = inner._hi + self.offset

    def sdf(self, pts: Vec) -> Vec:
        """Evaluate the inner solid at pts - offset."""
        return self.inner.sdf(np.asarray(pts, dtype=np.float64) - self.offset)


def _shrink_dims(shape: Dict[str, Any], r: float, updates: Dict[str, Any], lift: bool = True) -> Optional[Solid]:
    """Build a shrunk copy of `shape` with param `updates`; lift by r so the base stays centred."""
    new = dict(shape)
    new.update(updates)
    try:
        inner = make_solid(new)
    except ShapeError:
        return None
    return _Offset(inner, (0.0, r, 0.0)) if lift else inner


# ------------------------------------------------------------------------------------------------
# Primitives
# ------------------------------------------------------------------------------------------------
class Box(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Exact box SDF; box spans [-sx/2, sx/2] x [0, sy] x [-sz/2, sz/2]."""
        sx, sy, sz = self.shape["size"]
        half = np.array([sx / 2, sy / 2, sz / 2])
        c = np.array([0.0, sy / 2, 0.0])
        q = np.abs(pts - c) - half
        return _box_dist(q)

    def shrunk(self, r: float) -> Optional[Solid]:
        sx, sy, sz = self.shape["size"]
        if min(sx, sy, sz) <= 2 * r:
            return None
        return _shrink_dims(self.shape, r, {"size": [sx - 2 * r, sy - 2 * r, sz - 2 * r]})


def _capped_cone(q_r: Vec, q_y: Vec, half_h: float, r1: float, r2: float) -> Vec:
    """Quilez sdCappedCone: q=(radial, axial) with axial in [-half_h, half_h], r1 at bottom, r2 at top."""
    k1 = np.array([r2, half_h])
    k2 = np.array([r2 - r1, 2.0 * half_h])
    ca_x = q_r - np.minimum(q_r, np.where(q_y < 0.0, r1, r2))
    ca_y = np.abs(q_y) - half_h
    dot_k2 = float(k2 @ k2)
    t = ((k1[0] - q_r) * k2[0] + (k1[1] - q_y) * k2[1]) / dot_k2 if dot_k2 > 0 else np.zeros_like(q_r)
    t = np.clip(t, 0.0, 1.0)
    cb_x = q_r - k1[0] + k2[0] * t
    cb_y = q_y - k1[1] + k2[1] * t
    s = np.where((cb_x < 0.0) & (ca_y < 0.0), -1.0, 1.0)
    d2 = np.minimum(ca_x * ca_x + ca_y * ca_y, cb_x * cb_x + cb_y * cb_y)
    return s * np.sqrt(d2)


def _radial_axial(pts: Vec, axis: str, radius: float, height: float) -> Tuple[Vec, Vec]:
    """Split local points into (radial distance, axial coordinate centred on the solid's mid-length)."""
    if axis == "y":
        r = np.hypot(pts[:, 0], pts[:, 2])
        a = pts[:, 1] - height / 2.0
    elif axis == "x":
        r = np.hypot(pts[:, 1] - radius, pts[:, 2])
        a = pts[:, 0]
    else:
        r = np.hypot(pts[:, 0], pts[:, 1] - radius)
        a = pts[:, 2]
    return r, a


class Cylinder(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Capped cylinder / frustum along `axis`; horizontal cylinders rest on y=0."""
        s = self.shape
        R = max(s["radius"], s.get("radius_top") or 0.0)
        rr, aa = _radial_axial(pts, s["axis"], R, s["height"])
        r_top = s.get("radius_top")
        if r_top is None or abs(r_top - s["radius"]) < 1e-9:
            d = np.stack([rr - s["radius"], np.abs(aa) - s["height"] / 2.0], axis=1)
            return _box_dist(d)
        return _capped_cone(rr, aa, s["height"] / 2.0, s["radius"], r_top)

    def shrunk(self, r: float) -> Optional[Solid]:
        s = self.shape
        if s["radius"] <= r or s["height"] <= 2 * r:
            return None
        upd: Dict[str, Any] = {"radius": s["radius"] - r, "height": s["height"] - 2 * r}
        if s.get("radius_top") is not None:
            upd["radius_top"] = max(0.0, s["radius_top"] - r)
        return _shrink_dims(s, r, upd)


class Sphere(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Sphere centred at the origin; half=True keeps y >= 0."""
        d = _len(pts) - self.shape["radius"]
        if self.shape.get("half"):
            d = np.maximum(d, -pts[:, 1])
        return d

    def shrunk(self, r: float) -> Optional[Solid]:
        if self.shape["radius"] <= r:
            return None
        return _shrink_dims(self.shape, r, {"radius": self.shape["radius"] - r}, lift=bool(self.shape.get("half")))


class Ellipsoid(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Quilez ellipsoid bound (good to ~0.1 block at building scale)."""
        rad = np.asarray(self.shape["radii"])
        k0 = _len(pts / rad)
        k1 = _len(pts / (rad * rad))
        safe = np.where(k1 < 1e-9, 1.0, k1)
        d = k0 * (k0 - 1.0) / safe
        return np.where(k1 < 1e-9, -float(rad.min()), d)

    def shrunk(self, r: float) -> Optional[Solid]:
        rad = self.shape["radii"]
        if min(rad) <= r:
            return None
        return _shrink_dims(self.shape, r, {"radii": [v - r for v in rad]}, lift=False)


class Cone(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Cone / frustum: radius at y=0, radius_top at y=height."""
        s = self.shape
        rr = np.hypot(pts[:, 0], pts[:, 2])
        aa = pts[:, 1] - s["height"] / 2.0
        return _capped_cone(rr, aa, s["height"] / 2.0, s["radius"], s["radius_top"])

    def shrunk(self, r: float) -> Optional[Solid]:
        s = self.shape
        if s["radius"] <= r or s["height"] <= 2 * r:
            return None
        return _shrink_dims(s, r, {"radius": s["radius"] - r, "radius_top": max(0.0, s["radius_top"] - r), "height": s["height"] - 2 * r})


class Pyramid(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Rectangular pyramid / frustum via a height-lerped 2D box with slope-corrected distances."""
        s = self.shape
        sx, sz = s["base"]
        tx, tz = s["top"]
        h = s["height"]
        y = pts[:, 1]
        t = np.clip(y / h, 0.0, 1.0)
        hx = (sx / 2) * (1 - t) + (tx / 2) * t
        hz = (sz / 2) * (1 - t) + (tz / 2) * t
        cx = math.cos(math.atan2(abs(sx - tx) / 2.0, h))
        cz = math.cos(math.atan2(abs(sz - tz) / 2.0, h))
        qx = (np.abs(pts[:, 0]) - hx) * cx
        qz = (np.abs(pts[:, 2]) - hz) * cz
        d2 = _box_dist(np.stack([qx, qz], axis=1))
        dy = np.abs(y - h / 2.0) - h / 2.0
        return _box_dist(np.stack([d2, dy], axis=1))

    def shrunk(self, r: float) -> Optional[Solid]:
        s = self.shape
        if min(s["base"]) <= 2 * r or s["height"] <= 2 * r:
            return None
        return _shrink_dims(s, r, {"base": [v - 2 * r for v in s["base"]], "top": [max(0.0, v - 2 * r) for v in s["top"]], "height": s["height"] - 2 * r})


class Wedge(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Box intersected with a sloped half-space; height rises toward the +slope_axis end."""
        sx, sy, sz = self.shape["size"]
        half = np.array([sx / 2, sy / 2, sz / 2])
        c = np.array([0.0, sy / 2, 0.0])
        box = _box_dist(np.abs(pts - c) - half)
        sa = self.shape["slope_axis"]
        ax = 0 if sa.endswith("x") else 2
        sign = -1.0 if sa.startswith("-") else 1.0
        L = sx if ax == 0 else sz
        u = sign * pts[:, ax] + L / 2.0  # 0 at the low end, L at the high end
        # plane through (u=0, y=0) and (u=L, y=sy); outward normal (-sy, L)/norm in (u, y)
        norm = math.hypot(sy, L)
        plane = (-sy * u + L * pts[:, 1]) / norm
        return np.maximum(box, plane)

    def shrunk(self, r: float) -> Optional[Solid]:
        sx, sy, sz = self.shape["size"]
        if min(sx, sy, sz) <= 2 * r:
            return None
        return _shrink_dims(self.shape, r, {"size": [sx - 2 * r, sy - 2 * r, sz - 2 * r]})


class Prism(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Regular n-gon (circumradius `radius`, a vertex on +x) extruded from y=0 to height."""
        n = self.shape["sides"]
        R = self.shape["radius"]
        h = self.shape["height"]
        sector = 2.0 * math.pi / n
        ang = np.arctan2(pts[:, 2], pts[:, 0])
        a = np.mod(ang + sector / 2.0, sector) - sector / 2.0
        rho = np.hypot(pts[:, 0], pts[:, 2])
        px = rho * np.cos(a)
        py = rho * np.abs(np.sin(a))
        apothem = R * math.cos(math.pi / n)
        halfedge = R * math.sin(math.pi / n)
        ex = px - apothem
        ey = py - halfedge
        d2 = np.where(ey > 0.0, np.sqrt(np.maximum(ex, 0.0) ** 2 + ey * ey), ex)
        dy = np.abs(pts[:, 1] - h / 2.0) - h / 2.0
        return _box_dist(np.stack([d2, dy], axis=1))

    def shrunk(self, r: float) -> Optional[Solid]:
        s = self.shape
        apothem = s["radius"] * math.cos(math.pi / s["sides"])
        if apothem <= r or s["height"] <= 2 * r:
            return None
        return _shrink_dims(s, r, {"radius": s["radius"] - r / math.cos(math.pi / s["sides"]), "height": s["height"] - 2 * r})


class Torus(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Torus centred at the origin around `axis`."""
        R, m, ax = self.shape["major"], self.shape["minor"], self.shape["axis"]
        if ax == "y":
            q = np.stack([np.hypot(pts[:, 0], pts[:, 2]) - R, pts[:, 1]], axis=1)
        elif ax == "x":
            q = np.stack([np.hypot(pts[:, 1], pts[:, 2]) - R, pts[:, 0]], axis=1)
        else:
            q = np.stack([np.hypot(pts[:, 0], pts[:, 1]) - R, pts[:, 2]], axis=1)
        return _len(q) - m

    def shrunk(self, r: float) -> Optional[Solid]:
        if self.shape["minor"] <= r:
            return None
        return _shrink_dims(self.shape, r, {"minor": self.shape["minor"] - r}, lift=False)


def _segment_dist(pts: Vec, a: Vec, b: Vec) -> Vec:
    """Distance from points to segment ab (3D or 2D)."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return _len(pts - a)
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    return _len(pts - (a + t[:, None] * ab))


class Capsule(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Vertical capsule centred at the origin; `height` includes the caps."""
        r, h = self.shape["radius"], self.shape["height"]
        half = max(h / 2.0 - r, 0.0)
        a = np.array([0.0, -half, 0.0])
        b = np.array([0.0, half, 0.0])
        return _segment_dist(pts, a, b) - r

    def shrunk(self, r: float) -> Optional[Solid]:
        if self.shape["radius"] <= r:
            return None
        return _shrink_dims(self.shape, r, {"radius": self.shape["radius"] - r, "height": self.shape["height"] - 2 * r}, lift=False)


def polygon_sdf(pts2: Vec, poly: Vec) -> Vec:
    """Exact signed distance to a closed 2D polygon (Quilez), negative inside; pts2 [N,2], poly [M,2]."""
    n = len(poly)
    d = np.einsum("ij,ij->i", pts2 - poly[0], pts2 - poly[0])
    s = np.ones(len(pts2))
    for i in range(n):
        j = (i - 1) % n
        vi, vj = poly[i], poly[j]
        e = vj - vi
        w = pts2 - vi
        ee = float(e @ e)
        t = np.clip((w @ e) / ee, 0.0, 1.0) if ee > 0 else np.zeros(len(pts2))
        b = w - e[None, :] * t[:, None]
        d = np.minimum(d, np.einsum("ij,ij->i", b, b))
        c1 = pts2[:, 1] >= vi[1]
        c2 = pts2[:, 1] < vj[1]
        c3 = e[0] * w[:, 1] > e[1] * w[:, 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
    return s * np.sqrt(d)


class Extrude(Solid):
    def __init__(self, shape):
        super().__init__(shape)
        self._poly = np.asarray(self.shape["profile"], dtype=np.float64)
        self._inset = 0.0

    def sdf(self, pts: Vec) -> Vec:
        """2D polygon SDF in the xz plane extruded from y=0 to height."""
        d2 = polygon_sdf(pts[:, [0, 2]], self._poly) + self._inset
        h = self.shape["height"]
        dy = np.abs(pts[:, 1] - h / 2.0) - h / 2.0
        return _box_dist(np.stack([d2, dy], axis=1))

    def shrunk(self, r: float) -> Optional[Solid]:
        """Shrink the height exactly and inset the profile by r (approximate near convex corners)."""
        if self.shape["height"] <= 2 * r:
            return None
        new = Extrude(dict(self.shape, height=self.shape["height"] - 2 * r))
        new._inset = r
        return _Offset(new, (0.0, r, 0.0))


class Revolve(Solid):
    def __init__(self, shape):
        super().__init__(shape)
        prof = np.asarray(self.shape["profile"], dtype=np.float64)
        # Mirror the profile across the axis so points on r=0 are interior, not on an edge.
        mirrored = prof[::-1].copy()
        mirrored[:, 0] = -mirrored[:, 0]
        self._poly = np.concatenate([prof, mirrored], axis=0)
        self._inset = 0.0

    def sdf(self, pts: Vec) -> Vec:
        """Lathe: polygon SDF in (r, y) with r = sqrt(x^2 + z^2)."""
        r = np.hypot(pts[:, 0], pts[:, 2])
        q = np.stack([r, pts[:, 1]], axis=1)
        return polygon_sdf(q, self._poly) + self._inset

    def shrunk(self, r: float) -> Optional[Solid]:
        new = Revolve(dict(self.shape))
        new._inset = r
        return new


class Sweep(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Tube of `radius` along a polyline path (optionally closed)."""
        path = np.asarray(self.shape["path"], dtype=np.float64)
        segs = list(zip(path[:-1], path[1:]))
        if self.shape.get("closed") and len(path) > 2:
            segs.append((path[-1], path[0]))
        d = np.full(len(pts), np.inf)
        for a, b in segs:
            d = np.minimum(d, _segment_dist(pts, a, b))
        return d - self.shape["radius"]

    def shrunk(self, r: float) -> Optional[Solid]:
        if self.shape["radius"] <= r:
            return None
        return _shrink_dims(self.shape, r, {"radius": self.shape["radius"] - r}, lift=False)


class PlaneCut(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Half-space n·p - offset <= 0 (use with op=intersect to slice)."""
        n = np.asarray(self.shape["normal"])
        return pts @ n - self.shape["offset"]

    def shrunk(self, r: float) -> Optional[Solid]:
        return _shrink_dims(self.shape, r, {"offset": self.shape["offset"] - r}, lift=False)


class Block(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Unit cube [0,1)^3 (the rasterizer places `block` solids directly; this is for completeness)."""
        q = np.abs(pts - 0.5) - 0.5
        return _box_dist(q)


class Line(Solid):
    def sdf(self, pts: Vec) -> Vec:
        """Capsule between `from` and `to` with diameter `thickness`."""
        a = np.asarray(self.shape["from"], dtype=np.float64)
        b = np.asarray(self.shape["to"], dtype=np.float64)
        return _segment_dist(pts, a, b) - self.shape["thickness"] / 2.0

    def shrunk(self, r: float) -> Optional[Solid]:
        if self.shape["thickness"] <= 2 * r:
            return None
        return _shrink_dims(self.shape, r, {"thickness": self.shape["thickness"] - 2 * r}, lift=False)


PRIMITIVES = {
    "box": Box,
    "cylinder": Cylinder,
    "sphere": Sphere,
    "ellipsoid": Ellipsoid,
    "cone": Cone,
    "pyramid": Pyramid,
    "wedge": Wedge,
    "prism": Prism,
    "torus": Torus,
    "capsule": Capsule,
    "extrude": Extrude,
    "revolve": Revolve,
    "sweep": Sweep,
    "plane_cut": PlaneCut,
    "block": Block,
    "line": Line,
}


def make_solid(shape: Dict[str, Any]) -> Solid:
    """Build the Solid for a shape dict (validated via shapes.validate_shape)."""
    shape = validate_shape(shape)
    return PRIMITIVES[shape["type"]](shape)
