"""Rigid/scale transforms, Euler rotations and anchors. See CONTRACTS.md §2.

World -> local for an object: q = (R^-1 · (p - pos)) / scale + anchor_point_local.
Rotation R = Rz(rz) · Ry(ry) · Rx(rx) (X applied first). Angles in degrees.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

ANCHORS = ("bottom_center", "center", "bottom_min", "top_center", "bottom_max")


def rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_matrix(rot: Sequence[float]) -> np.ndarray:
    rx, ry, rz = (float(v) for v in rot)
    if rx == 0 and ry == 0 and rz == 0:
        return np.eye(3)
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def anchor_point(lo: np.ndarray, hi: np.ndarray, anchor: str) -> np.ndarray:
    """The local-space point that is placed at transform.pos."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    # Centred anchors snap to an integer offset from lo so that integer-sized shapes at integer
    # positions fill whole voxels: a 9-wide shape at x=34 occupies voxels 30..38 (centre voxel 34),
    # a 14-wide shape at x=17 occupies 10..23 (centre boundary at 17). Voxel v spans [v, v+1).
    c = lo + np.floor((hi - lo) / 2.0)
    if anchor == "bottom_center":
        return np.array([c[0], lo[1], c[2]])
    if anchor == "center":
        return c
    if anchor == "bottom_min":
        return lo.copy()
    if anchor == "bottom_max":
        return np.array([hi[0], lo[1], hi[2]])
    if anchor == "top_center":
        return np.array([c[0], hi[1], c[2]])
    raise ValueError(f"unknown anchor {anchor!r}; expected one of {ANCHORS}")


class Transform:
    """Object transform. `pos` is where the anchor point lands in world space."""

    __slots__ = ("pos", "rot", "scale", "anchor", "_R", "_Rinv")

    def __init__(self, pos=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), anchor="bottom_center"):
        self.pos = np.asarray(_vec3(pos), dtype=np.float64)
        self.rot = np.asarray(_vec3(rot), dtype=np.float64)
        s = scale
        if isinstance(s, (int, float)):
            s = (s, s, s)
        self.scale = np.asarray(_vec3(s), dtype=np.float64)
        if np.any(self.scale == 0):
            raise ValueError("scale components must be non-zero")
        if anchor not in ANCHORS:
            raise ValueError(f"unknown anchor {anchor!r}; expected one of {ANCHORS}")
        self.anchor = anchor
        self._R = euler_matrix(self.rot)
        self._Rinv = self._R.T

    @staticmethod
    def from_dict(d: dict | None) -> "Transform":
        d = d or {}
        return Transform(
            pos=d.get("pos", (0, 0, 0)),
            rot=d.get("rot", (0, 0, 0)),
            scale=d.get("scale", (1, 1, 1)),
            anchor=d.get("anchor", "bottom_center"),
        )

    def to_dict(self) -> dict:
        return {
            "pos": [_num(v) for v in self.pos],
            "rot": [_num(v) for v in self.rot],
            "scale": [_num(v) for v in self.scale],
            "anchor": self.anchor,
        }

    @property
    def R(self) -> np.ndarray:
        return self._R

    def is_axis_aligned(self) -> bool:
        return bool(np.allclose(np.mod(self.rot, 90.0), 0.0))

    # -- point mapping ---------------------------------------------------
    def world_to_local(self, pts: np.ndarray, anchor_pt: np.ndarray) -> np.ndarray:
        """pts [N,3] world -> local shape space (anchor_pt = anchor_point(lo, hi, anchor))."""
        q = pts - self.pos
        if not (self.rot == 0).all():
            q = q @ self._R  # (R^-1 p) for row vectors == p @ R
        q = q / self.scale
        return q + anchor_pt

    def local_to_world(self, pts: np.ndarray, anchor_pt: np.ndarray) -> np.ndarray:
        q = (pts - anchor_pt) * self.scale
        if not (self.rot == 0).all():
            q = q @ self._Rinv
        return q + self.pos

    def world_bbox(self, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Conservative world AABB of the local box [lo, hi] under this transform."""
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        a = anchor_point(lo, hi, self.anchor)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = self.local_to_world(corners, a)
        return w.min(axis=0), w.max(axis=0)

    def scale_factor(self) -> float:
        """Approximate isotropic distance scale (for SDF distance correction)."""
        return float(np.min(np.abs(self.scale)))

    def copy(self) -> "Transform":
        return Transform(self.pos.copy(), self.rot.copy(), self.scale.copy(), self.anchor)


def _vec3(v) -> Tuple[float, float, float]:
    v = list(v)
    if len(v) != 3:
        raise ValueError(f"expected 3 components, got {v}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _num(v: float):
    f = float(v)
    return int(f) if f.is_integer() else round(f, 4)


def rotate_point_about(p: np.ndarray, deg: float, axis: str, pivot: np.ndarray) -> np.ndarray:
    R = {"x": rot_x, "y": rot_y, "z": rot_z}[axis](deg)
    return (R @ (np.asarray(p, dtype=np.float64) - pivot)) + pivot


def snap_angle(deg: float, step: float = 90.0) -> float:
    return float(np.round(deg / step) * step) % 360.0
