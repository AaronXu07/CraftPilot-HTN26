"""Seeded coherent value noise (numpy only). Shared by modifiers (noise_displace) and materials."""
from __future__ import annotations

import numpy as np


def _hash3(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic integer lattice hash -> float in [0, 1)."""
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + iz.astype(np.int64) * 2147483647 + int(seed) * 1013904223) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0x1000000)


def _fade(t: np.ndarray) -> np.ndarray:
    """Quintic smoothstep."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise3(pts: np.ndarray, scale: float, seed: int = 0) -> np.ndarray:
    """Single-octave value noise in [0, 1] at points [N,3] with lattice spacing `scale` blocks."""
    p = np.asarray(pts, dtype=np.float64) / float(max(scale, 1e-6))
    i = np.floor(p).astype(np.int64)
    f = _fade(p - i)
    ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]

    def c(dx, dy, dz):
        return _hash3(ix + dx, iy + dy, iz + dz, seed)

    x00 = c(0, 0, 0) * (1 - fx) + c(1, 0, 0) * fx
    x10 = c(0, 1, 0) * (1 - fx) + c(1, 1, 0) * fx
    x01 = c(0, 0, 1) * (1 - fx) + c(1, 0, 1) * fx
    x11 = c(0, 1, 1) * (1 - fx) + c(1, 1, 1) * fx
    y0 = x00 * (1 - fy) + x10 * fy
    y1 = x01 * (1 - fy) + x11 * fy
    return y0 * (1 - fz) + y1 * fz


def noise3(pts: np.ndarray, scale: float, seed: int = 0, octaves: int = 2) -> np.ndarray:
    """Coherent fractal value noise in [-1, 1] at points [N,3]; deterministic for (scale, seed)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    total = np.zeros(len(pts))
    amp = 1.0
    norm = 0.0
    s = float(scale)
    for o in range(max(1, octaves)):
        total += amp * value_noise3(pts + 17.31 * o, s, seed + 101 * o)
        norm += amp
        amp *= 0.5
        s = max(s / 2.0, 0.5)
    return (total / norm) * 2.0 - 1.0
