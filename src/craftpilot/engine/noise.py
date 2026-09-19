"""Seeded, clustered noise fields for gradients and texturing."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def clustered(shape: tuple[int, ...], scale: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform-ish noise in [0, 1] with blobs of roughly `scale` blocks."""
    raw = rng.random(shape).astype(np.float32)
    if scale <= 0.5:
        return raw
    sm = ndimage.gaussian_filter(raw, sigma=scale / 2.0, mode="nearest")
    lo, hi = float(sm.min()), float(sm.max())
    if hi - lo < 1e-6:
        return raw
    # Rank-normalise so the distribution is uniform again.
    flat = sm.ravel()
    order = np.argsort(flat)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(flat.size)
    return (ranks / max(1, flat.size - 1)).reshape(shape).astype(np.float32)
