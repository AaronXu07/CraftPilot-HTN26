"""The hologram cloud the Fabric mod renders: every filled cell of a grid as a flat [x, y, z, rgb, ...] list."""

from __future__ import annotations

import numpy as np

from craftpilot.grid.semantic import SemanticGrid
from craftpilot.preview.render import block_color


def ghost_cloud(grid: SemanticGrid) -> tuple[list[int], int, int]:
    """Returns (flat cloud, trimmed height, block count) for a grid authored facing south; the mod rotates it."""
    colors = []
    for ref in grid.palette:
        r, g, b = block_color(ref.block_id)
        colors.append((r << 16) | (g << 8) | b)
    xs, ys, zs = np.nonzero(grid.block >= 0)
    if not xs.size:
        return [], 1, 0
    rgb = np.array(colors, dtype=np.int64)[grid.block[xs, ys, zs]]
    flat = np.stack([xs, ys, zs, rgb], axis=1).reshape(-1).tolist()
    return flat, int(ys.max()) + 1, int(xs.size)
