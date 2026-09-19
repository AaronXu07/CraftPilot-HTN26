"""Voxel circles and polygons the way Minecraft builders draw them."""

from __future__ import annotations

import math

import numpy as np


def circle_mask(width: int, depth: int, cx: float, cz: float, radius: float) -> np.ndarray:
    """Filled circle. Uses the (r + 0.5) rule that matches common circle generators."""
    xs = np.arange(width)[:, None]
    zs = np.arange(depth)[None, :]
    return (xs - cx) ** 2 + (zs - cz) ** 2 < (radius + 0.5) ** 2


def ngon_mask(width: int, depth: int, cx: float, cz: float, radius: float, sides: int,
              rotation: float = 0.0) -> np.ndarray:
    """Filled regular polygon with a flat side facing south by default."""
    sides = max(3, sides)
    rot = rotation + math.pi / sides + math.pi / 2
    verts = [(cx + radius * math.cos(rot + 2 * math.pi * i / sides),
              cz + radius * math.sin(rot + 2 * math.pi * i / sides)) for i in range(sides)]
    xs = np.arange(width)[:, None].astype(float)
    zs = np.arange(depth)[None, :].astype(float)
    inside = np.ones((width, depth), dtype=bool)
    # Convex polygon: point is inside if it is on the same side of every edge.
    for i in range(sides):
        ax, az = verts[i]
        bx, bz = verts[(i + 1) % sides]
        cross = (bx - ax) * (zs - az) - (bz - az) * (xs - ax)
        inside &= cross >= -0.5
    return inside


def perimeter(mask: np.ndarray) -> np.ndarray:
    """Cells in the mask with at least one 4-neighbour outside it."""
    padded = np.pad(mask, 1, constant_values=False)
    n = padded[1:-1, :-2] & padded[1:-1, 2:] & padded[:-2, 1:-1] & padded[2:, 1:-1]
    return mask & ~n


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Grow a mask by n blocks in all eight directions (square structure), so corners are kept."""
    if n <= 0:
        return mask.copy()
    from scipy import ndimage
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=n)
