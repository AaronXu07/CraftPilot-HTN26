"""Software renderer over a BlockMap (PNG via PIL) plus ASCII slices. plan.md §6.

Orthographic painter's-order projection of exposed, front-facing block faces. Stairs and slabs are
drawn with their real half/step geometry so the critic sees the sub-voxel smoothing. Deterministic.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .coretypes import BlockMap, parse_state, short_id

ColorFn = Callable[[str], Tuple[int, int, int]]

# view name -> (azimuth deg from +z toward +x, elevation deg); "top" is special-cased
VIEWS: Dict[str, Tuple[float, float]] = {
    "iso": (45.0, 35.0),
    "front": (0.0, 20.0),
    "back": (180.0, 20.0),
    "left": (-90.0, 20.0),
    "right": (90.0, 20.0),
    "top": (0.0, 90.0),
}
SHADE = {(0, 1, 0): 1.0, (0, -1, 0): 0.45, (0, 0, 1): 0.80, (0, 0, -1): 0.70, (1, 0, 0): 0.65, (-1, 0, 0): 0.60}
FACE_NORMALS = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
_K = 4096
_OFF = 2048


def _key(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Pack integer coords into a single int64 key."""
    return ((x.astype(np.int64) + _OFF) * _K + (y.astype(np.int64) + _OFF)) * _K + (z.astype(np.int64) + _OFF)


def camera_axes(view: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(d, right, up): d points from the scene toward the camera; right/up are screen axes."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {', '.join(VIEWS)}")
    az, el = VIEWS[view]
    if view == "top":
        return np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
    a, e = math.radians(az), math.radians(el)
    d = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
    f = -d
    right = np.cross(f, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    up /= np.linalg.norm(up)
    return d, right, up


def block_boxes(state: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sub-boxes (lo, hi in [0,1]^3) that a block state occupies: slabs, stairs, else the unit cube."""
    bid, props = parse_state(state)
    name = short_id(bid)
    if name.endswith("_slab"):
        t = props.get("type", "bottom")
        if t == "double":
            return [(np.zeros(3), np.ones(3))]
        if t == "top":
            return [(np.array([0.0, 0.5, 0.0]), np.array([1.0, 1.0, 1.0]))]
        return [(np.zeros(3), np.array([1.0, 0.5, 1.0]))]
    if name.endswith("_stairs"):
        facing = props.get("facing", "north")
        half = props.get("half", "bottom")
        if half == "top":
            slab = (np.array([0.0, 0.5, 0.0]), np.array([1.0, 1.0, 1.0]))
            y0, y1 = 0.0, 0.5
        else:
            slab = (np.zeros(3), np.array([1.0, 0.5, 1.0]))
            y0, y1 = 0.5, 1.0
        if facing == "east":
            step = (np.array([0.5, y0, 0.0]), np.array([1.0, y1, 1.0]))
        elif facing == "west":
            step = (np.array([0.0, y0, 0.0]), np.array([0.5, y1, 1.0]))
        elif facing == "south":
            step = (np.array([0.0, y0, 0.5]), np.array([1.0, y1, 1.0]))
        else:
            step = (np.array([0.0, y0, 0.0]), np.array([1.0, y1, 0.5]))
        return [slab, step]
    if name.endswith("_wall") or name.endswith("_fence") or name == "iron_bars" or name.endswith("_pane"):
        return [(np.array([0.25, 0.0, 0.25]), np.array([0.75, 1.0, 0.75]))]
    if name in ("lantern", "soul_lantern"):
        return [(np.array([0.3, 0.0, 0.3]), np.array([0.7, 0.55, 0.7]))]
    if name.endswith("torch") or name.endswith("_flower") or name in ("poppy", "dandelion", "grass", "short_grass", "fern"):
        return [(np.array([0.4, 0.0, 0.4]), np.array([0.6, 0.6, 0.6]))]
    return [(np.zeros(3), np.ones(3))]


def apply_cutaway(block_map: BlockMap, cutaway: Optional[Tuple[str, float]]) -> BlockMap:
    """Drop blocks beyond a cut plane: ("x", 10) keeps x <= 10; ("-x", 10) keeps x >= 10."""
    if not cutaway:
        return block_map
    axis, at = cutaway
    axis = str(axis)
    neg = axis.startswith("-")
    ai = "xyz".index(axis.lstrip("-+"))
    at = float(at)
    if neg:
        return {p: s for p, s in block_map.items() if p[ai] >= at}
    return {p: s for p, s in block_map.items() if p[ai] <= at}


def _build_faces(block_map: BlockMap, color_fn: ColorFn):
    """Vectorised face list: corners [F,4,3], normals [F,3], colors [F,3], on-boundary flags [F]."""
    if not block_map:
        return None
    coords = np.array(list(block_map.keys()), dtype=np.int64)
    states = list(block_map.values())
    uniq = {}
    for s in states:
        if s not in uniq:
            uniq[s] = (block_boxes(s), np.array(color_fn(s), dtype=np.float64))
    full_mask = np.array([len(uniq[s][0]) == 1 and np.all(uniq[s][0][0][0] == 0) and np.all(uniq[s][0][0][1] == 1) for s in states])
    full_keys = np.sort(_key(coords[full_mask, 0], coords[full_mask, 1], coords[full_mask, 2]))
    all_keys = np.sort(_key(coords[:, 0], coords[:, 1], coords[:, 2]))

    corners_l, normals_l, colors_l, boundary_l, cells_l = [], [], [], [], []
    # group blocks by state to vectorise
    by_state: Dict[str, List[int]] = {}
    for i, s in enumerate(states):
        by_state.setdefault(s, []).append(i)
    for s, idxs in by_state.items():
        boxes, color = uniq[s]
        cells = coords[idxs]  # [B,3]
        for lo, hi in boxes:
            for n in FACE_NORMALS:
                nx, ny, nz = n
                # face rectangle corners in local box coords
                if nx != 0:
                    xf = hi[0] if nx > 0 else lo[0]
                    c = np.array([[xf, lo[1], lo[2]], [xf, hi[1], lo[2]], [xf, hi[1], hi[2]], [xf, lo[1], hi[2]]])
                    on_b = (xf == 1.0) if nx > 0 else (xf == 0.0)
                elif ny != 0:
                    yf = hi[1] if ny > 0 else lo[1]
                    c = np.array([[lo[0], yf, lo[2]], [hi[0], yf, lo[2]], [hi[0], yf, hi[2]], [lo[0], yf, hi[2]]])
                    on_b = (yf == 1.0) if ny > 0 else (yf == 0.0)
                else:
                    zf = hi[2] if nz > 0 else lo[2]
                    c = np.array([[lo[0], lo[1], zf], [hi[0], lo[1], zf], [hi[0], hi[1], zf], [lo[0], hi[1], zf]])
                    on_b = (zf == 1.0) if nz > 0 else (zf == 0.0)
                cc = cells[:, None, :].astype(np.float64) + c[None, :, :]  # [B,4,3]
                corners_l.append(cc)
                normals_l.append(np.tile(np.array(n, dtype=np.float64), (len(cells), 1)))
                colors_l.append(np.tile(color, (len(cells), 1)))
                boundary_l.append(np.full(len(cells), bool(on_b)))
                cells_l.append(cells)
    corners = np.concatenate(corners_l)
    normals = np.concatenate(normals_l)
    colors = np.concatenate(colors_l)
    boundary = np.concatenate(boundary_l)
    cells = np.concatenate(cells_l)
    # hide boundary faces whose neighbour is a full block
    nb = cells + normals.astype(np.int64)
    nb_full = np.isin(_key(nb[:, 0], nb[:, 1], nb[:, 2]), full_keys, assume_unique=False)
    keep = ~(boundary & nb_full)
    corners, normals, colors, cells = corners[keep], normals[keep], colors[keep], cells[keep]
    # ambient occlusion: occupied cells around the air cell in front of the face
    nb = cells + normals.astype(np.int64)
    ao = np.zeros(len(cells))
    for ax in range(3):
        for sgn in (-1, 1):
            off = np.zeros(3, dtype=np.int64)
            off[ax] = sgn
            in_plane = normals[:, ax] == 0
            q = nb + off[None, :]
            occ = np.isin(_key(q[:, 0], q[:, 1], q[:, 2]), all_keys)
            ao += (occ & in_plane) * 0.07
    shade = np.array([SHADE[(int(n[0]), int(n[1]), int(n[2]))] for n in normals])
    shade = shade * (1.0 - np.minimum(ao, 0.28))
    colors = np.clip(colors * shade[:, None], 0, 255)
    return corners, normals, colors


def render_blocks(
    block_map: BlockMap,
    color_fn: ColorFn,
    view: str = "iso",
    size: int = 1024,
    cutaway: Optional[Tuple[str, float]] = None,
    background: Tuple[int, int, int] = (235, 238, 242),
    ground: bool = True,
    margin: int = 24,
) -> Image.Image:
    """Render a block map from a named view into a PIL image of at most `size` px."""
    bm = apply_cutaway(block_map, cutaway)
    img = Image.new("RGB", (size, size), background)
    if not bm:
        return img
    d, right, up = camera_axes(view)
    built = _build_faces(bm, color_fn)
    if built is None:
        return img
    corners, normals, colors = built
    front = (normals @ d) > 1e-9
    corners, colors = corners[front], colors[front]
    # projection
    flat = corners.reshape(-1, 3)
    sx = flat @ right
    sy = flat @ up
    depth = (corners.mean(axis=1) @ d)
    coords = np.array(list(bm.keys()), dtype=np.float64)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1.0
    # include ground plane corners in the framing
    gc = np.array([[x, lo[1], z] for x in (lo[0] - 1, hi[0] + 1) for z in (lo[2] - 1, hi[2] + 1)])
    gx, gy = gc @ right, gc @ up
    xmin, xmax = min(sx.min(), gx.min()), max(sx.max(), gx.max())
    ymin, ymax = min(sy.min(), gy.min()), max(sy.max(), gy.max())
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    scale = (size - 2 * margin) / span
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    def to_px(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return np.stack([(px - cx) * scale + size / 2, size / 2 - (py - cy) * scale], axis=-1)

    draw = ImageDraw.Draw(img)
    if ground and view != "top":
        g = to_px(gx, gy)
        gorder = [0, 1, 3, 2]
        gcol = tuple(int(v * 0.93) for v in background)
        draw.polygon([tuple(g[i]) for i in gorder], fill=gcol, outline=None)
    pix = to_px(sx, sy).reshape(-1, 4, 2)
    order = np.argsort(depth, kind="stable")
    icols = colors.astype(np.int64)
    for i in order:
        c = icols[i]
        poly = [(float(pix[i, k, 0]), float(pix[i, k, 1])) for k in range(4)]
        draw.polygon(poly, fill=(int(c[0]), int(c[1]), int(c[2])))
    return img


def contact_sheet(
    block_map: BlockMap,
    color_fn: ColorFn,
    views: Sequence[str] = ("iso", "front", "top"),
    cutaway: Optional[Tuple[str, float]] = None,
    size: int = 1024,
    background: Tuple[int, int, int] = (235, 238, 242),
) -> Image.Image:
    """2x2 sheet: the given views plus a cutaway (or 'right' view) so one vision call sees everything."""
    tile = size // 2
    panels: List[Tuple[str, Image.Image]] = []
    for v in list(views)[:3]:
        panels.append((v, render_blocks(block_map, color_fn, view=v, size=tile, background=background)))
    if cutaway is not None:
        panels.append((f"cutaway {cutaway[0]}<={cutaway[1]:g}", render_blocks(block_map, color_fn, view="iso", size=tile, cutaway=cutaway, background=background)))
    else:
        coords = np.array(list(block_map.keys())) if block_map else np.zeros((1, 3))
        mid = float(np.median(coords[:, 2])) if len(coords) else 0.0
        panels.append((f"cutaway z<={mid:g}", render_blocks(block_map, color_fn, view="iso", size=tile, cutaway=("z", mid), background=background)))
    sheet = Image.new("RGB", (tile * 2, tile * 2), background)
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(panels[:4]):
        x, y = (i % 2) * tile, (i // 2) * tile
        sheet.paste(im, (x, y))
        draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(180, 184, 190))
        draw.rectangle([x + 4, y + 4, x + 12 + 7 * len(label), y + 20], fill=(40, 40, 48))
        draw.text((x + 8, y + 6), label, fill=(240, 240, 240))
    return sheet


def ascii_slice(block_map: BlockMap, y: Optional[int] = None, axis: str = "y", at: Optional[float] = None, glyph_fn: Optional[Callable[[str], str]] = None) -> str:
    """ASCII slice of a block map: a y-layer viewed from above (north up) or a vertical x/z cut."""
    if not block_map:
        return "(empty)"
    coords = np.array(list(block_map.keys()), dtype=np.int64)
    if axis == "y":
        layer = int(y if y is not None else (at if at is not None else np.median(coords[:, 1])))
        sel = {p: s for p, s in block_map.items() if p[1] == layer}
        rows_axis, cols_axis, title = 2, 0, f"slice y={layer} (north up, east right)"
    else:
        ai = "xyz".index(axis)
        cut = int(at if at is not None else np.median(coords[:, ai]))
        sel = {p: s for p, s in block_map.items() if p[ai] == cut}
        cols_axis = 2 if axis == "x" else 0
        rows_axis, title = 1, f"slice {axis}={cut} (up is +y)"
    if not sel:
        return title + ": (no blocks)"
    names: Dict[str, int] = {}
    for s in sel.values():
        n = short_id(parse_state(s)[0])
        names[n] = names.get(n, 0) + 1
    ranked = sorted(names, key=lambda n: -names[n])
    glyphs = "#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    table = {n: (glyph_fn(n) if glyph_fn else glyphs[i % len(glyphs)]) for i, n in enumerate(ranked)}
    pts = np.array(list(sel.keys()), dtype=np.int64)
    r0, r1 = pts[:, rows_axis].min(), pts[:, rows_axis].max()
    c0, c1 = pts[:, cols_axis].min(), pts[:, cols_axis].max()
    grid = {(p[rows_axis], p[cols_axis]): table[short_id(parse_state(s)[0])] for p, s in sel.items()}
    lines = [title]
    row_range = range(r0, r1 + 1) if axis == "y" else range(r1, r0 - 1, -1)
    for r in row_range:
        lines.append(f"{r:4d} " + "".join(grid.get((r, c), ".") for c in range(c0, c1 + 1)))
    lines.append(f"cols {('x' if cols_axis == 0 else 'z')} {c0}..{c1}; legend: " + " ".join(f"{table[n]}={n}" for n in ranked[:12]))
    return "\n".join(lines)


def default_color_fn(state: str) -> Tuple[int, int, int]:
    """Deterministic fallback colour per block id (hash-based pastel) for renders without a registry."""
    name = short_id(parse_state(state)[0])
    h = 0
    for ch in name:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    r = 90 + (h & 0x7F)
    g = 90 + ((h >> 8) & 0x7F)
    b = 90 + ((h >> 16) & 0x7F)
    if "stone" in name or "cobble" in name or "deepslate" in name:
        return (128 + (h & 15), 128 + (h & 15), 130 + (h & 15))
    if "planks" in name or "log" in name or "wood" in name:
        return (150, 110, 70)
    if "glass" in name:
        return (190, 220, 235)
    return (r, g, b)
