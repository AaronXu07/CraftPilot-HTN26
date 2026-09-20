"""Design-rule linter (plan.md §8.2). Mechanical checks the critic can cite by rule number.

R1 blank façade · R2 roof overhang/slope · R3 entrance · R4 material variety · R5 floating/isolated
blocks · R6 props resting on air · R7 scale (dimension > 120) · R8 unknown material names ·
R9 materials stage: ≥ 3 distinct materials on solids and a ground gradient on the main wall material.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np

from .coretypes import BlockMap, FitResult, Kind, LintFinding, RasterResult, parse_state, short_id
from .registry import Registry

BLANK_W, BLANK_H = 8, 4
MAX_DIM = 120.0
ROOF_MIN_DEG, ROOF_MAX_DEG = 25.0, 60.0
_HANGING = ("chain", "vine", "lichen", "hanging", "wall_", "ladder", "cave_vines", "weeping", "roots", "bell", "end_rod", "button", "lever", "tripwire", "bars", "pane", "glass", "scaffolding", "cobweb", "amethyst", "dripstone", "spore", "lightning_rod", "water", "lava", "air")


def lint_scene_only(scene: Any) -> List[LintFinding]:
    """Cheap checks that need no raster: R7 (scale) and R8 (unknown materials)."""
    out: List[LintFinding] = []
    try:
        from .materials import PRESETS
    except Exception:  # noqa: BLE001
        PRESETS = {}
    known = set(getattr(scene, "materials", {}) or {}) | set(PRESETS)
    for o in getattr(scene, "objects", []):
        try:
            size = scene.object_bbox(o).size
        except Exception:  # noqa: BLE001
            continue
        if float(np.max(size)) > MAX_DIM:
            out.append(LintFinding("R7", "error", f"{o.id} spans {float(np.max(size)):.0f} blocks on one axis (> {MAX_DIM:.0f}); probably a units mistake", [o.id]))
        if o.op in ("add", "paint"):
            mn = o.material_name()
            if mn and mn not in known and not mn.startswith("inline") and o.shape.get("type") != "block":
                try:
                    from .registry import default_registry

                    if default_registry().has(mn):
                        continue
                except Exception:  # noqa: BLE001
                    pass
                out.append(LintFinding("R8", "warn", f"material {mn!r} on {o.id} is not defined (define_material or use a preset); resolver falls back to stone", [o.id]))
    return out


def lint(scene: Any, raster: Optional[RasterResult], fit: Optional[FitResult], block_map: Optional[BlockMap],
         registry: Optional[Registry] = None) -> List[LintFinding]:
    """Run every design rule. `raster`, `fit` and `block_map` may be None (then only scene rules run).

    Object scenes (brief kind "object": statues, creatures, vehicles, props) skip the architecture rules —
    R1 blank façade, R2 roofs, R3 entrance and R9's ground gradient — which would only feed the critic
    irrelevant findings; floating/isolated blocks, props on air, scale and unknown materials still apply."""
    building = not is_object_scene(scene)
    findings = lint_scene_only(scene) + _r9_materials(scene, gradient=building)
    if raster is None or fit is None:
        return findings
    kind = np.asarray(fit.kind)
    occ = kind > 0
    if not occ.any():
        return findings
    if building:
        findings += _r1_blank_facades(scene, raster, kind)
        findings += _r2_roofs(scene)
        findings += _r3_entrance(scene, raster, kind)
    if block_map is not None:
        findings += _r4_variety(block_map)
    findings += _r5_floating(scene, raster, kind)
    findings += _r6_props(raster, kind, registry)
    return findings


# ----------------------------------------------------------------------------------------------
def _owner_ids(scene: Any, raster: RasterResult, idx: np.ndarray) -> List[str]:
    owners = np.asarray(raster.owner)[idx[:, 0], idx[:, 1], idx[:, 2]]
    ids: List[str] = []
    objs = getattr(scene, "objects", [])
    for oi in np.unique(owners):
        if 0 <= int(oi) < len(objs):
            ids.append(objs[int(oi)].id)
    return ids


def _components(mask: np.ndarray) -> List[np.ndarray]:
    """4-connected components of a 2D boolean mask -> list of index arrays [k,2]. BFS; masks are small."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        q = deque([(y0, x0)])
        seen[y0, x0] = True
        cells = []
        while q:
            y, x = q.popleft()
            cells.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        comps.append(np.asarray(cells))
    return comps


def _outside_air(kind: np.ndarray, max_iter: int = 96) -> np.ndarray:
    """Air voxels reachable from the grid boundary (6-connected flood fill, iterative dilation)."""
    air = kind == 0
    out = np.zeros_like(air)
    out[0, :, :] = air[0, :, :]
    out[-1, :, :] = air[-1, :, :]
    out[:, 0, :] = air[:, 0, :]
    out[:, -1, :] = air[:, -1, :]
    out[:, :, 0] = air[:, :, 0]
    out[:, :, -1] = air[:, :, -1]
    for _ in range(max_iter):
        g = out.copy()
        g[1:, :, :] |= out[:-1, :, :]
        g[:-1, :, :] |= out[1:, :, :]
        g[:, 1:, :] |= out[:, :-1, :]
        g[:, :-1, :] |= out[:, 1:, :]
        g[:, :, 1:] |= out[:, :, :-1]
        g[:, :, :-1] |= out[:, :, 1:]
        g &= air
        if np.array_equal(g, out):
            break
        out = g
    return out


def _max_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int, int]:
    """Largest all-True axis-aligned rectangle in a 2D mask -> (area, r0, c0, h, w). Histogram method."""
    H, W = mask.shape
    best = (0, 0, 0, 0, 0)
    heights = np.zeros(W, dtype=np.int64)
    for r in range(H):
        heights = np.where(mask[r], heights + 1, 0)
        stack: List[int] = []
        for c in range(W + 1):
            h = heights[c] if c < W else 0
            start = c
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                th = int(heights[top])
                start = stack[-1] + 1 if stack else 0
                w = c - start
                area = th * w
                if area > best[0]:
                    best = (area, r - th + 1, start, th, w)
            stack.append(c)
    return best


def _r1_blank_facades(scene: Any, raster: RasterResult, kind: np.ndarray) -> List[LintFinding]:
    """Exposed vertical face regions containing a solid >= 8 wide x 4 tall rectangle of full blocks of
    one material in one depth plane (no windows, pilasters or steps inside it)."""
    out: List[LintFinding] = []
    full = kind == int(Kind.FULL)
    mat = np.asarray(raster.material)
    X, Y, Z = kind.shape
    air = _outside_air(kind)  # only exterior façades count; interior rooms are not façades
    ox, oy, oz = raster.origin
    dirs = {"east": (1, 0, 0), "west": (-1, 0, 0), "south": (0, 0, 1), "north": (0, 0, -1)}
    reported = 0
    for dname, (dx, dy, dz) in dirs.items():
        nb_air = np.ones_like(air)
        if dx == 1:
            nb_air[:-1, :, :] = air[1:, :, :]
        elif dx == -1:
            nb_air[1:, :, :] = air[:-1, :, :]
        elif dz == 1:
            nb_air[:, :, :-1] = air[:, :, 1:]
        else:
            nb_air[:, :, 1:] = air[:, :, :-1]
        exposed = full & nb_air
        planes = range(X) if dx else range(Z)
        for p in planes:
            face = exposed[p, :, :] if dx else exposed[:, :, p]
            pm = mat[p, :, :] if dx else mat[:, :, p]
            mask2d = face if dx else face.T  # rows = y, cols = z (or x)
            mat2d = pm if dx else pm.T
            if mask2d.sum() < BLANK_W * BLANK_H:
                continue
            for m in np.unique(mat2d[mask2d]):
                mm = mask2d & (mat2d == m)
                while True:
                    area, r0, c0, h, w = _max_rectangle(mm)
                    if h < BLANK_H or w < BLANK_W:
                        break
                    mname = raster.material_names[int(m)] if int(m) < len(raster.material_names) else "?"
                    rows = np.arange(r0, r0 + h)
                    cols = np.arange(c0, c0 + w)
                    rr, cc = np.meshgrid(rows, cols, indexing="ij")
                    if dx:
                        vox = np.stack([np.full(rr.size, p), rr.ravel(), cc.ravel()], axis=1)
                        where = f"x={p + ox}, y {r0 + oy}..{r0 + h - 1 + oy}, z {c0 + oz}..{c0 + w - 1 + oz}"
                    else:
                        vox = np.stack([cc.ravel(), rr.ravel(), np.full(rr.size, p)], axis=1)
                        where = f"z={p + oz}, y {r0 + oy}..{r0 + h - 1 + oy}, x {c0 + ox}..{c0 + w - 1 + ox}"
                    ids = _owner_ids(scene, raster, vox)
                    out.append(LintFinding("R1", "warn", f"blank {dname}-facing façade {w}x{h} of {mname} at {where}: add depth (recessed windows, pilasters, a string course) or material variation", ids))
                    reported += 1
                    if reported >= 8:
                        return out
                    mm[r0:r0 + h, c0:c0 + w] = False
    return out


def _r2_roofs(scene: Any) -> List[LintFinding]:
    out: List[LintFinding] = []
    objs = [o for o in getattr(scene, "objects", []) if o.op == "add" and o.visible]
    for o in objs:
        is_roof = "roof" in o.id or any("roof" in t for t in o.tags) or o.shape.get("type") in ("cone", "pyramid", "wedge")
        if not is_roof:
            continue
        bb = scene.object_bbox(o)
        # slope
        t = o.shape.get("type")
        angles: List[float] = []
        if t == "cone":
            angles.append(math.degrees(math.atan2(o.shape["height"], max(o.shape["radius"] - o.shape.get("radius_top", 0), 1e-6))))
        elif t == "pyramid":
            bx, bz = o.shape["base"]
            tx, tz = o.shape.get("top", [0, 0])
            angles.append(math.degrees(math.atan2(o.shape["height"], max((bx - tx) / 2, 1e-6))))
            angles.append(math.degrees(math.atan2(o.shape["height"], max((bz - tz) / 2, 1e-6))))
        elif t == "wedge":
            sx, sy, sz = o.shape["size"]
            run = sx if o.shape.get("slope_axis", "x").lstrip("-") == "x" else sz
            angles.append(math.degrees(math.atan2(sy, max(run, 1e-6))))
        for a in angles:
            if a < ROOF_MIN_DEG or a > ROOF_MAX_DEG:
                out.append(LintFinding("R2", "warn", f"roof {o.id} slope is {a:.0f}° (aim for {ROOF_MIN_DEG:.0f}–{ROOF_MAX_DEG:.0f}°): change height or footprint", [o.id]))
                break
        # overhang: find the supporting object directly beneath
        below = None
        for s in objs:
            if s is o or s.shape.get("type") == "block":
                continue
            sb = scene.object_bbox(s)
            if abs(sb.hi[1] - bb.lo[1]) <= 1.0 and sb.lo[0] < bb.hi[0] and sb.hi[0] > bb.lo[0] and sb.lo[2] < bb.hi[2] and sb.hi[2] > bb.lo[2]:
                if below is None or (sb.size[0] * sb.size[2]) > (scene.object_bbox(below).size[0] * scene.object_bbox(below).size[2]):
                    below = s
        if below is not None:
            sb = scene.object_bbox(below)
            overhang = min(sb.lo[0] - bb.lo[0], bb.hi[0] - sb.hi[0], sb.lo[2] - bb.lo[2], bb.hi[2] - sb.hi[2])
            if overhang < 1.0:
                out.append(LintFinding("R2", "warn", f"roof {o.id} has no overhang over {below.id} (min overhang {overhang:.1f}); widen the roof by ≥1 block per side", [o.id, below.id]))
    return out


def _r3_entrance(scene: Any, raster: RasterResult, kind: np.ndarray) -> List[LintFinding]:
    """Warn when a building-sized scene has no ground-level doorway (2-tall pass-through) or door."""
    objs = [o for o in getattr(scene, "objects", []) if o.op == "add" and o.visible and o.shape.get("type") != "block"]
    if not any(scene.object_bbox(o).size[1] >= 4 for o in objs):
        return []
    occ = kind > 0
    if occ.sum() < 200:
        return []
    for state in (raster.props or {}).values():
        if "_door" in state or "fence_gate" in state:
            return []
    X, Y, Z = kind.shape
    ys = np.nonzero(occ.any(axis=(0, 2)))[0]
    if len(ys) == 0:
        return []
    y0 = int(ys.min())
    if y0 + 1 >= Y:
        return []
    air = ~occ
    # pad one voxel of air around the grid so doorways on the boundary count
    a0 = np.pad(air[:, y0, :], 1, constant_values=True)
    a1 = np.pad(air[:, y0 + 1, :], 1, constant_values=True)
    s0 = np.pad(occ[:, y0, :], 1, constant_values=False)
    both = a0 & a1
    # doorway: 2-tall air column with solid on both sides along one axis and air on both sides along the other
    cand = np.zeros_like(both)
    cand[1:-1, 1:-1] |= both[1:-1, 1:-1] & s0[:-2, 1:-1] & s0[2:, 1:-1] & a0[1:-1, :-2] & a0[1:-1, 2:]
    cand[1:-1, 1:-1] |= both[1:-1, 1:-1] & s0[1:-1, :-2] & s0[1:-1, 2:] & a0[:-2, 1:-1] & a0[2:, 1:-1]
    if cand.any():
        return []
    return [LintFinding("R3", "warn", "no entrance at ground level: carve a 2-tall doorway (subtract a box) or place a door", [])]


MIN_MATERIALS = 3


def is_object_scene(scene: Any) -> bool:
    """True when the scene's brief (scene.meta["brief"]) says kind == "object"."""
    meta = getattr(scene, "meta", None) or {}
    brief = meta.get("brief") if isinstance(meta, dict) else None
    return isinstance(brief, dict) and brief.get("kind") == "object"


def _r9_materials(scene: Any, gradient: bool = True) -> List[LintFinding]:
    """Materials-stage rules (T4, P4/P5). Runs once the scene defines at least one material — i.e. the
    materials stage has begun — so blocking's placeholders do not trigger it."""
    mats = getattr(scene, "materials", {}) or {}
    if not mats:
        return []
    try:
        from .materials import PRESETS
    except Exception:  # noqa: BLE001
        PRESETS = {}
    volumes: dict = {}
    grounded: dict = {}  # material → volume of objects standing on the ground (the walls, not the roofs)
    solids = []
    for o in getattr(scene, "objects", []):
        if o.op != "add" or o.shape.get("type") == "block":
            continue
        name = o.material_name()
        if not name or name == "default":
            continue
        try:
            bb = scene.object_bbox(o)
            size = bb.size
            vol, lo_y = float(size[0] * size[1] * size[2]), float(bb.lo[1])
        except Exception:  # noqa: BLE001
            vol, lo_y = 1.0, 0.0
        solids.append((name, vol, lo_y))
        volumes[name] = volumes.get(name, 0.0) + vol
    if not volumes:
        return []
    floor = min(lo for _, _, lo in solids)
    for name, vol, lo_y in solids:
        if lo_y <= floor + 1.0:
            grounded[name] = grounded.get(name, 0.0) + vol
    out: List[LintFinding] = []
    if len(volumes) < MIN_MATERIALS:
        out.append(LintFinding("R9", "warn", f"only {len(volumes)} distinct material(s) on solids ({', '.join(sorted(volumes))}): wall, roof and trim must be three contrasting materials (P5)", []))
    pool = grounded or volumes
    main = max(pool, key=lambda k: pool[k])  # the wall material: biggest grounded volume
    spec = mats.get(main) or PRESETS.get(main) or {}
    if isinstance(spec, dict) and "preset" in spec and not spec.get("gradient"):
        spec = {**PRESETS.get(str(spec.get("preset")), {}), **spec}
    if gradient and isinstance(spec, dict) and not spec.get("gradient"):
        ids = [o.id for o in scene.objects if o.op == "add" and o.material_name() == main][:6]
        out.append(LintFinding("R9", "warn", f"main wall material {main!r} has no ground gradient (P4): define it with \"gradient\": {{\"axis\": \"y\", \"from\": 0, \"to\": 3, \"palette\": [[darker/rougher block, 1]]}}", ids))
    return out


def _r4_variety(block_map: BlockMap) -> List[LintFinding]:
    if len(block_map) <= 1000:
        return []
    ids = {parse_state(s)[0] for s in block_map.values()}
    if len(ids) < 3:
        return [LintFinding("R4", "warn", f"only {len(ids)} distinct block(s) in a {len(block_map)}-block build: use palettes, gradients and a contrasting trim/roof", [])]
    return []


def _r5_floating(scene: Any, raster: RasterResult, kind: np.ndarray) -> List[LintFinding]:
    out: List[LintFinding] = []
    occ = kind > 0
    X, Y, Z = occ.shape
    # isolated singles: no occupied 6-neighbour
    nb = np.zeros_like(occ)
    nb[1:, :, :] |= occ[:-1, :, :]
    nb[:-1, :, :] |= occ[1:, :, :]
    nb[:, 1:, :] |= occ[:, :-1, :]
    nb[:, :-1, :] |= occ[:, 1:, :]
    nb[:, :, 1:] |= occ[:, :, :-1]
    nb[:, :, :-1] |= occ[:, :, 1:]
    prop_idx: Set[Tuple[int, int, int]] = set()
    for (x, y, z) in (raster.props or {}):
        prop_idx.add(raster.world_to_index(x, y, z))
    iso = occ & ~nb
    if iso.any():
        pts = np.argwhere(iso)
        pts = np.asarray([p for p in pts if tuple(p) not in prop_idx]) if prop_idx else pts
        if len(pts):
            ox, oy, oz = raster.origin
            sample = ", ".join(f"({p[0] + ox},{p[1] + oy},{p[2] + oz})" for p in pts[:5])
            out.append(LintFinding("R5", "warn", f"{len(pts)} isolated single block(s), e.g. {sample}: fitting noise or a stray prop", _owner_ids(scene, raster, pts)))
    # floating components: occupied voxels not connected to the lowest occupied layer
    ys = np.nonzero(occ.any(axis=(0, 2)))[0]
    if len(ys) == 0:
        return out
    y0 = int(ys.min())
    grounded = np.zeros_like(occ)
    grounded[:, y0, :] = occ[:, y0, :]
    for _ in range(X + Y + Z):
        g = grounded.copy()
        g[1:, :, :] |= grounded[:-1, :, :]
        g[:-1, :, :] |= grounded[1:, :, :]
        g[:, 1:, :] |= grounded[:, :-1, :]
        g[:, :-1, :] |= grounded[:, 1:, :]
        g[:, :, 1:] |= grounded[:, :, :-1]
        g[:, :, :-1] |= grounded[:, :, 1:]
        g &= occ
        if np.array_equal(g, grounded):
            break
        grounded = g
    floating = occ & ~grounded & ~iso
    n = int(floating.sum())
    if n:
        pts = np.argwhere(floating)
        ox, oy, oz = raster.origin
        lo = pts.min(axis=0) + np.asarray(raster.origin)
        hi = pts.max(axis=0) + np.asarray(raster.origin)
        out.append(LintFinding("R5", "warn", f"{n} floating block(s) not connected to the ground, bbox ({lo[0]},{lo[1]},{lo[2]})..({hi[0]},{hi[1]},{hi[2]}): add supports or lower them", _owner_ids(scene, raster, pts)))
    return out


def _r6_props(raster: RasterResult, kind: np.ndarray, registry: Optional[Registry]) -> List[LintFinding]:
    out: List[LintFinding] = []
    bad: List[str] = []
    for (x, y, z), state in (raster.props or {}).items():
        bid, props = parse_state(state)
        s = short_id(bid)
        if props.get("hanging") == "true" or any(h in s for h in _HANGING):
            continue
        i, j, k = raster.world_to_index(x, y - 1, z)
        X, Y, Z = kind.shape
        below_solid = (0 <= i < X and 0 <= j < Y and 0 <= k < Z and kind[i, j, k] > 0) or ((x, y - 1, z) in raster.props)
        if not below_solid:
            bad.append(f"{s}@({x},{y},{z})")
    if bad:
        out.append(LintFinding("R6", "warn", f"{len(bad)} prop(s) resting on air: {', '.join(bad[:6])}{' …' if len(bad) > 6 else ''}", []))
    return out


def findings_text(findings: Sequence[LintFinding], limit: int = 12) -> str:
    if not findings:
        return "lint: no findings"
    lines = [str(f) for f in findings[:limit]]
    if len(findings) > limit:
        lines.append(f"... {len(findings) - limit} more")
    return "\n".join(lines)
