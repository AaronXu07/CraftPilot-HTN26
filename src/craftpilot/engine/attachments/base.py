"""Attachment protocol, registry, and the placement loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from craftpilot.grid.semantic import SemanticGrid
from craftpilot.program.model import AttachmentKind, AttachmentRequest, BuildProgram

Level = Literal["mass", "roof", "facade"]
Placer = Callable[[SemanticGrid, BuildProgram, AttachmentRequest], int]

_REGISTRY: dict[AttachmentKind, tuple[Level, Placer]] = {}


def register(kind: AttachmentKind, level: Level):
    def deco(fn: Placer) -> Placer:
        _REGISTRY[kind] = (level, fn)
        return fn
    return deco


def run_level(grid: SemanticGrid, program: BuildProgram, level: Level) -> None:
    placed: dict[str, int] = {}
    for req in program.attachments:
        entry = _REGISTRY.get(req.kind)
        if entry is None:
            if level == "roof":
                grid.note(f"Attachment '{req.kind.value}' is not implemented yet and was skipped.")
            continue
        lvl, fn = entry
        if lvl != level:
            continue
        try:
            n = fn(grid, program, req)
        except Exception as exc:  # closure: an attachment must never break the build
            grid.note(f"Attachment '{req.kind.value}' failed and was skipped ({exc}).")
            n = 0
        placed[req.kind.value] = placed.get(req.kind.value, 0) + n
    grid.stage_done(f"attachments:{level}", placed=placed)


def reserve(grid: SemanticGrid, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> None:
    x0, x1 = max(0, x0), min(grid.W - 1, x1)
    y0, y1 = max(0, y0), min(grid.H - 1, y1)
    z0, z1 = max(0, z0), min(grid.D - 1, z1)
    if x1 >= x0 and y1 >= y0 and z1 >= z0:
        grid.reserved[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True


def is_reserved(grid: SemanticGrid, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> bool:
    x0, x1 = max(0, x0), min(grid.W - 1, x1)
    y0, y1 = max(0, y0), min(grid.H - 1, y1)
    z0, z1 = max(0, z0), min(grid.D - 1, z1)
    if x1 < x0 or y1 < y0 or z1 < z0:
        return True
    return bool(grid.reserved[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1].any())


def target_part(grid: SemanticGrid, program: BuildProgram, req: AttachmentRequest):
    if req.on:
        p = grid.part_by_name(req.on)
        if p is not None:
            return p
    root = program.root().name
    return grid.part_by_name(root) or grid.parts[0]


def spread(n: int, lo: int, hi: int, rng: np.random.Generator, regular: bool, min_gap: int) -> list[int]:
    """n positions in [lo, hi], evenly spaced (regular) or jittered, at least min_gap apart."""
    if hi < lo or n <= 0:
        return []
    span = hi - lo
    if n == 1:
        return [(lo + hi) // 2]
    if (n - 1) * min_gap > span:
        n = max(1, span // min_gap + 1)
        if n == 1:
            return [(lo + hi) // 2]
    step = span / (n - 1)
    pos = [int(round(lo + i * step)) for i in range(n)]
    if not regular and step > min_gap + 1:
        jit = int((step - min_gap) // 2)
        pos = [p + int(rng.integers(-jit, jit + 1)) if 0 < i < n - 1 else p for i, p in enumerate(pos)]
    return sorted(set(max(lo, min(hi, p)) for p in pos))


DOMINANT = {AttachmentKind.tower, AttachmentKind.chimney, AttachmentKind.cupola, AttachmentKind.spire, AttachmentKind.flagpole}
MEDIUM = {AttachmentKind.dormer, AttachmentKind.balcony, AttachmentKind.porch, AttachmentKind.bartizan,
          AttachmentKind.gallery, AttachmentKind.jetty}
SMALL = {AttachmentKind.awning, AttachmentKind.buttress, AttachmentKind.arch, AttachmentKind.colonnade}
SLOPED = {"gable", "hip", "gambrel", "mansard"}


def auto_budget(grid: SemanticGrid, program: BuildProgram) -> list[str]:
    """Spend the silhouette budget on classes the program left empty. Mutates program.attachments."""
    kinds = {a.kind for a in program.attachments}
    root = program.root()
    roof = root.roof.type.value
    added: list[str] = []
    b = program.budget
    if b.dominant >= 1 and not (kinds & DOMINANT):
        if roof in SLOPED:
            program.attachments.append(AttachmentRequest(kind=AttachmentKind.chimney, count=1))
            added.append("chimney")
        elif roof in ("flat", "parapet"):
            program.attachments.append(AttachmentRequest(kind=AttachmentKind.flagpole, count=1))
            added.append("flagpole")
    if b.medium_min >= 1 and not (kinds & MEDIUM):
        rp = grid.part_by_name(root.name)
        wide = rp is not None and min(rp.width, rp.depth) >= 11
        if roof in SLOPED and wide:
            program.attachments.append(AttachmentRequest(kind=AttachmentKind.dormer, count=min(b.medium_min, 3)))
            added.append("dormer")
        elif roof == "parapet" and root.floors >= 2:
            program.attachments.append(AttachmentRequest(kind=AttachmentKind.bartizan, count=2))
            added.append("bartizan")
    if b.small_min >= 1 and not (kinds & SMALL):
        if roof in SLOPED:
            program.attachments.append(AttachmentRequest(kind=AttachmentKind.awning))
            added.append("awning")
        else:
            program.attachments.append(AttachmentRequest(kind=AttachmentKind.buttress))
            added.append("buttress")
    if added:
        grid.note("Silhouette budget added: " + ", ".join(added) + ".")
    return added
