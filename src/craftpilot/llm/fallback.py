"""Offline fallback: nearest exemplar, patched by simple regexes."""

from __future__ import annotations

import copy
import re

from craftpilot.blocks import catalog
from craftpilot.program.exemplars import Exemplar, retrieve
from craftpilot.program.model import Bounds, BuildProgram, FamilyWeight, RolePalette

_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}

# Word -> family for the primary wall material.
_MATERIALS = {
    "oak": "oak", "spruce": "spruce", "birch": "birch", "dark oak": "dark_oak", "jungle": "jungle",
    "acacia": "acacia", "cherry": "cherry", "mangrove": "mangrove", "bamboo": "bamboo",
    "cobblestone": "cobblestone", "stone brick": "stone_bricks", "stone": "stone_bricks", "brick": "bricks",
    "sandstone": "sandstone", "quartz": "quartz", "deepslate": "deepslate_bricks", "blackstone": "blackstone",
    "concrete": "white_concrete", "mud": "mud_bricks", "prismarine": "prismarine", "nether brick": "nether_bricks",
}


def _number(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    if not m:
        return None
    tok = m.group(1)
    return int(tok) if tok.isdigit() else _NUM.get(tok)


def fallback_program(text: str, exemplars: list[Exemplar]) -> tuple[BuildProgram, list[str]]:
    notes: list[str] = []
    if not exemplars:
        raise RuntimeError("No exemplars available for the fallback parser.")
    best = retrieve(exemplars, text, 1)[0]
    program = copy.deepcopy(best.program)
    program.label = text.strip()[:60] or best.program.label
    notes.append(f"Composed offline from the '{best.name}' exemplar.")
    low = text.lower()

    floors = _number(low, r"(\d+|one|two|three|four|five|six|seven|eight)[\s-]*(?:stor(?:e)?y|stories|storeys|floors?)")
    if floors:
        program.root().floors = max(1, min(12, floors))
    m = re.search(r"(\d+)\s*(?:x|by)\s*(\d+)(?:\s*(?:x|by)\s*(\d+))?", low)
    if m:
        w, second = int(m.group(1)), int(m.group(2))
        if m.group(3):
            program.bounds = Bounds(width=w, height=second, depth=int(m.group(3)))
        else:
            program.bounds = Bounds(width=w, height=program.bounds.height, depth=second)
    from craftpilot.program.palette import library_match, library_palette
    m2 = library_match(text)
    if m2 is not None and m2[0] not in best.description.lower():
        lib = library_palette(m2[0])
        if lib is not None:
            program.palette = lib
            program.palette_name = m2[0]
            notes.append(f"Used the '{m2[0]}' palette.")
    for word, fam in sorted(_MATERIALS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", low) and catalog.family(fam):
            program.palette.primary = RolePalette(families=[FamilyWeight(family=fam, weight=1.0)],
                                                  texture_rate=0.2, weathering=0.3)
            break
    if "tower" in low and not any(a.kind.value == "tower" for a in program.attachments):
        from craftpilot.program.model import AttachmentKind, AttachmentRequest
        n = _number(low, r"(\d+|one|two|three|four|five|six|seven|eight)\s+towers?")
        program.attachments.append(AttachmentRequest(kind=AttachmentKind.tower, count=n))
    return program, notes
