"""A BuildProgram as a dozen chat lines, so the player can check the plan before it is built.

Deterministic: no LLM, only the program. Kept short because it is printed in the game's chat.
"""

from __future__ import annotations

from craftpilot.program.model import Bounds, BuildProgram, PartSpec, RolePalette

_ROLE_LABELS = (("primary", "walls"), ("roof", "roof"), ("secondary", "secondary"), ("framing", "framing"),
                ("trim", "trim"), ("foundation", "base"), ("accent", "accent"))


def _families(role: RolePalette | None) -> str:
    if role is None or not role.families:
        return ""
    fams = sorted(role.families, key=lambda f: -f.weight)
    names = [f.family for f in fams[:2]]
    text = "/".join(names)
    if role.gradient.value != "none":
        text += f" ({role.gradient.value} gradient)"
    return text


def _part_line(p: PartSpec, root: PartSpec) -> str:
    bits = []
    if p is root:
        bits.append(p.shape.value if p.shape.value != "rect" else "main block")
    else:
        a = p.attach
        where = f"stacked on {a.to}" if a.side.value == "top" else f"attached {a.side.value} of {a.to}"
        bits.append(f"{p.shape.value} {where}" if p.shape.value != "rect" else where)
        if p.size.width != 1.0 or p.size.depth != 1.0:
            bits.append(f"{p.size.width:.0%} x {p.size.depth:.0%} of it")
    bits.append(f"{p.floors} floor{'s' if p.floors != 1 else ''} x {p.floor_height}")
    if p.taper:
        bits.append(f"tapering {p.taper:.0%}/floor")
    if p.wall_thickness > 1:
        bits.append("thick walls")
    roof = p.roof.type.value + (" roof" if p.roof.type.value != "none" else "")
    if p.roof.type.value in ("gable", "hip", "shed", "gambrel", "mansard") and p.roof.pitch != 1.0:
        roof += f" pitch {p.roof.pitch:g}"
    if p.roof.type.value == "parapet" and p.roof.crenellated:
        roof += " crenellated"
    if p.roof.type.value == "pagoda" and p.roof.tiers > 1:
        roof += f" x{p.roof.tiers}"
    bits.append(roof)
    if p.role_hint:
        bits.append(f"({p.role_hint})")
    return f"{p.name}: " + ", ".join(bits)


def describe(program: BuildProgram, bounds: Bounds | None = None) -> list[str]:
    """Lines for chat: size, every part, facade, materials, attachments, and what could not be expressed."""
    b = bounds or program.bounds
    root = program.root()
    lines = [f"{program.label} - {b.width} wide x {b.height} tall x {b.depth} deep"]
    for p in program.parts:
        lines.append(_part_line(p, root))
    f = program.facade
    facade = [f"{f.window.value} windows {f.window_width}x{f.window_height}",
              f"{f.framing.value} framing" if f.framing.value != "none" else "no framing",
              f"{f.entrance} entrance"]
    if f.shutters:
        facade.append("shutters")
    if f.ground_floor_taller:
        facade.append(f"taller ground floor +{f.ground_floor_taller}")
    lines.append("facade: " + ", ".join(facade))
    mats = [f"{label} {_families(getattr(program.palette, attr))}" for attr, label in _ROLE_LABELS
            if _families(getattr(program.palette, attr))]
    if program.palette_name:
        mats.insert(0, f"'{program.palette_name}' palette")
    lines.append("materials: " + ", ".join(mats))
    if program.attachments:
        att = []
        for a in program.attachments:
            n = f"{a.count} " if a.count else ""
            kind = a.kind.value
            if a.count and a.count > 1:
                kind += "es" if kind.endswith("s") else "s"
            on = f" on {a.on}" if a.on else ""
            att.append(f"{n}{kind}{on}")
        lines.append("attachments: " + ", ".join(att))
    else:
        lines.append("attachments: none asked for (the engine adds a few on its own)")
    interior = [k for k, v in (("stairs", program.interior.stairs), ("doorways", program.interior.doorways),
                               ("partitions", program.interior.partitions), ("lighting", program.interior.lighting)) if v]
    lines.append("interior: " + (", ".join(interior) if interior else "empty shell"))
    if program.design.strip():
        lines.append("design: " + " ".join(program.design.split())[:200])
    if program.material_reasoning.strip():
        lines.append("why these materials: " + " ".join(program.material_reasoning.split())[:160])
    if program.notes.strip():
        lines.append("could not express: " + " ".join(program.notes.split())[:160])
    return lines
