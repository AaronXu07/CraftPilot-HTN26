# Stage 0 — Interpret

Turn the player's request into a design brief. Do not build anything in this stage.

## Goal
Produce a brief JSON that a builder can execute without asking questions. Be decisive: choose a
style, a footprint and heights, and write an explicit **silhouette plan** — the massing in one
paragraph, with numbers. Fill gaps with the strongest conventional choice for the build type.

## Output — call `set_brief(brief)` with exactly this shape
```json
{
  "name": "castle_v1",
  "build_type": "castle",
  "style": "medieval stone",
  "footprint": [40, 32],
  "height": 26,
  "facing": "south",
  "key_features": ["L-shaped keep", "four round corner towers", "gatehouse on the south side"],
  "constraints": ["flat ground", "entrance faces the player (+z)"],
  "materials_intent": "stone bricks with cobblestone base, dark oak roofs, spruce trim",
  "silhouette_plan": "L-shaped keep 20x14, 18 tall, at the north-west; four round towers r=4 h=24 at the corners of a 40x32 curtain wall h=10 t=2; gatehouse 8x6x12 centred on the south wall with a 3x4 arched gate (the entrance); cone roofs r=5 h=6 on the towers (overhang 1); hip roof on the keep, overhang 1, 40° pitch.",
  "stages": ["blocking", "detailing", "materials", "decoration"]
}
```
- `footprint` is [x-extent, z-extent] in blocks; `height` is the tallest point.
- Keep footprints ≤ 60×60 and heights ≤ 60 unless the player asks for more.
- `facing` is where the entrance faces; default "south" (toward the player).
- `key_features` must include every feature the player named, verbatim in spirit.
- `silhouette_plan` must mention every primary mass with its shape, size and position relative to the
  others, plus roof shapes. Numbers, not adjectives. It MUST state: (a) the entrance — where the
  door/gate is on the `facing` side and its size; (b) each roof's overhang (≥ 1) or "flat roof" with a
  parapet; (c) for every tower/turret/spire a height larger than its width (`r=4 h=24`, `6x6 h=20`).
  A brief that breaks (a)–(c) is rejected and you will be asked to rewrite it — get it right first.
- Choose `stages`: all four for a new build; drop `decoration` for tiny builds (< 300 blocks).

## Checklist
- [ ] Every named feature appears in key_features and silhouette_plan.
- [ ] Proportions obey P1 (every tower taller than wide; roof pitch 30–45°, overhang ≥ 1 stated).
- [ ] The entrance (door/gate/portico) is stated with its side and size (P6).
- [ ] The footprint fits the request ("small" ≈ 10–16, "large" ≈ 40–60).
- [ ] The style implies a concrete material intent (stone/timber/sandstone/quartz/…).

## Examples of good silhouette plans
- Lighthouse: "Tapered cylinder r=4→3 h=30 on a 12x12x3 stone plinth; octagonal lantern room r=3.5
  h=4 on top with a shallow cone cap (overhang 1); keeper's cottage 9x6x5 with a gable roof (overhang 1)
  attached on the west; entrance: 2x3 door in the cottage's south wall."
- Japanese pagoda: "Five square tiers, 14→8 wide, each 5 tall, each with a hip roof pyramid 2 wider
  than its tier (overhang 1 each side) and 2.5 tall, up-curved eaves implied by a 1-block slab lip;
  central spire r=1 h=6; entrance: 2x3 door centred on the south face of the bottom tier, up 3 plinth steps."
