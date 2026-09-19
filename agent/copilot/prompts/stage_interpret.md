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
  "silhouette_plan": "L-shaped keep 20x14, 18 tall, at the north-west; four round towers r=4 h=24 at the corners of a 40x32 curtain wall h=10 t=2; gatehouse 8x6x12 centred on the south wall with a 3x4 arched gate; cone roofs on the towers; hip roof on the keep.",
  "stages": ["blocking", "detailing", "materials", "decoration"]
}
```
- `footprint` is [x-extent, z-extent] in blocks; `height` is the tallest point.
- Keep footprints ≤ 60×60 and heights ≤ 60 unless the player asks for more.
- `facing` is where the entrance faces; default "south" (toward the player).
- `key_features` must include every feature the player named, verbatim in spirit.
- `silhouette_plan` must mention every primary mass with its shape, size and position relative to the
  others, plus roof shapes. Numbers, not adjectives.
- Choose `stages`: all four for a new build; drop `decoration` for tiny builds (< 300 blocks).

## Checklist
- [ ] Every named feature appears in key_features and silhouette_plan.
- [ ] Proportions obey P1 (towers taller than wide; roof pitch 30–45°).
- [ ] The footprint fits the request ("small" ≈ 10–16, "large" ≈ 40–60).
- [ ] The style implies a concrete material intent (stone/timber/sandstone/quartz/…).

## Examples of good silhouette plans
- Lighthouse: "Tapered cylinder r=4→3 h=30 on a 12x12x3 stone plinth; octagonal lantern room r=3.5
  h=4 on top with a shallow cone cap; keeper's cottage 9x6x5 with a gable roof attached on the west."
- Japanese pagoda: "Five square tiers, 14→8 wide, each 5 tall, each with a hip roof pyramid 2 wider
  than its tier and 2.5 tall, up-curved eaves implied by a 1-block slab lip; central spire 6 tall."
