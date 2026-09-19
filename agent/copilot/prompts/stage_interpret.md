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
- `kind`: **"building"** for architecture (anything with walls/roof/entrance) or **"object"** for a
  statue, creature, animal, vehicle, weapon, prop or sculpture. Objects are built differently, so get
  this right.

## Objects (kind = "object") — the brief is a PARTS LIST, not a floor plan
Rules (a)–(c) above do not apply. Instead `silhouette_plan` must list every part with its primitive,
size and position, the long axis, and which way the front/head faces (default: +z, toward the player).
Use ellipsoids for bodies/heads, lines (capsules between two points) for necks/tails/limbs, cylinders
(axis x) for wheels, thin wedges/boxes for wings/fins, and a plinth for statues. `stages` defaults to
["blocking", "detailing", "materials"] (no decoration). Example plan for "a dragon statue":
"Plinth box 14x3x22 at origin. Body ellipsoid radii [3,3,7] centred (0,9,0), long axis z, head faces
+z. Neck line (0,10,6)→(0,15,11) t=3; head ellipsoid [2,2,3.5] centred (0,15.5,13); two horns cone
r=0.7 h=3 on the head. Tail two lines (0,9,-6)→(0,8,-12) t=2.4 and →(0,10,-17) t=1.8. Four legs lines
t=2 from the belly (±2,8,±4) down to the plinth top y=3. Two wings: wedges 10x1x6 from the shoulders
(±2,11,1) rotated [0,0,±35] so they sweep up and out; wingspan 24." A car: "Body ellipsoid radii
[3,1.6,8] centred (0,2.6,0) plus cabin ellipsoid [2.4,1.4,3.5] centred (0,4,−0.5); nose faces +z; four
wheels cylinder axis x r=1.5 h=1 at (±3,1.5,±5); spoiler box 7x0.5x1.5 at (0,4.5,−7.5) on two short lines."


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
