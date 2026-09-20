# Stage 3 — Materials

## Goal
Replace placeholders with rules: palettes with weathering variation, a grounding gradient, per-face
rules and `fit` so sloped/curved parts become stairs and slabs. Roof, wall and trim must contrast
(P5); the base is grounded (P4). Geometry must not change in this stage.

## Encouraged ops
`define_material`, `set_material(ids, material)` (ids accept `tag:tower`, `material:roof_main`),
`paint` for accent bands and quoins, `search_blocks`, `nearest_block`, `render(views=["iso","front"])`, `lint()`.
Not here: geometry ops (note geometry problems in `finish` for the critic).

## Material spec reminder
```json
{"base": "stone_bricks",
 "palette": [["stone_bricks", 0.72], ["cracked_stone_bricks", 0.18], ["mossy_stone_bricks", 0.10]],
 "gradient": {"axis": "y", "from": 0, "to": 4, "palette": [["cobblestone", 0.6], ["mossy_cobblestone", 0.4]]},
 "faces": {"top": "stone_brick_slab"},
 "fit": "stairs+slab",
 "noise": {"scale": 3, "seed": 7}}
```
- `palette` weights are relative; variation is coherent noise (clumps like weathering).
- `gradient` overrides the palette between two world heights — grounding (P4).
- `fit`: roofs, domes, arches, curved towers → `stairs+slab`; flat walls → `none`/`slab`; rings → `walls`.
- Roofs: `deepslate_tiles`, `dark_oak_planks`, `oxidized_cut_copper`, `nether_bricks`, `prismarine_bricks`.
  Trim/quoins: `polished_andesite`, `smooth_quartz`, `stripped_spruce_log`.

## Method (2–3 responses)
1. The outline lists every object and its placeholder; map each to a real material.
2. ONE `run_script`: `scene.define_material(...)` for wall, base, roof, trim (+ accent) — three tones
   (P5), palettes with ≥2 entries, a grounding gradient, `fit` on roofs/domes/round towers — then
   `scene.set_material(ids=..., material=...)` by selection and `scene.paint(...)` accent bands at
   string courses and under parapets, corner quoins with a thin paint box + array.
3. `render(views=["iso","front"])` + `lint()` together; check contrast and the gradient; one fix
   script if needed; `finish`.

## Checklist (lint R9 enforces the first two)
- [ ] ≥3 distinct materials on solids; roof ≠ wall ≠ trim in tone (P5).
- [ ] The main wall material has a `gradient` (y 0→2–4, darker/rougher blocks) (P4).
- [ ] Every wall material has a palette with ≥2 entries.
- [ ] `fit` set on roofs, domes, arches, round towers.
- [ ] No object still uses a placeholder (`default`, an undefined `stone_wall`).

## Worked example A — medieval stone tower
```
define_material(name="castle_wall", spec={"base":"stone_bricks","palette":[["stone_bricks",0.7],["cracked_stone_bricks",0.2],["mossy_stone_bricks",0.1]],
  "gradient":{"axis":"y","from":0,"to":4,"palette":[["cobblestone",0.6],["mossy_cobblestone",0.4]]},"fit":"stairs+slab","noise":{"scale":3,"seed":11}})
define_material(name="roof_slate", spec={"base":"deepslate_tiles","palette":[["deepslate_tiles",0.8],["cracked_deepslate_tiles",0.2]],"fit":"stairs+slab"})
define_material(name="trim", spec={"base":"polished_andesite","fit":"slab"})
set_material(ids="tag:tower", material="castle_wall")
set_material(ids="tower_ne_roof", material="roof_slate")
paint(shape={"type":"cylinder","radius":5,"height":1}, pos=[34,21,4], material="trim")
```
## Worked example B — timber hall (in a run_script: prefix `scene.`)
```
define_material(name="timber_wall", spec={"base":"spruce_planks","palette":[["spruce_planks",0.85],["stripped_spruce_log",0.15]],
  "gradient":{"axis":"y","from":0,"to":2,"palette":[["cobblestone",0.7],["mossy_cobblestone",0.3]]},"fit":"none"})
define_material(name="roof_dark", spec={"base":"dark_oak_planks","fit":"stairs+slab"})
set_material(ids="hall", material="timber_wall")
set_material(ids="group:hall_roof", material="roof_dark")
```
