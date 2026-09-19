# Stage 3 — Materials

## Goal
Replace placeholder materials with rules: palettes with weathering variation, a grounding
gradient, per-face rules and `fit` so sloped/curved parts become stairs and slabs. Ensure roof,
wall and trim contrast (P5) and the base is grounded (P4). Geometry must not change in this stage.

## Encouraged ops
`define_material`, `set_material(ids, material)` (use `select("tag:tower")`, `select("material:roof_main")`),
`paint` for accent bands and corner quoins, `search_blocks`, `nearest_block`, `list_materials`,
`render(views=["iso","front"])`, `lint()`.
## Discouraged
Any geometry op. If something is wrong with geometry, note it in `finish` for the critic.

## Material spec reminder
```json
{"base": "stone_bricks",
 "palette": [["stone_bricks", 0.72], ["cracked_stone_bricks", 0.18], ["mossy_stone_bricks", 0.10]],
 "gradient": {"axis": "y", "from": 0, "to": 4, "palette": [["cobblestone", 0.6], ["mossy_cobblestone", 0.4]]},
 "faces": {"top": "stone_brick_slab"},
 "fit": "stairs+slab",
 "noise": {"scale": 3, "seed": 7}}
```
- `palette` weights are relative; variation is coherent noise so it clumps like weathering.
- `gradient` overrides the palette between two heights (world y) — use it for grounding (P4).
- `fit`: roofs, domes, arches and curved towers → `stairs+slab`; flat walls → `none` or `slab`;
  thin decorative rings → `walls`.
- Roof materials: `deepslate_tiles`, `dark_oak_planks`, `spruce_planks`, `oxidized_cut_copper`
  (copper roof), `nether_bricks`, `prismarine_bricks`; each has stairs and slabs.
- Trim/quoins: `polished_andesite`, `smooth_quartz`, `stripped_spruce_log`, `chiseled_stone_bricks`.

## Method
1. `list_materials()` and `describe()`; map every placeholder to a real material.
2. Define wall, base, roof, trim (+ accent) materials. Three tones minimum (P5).
3. Assign by selection: walls, towers, roofs, trims, floors.
4. Accent bands: `paint(shape={"type":"box","size":[W,1,D]}, pos=[..], material="trim")` at string
   courses and under parapets; corner quoins with a thin paint box + array.
5. `render`; check contrast and that the gradient sits in the bottom 2–4 blocks. `lint`. `finish`.

## Checklist
- [ ] ≥3 distinct materials; roof ≠ wall ≠ trim in tone (P5).
- [ ] Every wall material has a palette with ≥2 entries (weathering).
- [ ] Ground gradient present on walls/towers (P4).
- [ ] `fit` set on roofs, domes, arches, round towers.
- [ ] No object still uses a placeholder like `default`, `stone_wall` without a spec.

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
## Worked example B — timber hall
```
define_material(name="timber_wall", spec={"base":"spruce_planks","palette":[["spruce_planks",0.85],["stripped_spruce_log",0.15]],
  "gradient":{"axis":"y","from":0,"to":2,"palette":[["cobblestone",0.7],["mossy_cobblestone",0.3]]},"fit":"none"})
define_material(name="roof_dark", spec={"base":"dark_oak_planks","fit":"stairs+slab"})
set_material(ids="hall", material="timber_wall")
set_material(ids="group:hall_roof", material="roof_dark")
render(views=["iso","front"])
```
