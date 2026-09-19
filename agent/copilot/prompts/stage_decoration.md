# Stage 4 — Decoration (props)

## Goal
Make it lived-in and legible at night: lighting at the entrance and along walls (P7), banners and
flags, flower boxes, paths, a few interior furnishings. Props are `block` solids (single explicit
block states) and small arrays. Scale matters: a handful of well-placed props beats fifty.

## Encouraged ops
`add` with `{"type":"block","state":"..."}` (+ `array` modifier for rows), `top_of`, `side_of`,
`run_script` for prop rows, `search_blocks` (to find states), `render`, `lint`.
## Discouraged
Geometry or material changes (report them via `finish` instead).

## Useful states (all validated by the runtime; connectivity props are filled by the game)
- `lantern[hanging=false]` on a wall-top or post; `lantern[hanging=true]` under an overhang
- `wall_torch[facing=south]` (facing = the direction the torch points AWAY from the wall)
- `oak_door[facing=south,half=lower,hinge=left]` + `[half=upper]` one above
- `red_banner[rotation=8]` (floor banner) / `red_wall_banner[facing=south]`
- `flower_pot`, `potted_red_tulip`, `oak_leaves[persistent=true]` (hedges), `campfire[lit=true]`
- `chain` under lanterns, `iron_bars`, `glass_pane` in windows, `oak_fence` railings
- `dirt_path`, `gravel`, `coarse_dirt` for paths (paint a thin box at y=-1..0)
- `spruce_stairs[facing=north,half=bottom]` as seats, `oak_trapdoor[half=top,open=false]` as table tops

## Method
1. `describe()`; find the entrance (`side_of(door_cut, "south")`) and wall tops (`top_of`).
2. Entrance: two lanterns/torches flanking the door at y+2, a banner over it, a door pair.
3. Walls: a lantern every 6–8 blocks along parapets or under eaves (`array`).
4. Interior: 2–4 light sources per floor; simple furniture.
5. Ground: a path from the door toward +z (paint band of gravel/dirt_path), flower boxes
   under windows.
6. `render`, `lint` (props must rest on something), `finish`.

## Checklist
- [ ] Entrance lit and framed (P6, P7).
- [ ] Light every 6–8 blocks on the exterior; interiors lit.
- [ ] No prop floats (lint R6).
- [ ] Prop count proportional to the build (≈1 per 60 blocks of surface).

## Worked example A — lighting the tower parapet
```
add(id="tower_ne_lanterns", shape={"type":"block","state":"lantern[hanging=false]"}, pos=[30,22,4],
    modifiers=[{"type":"array","count":2,"offset":[8,0,0]}])
add(id="tower_ne_banner", shape={"type":"block","state":"blue_wall_banner[facing=south]"}, pos=[34,12,9])
```
## Worked example B — the hall entrance
```
side_of(id="hall_door_cut", dir="south")           # → z=7.5 face centre (0,0,7.5)
add(id="hall_door_l", shape={"type":"block","state":"spruce_door[facing=south,half=lower,hinge=left]"}, pos=[0,0,5])
add(id="hall_door_u", shape={"type":"block","state":"spruce_door[facing=south,half=upper,hinge=left]"}, pos=[0,1,5])
add(id="hall_lantern_l", shape={"type":"block","state":"wall_torch[facing=south]"}, pos=[-2,3,6],
    modifiers=[{"type":"array","count":2,"offset":[4,0,0]}])
paint(shape={"type":"box","size":[3,1,8]}, pos=[0,-1,10], material="path_gravel")
```
