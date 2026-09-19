# Stage 2 — Detailing (secondary geometry)

## Goal
Add depth and rhythm: openings, pilasters, quoins, string courses, overhangs, battlements, roof trim,
interior floors. After this stage no façade is a flat plane and no wall region exceeds 8×4 (P2, P3, P8).

## Ops
`subtract` (windows, doors, arches — with `array`), thin `add` boxes for pilasters/plinths/string
courses (`array`/`mirror`), `shell`, `round`, `wedge`/`pyramid`/`cone`, `extrude` (battlement rings),
`reorder` (a cut comes AFTER the wall it carves), `run_script`. Not here: primary masses, props, palettes.

## Method (3 responses, not 30)
1. Read the outline in the message (it is current) and blocking's lint findings; skip `describe()`.
2. ONE `run_script` that adds all the secondary geometry at once:
   - openings: one `subtract` box per façade with `array` (windows 1×2 or 2×3, doors 2×3, P6);
     arches: a box plus a horizontal cylinder (`axis="x"`/`"z"`) subtract on top, or a `torus` half;
   - depth: pilasters (1-proud boxes with `array` along every façade > 8), buttresses (boxes with
     `taper`), a plinth (box 1 wider than the wall, 2 tall, at y=0), a string course (thin box ring or a
     `paint` band), corbels under overhangs; quoins at every outside corner (`mirror`);
   - roofs: ridge (`line`/thin box in trim), eave trim ring, dormers (small box + wedge), chimneys;
   - battlements: merlon box with `array` along each wall top (odd count, gap = merlon width);
   - interiors: floor slabs every 4–5 blocks (`box` 1 tall inside the shell), a stair `wedge`.
3. `render(views=["iso","front"])` + `lint()` in one response; one fix script if needed; `finish`.

## Checklist (lint and the critic check every line)
- [ ] Every façade > 8 long has DEPTH (pilasters 1 proud every 4–6 via `array`, or 1-deep window
      reveals) and RHYTHM (odd count of equal-spaced openings, aligned across floors) (P2, P3).
- [ ] Every outside corner has quoins: a 1×N×1 `trim` column mirrored to all corners.
- [ ] Every roof: overhang ≥1, pitch 30–45°, a trim ridge line and a 1-thick eave trim ring (P1).
- [ ] Entrance 2 tall, framed (trim jambs + lintel), on the `facing` side, reachable from y=0 (P6).
- [ ] Cuts ordered after the walls they carve and going fully through the shell.
- [ ] No blank region > 8×4 (P8) — `lint()` confirms.

## Worked example A — slits, merlons and a string course on the tower from blocking
```
add(id="tower_ne_win", shape={"type":"box","size":[1,2,3]}, pos=[38,6,4], op="subtract",
    modifiers=[{"type":"array","count":3,"offset":[0,6,0]}])           # 3 slits up the east face
add(id="tower_ne_merlon", shape={"type":"box","size":[1,2,1]}, pos=[34,22,-1], material="stone_wall",
    modifiers=[{"type":"array","count":3,"offset":[3,0,0]}])
add(id="tower_ne_band", shape={"type":"cylinder","radius":5,"height":1}, pos=[34,21,4], material="trim")
```
## Worked example B — the hall: door, window row, plinth, quoins, ridge
```
add(id="hall_door_cut", shape={"type":"box","size":[2,3,3]}, pos=[0,0,6], op="subtract")
add(id="hall_win_cut_s", shape={"type":"box","size":[2,2,3]}, pos=[-6,3,6], op="subtract",
    modifiers=[{"type":"array","count":2,"offset":[12,0,0]}])          # one each side of the door
add(id="hall_plinth", shape={"type":"box","size":[22,2,14]}, pos=[0,0,0], material="stone_base",
    modifiers=[{"type":"shell","thickness":1}])
reorder(id="hall_plinth", before="hall_door_cut")                        # the cut must carve the plinth too
add(id="hall_quoin", shape={"type":"box","size":[1,12,1]}, pos=[-11,0,-7], material="trim",
    modifiers=[{"type":"mirror","axis":"x","plane":0},{"type":"mirror","axis":"z","plane":0}])  # 4 corners
add(id="hall_ridge", shape={"type":"box","size":[22,1,1]}, pos=[0,15,0], material="trim")
```
