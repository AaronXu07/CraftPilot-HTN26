# Stage 2 — Detailing (secondary geometry)

## Goal
Add depth and rhythm: openings, buttresses, string courses, overhangs, battlements, roof geometry,
interior floors. After this stage no façade should be a flat plane and no exposed wall region
should exceed 8×4 blocks without a break (P2, P3, P8).

## Encouraged ops
`subtract` (windows, doors, arches — with `array` for rows), `add` thin boxes for pilasters/plinths/
string courses (`array`/`mirror`), `shell`, `round`, `wedge`/`pyramid`/`cone` roof refinements,
`extrude` for battlement rings, `reorder` (a cut must come AFTER the wall it carves),
`run_script` for anything repeated, `render(views=["iso","front"])`, `lint()`.
## Discouraged
Changing primary masses (that was blocking) unless the critic asked; `block` props; palettes.

## Method
1. `describe()` and `lint()` to see the blank faces.
2. Openings: one `subtract` box per façade with `array`. Windows 1×2 or 2×3, doors 2×3 (P6). Arches:
   a box plus a horizontal cylinder (`axis="x"` or `"z"`) subtract on top, or a `torus` half.
3. Depth: buttresses (boxes with `taper`), a plinth (box 1 wider than the wall, 2 tall, at y=0), a
   string course (a thin box ring or a `paint` band 1 tall), corbels under overhangs.
4. Roofs: check pitch; add a ridge (`line`/thin box), dormers (small box + wedge), chimneys.
5. Battlements: `add` merlon box with `array` along each wall top (odd count, gap = merlon width).
6. Interiors: floor slabs every 4–5 blocks (`box` 1 tall inside the shell), a stair `wedge`.
7. `render`, `lint`, fix, `finish`.

## Checklist
- [ ] Every façade has openings at consistent spacing (P3) and ≥2 depth layers (P2).
- [ ] Entrance is 2 tall, framed, on the `facing` side, reachable from y=0 (P6).
- [ ] Cuts are ordered after the walls they carve; carved shapes go fully through the shell.
- [ ] No blank region > 8×4 (P8) — run `lint()` to confirm.
- [ ] Roof overhang ≥1 and pitch 30–45° (P1).

## Worked example A — windows, door and battlements on the tower from blocking
```
add(id="tower_ne_win", shape={"type":"box","size":[1,2,3]}, pos=[38,6,4], op="subtract",
    modifiers=[{"type":"array","count":3,"offset":[0,6,0]}])           # 3 slits up the east face
add(id="tower_ne_merlon", shape={"type":"box","size":[1,2,1]}, pos=[34,22,-1], material="stone_wall",
    modifiers=[{"type":"array","count":3,"offset":[3,0,0]}])           # merlons along the parapet
add(id="tower_ne_band", shape={"type":"cylinder","radius":5,"height":1}, pos=[34,21,4], material="trim")  # string course, 0.5 proud
```
## Worked example B — the hall: door, window row, plinth, ridge beam
```
add(id="hall_door_cut", shape={"type":"box","size":[2,3,3]}, pos=[0,0,6], op="subtract")
add(id="hall_win_cut_s", shape={"type":"box","size":[2,2,3]}, pos=[-6,3,6], op="subtract",
    modifiers=[{"type":"array","count":2,"offset":[12,0,0]}])          # one each side of the door
add(id="hall_plinth", shape={"type":"box","size":[22,2,14]}, pos=[0,0,0], material="stone_base",
    modifiers=[{"type":"shell","thickness":1}])
reorder(id="hall_plinth", before="hall_door_cut")                        # the cut must carve the plinth too
add(id="hall_ridge", shape={"type":"box","size":[22,1,1]}, pos=[0,15,0], material="trim")
lint()
```
