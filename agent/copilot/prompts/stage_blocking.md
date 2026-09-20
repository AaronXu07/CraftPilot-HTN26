# Stage 1 — Blocking (massing)

## Goal
Get the silhouette and proportions right with PRIMARY solids only, one placeholder material per
volume (e.g. `stone_wall`, `roof_main`, `timber_wall`). No windows, no props, no palettes yet. When
you render, the shape should already read as the requested building from 30 blocks away.

## Encouraged ops
`add` (box, extrude, cylinder, prism, cone, pyramid, wedge, sphere half), `stack`, `align`,
`mirror_copy`, `array` modifier, `group`, `run_script` for repeated masses, `render(views=["iso","front"])`.
## Discouraged
`subtract` (except a big arch/gate opening that defines the silhouette), `block`, `paint`,
`define_material` with palettes (that is the materials stage).

## Method (3 responses, not 30)
1. Read the brief's silhouette_plan and plan the ground plan in your head: walls/keep/hall footprints,
   towers at corners, roofs on top. Then write ONE `run_script` that creates every mass: placeholder
   materials, hollow main volumes with `shell(1 or 2)`, roofs as separate objects with overhang
   (≥1 block wider than the wall they cover), `scene.stack`/loops for repeated parts, groups
   (`group="towers"`, `group="keep"`).
2. In the same response as the script (or the next), call `render(views=["iso","front"])` and `lint()`
   together. Compare with the brief.
3. Fix proportions in one more script or a few `set_shape`/`move`/`scale` calls — do not delete and
   re-add. Then `finish`.

## Checklist
- [ ] Every mass in silhouette_plan exists with the right size and position.
- [ ] Entrance side faces the brief's `facing` (default +z/south).
- [ ] Towers taller than wide; roof pitch 30–45° with overhang (P1).
- [ ] Nothing floats; every roof rests on a wall (use `stack`).
- [ ] Ids are semantic; parts are grouped.

## Worked example A — round tower with a cone roof (ops, not prose)
```
define_material(name="stone_wall", spec={"base": "stone_bricks"})
define_material(name="roof_slate", spec={"base": "deepslate_tiles", "fit": "stairs+slab"})
add(id="tower_ne", shape={"type":"cylinder","radius":4.5,"height":22}, pos=[34,0,4],
    material="stone_wall", modifiers=[{"type":"shell","thickness":1}], tags=["tower"], group="towers")
add(id="tower_ne_roof", shape={"type":"cone","radius":6,"height":8}, pos=[34,22,4], material="roof_slate")
stack(id="tower_ne_roof", on="tower_ne")
```
## Worked example B — gable-roofed hall from two wedges
```
add(id="hall", shape={"type":"box","size":[20,8,12]}, pos=[0,0,0], material="timber_wall",
    modifiers=[{"type":"shell","thickness":1}])
add(id="hall_roof_w", shape={"type":"wedge","size":[12,7,14],"slope_axis":"x"},  pos=[-6,8,0], material="roof_main")
add(id="hall_roof_e", shape={"type":"wedge","size":[12,7,14],"slope_axis":"-x"}, pos=[6,8,0],  material="roof_main")
group(ids=["hall_roof_w","hall_roof_e"], group_id="hall_roof")
render(views=["iso","front"])
```
(the wedges rise toward the ridge at x=0; each is 1 block wider than its half of the hall and 1 block
deeper in z, giving the overhang.)

## Four corner towers with one script
```
run_script(python="""
for name,(x,z) in {"ne":(20,-16),"nw":(-20,-16),"se":(20,16),"sw":(-20,16)}.items():
    scene.add(id=f"tower_{name}", shape={"type":"cylinder","radius":4.5,"height":24}, pos=[x,0,z],
              material="stone_wall", modifiers=[{"type":"shell","thickness":1}], tags=["tower"], group="towers")
    scene.add(id=f"tower_{name}_roof", shape={"type":"cone","radius":6,"height":8}, pos=[x,24,z], material="roof_slate")
""")
```
