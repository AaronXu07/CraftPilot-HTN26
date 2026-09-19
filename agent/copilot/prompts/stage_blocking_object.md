# Stage 1 — Blocking an OBJECT (statue, creature, vehicle, prop)

This is not a building: no walls, roofs, windows or entrance. You are sculpting a **skeleton of
rounded parts** that reads as the subject from 30 blocks away. Think armature first, skin later.

## Goal
Every part in the brief's `silhouette_plan` exists as ONE primitive with the right size, position and
tilt, joined to its neighbour (overlap by ≥ 1 block — nothing floats, nothing touches only at a corner).
Placeholder material per part family (`body_mat`, `limb_mat`, `detail_mat`, `base_mat`), no palettes.

## The parts vocabulary (use these, not boxes)
- **Body / torso / head / hull**: `ellipsoid` radii=[rx,ry,rz] (anchor `center`, so pos is the centre).
  Elongate along the subject's long axis; a box body is the #1 reason objects look like crates.
- **Neck / tail / limbs / arms / fingers / horns**: `line` from=[x,y,z] to=[x,y,z] thickness=t — a
  capsule between two points. Chain two or three lines for a bent neck or a tail that curves.
  A curving tail or neck can also be one `sweep` radius=r path=[[...],[...],[...]].
- **Wheels / eyes / joints**: `cylinder` axis="x" (wheels of a car facing +z) or `sphere`.
- **Wings / fins / blades / ears**: thin `wedge` or `box` size=[w,1,d] with `rot` to tilt them; wings
  attach at the shoulder and sweep up-and-out.
- **Cones/spikes**: `cone`; **rings**: `torus`.
- **Base / plinth**: a `box` 2–3 tall, ≥ 2 blocks wider than the subject's footprint, material `base_mat`.
- `rot=[rx,ry,rz]` (degrees) tilts any part — lean a body forward, angle a wing, rake a windshield.

## Orientation (memorise)
The front / head / nose faces **+z (south, toward the player)** unless the brief says otherwise.
Length runs along z, width along x, height along y. A car 18 long is size [6, 5, 18], not [18, 5, 6].
Wheels are cylinders with axis="x" at z = ±(length/2 − 3), both sides at x = ±(width/2).

## Method (2–3 responses)
1. Read the brief's parts list. Write ONE `run_script` that defines the placeholder materials
   (`scene.define_material(name="body_mat", spec={"base": "stone"})`), creates the base, then every part on the
   RIGHT side only (x > 0) plus the centre parts (body, head, tail), each with a semantic id
   (`body`, `head`, `neck_1`, `leg_fr`, `wing_r`, `wheel_fr`), a `group` (`limbs`, `wings`, `wheels`),
   overlapping its neighbour. Then `scene.mirror_copy(ids=[...right-side ids...], axis="x", plane=0,
   new_suffix="_l")` for bilateral symmetry — never hand-place both sides.
2. `render(views=["iso","front"])` + `lint()` in the same response. Check: does the silhouette read
   as the subject from every view? Are proportions right (head : body : legs)? Anything floating?
3. Fix with `set_shape` / `move` / `rotate` on the parts that are off — do not delete and re-add.
   Then `finish`.

## Checklist
- [ ] Every part in the plan exists, with the plan's numbers.
- [ ] Nothing floats: each limb/tail/neck overlaps the body by ≥ 1 block (lint R5 clean).
- [ ] Front faces +z; the long axis is z (unless the brief says otherwise).
- [ ] Symmetric parts come from `mirror_copy`, not two hand-placed copies.
- [ ] Body is an ellipsoid/capsule, not a box; limbs are lines, not boxes.
- [ ] No part thinner than 1.8 blocks (thinner capsules rasterise into disconnected voxels).

## Worked example — quadruped statue (dragon body, no wings) in one script
```
run_script(python="""
for name, base in (("base_mat","polished_andesite"), ("body_mat","stone"), ("limb_mat","stone"), ("detail_mat","deepslate")):
    scene.define_material(name=name, spec={"base": base})
scene.add(id="plinth", shape={"type":"box","size":[14,3,22]}, pos=[0,0,0], material="base_mat")
# body: ellipsoid centred at y=9, long along z
scene.add(id="body", shape={"type":"ellipsoid","radii":[3,3,7]}, pos=[0,9,0], anchor="center", material="body_mat")
# neck rises forward (+z) and up; head is a smaller ellipsoid at the end
scene.add(id="neck", shape={"type":"line","from":[0,10,6],"to":[0,15,11],"thickness":3}, material="body_mat")
scene.add(id="head", shape={"type":"ellipsoid","radii":[2,2,3.5]}, pos=[0,15.5,13], anchor="center", material="body_mat", group="head")
# tail curves back (-z) and down in two segments
scene.add(id="tail_1", shape={"type":"line","from":[0,9,-6],"to":[0,8,-12],"thickness":2.4}, material="body_mat", group="tail")
scene.add(id="tail_2", shape={"type":"line","from":[0,8,-12],"to":[0,10,-17],"thickness":1.8}, material="body_mat", group="tail")
# right legs only: front and back, from the belly down to the plinth (y=3)
for name, z in (("fr", 4), ("br", -4)):
    scene.add(id=f"leg_{name}", shape={"type":"line","from":[2,8,z],"to":[2.5,3,z],"thickness":2}, material="limb_mat", group="limbs")
scene.add(id="horn_r", shape={"type":"cone","radius":0.7,"height":3}, pos=[1,17,12], rot=[-30,0,0], material="detail_mat", group="head")
scene.mirror_copy(ids=["leg_fr","leg_br","horn_r"], axis="x", plane=0, new_suffix="_l")
""")
render(views=["iso","front"])
lint()
```
