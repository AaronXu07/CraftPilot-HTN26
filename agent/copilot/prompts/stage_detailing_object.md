# Stage 2 — Detailing an OBJECT

The skeleton is in place. Now add the **secondary forms** that make the subject unmistakable, and
break up any large smooth area. Still no palettes (materials stage next). Objects need FEW, well-placed
details — five good ones beat twenty specks.

## Encouraged ops
`add` (ellipsoid, line, cone, wedge, sphere, cylinder), `add_modifier` (`round`, `taper`, `array`),
`mirror_copy`, `set_shape`, `run_script`, `render`, `lint`.

## What to add, by subject
- **Creatures**: eyes (2 spheres r≈0.6 or single dark blocks), jaw/snout (a smaller ellipsoid or
  wedge under the head), ears/horns (cones), spine ridge (array of small wedges along the back),
  claws/feet (flattened ellipsoids at the end of each leg), wing membrane (thin box between two
  `line` wing-bones), chest plates (a lighter ellipsoid on the belly, op="paint" later).
- **Humanoids**: shoulders (spheres), hands (spheres), feet (boxes), belt (a torus or paint band),
  face plane (a flat ellipsoid), a held item (line + box).
- **Vehicles**: windshield/windows (subtract a thin box, or a glass-tagged box), wheel arches
  (subtract a horizontal cylinder slightly larger than the wheel), headlights/taillights (blocks),
  spoiler (thin box on two short lines), grille (subtract shallow box), exhaust (small cylinders).
- **Weapons / props**: edge bevel (`round` modifier), fuller/groove (subtract a thin box), guard and
  pommel (torus / sphere), engraving line (thin subtract).
- **Any large smooth area > 6×6**: add ONE secondary form (a ridge, a plate, a seam) rather than noise.

## Rules
- Details attach: every new part overlaps the part it belongs to by ≥ 1 block.
- Right side first, then `mirror_copy(axis="x", plane=0)` — eyes, ears, wheels, arms, wings.
- Keep the silhouette: no detail may make the overall outline read as something else.
- Scale: a detail is 5–20 % of its parent part; eyes on a 4-block head are ≤ 1 block.
- Do NOT add lanterns, doors, windows-as-in-buildings, torches or props around the subject.

## Method (2 responses)
1. ONE `run_script` with all details for this stage (right side + centre), then the `mirror_copy`,
   then `render(views=["iso","front"])` and `lint()` in the same response.
2. Fix anything floating or out of scale with `set_shape`/`move`; `finish` with a one-line summary
   ("added eyes, jaw, spine ridge ×7, claws; mirrored").
