# Stage 3 — Materials for an OBJECT

Give the subject a readable palette: **one dominant material, one darker shadow/underside material,
one accent** (eyes, trim, wheels, blade edge). Sculptures read through value contrast, not variety.

## Encouraged ops
`define_material`, `set_material`, `paint` (a shape that recolours what it overlaps), `select`,
`search_blocks`, `nearest_block`, `list_materials`, `render`.

## Method (2 responses)
1. ONE `run_script`: define 3–4 materials, then assign by group/id:
   - **Dominant** on body/head/limbs — a 2–3 block palette of the same hue so surfaces are not flat
     (e.g. `{"palette": [["stone", 3], ["andesite", 1], ["cobblestone", 1]]}`; scales → prismarine
     family; metal → iron/light_gray concrete; wood → planks + stripped logs; red car → red concrete
     + red terracotta).
   - **Shadow** on the underside/belly/inner legs/wheel wells/plinth base — 1–2 values darker
     (deepslate, dark polished blackstone, gray concrete). Use a `paint` ellipsoid/box under the body
     for the belly rather than repainting whole parts.
   - **Accent** on eyes, horn tips, claws, teeth, lights, trim, blade edge — one strong contrast
     (glowstone/sea_lantern for glowing eyes, gold_block, quartz, black concrete for tyres).
   - Curved/sloped parts get `"fit": "stairs+slab"` (dominant material) so the surface looks smooth.
2. `render(views=["iso","front"])`, check the three values read from a distance, adjust one or two
   assignments, `finish`.

## Rules
- Whole-object one-material = fail; more than five materials on a small subject = noise.
- Plinth/base: a stone family clearly different from the subject (subject stone → plinth polished
  or dark; subject coloured → plinth grey).
- No gradients required; no lanterns/props — this is a sculpture, not a house.
