# Minecraft Copilot — CAD-style agentic build engine

## Full implementation plan

**Scope:** an in-game copilot that designs and edits builds the way a Blender user would: by placing parametric solids, combining them with booleans, stacking modifiers, assigning materials, and iterating with visual feedback. No templates, no retrieval, no external 3D APIs. Every block in Minecraft is available as a material. LLM: OpenAI model on Azure (vision-capable deployment required for the critic).

The document is organized so each section maps to one owner and one milestone. Sections 2–6 are the geometry engine and are pure Python, testable without the game or an LLM.

---

## 0. Why this design produces good-looking builds (read first)

Earlier attempts looked bad for three reasons, and each has a specific fix baked into this design:

| Problem | Cause | Fix in this plan |
|---|---|---|
| Blocky, stepped curves and roofs | Solids rasterized to full cubes | **Sub-voxel surface fitting** (§5): every surface voxel is sampled at 8 sub-points and resolved to a full block, slab, or stairs with correct facing/half. Cones, domes, sloped roofs and arches come out smooth for free. |
| Flat, dead walls | Single block per surface | **Materials are rules, not blocks** (§4): weighted palettes, vertical gradients, per-face rules (top/side/bottom), and paint bands. |
| LLM loses track of what it built | Coordinates in prose | **A named scene graph** (§2): objects have names, params, transforms and modifiers. Edits are `move("tower_ne", ...)`, not "re-emit 4,000 blocks". |
| Builds ignore proportion and detail | No feedback | **Staged pipeline + vision critic + design linter** (§7–8): blocking → detailing → materials → decoration, each followed by a rendered critique against an explicit rubric. |
| LLM invents block names/states | No validation | **Registry-driven resolver** (§4.4): every block ID and property value is validated against the game's own registry dump; families (planks → stairs/slab) are derived automatically. |

---

## 1. System overview

```
Player chat  ──/cp──▶  Fabric mod (thin bridge, 7 HTTP endpoints)
                             ▲            │
                    setblocks│            │ scan / player / blocks / say / camera
                             │            ▼
                      Python agent  ◀──────────▶  Azure OpenAI (chat + vision)
                       ├─ session (scene document, history, undo stack)
                       ├─ pipeline (interpret → block → detail → materials → decorate → critic)
                       ├─ tools (scene ops, scripts, render, place, export)
                       └─ engine
                            ├─ scene model      (§2)
                            ├─ solids as SDFs   (§3)
                            ├─ materials        (§4)
                            ├─ rasterizer + fit (§5)
                            └─ render           (§6)
```

**Repository layout**

```
minecraft-copilot/
  mod/                       Fabric client mod (Java)          — Track 1
  agent/
    copilot/
      server.py              FastAPI /chat                       — Track 2
      llm.py                 Azure client, tool loop, vision     — Track 2
      session.py             scene doc, history, undo/redo       — Track 2
      pipeline/              stage prompts + orchestration       — Track 2
      tools/                 tool schemas + dispatch             — Track 2
      engine/
        scene.py             Scene, Object, Transform, Modifier  — Track 3
        solids.py            SDF primitives                      — Track 3
        modifiers.py         array, mirror, shell, round, ...    — Track 3
        raster.py            SDF → voxel grid                    — Track 3
        fit.py               sub-voxel → block/slab/stairs       — Track 3
        materials.py         Material rules → material ids       — Track 4
        resolver.py          material id + context → block state — Track 4
        registry.py          block dump, families, colors        — Track 4
        render.py            software renderer (PNG + ASCII)     — Track 3
        lint.py              design-rule checks                  — Track 4
        diff.py              scene/world diffs                   — Track 3
        schematic.py         .litematic export                   — Track 4
      bridge.py              HTTP client for the mod             — Track 2
    mock_mod/                in-memory fake bridge               — Track 2
    tests/
    bench/                   10 prompts + scoring harness        — Track 4
  scripts/
```

---

## 2. Scene model (the document the agent edits)

A `Scene` is an **ordered list of objects**, evaluated top to bottom like layers in a CSG stack (painter's model). Order matters: a `subtract` only carves what came before it.

### 2.1 Object

```json
{
  "id": "tower_ne",
  "shape": {"type": "cylinder", "radius": 4.5, "height": 22},
  "transform": {"pos": [30, 0, 0], "rot": [0, 0, 0], "scale": [1, 1, 1], "anchor": "bottom_center"},
  "op": "add",
  "material": "stone_wall",
  "modifiers": [
    {"type": "shell", "thickness": 1},
    {"type": "round", "radius": 0.5}
  ],
  "tags": ["tower", "exterior"],
  "group": "outer_wall",
  "visible": true
}
```

- `op`: `add` (union), `subtract` (carve), `intersect` (keep overlap with existing), `paint` (change material of existing voxels inside the shape; no geometry change).
- `anchor`: which point of the shape's local bbox sits at `pos`. `bottom_center` is the default because builds sit on the ground. Others: `center`, `bottom_min` (corner), `top_center`.
- `rot`: Euler degrees, any angle. Solids are implicit, so arbitrary rotation is exact; block *states* are resolved after rasterization, so orientation is always consistent.
- `material`: a material name from the scene's material table (§4) or an inline material spec.
- `group`: objects in a group move/rotate/scale together; groups nest.
- Coordinates: **x east, y up, z south**, units = blocks, scene origin = ground level at the build anchor. The system prompt states this once, with a diagram.

### 2.2 Scene

```json
{
  "name": "castle_v3",
  "materials": {"stone_wall": {...}, "roof": {...}},
  "objects": [ ... ordered ... ],
  "groups": {"outer_wall": {"parent": null, "transform": {...}}},
  "meta": {"style": "medieval", "brief": {...}}
}
```

### 2.3 Operations (pure functions `Scene → Scene`)

Every edit is an operation recorded in the session history; undo/redo replays. This is the CAD API. The same operations are exposed as LLM tools (§7.3) and as a Python scripting API (`run_script`).

Create / delete
- `add(id, shape, pos, rot=0, scale=1, anchor, op, material, tags, group)`
- `delete(ids)` · `duplicate(id, new_id, offset)` · `rename(id, new_id)`

Transform
- `move(ids, delta)` · `move_to(ids, pos)` · `rotate(ids, deg, axis="y", pivot="self"|"scene"|[x,y,z])` · `scale(ids, factor|[sx,sy,sz], pivot)`
- `align(ids, axis, mode="min|center|max", to=id|value)` · `stack(id, on=id, gap=0)` (place bottom of A on top of B)
- `mirror_copy(ids, axis, plane_value, new_suffix)` (creates mirrored duplicates)

Shape editing ("edit mode")
- `set_shape(id, **params)` — change any shape parameter (radius, height, profile points...)
- `set_op(id, op)` · `set_material(ids, material)` · `set_anchor(id, anchor)`
- `reorder(id, before=id|after=id)` — CSG order

Modifiers (ordered stack per object, like Blender)
- `add_modifier(id, modifier, index=None)` · `remove_modifier(id, index)` · `set_modifier(id, index, **params)`

Grouping & selection
- `group(ids, group_id)` · `ungroup(group_id)` · `select(query)` → ids, where query supports `tag:tower`, `group:outer_wall`, `name:tower_*`, `material:roof`, bbox filters (`above_y:20`), and combinations.

Introspection
- `describe(ids|all, detail="outline|full")` → compact text outline (§7.4) · `bbox(ids)` · `measure(id_a, id_b)` (gap/overlap per axis) · `top_of(id)`, `side_of(id, dir)` → coordinates for placement

Materials
- `define_material(name, spec)` · `list_materials()` · `paint(shape, pos, material, ...)` (sugar for `add(op="paint")`)

History
- `undo(n=1)` · `redo(n=1)` · `snapshot(label)` · `restore(label)`

Every operation validates its inputs against the scene (unknown id, invalid param) and returns a one-line result plus any lint warnings triggered (§8.2), so the model gets immediate feedback.

---

## 3. Solids (signed distance functions)

Each primitive implements `sdf(points: np.ndarray[N,3]) -> np.ndarray[N]` in local space and `local_bbox()`. Rotation/scale are applied by transforming query points into local space (inverse transform), so any rotation works. Non-uniform scale on an SDF is approximate but visually fine at block resolution; document it.

### 3.1 Primitive catalog

| Shape | Params | Notes |
|---|---|---|
| `box` | `size=[sx,sy,sz]` | The workhorse |
| `cylinder` | `radius, height, axis="y"`, optional `radius_top` (→ frustum) | Towers, columns, arches when `axis="x|z"` and used as subtract |
| `sphere` | `radius` | Domes via `intersect` with a box or `half=True` |
| `ellipsoid` | `radii=[rx,ry,rz]` | |
| `cone` | `radius, height`, optional `radius_top` | Tower roofs |
| `pyramid` | `base=[sx,sz], height`, optional `top=[tx,tz]` | Hip roofs, spires |
| `wedge` | `size=[sx,sy,sz], slope_axis="x|z"` | Gable roof halves, ramps |
| `prism` | `sides=n, radius, height` | Octagonal towers, hex floors |
| `torus` | `major, minor, axis` | Rings, arches when half |
| `capsule` | `radius, height` | Rounded columns |
| `extrude` | `profile=[[x,z],...], height` | Any 2D polygon → walls with L/T/U plans; **the second workhorse** |
| `revolve` | `profile=[[r,y],...]` | Lathe: onion domes, vases, balusters |
| `sweep` | `profile=[[u,v],...], path=[[x,y,z],...], closed=false` | Curtain walls along a polyline, bridges, pipes |
| `plane_cut` | `normal, offset` (used as `intersect`) | Slice anything |
| `block` | `state="minecraft:lantern[hanging=true]"` | A single placed block with explicit state: lanterns, banners, doors, torches, flowers. Props. |
| `line` | `from, to, thickness` | Beams, ropes, diagonal supports |

Missing something is fine: `extrude` + `revolve` + `sweep` + booleans cover almost every architectural form. Do **not** add domain shapes ("house", "tower_with_roof"); the whole point is that the agent composes.

### 3.2 SDF formulas
Use the standard Inigo Quilez formulas for box, sphere, cylinder, cone, torus, capsule, and polygon extrusion (2D polygon SDF × height). Revolve: 2D polygon SDF in `(r, y)` with `r = sqrt(x²+z²)`. Sweep: distance to polyline segments minus profile radius (use a circular profile in v1; polygonal profiles are a stretch). Precision only needs to be good to ~0.1 block.

---

## 4. Materials — every block in Minecraft

### 4.1 Registry pipeline (`registry.py`, runs once at startup, cached)

1. `GET /blocks` from the mod → every block ID with its properties and allowed values (~1,100 blocks).
2. **Families** by name pattern, with a small hand-maintained override file: `oak_planks → {full: oak_planks, stairs: oak_stairs, slab: oak_slab, fence: oak_fence, door: oak_door, trapdoor: oak_trapdoor, log: oak_log}`, `stone_bricks → {full, stairs, slab, wall, cracked: cracked_stone_bricks, mossy: ...}`, `red_concrete → {full, powder}`, `deepslate_tiles → {...}` etc. The rule: strip `_planks/_bricks/_tiles/_block/_stairs/_slab/_wall` suffixes to a base, then look for siblings. Log the ~40 that don't auto-resolve and fix them by hand.
3. **Colors**: extract `assets/minecraft/textures/block/*.png` from the client jar (path known from the mod's `/health`), average the RGB of each texture, map to block IDs by name (top texture for grass/logs when present). Used by the renderer and by `nearest_block(color, categories=...)`.
4. **Catalog for the prompt** (~2.5k tokens): grouped, compressed — wood types × forms, stone types × forms, 16 colors × {wool, concrete, terracotta, glass, glazed}, nature, ores/metals, decorative props (with their key properties), lighting, functional. The full list is never in the prompt; `search_blocks(query)` covers the tail.

### 4.2 Material spec

A material maps a voxel to a *material id* first, then the resolver (§4.4) maps material id + geometry context to a concrete block state.

```json
"stone_wall": {
  "base": "stone_bricks",
  "palette": [["stone_bricks", 0.72], ["cracked_stone_bricks", 0.18], ["mossy_stone_bricks", 0.10]],
  "gradient": {"axis": "y", "from": 0, "to": 6, "palette": [["cobblestone", 0.6], ["mossy_cobblestone", 0.4]]},
  "faces": {"top": "stone_brick_slab", "bottom": null},
  "fit": "stairs+slab",
  "noise": {"scale": 3, "seed": 7}
}
```

- `base`: the family used for fitting (stairs/slab variants come from its family).
- `palette`: weighted random choice, using coherent noise (Perlin-ish, `scale` in blocks) rather than white noise so variation clumps naturally like weathering.
- `gradient`: overrides the palette between two heights (or along any axis), blending by noise across the boundary.
- `faces`: optional per-face override for exposed top/side/bottom voxels (a roof material can put `stairs` on sloped faces and `planks` on interior).
- `fit`: `none | slab | stairs | stairs+slab | walls` — which sub-voxel fitting the resolver may use for this material (§5).
- Presets ship for ~25 common looks (`medieval_stone`, `spruce_timber`, `sandstone_desert`, `blackstone_dark`, `quartz_modern`, `copper_roof`, `slate_roof`, ...) but the LLM can define any material inline. Presets are materials, not templates.

### 4.3 Props (non-solid blocks)
`block` solids carry explicit states. The resolver validates properties against the registry and fills defaults. Connectivity properties (fence/wall/pane `north/east/...`, stairs `shape`, redstone) are **left to the game**, which recomputes them on placement, so the agent never sets them.

### 4.4 Resolver
Input: material id per voxel, plus the fit result (§5) per surface voxel. Output: block state string. Steps: pick palette entry via noise → apply gradient/faces override → if fit says `stairs(facing, half)` or `slab(type)` and the family has that form, substitute → validate against the registry → emit `minecraft:id[props]`.

---

## 5. Rasterizer and sub-voxel fitting (the quality core)

### 5.1 Rasterize
- Compute the scene bbox from all `add` objects (padding 1). Allocate `material_id: int16[X,Y,Z]` (0 = air) and `sdf_min: float32[X,Y,Z]` (the SDF of whichever object owns each voxel).
- For each object in order, evaluate its SDF only inside its own bbox (numpy, vectorized, points at voxel centers):
  - `add`: inside where `f ≤ 0`; set material, store `f`.
  - `subtract`: where `f ≤ 0`, clear material; also record `-f` so carved surfaces fit correctly.
  - `intersect`: clear voxels outside `f ≤ 0` within the object bbox.
  - `paint`: where `f ≤ 0` and material ≠ 0, set material.
- Modifiers transform the SDF before evaluation: `shell(t)`: `max(f, -(f + t))`; `round(r)`: `f - r` on a shrunk shape; `array(count, offset)`: explicit instances (simple and correct); `mirror(axis, plane)`: evaluate at reflected points too (min of both); `taper(top_scale)`: scale xz by a function of y before evaluating; `twist(deg_per_block)`: rotate xz by y before evaluating; `noise_displace(amplitude, scale)`: `f + amp·noise(p)` for rocks/organic; `boolean(target_id, op)`: per-object boolean with a specific target instead of the whole stack (evaluated as a combined SDF).
- Size budget: 200×120×200 = 4.8M voxels; each object touches only its bbox; whole scene under 1 s for typical builds, a few seconds for large ones. Cache per object; re-rasterize only objects whose bbox intersects a changed object's bbox (needed for interactive editing).

### 5.2 Sub-voxel fitting (`fit.py`)
For every occupied voxel that has at least one exposed face, and whose material allows fitting, sample the owning SDF at the 8 sub-voxel centers `(±0.25, ±0.25, ±0.25)` around the voxel center. Let `inside` be the 8-bit occupancy pattern.

| Pattern | Result |
|---|---|
| 8 inside | full block |
| bottom 4 inside only | `slab[type=bottom]` |
| top 4 inside only | `slab[type=top]` |
| 6 inside, missing the top pair on one horizontal side | `stairs[half=bottom, facing=<away from the missing side>]` |
| 6 inside, missing the bottom pair on one horizontal side | `stairs[half=top, facing=<away from the missing side>]` |
| 4 inside forming a vertical half (left/right) | full block if material has no `walls`, else `wall` for thin (1-wide) geometry |
| ≤ 3 inside | air, unless the voxel is needed for support (§5.3) |
| anything else | full block if ≥ 4 inside else air |

The stairs `facing` direction convention is **calibrated on day one**: Track 1 places one stair of each facing by hand, Track 3 scans them and writes the 4-entry lookup. Same for slab `type` and for `half`. Never trust memory for these.

Also run fitting on **carved** surfaces (arches, window reveals) using the recorded subtract SDF, so an arch cut with a cylinder gets stairs along its curve.

### 5.3 Structural sanity pass
- Remove isolated single voxels produced by fitting noise.
- Keep voxels that fitting would delete if they are the only support of a prop above (lanterns, torches).
- Optional `gravity` warning: blocks with nothing below and nothing adjacent (floating) → lint warning, not error (Minecraft allows it, but it usually looks wrong).

---

## 6. Renderer (`render.py`)

Software renderer over the voxel grid using block colors from the registry:

- Views: `iso` (default, 45°/35°), `front`, `back`, `left`, `right`, `top`, plus `slice(y)` and `cutaway(axis, at)` for interiors.
- Method: painter's-order projection of exposed faces as parallelograms (numpy → PIL), directional shading (top 1.0, one side 0.8, other 0.65) and cheap ambient occlusion (darken faces with occupied diagonal neighbors). Stairs/slabs drawn as their real half/step geometry so the critic sees the smoothing.
- Output ≤ 1024 px PNG, plus an ASCII slice mode for text-only fallback.
- Target < 0.5 s for a 150³ scene. Renders are cached by scene hash.
- A **contact sheet** helper combines iso + front + top + one cutaway into one image, so a critique needs a single vision call.

---

## 7. The agent

### 7.1 Session
Holds: current `Scene`, operation history with undo/redo, named snapshots, the design brief, the placed-world diff record (for in-place updates and `undo` in the game), and chat history (last ~30 turns). One session per player.

### 7.2 Pipeline
Every stage uses the same tools and sees the same scene; they differ by system prompt, checklist, and which operations are encouraged. Stages run automatically on a new build; on an edit request the router picks the stage(s) that apply.

0. **Interpret** — request → brief JSON: build type, style, footprint & height targets, key features the user named, constraints (superflat, facing), and an explicit *silhouette plan* ("L-shaped keep 20×14×18, four round towers r=4 h=24 at corners, curtain walls h=10, gatehouse on south"). The brief is shown to the player in chat before building so they can correct it.
1. **Blocking (massing)** — primary solids only, one material placeholder per volume. Goal: correct silhouette and proportions. Encouraged ops: `add`, `extrude`, `stack`, `align`, `mirror_copy`, `array`. Critic checks silhouette against the brief from `iso` + `front`.
2. **Detailing** — secondary geometry: carve windows/doors/arches (`subtract`), add buttresses, string courses (thin `paint` bands or shallow boxes), overhangs, battlements (`array` of boxes), roof geometry with `wedge/pyramid/cone`, interior floors and stairs. Encouraged: `subtract`, `array`, `shell`, `round`. Critic checks depth, rhythm, and that no wall face longer than ~8 blocks is featureless.
3. **Materials** — define materials with palettes/gradients/faces/fit and assign them; paint accent bands; set `fit` on roofs and curved parts. Critic checks contrast between roof/wall/trim, texture variation, and gradient at ground contact.
4. **Decoration** — props: lanterns, banners, flower boxes, torches, paths, interior furniture via `block` solids and small arrays. Critic checks lighting, entrances, and scale of props.
5. **Final critique + fixes** — up to 2 rounds.

Between stages the agent calls `say()` with a one-line summary and places the intermediate result in-game when `live_preview` is on (demo mode), so spectators watch the build evolve.

### 7.3 Tools (LLM-facing)
Scene ops from §2.3, each as a tool with a strict JSON schema, plus:
- `run_script(python)` — bulk creation with loops (four towers, a colonnade). The script uses the same ops through a `scene` object; it runs in a subprocess with a 10 s limit and no I/O; the ops it performed are recorded individually in history so undo works.
- `render(views=[...], cutaway=None)` → image(s) attached to the next model message (vision) + a short text summary (bbox, block count, top materials).
- `lint()` → design-rule findings (§8.2).
- `place(mode="diff|full", animate=True)` · `undo_world()` · `export_schematic(name)` · `materials_list()`.
- `get_player()` · `say(text)` · `search_blocks(query)` · `nearest_block(rgb, category)`.

Tool descriptions are the most important prompt text in the system; write them with examples and constraints (e.g. `move`: "delta in blocks; positive z is south").

### 7.4 Scene outline format (what the model reads)
`describe()` returns a compact, stable outline, never raw JSON:

```
scene castle_v3  bbox 0..48 x 0..30 y 0..40 z  (11 objects, 3 groups, 5 materials)
[outer_wall]
  keep         box 14x18x14 @ (17,0,13) mat stone_wall  mods: shell(1)
  tower_ne     cylinder r4.5 h22 @ (34,0,4) mat stone_wall  mods: shell(1) round(0.5)
  tower_ne_roof cone r5.5 h7 @ (34,22,4) mat slate_roof(fit=stairs)
  gate_cut     box 4x6x3 @ (24,0,-1) SUBTRACT
...
```
Ids are short and semantic because the model chooses them; the system prompt requires `snake_case` semantic names.

### 7.5 Router for conversational edits
"make the north wall taller and add arrow slits" → the router (same model, cheap call) maps to `set_shape(wall_n, size=[..])` + Detailing-stage sub-run scoped to `select(name:wall_n)`. "make it darker" → Materials stage on all. "move the whole thing 10 blocks east" → `move(all, [10,0,0])` then `place(mode="diff")`. Every edit ends with a render and, if the user asked for a change, a one-line confirmation of exactly what changed.

### 7.6 Model configuration
- Azure OpenAI chat completions with function calling; temperature 0.2 for ops, 0.6 for the Interpret stage (creativity in the brief, precision in the ops).
- Vision: pass the contact sheet PNG as an image message to the critic. If the deployment lacks vision, the critic uses ASCII slices + lint output (works, weaker).
- Parallel tool calls on. Cap 40 tool calls per stage; scripts count as one.
- Log every call with timing to `runs/<session>/<turn>.jsonl`; the bench (§8.3) replays these.

---

## 8. Quality system

### 8.1 Design principles (in the system prompt, ~600 tokens)
Proportions (towers taller than wide; roofs 30–45° with ≥1 overhang), depth (every façade needs ≥2 depth layers: pilasters, recessed windows, trim), rhythm (repeat openings at consistent spacing; odd counts read better), grounding (darker/rougher material in the bottom 2–4 blocks), roof/wall/trim contrast, entrances at human scale (2 tall, framed), lighting at night, and the "no blank face longer than 8 blocks" rule. Written as rules the critic can cite by number.

### 8.2 Linter (`lint.py`) — mechanical checks after every stage
- blank façade: exposed vertical face region > 8×4 with a single material and no depth change
- roof without overhang; roof slope outside 25–60°
- no entrance reachable from ground
- material variety: fewer than 3 distinct materials in a build > 1,000 blocks
- floating blocks; isolated single blocks
- door/prop validity (props resting on air)
- scale: any object with a dimension > 120 (probably a units mistake)
Each finding cites the rule number and the object ids involved so the model can act on it.

### 8.3 Bench
`bench/prompts.json`: 10 varied requests (medieval castle, modern villa, Japanese pagoda, lighthouse, bridge, cathedral, desert temple, treehouse, gothic tower, greenhouse). `bench/run.py` builds each headless (mock bridge), renders contact sheets, and asks the vision model to score 1–10 on silhouette, detail, materials, and fidelity to the prompt, writing a report. Run it every 4 hours; it tells you whether a prompt change helped.

---

## 9. Placement, undo, export

- **Scene → world**: anchor = 2 blocks in front of the player at ground level; rotation snapped to the player's facing (0/90/180/270) applied at the block-state level (facing/axis/rotation properties remapped by a table).
- **Diff placement**: keep the last placed block map per scene; compute added/removed/changed blocks; send only those. Removed → `minecraft:air`. This is what makes in-conversation edits feel live.
- **Animation**: bottom-up layer chunks, ~1,500 blocks per chunk, 60 ms apart; big builds finish in under 60 s.
- **World undo**: the mod's `/scan` snapshot before the first placement of a scene; `undo_world` restores it.
- **Export**: `.litematic` via `litemapy` from the resolved block map; `materials_list()` counts by block for survival players.

---

## 10. Mod bridge (unchanged in spirit; 7 endpoints)

`GET /health` · `GET /player` (pos, facing, looking_at) · `POST /scan` · `POST /setblocks` (accepts `chunks` with delays for animation) · `POST /say` · `GET /blocks` · `POST /camera` (`orbit` around a bbox in spectator for N seconds, then return; used for demo and for optional in-game screenshots). All world access on the server thread via `server.submit`. No logic in Java.

---

## 11. Prompts (what Track 2 writes)

- `system_core.md`: role, coordinate system, scene outline format, op semantics, naming rules, the design principles, materials catalog summary, "compose solids; never describe blocks by coordinates in prose".
- `stage_interpret.md`, `stage_blocking.md`, `stage_detailing.md`, `stage_materials.md`, `stage_decoration.md`: goal, allowed/encouraged ops, a checklist, and 2 short worked examples each (an ops sequence for a tower, a gable-roofed hall).
- `critic.md`: rubric with the numbered rules, output format `{score, top_3_fixes: [{rule, objects, op_suggestion}]}`.
- `router.md`: edit request → stage(s) + selection query.
- Few-shot examples are **op sequences**, never full builds.

---

## 12. Testing strategy

Pure engine (no game, no LLM):
- SDF unit tests: sample points inside/outside each primitive; rotation invariance.
- Modifier tests: shell thickness; array count; mirror symmetry.
- Fitting golden tests: a cone of radius 6 produces stairs on its slope with facings pointing outward; a half-sphere dome uses slabs at the crown; an arch cut with a cylinder yields stairs along the curve.
- Resolver: every emitted state validates against the registry; families resolve for all wood and stone types.
- Render: golden PNG hashes for 3 scenes.
- Diff: editing one object re-emits only its bbox.
Agent (mock LLM with scripted tool calls): blocking → detailing → place → edit → diff-place → undo.
Live (game): bridge verify script; facing calibration; one full castle from chat.

---

## 13. Milestones (36 h, 4 tracks)

Contracts locked at hour 1: bridge JSON, Scene JSON, Material spec, tool schemas, outline format.

| Hour | Track 1 (mod) | Track 2 (agent) | Track 3 (geometry) | Track 4 (materials/quality) |
|---|---|---|---|---|
| 1–6 | mod builds, `/cp`, `/say`, `/setblocks`, `/scan`, `/player` | Azure client, tool loop, session, mock bridge, `run_script` | Scene model + ops + undo; box/cylinder/extrude SDFs; rasterizer | registry pipeline: families, colors, catalog |
| **6** | 🎮 chat round-trips; verify script passes | | | |
| 6–12 | `/blocks`, chunked animated setblocks, facing calibration scans | Interpret + Blocking stage prompts; describe/outline; place via diff | all primitives; shell/round/array/mirror; renderer v1 | material spec, resolver, 25 presets, lint v1 |
| **12** | 🎮 `/cp build a castle with four towers` → massing appears animated in-game | | | |
| 12–20 | `/camera` orbit; world undo | Detailing, Materials, Decoration stages; critic with vision; router for edits | sub-voxel fitting + calibration table; taper/twist/noise; diff engine; render contact sheet | schematic export, materials list, bench harness, first bench run |
| **20** | 🎮 full pipeline; smooth cone roofs; live edits ("taller", "darker", "add a bridge between towers") | | | |
| 20–28 | polish, screenshots for pitch | prompt tuning from bench; latency work (parallel tool calls, script bulk ops) | perf: incremental re-raster; fit on carved surfaces | bench every 4 h; lint tuning; demo world; pitch |
| **28** | 🎮 demo rehearsal ×2 from cold start; backup video | | | |
| 28–36 | freeze, sleep, present | | | |

**Cut order if behind:** twist/taper/noise modifiers → decoration stage → camera orbit → cutaway renders → per-face materials. Never cut: sub-voxel fitting, palettes, the critic loop, diff placement. Those four are the difference between "boxes" and "builds".

---

## 14. Demo (3 minutes, template-free by construction)

1. **"build a castle with an L-shaped keep, four round towers and a gatehouse facing me"** — brief appears in chat; massing rises; detailing carves windows and battlements; materials stage textures it; roofs come out as smooth stair cones.
2. **"make the northeast tower 8 blocks taller and give it a copper roof"** — one tower changes, live, in ~15 s (diff placement).
3. **"cut an arched bridge between the two north towers"** — `sweep` + `subtract` arch; stairs follow the curve.
4. **"swap the walls to deepslate with a mossy base"** — materials only; geometry untouched.
5. **"add lanterns along the walls and a banner over the gate", then "export for survival"** — decoration pass, materials list, `.litematic`.

Ask a judge for a shape you didn't rehearse ("make the keep hexagonal"). `set_shape(keep, type=prism, sides=6)` is a single op; that's the moment that proves it's CAD, not templates.

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Stair/slab facing conventions wrong | Calibrate from real scans at hour 6–12; one lookup table; golden tests |
| LLM produces ugly massing | Interpret stage must emit a silhouette plan; critic on `front`+`iso` after blocking; bench prompts |
| Latency: 5 stages × many tool calls | `run_script` for bulk ops; parallel tool calls; place intermediate stages so the wait is the show |
| Big scenes slow to rasterize | bbox-limited SDF eval; incremental re-raster; 200³ hard cap with a clear error |
| Vision critic too vague | Rubric with numbered rules; output must name object ids and an op; cap at 3 fixes per round |
| Material catalog too big for context | 2.5k-token compressed catalog + `search_blocks`; presets |
| Azure quota/latency on the day | Test deployment TPM early; keep a second deployment in another region; replay logs as a backup |