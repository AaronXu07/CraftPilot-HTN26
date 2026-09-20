# CraftPilot Implementation Plan

> **Status (2026-09-19):** milestones M0 to M5 are implemented. M4 landed as a thin bridge: the Fabric
> mod hosts an HTTP server inside the game and Python pushes world-space blocks to it (`/setblocks`),
> so the mod never parses a `.litematic`; `/build` in chat calls the service, which places the result
> in front of the player layer by layer. An animated wireframe of the build box shows while a build is in
> flight (placeholder at once, exact box from the service, scan line tracking placed rows). Wand selection
> and undo are deferred.
> Engine: layout with side and stacked attachment, massing with taper, jetty and thick walls, all twelve
> roof types, all sixteen attachment kinds, the silhouette budget, facade grammar with six window
> styles and four framing modes, depth pass, basic interiors (stairs, ladders, doorways, partitions),
> palettes with vertical, band and horizontal gradients, weathering and texturing, rotation to face
> the player, litemapy export, isometric preview. LLM: Azure OpenAI compose and edit with strict
> structured outputs, exemplar retrieval (10 exemplars), offline fallback. Block catalog generated from
> the 26.2 jar (`scripts/gen_catalog.py`, `data/blocks_26.2.json`). Tests: unit plus a closure
> property test. After the first in-game review (2026-09-19): panes stay in the wall plane, windows
> are gated by visibility so they never sit on junctions, rooms are verified reachable by flood fill
> and doorways are carved where needed, palettes are checked against 60-30-10 colour rules, context
> tags and texture noisiness (with a curated library of 28 named palettes), and a final detail pass
> adds inline wall details (stairs and slabs set into the wall) plus vines and leaf bushes.
> Second review round: staircases rebuilt with proper head room (straight runs, 3x3 and 2x2 spirals,
> ladders), entrances (double doors, portals with pillars and steps, castle gates), window styles
> (stair slits, boarded, fence gates), buttons and flush trapdoors matched to the wall, sills as
> stairs or trapdoor ledges, dormers carved into the attic with flat floors, roof bodies filled so
> corner stairs never show gaps, block families carrying colour, texture noise, material and style
> tags that the model sees in the catalog, and a `material_reasoning` field the model fills before
> choosing a palette. Deferred: the critique loop from preview images and WFC facade fill (M6).

Procedural Minecraft building generator. A player types `/build <natural
language>` in a singleplayer world, a Fabric mod forwards it to a local Python
service, an LLM composes a structured **build program** from the text, a
deterministic engine renders the program as a voxel grid inside a bounding box,
litemapy writes a `.litematic`, and the service streams the blocks back through
the mod, which places them in the world one layer per tick.

Scope for now: buildings of any kind. No terrain generation, roads, or settlements
(buildings are seated on the existing terrain, §8.4).

---

## 1. Decisions and assumptions

| Area | Decision | Notes |
|---|---|---|
| Minecraft | Java Edition **1.21.1** in-game (mod); catalogs shipped for 1.21.1 and 26.2 | The block catalog is generated from the game jar and selected by `CRAFTPILOT_MC_VERSION`, so a version bump is `scripts/gen_catalog.py` plus the mod's `gradle.properties`. |
| World | Singleplayer, integrated server | All in-game integration goes through a Fabric mod. |
| Mod | Fabric mod (`mod/`), Gradle + Fabric Loom, client side | Hosts the HTTP bridge on 7777 (`/player`, `/setblocks`, `/scan`, `/say`); `/build` chat command calls the service; `/outline` animates the in-progress box. Wand, undo and auto-launch deferred. |
| Language | Python 3.12+, `uv` | `pyproject.toml`, hatchling, ruff, pytest. numpy, scipy, pydantic, FastAPI. |
| Schematic | litemapy 0.11.x | Confirm the data version it writes matches 26.2. |
| LLM | Azure OpenAI, GPT-5.4 line, Responses API, structured outputs | Deployment names come from env. `gpt-5.4` composes the build program; `gpt-5.4-mini` handles follow-up edits. See section 6. |
| Generality | **No closed style list.** The LLM writes a `BuildProgram` over a primitive vocabulary; the engine renders any valid program. | House, mansion, castle, modern are exemplars, not special cases. New building types need new exemplars, not new code, until they need a primitive the vocabulary lacks. |
| Bounds | Every build has an explicit bounding box | From the wand selection, the command text, or a default. Configurable safety limit only. |
| Determinism | Reproducible from `(program, bounds, seed)` | Golden tests, "same again but taller". |

Coordinates follow Minecraft: `x` east, `y` up, `z` south, north is `-z`.

---

## 2. Architecture

```
/build a stone lighthouse with a red and white stripe and a glass lantern room
   |
   v
[mod]     command handler -> POST /build {text, player, pos, yaw, bounds?}
   |
   v
[service] FastAPI on localhost (craftpilot serve)
   |
   |-- [llm]    gpt-5.4 + retrieved exemplars -> BuildProgram   (fallback: nearest exemplar)
   |
   |-- [engine] BuildProgram + bounds + seed -> SemanticGrid -> resolved block grid
   |            layout -> massing -> mass attachments -> roof -> roof attachments
   |            -> facade -> depth -> materials -> postprocess
   |
   |-- [export] litemapy -> .minecraft/schematics/craftpilot/build_<id>.litematic
   |
   |-- [place]  grid -> world blocks anchored in front of the player -> POST /setblocks to the mod
   |
   v
[mod]     drains the queue one layer-chunk per tick with FORCE_STATE (states arrive fully baked)
```

Three hard boundaries:

1. **LLM vs engine.** The LLM produces a `BuildProgram`. The engine never
   calls the LLM. Generation is fast, testable, and works offline from
   exemplars.
2. **Semantic grid vs blocks.** Geometry stages write roles and tags. Only the
   materials stage writes concrete block states. Every Minecraft build
   technique (gradient, texturing, depth) is a function over tags.
3. **Python vs mod.** The mod's HTTP bridge (`mod/README.md`) is the only contract: Python sends
   fully specified block states; the mod never computes anything.

The key property the engine must have: **closure**. Every `BuildProgram` that
passes schema validation renders to a complete, valid building. Ugly is
acceptable, broken is not. This is what lets the LLM be creative.

---

## 3. Repository layout

```
craftpilot/
  pyproject.toml  uv.lock  README.md  PLAN.md
  exemplars/                   # complete BuildPrograms, used as few-shot + fallback + goldens
    cottage.yaml  mansion.yaml  castle.yaml  modern_house.yaml
    church.yaml  barn.yaml  lighthouse.yaml  pagoda.yaml
    watchtower.yaml  townhouse.yaml  ...
  data/
    blocks_26.2.json           # generated block catalog
  craftpilot/
    cli.py  serve.py  config.py
    program/
      model.py                 # BuildProgram and all vocabulary enums (the contract)
      validate.py              # semantic checks beyond the schema
      exemplars.py             # load, index, retrieve exemplars
    llm/
      azure.py
      compose.py               # text (+ exemplars, + previous program) -> BuildProgram
      edit.py                  # "same but taller" -> patched BuildProgram (mini)
      fallback.py              # keyword match -> nearest exemplar, patched by regex
      prompts/
    grid/
      semantic.py  enums.py  ops.py  anchors.py
    engine/
      pipeline.py
      layout.py                # relational part layout -> footprint masks in bounds
      massing.py
      roof.py                  # one function per RoofType
      facade.py
      depth.py
      materials.py  noise.py
      postprocess.py
      attachments/
        base.py  registry.py
        tower.py  chimney.py  dormer.py  balcony.py  cupola.py
        buttress.py  bartizan.py  porch.py  colonnade.py  arch.py
        gallery.py  crenellation.py  spire.py  flagpole.py  awning.py
    blocks/
      catalog.py  state.py  circles.py
    export/litematic.py
    preview/render.py  ghost.py
    route.py                   # building or object? (section 9.1)
    objects/                   # image->3D objects path (section 9.2): brief, imagegen, recon, voxelize, pipeline, orient, flow
    place/                     # in-game placement: bridge, anchor, placer (terrain seating), undo
    terrain.py
  tools/                       # the local 3D reconstruction worker (recon_server.py) and its setup notes
  mod/                         # Fabric mod (see section 8)
  tests/
    unit/  golden/  llm_eval/
```

---

## 4. The BuildProgram vocabulary

This is the contract between the LLM and the engine. It is deliberately a
small language of architectural primitives rather than a list of styles.
Anything the LLM can say in it, the engine can build.

### 4.1 Parts and layout

```python
class Shape(str, Enum):
    rect = "rect"; circle = "circle"; ngon = "ngon"; cross = "cross"; ring = "ring"

class Side(str, Enum): north = "north"; east = "east"; south = "south"; west = "west"; top = "top"
class Align(str, Enum): start = "start"; center = "center"; end = "end"

class Attach(BaseModel):
    to: str                    # name of another part
    side: Side
    align: Align = Align.center
    overlap: int = 1           # blocks the two parts share; >1 embeds
    offset: int = 0            # slide along the side

class RoofType(str, Enum):
    gable = "gable"; hip = "hip"; flat = "flat"; parapet = "parapet"
    shed = "shed"; mansard = "mansard"; gambrel = "gambrel"
    cone = "cone"; dome = "dome"; spire = "spire"; pagoda = "pagoda"; none = "none"

class RoofSpec(BaseModel):
    type: RoofType
    pitch: float = 1.0         # blocks up per block in; 0.5 = slabs
    overhang: int = 1
    tiers: int = 1             # pagoda / stepped
    crenellated: bool = False  # parapet only
    ridge_axis: Literal["auto", "x", "z"] = "auto"

class PartSpec(BaseModel):
    name: str
    shape: Shape = Shape.rect
    sides: int = 8             # ngon only
    size: tuple[float, float]  # (width, depth) as fraction of the root part; root uses (1, 1)
    floors: int = 2
    floor_height: int = 4
    taper: float = 0.0         # fraction of width lost per floor (lighthouse, skyscraper setbacks)
    odd_dims: bool = True
    roof: RoofSpec
    attach: Attach | None = None   # None marks the root; exactly one root
    role_hint: str = ""        # "nave", "keep", "wing" — for the LLM's own bookkeeping
```

The root part scales to fill the bounds (minus foundation outset and roof
overhang). Other parts are sized relative to it and placed by `Attach`. The
layout stage resolves this into absolute footprints, then shrinks the whole
composition uniformly if any part would leave the bounds.

### 4.2 Facade and depth rules

```python
class Framing(str, Enum): none = "none"; corners = "corners"; bays = "bays"; tudor = "tudor"
class WindowStyle(str, Enum): plain = "plain"; arched = "arched"; tall = "tall"; slit = "slit"; wall = "wall"; round = "round"

class FacadeRules(BaseModel):
    bay_width: tuple[int, int] = (3, 5)
    window: WindowStyle = WindowStyle.plain
    window_size: tuple[int, int] = (1, 2)
    shutters: float = 0.0
    sills: bool = True
    framing: Framing = Framing.corners
    symmetry: bool = True
    max_flat_run: int = 7
    ground_floor_taller: int = 0

class DepthRules(BaseModel):
    frame_protrude: int = 1
    window_inset: int = 1
    eave_trim: bool = True
    foundation_rise: int = 1
    foundation_outset: int = 1
    floor_lips: bool = False   # modern slab lips at storey boundaries
    corbels: bool = True
```

### 4.3 Attachments and silhouette

```python
class AttachmentKind(str, Enum):
    tower = "tower"; chimney = "chimney"; dormer = "dormer"; balcony = "balcony"
    cupola = "cupola"; buttress = "buttress"; bartizan = "bartizan"; porch = "porch"
    colonnade = "colonnade"; arch = "arch"; gallery = "gallery"   # gallery = ring balcony
    spire = "spire"; flagpole = "flagpole"; awning = "awning"; jetty = "jetty"
    cantilever = "cantilever"

class AttachmentRequest(BaseModel):
    kind: AttachmentKind
    count: int | Literal["auto"] = "auto"
    on: str | None = None              # part name, or None for anywhere
    prefer: list[AnchorType] = []      # anchor preference
    spacing: Literal["regular", "irregular"] = "irregular"
    params: dict[str, float | int | str] = {}   # kind-specific, validated per kind

class SilhouetteBudget(BaseModel):
    dominant: int = 1
    medium: tuple[int, int] = (1, 3)
    small: tuple[int, int] = (2, 5)
```

### 4.4 Palette

Same as before: per-role `RolePalette` of `(family, weight)` with gradient and
texture rate, composed from the block catalog's families and tones.

### 4.5 BuildProgram

```python
class BuildProgram(BaseModel):
    label: str                          # free text: "lighthouse", "tudor inn"
    parts: list[PartSpec]
    facade: FacadeRules = FacadeRules()
    depth: DepthRules = DepthRules()
    attachments: list[AttachmentRequest] = []
    budget: SilhouetteBudget = SilhouetteBudget()
    palette: PaletteSpec
    requested_bounds: Bounds | None = None
    notes: str = ""                     # what could not be expressed; shown to player
```

`validate.py` adds checks the schema cannot: exactly one root, `attach.to`
names exist and form a tree, `cone`/`dome` only on circle or ngon parts,
crenellation only on `parapet`, roof and trim families provide stairs and
slabs, attachment params in range. Invalid pieces are repaired to defaults
and reported in `notes` rather than failing the build.

### 4.6 Extending the vocabulary

When a request needs something the vocabulary cannot say, the LLM approximates
and writes what it wanted into `notes`. The service logs these. Each new
primitive is one enum value, one engine function, and one exemplar that uses
it. The rule for scope: primitives describe **form** (a roof type, a footprint
shape, an attachment), never a building type. "Windmill sails" would be an
attachment kind; "windmill" stays an exemplar.

---

## 5. Engine stages

Each stage is `stage(grid, program, bounds, rng) -> None`. Order:

1. layout
2. massing (+ foundation)
3. mass-level attachments
4. roof
5. roof-level attachments
6. facade
7. depth
8. materials
9. postprocess

### 5.1 Layout

- Root part sized to bounds minus foundation outset and max roof overhang.
- Walk the attach tree. For each child, compute width and depth from `size`
  fractions, snap to odd if `odd_dims`, place against the parent's side with
  `align`, `overlap`, `offset`.
- Rasterize each part: rect, circle (from `circles.py`), ngon, cross, ring.
- If the union leaves the bounds, scale all fractions down together and redo.
- Output: per-part masks, union mask, part adjacency.

### 5.2 Massing

- Extrude each part by `floors * floor_height`, applying `taper` by shrinking
  the mask per floor. Perimeter `WALL`, inside `INTERIOR`, `FLOOR` per storey.
- Foundation via dilation by `foundation_outset`, `foundation_rise` tall.
- Compute `normal`, `edge_dist`, `h_norm`; publish `CORNER` anchors.
- Height budget: reserve space above the tallest part for roof plus the
  dominant attachment so it fits `bounds.height`; reduce floors if needed.

### 5.3 Roof

One function per `RoofType`, all producing a per-part height map merged by
`max`:

| Type | Height map |
|---|---|
| hip | `pitch * chessboard_dist(mask dilated by overhang)` |
| gable | distance to the two edges perpendicular to `ridge_axis`, capped at half width |
| shed | linear ramp across the part |
| mansard / gambrel | piecewise pitch: steep lower band, shallow upper band |
| flat / parapet | 0, plus a 1-block parapet ring; merlons if `crenellated` |
| cone | `pitch * (radius - euclid_dist)` |
| dome | `sqrt(r^2 - d^2)` scaled to pitch |
| spire | cone with pitch >= 2, finial on top |
| pagoda | `tiers` stacked hip roofs with decreasing masks and an upturned eave row |

Quantization, overhang, stair facing, and anchor publishing are shared.

### 5.4 Attachments

Registry maps `AttachmentKind` to an implementation with `accepts`, `level`,
`fits`, `apply`, and a params schema. Placement loop per level: honor explicit
`AttachmentRequest`s first (count, `on`, `prefer`, `spacing`), then spend any
remaining silhouette budget on kinds the program allows. Occlusion and bounds
checks reject bad placements. Shared corbel rule for overhangs.

New kinds since the earlier draft: `colonnade` (row of pillars along a face or
porch), `arch` (arched openings on ground floor bays), `gallery` (balcony ring
around a circular part, the lighthouse case), `spire` as an attachment on any
ridge or tower top, `flagpole`, `awning`, `jetty` (jettied floor), `cantilever`.

### 5.5 Facade

As before, driven by `FacadeRules`. `WindowStyle` adds arched (stair-topped
openings), tall (1x3), slit (castle), wall (modern glass), round (portholes).
`ground_floor_taller` supports shops and halls.

### 5.6 Depth, 5.7 Materials, 5.8 Postprocess

Unchanged in mechanism, now reading `DepthRules` and `PaletteSpec` from the
program instead of a preset. Every emitted block state is validated against
the catalog before export.

---

## 6. LLM layer (Azure OpenAI, GPT-5.4)

### 6.1 Steps and models

| Step | Deployment | Effort | When |
|---|---|---|---|
| Compose `BuildProgram` | `gpt-5.4-mini` | `low` (`CRAFTPILOT_COMPOSE_EFFORT`) | Every new `/build`. Architectural reasoning plus palette design in one call, ~8 s. |
| Edit program | `gpt-5.4-mini` | `low` | "same but taller", "make the roof red": patch the previous program. |
| Route (unsure cases only) | `gpt-5.4-mini` | `low` | Building or object, when the word rules are not confident (section 9.1). |

Palette composition is folded into the compose call because palette choices
depend on the form (a pagoda roof wants a different family than a gable) and
one call is simpler than two. Deployment names come from
`AZURE_OPENAI_COMPOSE_DEPLOYMENT` and `AZURE_OPENAI_EDIT_DEPLOYMENT`
(`AZURE_OPENAI_DEPLOYMENT` is accepted for both).

Why `low` works: with strict structured outputs the model emits keys in schema order, so `BuildProgram.design`
- three or four sentences deciding the massing, roofs, materials and where each named feature goes - is
written *first* and the rest of the program follows from it. That visible plan (~100 tokens) replaces the
thousands of hidden reasoning tokens `medium` spent (6k-16k per call, 30-100 s) for programs of the same
fidelity; every exemplar carries a `design` so the few-shot turns show it filled. An answer that fails
validation is retried one effort level up before the offline fallback. The service also starts composing the
moment the mod sees `/build <text>` (`POST /prepare`, single-flight with the real request), so the call
overlaps the seconds the player spends aiming, and the G commit places the grid rendered for the hologram
instead of generating it again.

### 6.2 Client

```python
client = AzureOpenAI(api_key=..., api_version=..., azure_endpoint=...)
resp = client.responses.parse(
    model=COMPOSE_DEPLOYMENT,
    input=[{"role": "system", "content": SYSTEM}, *exemplar_messages, {"role": "user", "content": text}],
    text_format=BuildProgram,
    reasoning={"effort": SETTINGS.compose_effort},
)
program = resp.output_parsed
```

Responses API with structured outputs. GPT-5.4 rejects `temperature`, so
determinism comes from the schema, validation, and the exemplars.

### 6.3 Compose prompt

- System prompt: what each primitive means in Minecraft terms, the Minecraft
  build rules (palette of 3 to 6 families, at most 2 loud, gradients,
  no flat runs, odd widths for symmetry, silhouette budget), the catalog's
  families grouped by tone with the shapes each provides, and the bounds.
- **Exemplar retrieval**: `exemplars.py` embeds each exemplar's description
  and retrieves the top 3 to 5 for the request as few-shot pairs. A request
  for a "lighthouse" pulls lighthouse, watchtower, and pagoda; a request for
  a "tudor inn" pulls cottage, townhouse, and mansion. This is how the system
  generalizes: the model sees nearby working programs and composes a new one.
- Output is validated and repaired; repairs go to `notes`.

### 6.4 Exemplar library

- Each exemplar is a full `BuildProgram` plus a one-paragraph description and
  a golden PNG.
- Initial set of about ten chosen to span the vocabulary: cottage, mansion,
  castle, modern house, church (nave + tower + apse), barn (gambrel + silo),
  lighthouse (circle, taper, cone, gallery), pagoda (tiers), watchtower
  (ngon, parapet, bartizans), townhouse (narrow, tall, shed dormers).
- When a generated program looks good in-game, `/build save <name>` stores it
  as a new exemplar. The library grows with use, and retrieval improves.

### 6.5 Guardrails

- 15 s timeout, one retry, then `fallback.py`: nearest exemplar by keyword,
  patched by regexes for numbers and materials. The player is told.
- Cache programs by normalized text.
- Log prompt, response, validation repairs, and `notes` with request ids.
- The LLM never sees or produces coordinates or block placements.

---

## 7. Export with litemapy

Unchanged: `Region` filled from the grid's palette of `BlockState` objects,
saved to `.minecraft/schematics/craftpilot/build_<label>_<seed>_<id>.litematic`.
Description carries the program, bounds, and seed. Verify the data version.

---

## 8. Fabric mod

### 8.1 Commands

| Command | Behavior |
|---|---|
| `/build <text>` | (implemented) An outline box follows the player; the lock key (G) freezes it and sends `/build {ghost:true}` with that pose. The service composes and generates and answers with a voxel cloud; the mod shows it as a translucent hologram (`GhostRender`) where the box was locked. G again commits via `/regenerate {seed, place}` (same seed, same grid); H lets the hologram follow the player first. Blocks stream in bottom-up and hide the hologram row by row. |
| `/build plan <text>` | (implemented) Compose only; the service describes the program (`program/describe.py`) and the mod prints it. `/build edit` then refines the plan without building; `/build go` aims a box of the plan's size and builds it via `/regenerate`. |
| `/build wand` | Gives the selection wand (a stick with a custom name and NBT tag). Left click sets corner 1, right click sets corner 2. |
| `/build bounds <w> <h> <d>` | Selection without corners, anchored at the player. |
| `/build clear` | Clears the selection. |
| `/build undo` | Restores the last placement for this player. |
| `/build again [seed]` | Regenerates the last program with a new or given seed. |
| `/build edit <text>` | Sends the last program plus the edit text to the edit endpoint, then places. |
| `/build preview <text>` | Generates and saves the schematic without placing. |
| `/build save <name>` | Saves the last program as an exemplar. |

### 8.2 Build outline (implemented)

`BuildOutline` draws a wireframe of the box a build will fill, in `WorldRenderEvents.BEFORE_DEBUG_RENDER`
(vanilla flushes the lines layer right after it). `/build` shows a placeholder at once (default bounds,
the same anchor math as `place/anchor.py`); the service posts the exact box through `POST /outline` after
compose (`generating`) and again with the trimmed box before queueing (`placing`); a scan line sweeps
while waiting and tracks the highest placed row while blocks land; the box flashes green and clears when
the queue drains, on cancel, or on a service error. A wand *selection* outline (persisted corners,
dimensions in the action bar) is still deferred.

### 8.3 Service client (implemented)

- `/build ...` posts to `craftpilot serve` on 7778 from a worker thread and prints `summary`, `notes`
  and any `invalid` block-state warning in chat. No auto-launch: the player starts the service.

### 8.4 Placement (implemented as a bridge)

- The mod hosts `POST /setblocks` on 7777. The service (`src/craftpilot/place/`) reads `/player`,
  rotates the grid to face them, anchors it `gap` blocks ahead at feet level, converts the grid to
  `[x, y, z, "minecraft:id[props]"]`, and groups blocks into whole bottom-up layers of at most 1500
  blocks (attachables last within a layer). One chunk is placed per tick with
  `NOTIFY_LISTENERS | FORCE_STATE` and no post-process, because the engine already bakes stair
  shapes, hinges, connections and support into every state. The `.litematic` is still written.
- **Terrain** (`src/craftpilot/terrain.py`): before queueing, the service surveys the footprint plus
  an apron through `POST /heightmap` (per column: the ground — the top motion-blocking block that is
  not a tree or a plant — its state, and the highest block standing in the column) and seats the
  plinth row at the 40th percentile of the ground heights under the base — sunk further to stay under
  y=319, refused if that means burying it more than 16 blocks. It streams with the building: air for
  the hill and for whole trees/plants inside the base columns, a foundation down to the terrain in
  the building's own foundation block (solid plinth up to a 5-block range, perimeter wall + pillar
  grid beyond), and the apron (3–8 columns, by range) ramped one block per column, cleared of trees
  and plants, and re-topped with each column's surface block. Water and unsampled columns are left
  alone. `CRAFTPILOT_PLACE_TERRAIN=0`, an older mod without `/heightmap`,
  or an empty survey fall back to feet-level placement with the bounding box cleared.
- Undo is not implemented yet; `/build cancel` drops what is still queued.

---

## 9. Local service

`craftpilot serve` on `127.0.0.1:7778` (the mod owns 7777):

- `POST /plan` -> `{text, player, bounds?, use_llm?, kind?, height?}`: compose (or brief) only; `{plan, bounds,
  summary, notes, route}` out.
- `POST /build` -> `BuildRequest` in (`ghost: true` answers a hologram cloud, `place: true` streams the blocks into
  the game), `{schematic, summary, notes, seed, bounds, placement?, route}` out.
- `POST /edit` -> `{player, text, plan?|ghost?}` refines the pending plan / rebuilds the last thing with a change.
- `POST /regenerate` -> `{player, seed?, ghost?|place?}`: the hologram for a pending plan (`/build go`), a new take
  (`/build again`), or the G commit that places the previewed build where the box was locked.
- `POST /cancel` -> drops the mod's placement queue. `POST /exemplars` -> saves the last building. `GET /health`.

Every `/plan` and `/build` is **routed** first (`craftpilot/route.py`, section 9.1). The service keeps one
`Pending` per player (kind, text, the program or the object brief + cached draw) so `/edit`, `/regenerate` and
the G commit act on the right thing without the mod knowing which path answered: both paths speak the same
`ghost` / `plan` / `placement` / `summary` / `notes` shapes.

CLI mirrors it: `craftpilot build "..." [--kind auto|building|object] [--place]`, `craftpilot plan`,
`craftpilot object "..."` (= `build --kind object`), `craftpilot route "..."`, `craftpilot render`.

### 9.1 Routing: building or object

Two generators share `/build`. The **building generator** (sections 4-6) is right for architecture its part
grammar can express; the **objects path** (`src/craftpilot/objects/`, section 9.2) is right for anything whose
identity is a silhouette. `route.classify(text)`:

1. an explicit override (`/build object ...`, `/build building ...`, `--kind`, or an `object:` / `building:`
   text prefix) wins;
2. **landmarks** - a curated list of named places and famous things (Eiffel Tower, Big Ben, Golden Gate Bridge,
   Hogwarts, Titanic...) - always go to objects, whatever generic words they contain;
3. otherwise the **head noun** decides (last keyword before `with|of|on|...`): building words are what the
   grammar can build (house, cottage, castle, tower, church, barn, factory...); object words are statues,
   creatures, vehicles, props, and the structures the grammar cannot express (bridge, windmill, pyramid, arch,
   fountain). A theme word scores nothing ("dragon-themed castle"), "shaped like X" is a silhouette request,
   and storeys / `NxMxK` dimensions / facade features are building evidence. Ties go to the building generator.
4. Only when the margin is small (confidence < 0.8: "a house shaped like a pineapple", "a dog house", "build me
   something cool") does one small gpt-5.4-mini call decide (effort low, 12 s timeout, cached on disk). Any
   failure of that call falls back to the rule result; the router never raises.

The decision is the first note in every answer (`routed to objects: landmark "eiffel tower"`) and the full
`Route` is returned as `route`. `craftpilot route TEXT` prints it with its evidence and scores.

### 9.2 Objects on the mod flow

`objects/flow.draw()` runs `objects/pipeline.build_object` (brief -> FLUX reference image -> Hunyuan3D mesh ->
coloured voxels, 15-35 s, plus a one-off 3D worker start) and returns a `SemanticGrid` turned to face south
(`objects/orient.face_south`), so from then on it is indistinguishable from an engine-built building: the same
`ghost_cloud`, the same `rotate_cw(grid, quarter_turns_for_facing(...))` toward the player, the same
`place/placer.place_grid` with terrain seating (a plinth stands like a foundation ring; a creature on legs gets a
level pad under its whole footprint) and one `/setblocks`. The draw is cached per player: the G commit never
regenerates (FLUX and the reconstruction are not deterministic; `seed` is only the token the mod echoes back),
`/build again` draws a new take, `/build plan` composes the brief only and `/build go` draws it, `/build edit`
edits the brief (`objects/brief.edit_brief`). Every object placement writes `placed.json` next to its artefacts;
`craftpilot object-undo` sets those positions back to air. Failures are one chat line (503 when the 3D worker or
image service is missing, 422 when the content filter refuses a name) - never a silent fallback to the building
generator; only an *auto* route on a machine without the 3D tooling is downgraded, with a note.

---

## 10. Testing

- **Unit**: layout tree resolution and bounds scaling, every `RoofType`
  height map on rect / circle / ngon parts, stair facing on every slope,
  corbel rule, occlusion, rotation remapping, catalog validation.
- **Closure / property**: random valid `BuildProgram`s from a generator
  (hypothesis) with random bounds and seeds must always render: no exception,
  nothing outside bounds, every stair valid, every attachable supported, door
  reachable. This test is what guarantees LLM creativity cannot break the
  build.
- **Golden**: every exemplar renders to a known hash and PNG.
- **LLM eval**: 40 to 60 commands spanning building types not in the exemplar
  set (inn, chapel, silo, gatehouse, bakery, skyscraper, greenhouse). Check
  that the program is valid, uses sensible primitives, and that a human
  gallery review passes. Run on demand, not in CI.
- **Mod**: manual test world checklist per milestone; undo round-trip; wand
  outline.

---

## 11. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | uv scaffold, catalog generator for 26.2, SemanticGrid, litemapy export, preview renderer | A hard-coded box exports and pastes via Litematica |
| M1 | `BuildProgram` model, layout, massing, gable/hip/flat roofs, facade, depth, materials, postprocess | The cottage exemplar renders well at three bounds |
| M2 | Attachments and remaining roof types | Chimney, dormer, balcony, porch, cupola, tower, cone, dome, pagoda, shed, mansard; closure test passes |
| M3 | Azure compose + edit, exemplar retrieval, fallback, CLI, service | `craftpilot build "a lighthouse"` produces a valid program and building with no lighthouse-specific code |
| M4 | Fabric mod | `/build` in-game places the building in front of the player (done); wand selection, outline, auto-launch and undo deferred |
| M5 | Exemplar library to ten, remaining attachment kinds | All ten exemplars pass gallery review; LLM eval on unseen building types |
| M6 | Polish | Basic interiors, critique loop from preview PNG, WFC facade fill |

---

## 12. Resolved

- Deployment names and Azure config come from the environment.
- Bounds selection uses a wand with a rendered outline.
- The mod auto-launches the Python service.
- Building types are open-ended: the LLM composes a program over a primitive
  vocabulary, exemplars guide it, and the closure test keeps the engine safe.
