# How CraftPilot works

CraftPilot turns a sentence typed in Minecraft chat into a build in the world. Two generators sit behind one
command: a **procedural building generator** for architecture and an **image-to-3D object path** for statues,
creatures, vehicles, props and landmarks. A router decides which one handles a request. Both produce the same
kind of block grid, so previewing, placing, undoing and exporting work the same way for either.

This document describes the system as of commit `161ab1c` (2026-09-19). Minecraft 26.2, Fabric, macOS on an
M4 Pro with 24 GB is the machine everything was measured on.

---

## 1. Processes and where they run

Everything runs on the player's computer. Four things talk to each other over localhost:

| Process | What it is | Port | Started by |
|---|---|---|---|
| **Minecraft + the `copilot` Fabric mod** (`mod/`, Java) | The game. The mod adds the `/build` chat command, an HTTP bridge that exposes the world, a hologram preview, and a block placement queue. | bridge on **7777** | the player, via the `fabric-loader-26.2` launcher profile |
| **The service** (`uv run craftpilot serve`, Python/FastAPI) | Routing, both generators, terrain seating, placement planning, schematics, undo records. Holds the player's pending build. | **7778** | the player, from the repo root |
| **The 3D worker** (`tools/recon_server.py`, Python, its own venv) | Keeps the image-to-3D models loaded on the GPU and turns a picture into a mesh. | **7790** | the service, automatically, the first time an object is requested |
| **Azure AI Foundry** (cloud) | The LLM (`gpt-5.4-mini`) that writes the building program or the object brief, and the image model (`FLUX-1.1-pro`) that draws the object reference. | HTTPS | — |

The mod contains no build logic, no LLM calls and no keys. If the service is down the mod says
`service not running at http://127.0.0.1:7778 (start it with: uv run craftpilot serve)`.

Data flow, in one line: chat → mod → service → (LLM / image model / 3D worker) → service → mod bridge → world.

---

## 2. One request, end to end

The player types `/build a stone owl statue on a rock`.

1. **Mod → service.** The mod's `ServiceClient` posts `{"text": "...", "player": "<name>", "kind": "auto",
   "ghost": true, "pos": [x, y, z], "yaw": ...}` to `POST http://127.0.0.1:7778/build`. `ghost: true` means
   "don't place anything yet, give me a preview".
2. **Router.** `craftpilot.route.classify()` decides `building` or `object` (section 3). Here: `statue` is an
   object head noun → `object`.
3. **Generation.** The object path runs (section 5): LLM brief → FLUX image → 3D worker mesh → coloured voxel
   grid. For a building the building path runs instead (section 4). Either way the result is a `SemanticGrid`
   (section 6) authored facing south (+z).
4. **Hologram.** The service returns a `ghost` payload: every filled cell of the grid as `[x, y, z, rgb]` plus
   its size. The mod draws it as a translucent cloud in front of the player, rotated to face them. The cloud
   follows the player until they press **G** (lock) — **H** lets it follow again.
5. **Commit.** On **G** the mod stamps the locked position and yaw into the request and posts
   `POST /regenerate {"place": true, "pos": ..., "yaw": ...}`. The service takes the pending result, rotates a
   copy toward the player, and:
   - asks the mod for a **heightmap** of the footprint plus an apron (`POST /heightmap` on the bridge),
   - computes the ground level from the terrain under the footprint (a little below the median surface, so the
     build reads as sunk into the slope rather than perched on it),
   - computes terrain edits: cut air for anything inside the footprint above ground (trees included), fill a
     foundation down to the terrain, grade an apron,
   - writes an undo record (world positions + previous states),
   - streams blocks to the bridge with `POST /setblocks`, bottom-up, in chunks (default 1500 blocks, 60 ms
     apart), attachable blocks (doors, torches, lanterns) last within each layer.
6. **Placement.** The mod's `BlockPlacer` drains the queue one chunk per server tick, then re-evaluates each
   placed block's connectivity (stairs shape, fence/wall/pane sides) so the last stair in a row is not left
   `straight`.
7. **Schematic.** Every result is also written to
   `~/Library/Application Support/minecraft/schematics/craftpilot/<label>_<stamp>.litematic`, loadable with
   Litematica (installed for 26.2 alongside MaLiLib).

Other commands: `/build preview <text>` (draw and report, no hologram), `/build plan <text>` then `/build go`
(plan first, hologram later), `/build again [seed]` (same program, new seed), `/build edit <change>` (patch the
pending program), `/build object <text>` / `/build building <text>` (force the router), `/build status`,
`/build cancel`.

---

## 3. The router (`src/craftpilot/route.py`)

`classify(text)` returns `Route(kind, source, confidence, reason)` in three layers, cheapest first:

1. **Override.** `/build object …` or `/build building …` sets `kind` directly.
2. **Word rules.** A landmark list (Eiffel Tower, Statue of Liberty, …) always routes to objects. Otherwise the
   head noun of the request is scored: building nouns (house, castle, tower, church, barn, hall…) vs object
   nouns (statue, creature names, vehicle names, props). Shape cues ("rearing", "perched") and structural cues
   ("two storeys", "with a gable roof", dimensions) add points to one side.
3. **LLM tie-break.** Only when the rule score's confidence is below 0.8 and `CRAFTPILOT_ROUTE_LLM=1`, one small
   cached call to `gpt-5.4-mini` decides. If that call fails the rule result stands. The router never raises.

The route and its reason travel with the response (`route.note`, e.g. `routed to objects: "statue" is an
object`) and are printed in chat.

---

## 4. The building path

**Text → BuildProgram (LLM).** `llm/compose.py` retrieves the k most similar exemplars from `exemplars/*.yaml`
(13 complete programs: cottage, castle, church, pagoda, lighthouse, …), sends them as few-shot examples with a
strict JSON schema to `gpt-5.4-mini` on the Responses API, and gets back a `BuildProgram`. Responses are
cached by content hash under `logs/cache/`. If Azure is unavailable, `llm/fallback.py` picks the nearest
exemplar by keywords and patches it (`source: fallback`); the chat line says so.

**BuildProgram** (`program/model.py`, pydantic) is a vocabulary, not a style list:
- `parts`: rectangular / polygonal masses with size (fractions of the root), floors, floor height, taper, wall
  thickness, a `roof` spec (12 roof types: gable, hip, pagoda tiers, parapet with merlons, onion…; pitch,
  overhang, ridge axis, curl, edge palettes), and an `attach` relation to a parent part (side, alignment,
  overlap, offset). Exactly one root part.
- `facade`: bay width, window style (6 styles), framing mode, door placement.
- `depth`: pilasters, string courses, corbels, recessed openings.
- `attachments`: 16 kinds (chimneys, balconies, buttresses, dormers, porches…), with a silhouette budget so
  the engine does not over-decorate.
- `interior`: floors, stairs, ladders, doorways, partitions.
- `palette`: named block families for wall/roof/trim/floor with gradients, banding and weathering, checked
  against 60-30-10 colour rules; 28 curated palettes to start from.
- `bounds`: a suggested bounding box; the player's selection overrides it.

**BuildProgram → blocks (deterministic engine).** `engine/pipeline.py` runs, in order:
`layout` (part footprints in the bounds) → `auto_budget` → mass-level attachments → `massing` (walls, floors,
taper, jetties) → `roof` → roof-level attachments → `facade` (windows on bays, gated by visibility so none sit
on junctions, doors) → facade-level attachments → `interior` (rooms verified reachable by flood fill, doorways
carved where needed) → `depth` → `detail` (inline wall details, vines, bushes) → `materials` (the only stage
that writes concrete block states: palettes, gradients, texturing noise, stair/slab orientation) →
`postprocess`.

Every stage before `materials` writes **roles and tags** (wall, roof, floor lip, corbel…) into the grid, not
block ids. Same program + same bounds + same seed = same building. The engine's contract is closure: any
program that passes schema validation renders to a complete, valid building. Block names come from a catalog
generated from the 26.2 game jar (`data/blocks_26.2.json`, `scripts/gen_catalog.py`).

---

## 5. The object path (`src/craftpilot/objects/`)

Used for anything whose identity is a silhouette rather than a floor plan. No LLM composes the geometry; the
LLM's only job is a short brief.

### 5.1 Brief (`brief.py`, ~2–3 s)
`gpt-5.4-mini` turns the request into JSON:
- `subject`: one sentence for the image model — pose, material, colours, 2–3 distinctive features; bold chunky
  forms; **never a trademarked name** (the image service rejects those words — "Pikachu" is written as "a
  chubby yellow mouse-like creature with red circular cheeks…"). The `label` may keep the name.
- `plinth`: true for statues/monuments, false for vehicles and animals on the ground.
- `height`: the object's **largest dimension** in blocks (a statue's height, a ship's length). Defaults 40 for
  a statue, 32 for a creature, 36 for a vehicle; "big" ×1.5, "huge/giant" ×2; a number in the request wins.
- `palette`: `stone` only when the player asked for stone/marble; otherwise `colorful`/`auto`. Only `stone`
  restricts the block set; everything else trusts the image's colours.
- `style`, `label`.
Without Azure a keyword fallback produces the same structure.

### 5.2 Reference image (`imagegen.py`, ~2–3 s)
One call to `FLUX-1.1-pro` through Azure's OpenAI-compatible images API. The prompt leads with the camera
("three-quarter view at eye level, front-left corner, 45°" — a view from above makes the reconstructor
return a flat relief), then the subject, then "single object, plain white background, no shadow, no text",
then a style clause asking for a chunky toy-like model with no thin poles, ropes or rigging (thin parts do not
survive at block scale), and "vivid saturated colours" for colourful subjects.

Azure applies two filters. A **prompt blocklist of protected names** (characters, brands, people): on a hit the
brief is rewritten by appearance once and retried. A **violence classifier on the generated image**: any blade
in the picture is rejected regardless of framing; the deployment's content-filter policy (`StatueImages`,
violence threshold "high") did not change that, so weapons currently fail and the chat says which filter hit.
Other rejections degrade through a cascade: drop the style clause → use the player's own words → museum-
sculpture framing.

### 5.3 Matte and mesh (`recon.py` → `tools/recon_server.py`, ~11–14 s)
The service posts the image path to the 3D worker. The worker:
1. **Mattes** the subject with `u2net` (`rembg`, CPU, 0.2 s once loaded). A flood-fill matte from the image
   border is the fallback when rembg is unavailable; it cannot separate white marble from a white background,
   which is why the neural one is primary.
2. **Reconstructs** with **Hunyuan3D-2 mini-turbo** (Tencent, 0.6 B-parameter flow-matching shape model,
   fp16 on Metal, FlashVDM decoder, 5 steps, octree 256): ~10 s inference. This is the default engine because
   it recovers a real body — a rearing horse gets four legs and 16 blocks of depth. The older **TripoSR** (2 s)
   is kept as a fallback engine; it returns a shallow relief that only reads from one angle.
3. **Colours** the mesh. Hunyuan3D outputs geometry only, so the reference image is projected onto it
   orthographically along the camera axis (image-left = −x, image-up = +y, camera on +z in its frame). A
   vertex whose projection lands outside the silhouette (the far side, a foreshortened limb) takes the colour of
   the nearest vertex that did hit, in 3D, rather than the darkest outline pixel.
4. Returns a PLY with vertex colours plus `up_axis` and `front_axis` for that engine (Hunyuan: y up, front +z;
   TripoSR: z up, front +x).

The worker loads both models at start (~28 s once per machine session) and keeps them warm; the first object
after a reboot pays that. A cold subprocess fallback (`tools/recon_worker.py`, TripoSR only) exists if the
server cannot start.

### 5.4 Voxelisation (`voxelize.py`, <1 s)
1. Orient to y-up, rotate by the requested yaw, scale so the largest dimension equals `height`, cap width and
   depth at 96.
2. Rasterise at **half-block pitch** and fill the interior. Each block owns a 2×2×2 group of sub-voxels and
   gets a **shape** from the pattern: 7–8 filled → full block; a filled lower half → bottom slab (upper half →
   top slab); a filled half plus the two sub-voxels along one side of the other half → a **stair** whose
   full-height side faces that way; ≥5 filled → full; fewer → air. Slopes and curves come out as quarter-steps
   the way builders smooth by hand. (All 70 stair/slab variants used exist in 26.2; concrete/wool/terracotta
   have none, so colourful subjects stay full cubes on slopes.)
3. Colour each block from the average of its 24 nearest mesh vertices.
4. Match colours to blocks in **CIELAB** against a palette of ~130 opaque full blocks (stone families, wood,
   concrete, wool, terracotta, metals, ice, obsidian, and emissive blocks: glowstone, sea lantern, shroomlight,
   froglights, magma). A block may be duller than the sample but is penalised for being more saturated; each
   sample's chroma is clamped to 1.5× the object's 95th-percentile chroma (kills the magenta a reconstructor
   hallucinates on a grey statue, keeps a red car red). An emissive block that would cover more than a quarter
   of the surface is re-matched to ordinary blocks (glow stays an accent and lights up in game). The palette is
   capped at 6 block types for low-chroma subjects and 10 for colourful ones, by frequency.
5. A majority-vote pass removes isolated odd blocks. Interior voxels get a filler block (stone).
6. The result is written into a `SemanticGrid` with stair/slab block states (`facing`, `half`, `type`), and
   `report["object_front"]` records which side faced the camera. `orient.face_south()` turns the grid so that
   side faces +z, after which it is indistinguishable from a building for the hologram, placement and export.

A relief gate catches the single-image failure mode (a pancake thin along up with a square footprint) and
retries the image with a different camera phrasing, keeping the better mesh. Every stage's artefact is kept
in `<schematics>/objects/<label>_<stamp>/`: `brief.json`, `prompt.txt`, `reference.png`, `mesh_input.png`
(after matting), `mesh.ply`, `preview.png`, `report.json` with timings.

Typical total: brief 2–3 s + image 2–3 s + mesh 11–14 s + voxels <1 s ≈ **15–20 s**, then placement.

### 5.5 What the object path cannot do
Single-image reconstruction guesses the back and cannot recover several thin parts. Compact subjects
(statues, busts, animals, characters, vehicles, furniture, trophies) work; rigging, blades, poles, wires do not.
Faces need scale: at 40 blocks a face is ~8 blocks across.

---

## 6. Shared machinery

**SemanticGrid** (`grid/semantic.py`): numpy arrays over a W×H×D box — per cell a role, a shape (full, stair,
upside-down stair, bottom/top slab, wall block, …), a facing normal, a part id, height fraction, flags, and a
`block` index into a palette of `BlockRef`s (block id + properties). Objects write `block` directly; buildings
write roles first and let `materials` resolve them.

**Rotation** (`grid/ops.rotate_cw`): quarter turns about the vertical axis with block-state properties
remapped (`facing`, `axis`, rail shapes, connectivity). Used to turn a south-facing grid toward the player.

**Preview** (`preview/render.py`): an isometric PNG of a grid, colours from the catalog's measured data;
`preview/ghost.py` produces the hologram cloud.

**Placement** (`place/`): `anchor.plan_origin` puts the build `CRAFTPILOT_PLACE_GAP` (2) blocks in front of the
player, centred, facing them; `terrain.py` surveys and terraforms; `placer.place_grid` streams chunks and
polls `/setblocks/status`; `undo.py` writes and replays undo records.

**Export** (`export/litematic.py`): a `.litematic` via litemapy, data version 4903 (26.2), trimmed above the
build. Litematica: `M` → Load Schematics → `craftpilot`.

---

## 7. The mod (`mod/`, Java, Fabric 26.2)

Client-side only. Six classes that matter:
- `CopilotClientMod` — registers `/build …` and its subcommands, starts the bridge, prints chat lines (tagged
  lines keep their tag in gold).
- `ServiceClient` — posts to the service on a background thread; arms the hologram when the answer carries a
  `ghost`; prints service errors.
- `PendingBuild` / `BuildOutline` / `GhostRender` — the preview: a box outline that follows the player, the
  hologram cloud once the service answers, **G** locks and commits, **H** releases.
- `HttpBridgeServer` — the bridge on `127.0.0.1:7777`: `GET /health`, `GET /player` (position, yaw, facing,
  crosshair target), `POST /scan` (inclusive box → palette-indexed blocks), `POST /heightmap` (surface survey
  ignoring trees and plants), `POST /setblocks` (chunks with delays), `GET /setblocks/status`,
  `POST /setblocks/cancel`, `POST /say`, `GET /blocks` (the live block registry with properties — the service
  refreshes its catalog from it), `POST /camera` (orbit), `POST /outline`.
- `BlockPlacer` — the tick-driven queue; connectivity post-process after each chunk.
- `WorldOps` — state string parsing/serialisation, scans, registry dump; runs on the server thread.

Build: JDK 25 (`~/.jdks/jdk-25.0.4.1+1`), Gradle 9.5.1, Loom 1.17 (Mojang mappings), Fabric API 0.161.0+26.2:
`cd mod && JAVA_HOME=~/.jdks/jdk-25.0.4.1+1/Contents/Home ./gradlew -Dorg.gradle.java.home=$JAVA_HOME build`.
The jar goes in `~/Library/Application Support/minecraft/mods/` next to Fabric API, MaLiLib and Litematica.

---

## 8. Configuration

`.env` in the repo root (never committed):

| Variable | Meaning |
|---|---|
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` | The Azure AI Foundry resource (`htn26`). The v1 OpenAI-compatible surface is used for chat, structured output and images. |
| `AZURE_OPENAI_DEPLOYMENT` / `AZURE_OPENAI_COMPOSE_DEPLOYMENT` | LLM deployment (`gpt-5.4-mini`). |
| `CRAFTPILOT_IMAGE_DEPLOYMENT` | Image deployment (default `FLUX-1.1-pro`). |
| `CRAFTPILOT_RECON_ENGINE` | `hunyuan` (default) or `triposr`. |
| `CRAFTPILOT_PORT`, `CRAFTPILOT_MOD_URL` | Service port (7778) and the mod bridge (7777). |
| `CRAFTPILOT_MC_DATA_VERSION` | Schematic data version (4903 = 26.2). |
| `CRAFTPILOT_PLACE_*` | Gap, sink, chunk size, delay, terrain seating on/off. |
| `CRAFTPILOT_ROUTE_LLM`, `CRAFTPILOT_ROUTE_TIMEOUT` | Router tie-break LLM. |
| `AZURE_OPENAI_TIMEOUT_S`, `AZURE_OPENAI_REASONING_HEADROOM_TOKENS` | Per-call timeout (60 s for demos) and extra output tokens for reasoning models (reasoning tokens count against `max_output_tokens`; without headroom a call can spend it all thinking and return nothing). |

Azure: the subscription is an **Azure Sponsorship**. That grants quota for `gpt-5.4-mini`, FLUX and the
"direct from Azure" open models, but zero real-time quota for `gpt-5.4`, `gpt-5.5` and Claude in every region,
and sponsorship subscriptions cannot request increases. Deployments on `HTN26` (eastus): `gpt-5.4-mini`
(500 K TPM), `FLUX-1.1-pro` (capacity 30, content-filter policy `StatueImages`), `DeepSeek-V4-Pro` (unused;
a stronger model did not help the old tool-calling pipeline). A `gpt-5.4` deployment exists but is batch-only.

---

## 9. Running it

```sh
cd ~/Documents/CraftPilot-HTN26
uv sync                      # one shared .venv (Python 3.12) for the whole package
uv run craftpilot serve      # the service on 7778; starts the 3D worker on first object
```
Launch Minecraft with the `fabric-loader-26.2` profile, open a world, type `/build …`.

CLI without the game: `uv run craftpilot build "a stone cottage" --preview` (building),
`uv run craftpilot object "a stone owl statue" [--place] [--height 48] [--engine triposr]` (object),
`uv run craftpilot render` (all exemplars to `out/exemplars/`), `uv run craftpilot catalog`.

Tests: `uv run pytest -q tests` (168, ~50 s; the object tests use synthetic meshes and never call the network).
Lint: `uv run ruff check src tests`.

One-time setup of the 3D worker (torch, TripoSR, Hunyuan3D-2, weights ~3 GB): `tools/README.md`.

---

## 10. Timings (M4 Pro, 24 GB)

| | |
|---|---|
| Router | 0 s (rules) or ~2 s (LLM tie-break) |
| Building: compose + engine | 5–15 s (LLM) + <1 s |
| Object: brief | 2–3 s |
| Object: image | 2–3 s |
| Object: matte + mesh (Hunyuan, warm) | 11–14 s |
| Object: voxels + match | <1 s |
| 3D worker cold start | ~28 s, once |
| Placement | ~1 s per 1500 blocks at the default pacing |

---

## 11. Known limits

- Weapons: rejected by Azure's image violence classifier; the remaining lever is setting that filter to
  annotate-only in the `StatueImages` policy, which the account owner has to do.
- Protected names never reach the image model; the description-by-appearance rewrite handles the common cases.
- Thin/multi-part subjects (ships with rigging, swords, bridges) reconstruct poorly from one image.
- The back of an object is the reconstructor's guess; colours there mirror the front.
- Object previews (`preview.png`) draw stairs and slabs as cubes; the game shows the real shapes.
- The first reconstruction of an unseen tensor shape can take much longer (Metal kernel compilation was seen
  once at 54 s); repeat runs are 10–11 s.

---

## 12. Repository map

```
src/craftpilot/
  serve.py           FastAPI service: /health /plan /build /edit /regenerate /cancel /exemplars
  route.py           building or object
  cli.py             craftpilot build | object | object-undo | render | serve | catalog
  config.py          .env → SETTINGS
  program/           BuildProgram model, validation, palettes, exemplar retrieval, description
  llm/               Azure client, compose (text → program), offline fallback, prompts
  engine/            layout, massing, roof, attachments, facade, interior, depth, detail, materials, postprocess
  objects/           brief, imagegen, recon, voxelize, orient, place, pipeline, flow
  grid/              SemanticGrid, enums, rotation ops
  place/             anchor, bridge client, placer, undo
  terrain.py         survey, ground level, terraform
  preview/           isometric renderer, hologram cloud
  export/            litematic
  blocks/            block catalog (from the 26.2 jar), family colours, circle templates
exemplars/           13 complete BuildPrograms (few-shot examples, fallback, goldens)
data/blocks_26.2.json
mod/                 the Fabric mod (Java 25, Minecraft 26.2)
tools/               recon_server.py, recon_worker.py, .venv-3d (torch), triposr/, hunyuan3d/ (clones, ignored)
tests/               168 tests
docs/                this file, demo notes
```
