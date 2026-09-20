# CraftPilot Architecture

CraftPilot turns a chat message in Minecraft into a finished build placed in the player's world. It is
three processes talking over localhost: a **Fabric mod** inside the game, a **Python service** that does
the generating, and (for statues and other objects) a **3D reconstruction worker** in its own environment.

```
 Minecraft (Java)                    Python service                        Cloud / local models
 ┌──────────────────────┐   HTTP    ┌──────────────────────────┐          ┌─────────────────────────┐
 │ Fabric mod           │ ───────▶  │ FastAPI  :7778           │ ───────▶ │ Azure OpenAI (GPT-5.4)  │
 │  /build chat command │           │  route.py  building?     │          │  compose / edit / brief │
 │  hologram + outline  │ ◀───────  │            object?       │ ───────▶ │ FLUX 1.1 pro (images)   │
 │  HTTP bridge :7777   │  blocks   │  ┌────────┐ ┌──────────┐ │          └─────────────────────────┘
 │  layer-per-tick      │           │  │engine/ │ │objects/  │ │ ───────▶ ┌─────────────────────────┐
 │  placement + undo    │           │  └────────┘ └──────────┘ │          │ TripoSR / Hunyuan3D     │
 └──────────────────────┘           │  litematic + PNG preview │          │ tools/.venv-3d  :7790   │
                                    └──────────────────────────┘          └─────────────────────────┘
```

## Stack at a glance

| Layer | Technology |
| --- | --- |
| Game integration | Java 25, Fabric Loader 0.19 + Fabric API, Minecraft 26.2 (1.21.1 catalog also shipped), Gradle + Loom |
| Service | Python 3.12, FastAPI + Uvicorn, Pydantic v2, Typer CLI, `uv` for packaging |
| LLMs | Azure OpenAI Responses API (`gpt-5.4-mini`) with strict structured outputs, via the `openai` SDK |
| Image generation | FLUX 1.1 pro on Azure AI Foundry (OpenAI images API) |
| Image → mesh | TripoSR or Hunyuan3D running locally on PyTorch (Apple MPS), isolated in `tools/.venv-3d` |
| Geometry | NumPy, SciPy (`cKDTree` colour matching), trimesh (voxelisation) |
| Export / preview | litemapy (`.litematic` for Litematica), Pillow (isometric PNG render) |
| Tests / tooling | pytest, Hypothesis (property tests), ruff, mypy |
| Development | Built with Claude Code |

## 1. The Fabric mod (`mod/`)

A client-side mod, `dev.craftpilot.copilot`, that is the only thing the player touches.

- `CopilotClientMod` registers the `/build` chat command (`plan`, `edit`, `go`, `again`, `preview`,
  `cancel`, `object|building` overrides) and the keybinds: **G** to lock/commit a build, **H** to move it.
- `ServiceClient` posts requests to the Python service.
- `HttpBridgeServer` runs a tiny `com.sun.net.httpserver` on `127.0.0.1:7777` so Python can drive the world:
  `/player`, `/heightmap`, `/scan`, `/setblocks`, `/setblocks/status|cancel`, `/blocks`, `/camera`, `/outline`, `/say`.
- `BlockPlacer` + `WorldOps` place blocks on the render thread, bottom-up, a chunk of rows per game tick,
  using command-syntax block states (`minecraft:oak_stairs[facing=north]`) written verbatim (no neighbour
  updates — the engine already bakes stair shapes, hinges and connections).
- `GhostRender` / `BuildOutline` / `CameraOrbit` draw the hologram preview, the animated wireframe box with a
  scan line tracking placed rows, and the orbiting camera.
- `PendingBuild` tracks the plan → edit → go flow so the player can iterate before committing.

## 2. The Python service (`src/craftpilot/`)

`serve.py` is a FastAPI app on `127.0.0.1:7778` with `/plan`, `/build`, `/edit`, `/regenerate`, `/cancel`,
`/exemplars`. `cli.py` (Typer) exposes the same pipelines from a terminal (`craftpilot build|object|serve|render`).

### Routing (`route.py`)

One chat command, two generators. `classify()` decides building vs object in three layers: an explicit
override, word rules (landmark list, head noun, shape and structural cues), and only for low-confidence
cases a small cached LLM call. Any LLM failure falls back to the rules, so routing never raises.

### Building path: LLM composes, engine renders

The model never places blocks. It writes a **`BuildProgram`** — a Pydantic schema (`program/model.py`)
describing massing, roof types, attachments, facade grammar, palette and materials — and a deterministic
engine turns that into voxels.

- `llm/` — Azure OpenAI client (`azure.py`) with a JSON-schema `response_format`, per-request logging and
  an on-disk cache; `compose.py` (text → program) and edit (program + instruction → program);
  `fallback.py` picks the nearest exemplar and patches it by keyword when no LLM is configured.
- `program/` — the schema, validation, 13 hand-written exemplar programs in `exemplars/*.yaml` used for
  retrieval-augmented prompting, palette rules (60-30-10 colour checks, 28 curated named palettes).
- `engine/` — the renderer pipeline: `layout` (side and stacked attachments) → `massing` (taper, jetty,
  thick walls) → `roof` (twelve roof types, filled bodies) → `facade` (six window styles, four framing
  modes, visibility-gated placement) → `interior` (stairs, ladders, partitions, flood-fill reachability
  with doorways carved where needed) → `depth` → `materials` / `noise` (gradients, weathering, texture)
  → `detail` (inset stairs/slabs, vines, bushes) → `postprocess`.
- `grid/` — `SemanticGrid`: a NumPy voxel grid carrying roles and shapes, not just block ids, so later
  passes can reason about walls vs roofs vs openings.
- `blocks/` — the block-state catalog generated from the game jar (`data/blocks_<version>.json`,
  `scripts/gen_catalog.py`) with colour, texture-noise, material and style tags the model sees in its
  prompt; every emitted state is validated against it.

### Object path: text → image → mesh → voxels (`objects/`)

For anything whose identity is a silhouette (statues, creatures, vehicles, landmarks).

1. `brief.py` — one small LLM call turns the request into an image prompt, target height, plinth and
   palette hint (regex fallback offline).
2. `imagegen.py` — FLUX 1.1 pro renders a three-quarter, white-background, "chunky toy-like" reference
   image tuned for single-image reconstruction.
3. `recon.py` — hands the image to the local worker (`tools/recon_server.py`, auto-started, persistent so
   model load is paid once) which runs TripoSR or Hunyuan3D and returns a mesh.
4. `voxelize.py` — trimesh rasterises the mesh (surface + filled interior) at the requested height, every
   voxel is coloured from its nearest vertex via a `cKDTree`, then snapped to the closest opaque full block
   in a curated palette. `orient.py` faces it toward the player; `flow.py` / `place.py` wire it into the same
   preview and placement flow as buildings.

Every stage writes its artefact (image, mesh, preview, schematic, timings) to a work directory for debugging.

### Placement on real terrain (`place/`, `terrain.py`)

- `place/anchor.py` — origin and facing from the player's position and yaw.
- `terrain.py` — asks the mod for a surface survey (`/heightmap`), sets ground level from the median
  under the footprint, then terraforms: cuts trees and terrain inside the box, builds a foundation plinth or
  pillar grid down to the ground, and grades an apron around it.
- `place/bridge.py` (httpx) + `place/placer.py` — stream world-space blocks to the mod in chunks sized to
  finish in ~6 s; `place/undo.py` snapshots what was there.

### Output (`export/`, `preview/`)

Every build is also written as a `.litematic` (litemapy) into the Litematica schematics folder, plus an
isometric PNG preview (Pillow) and the composed `.program.json`. `preview/ghost.py` produces the hologram
cloud the mod renders before the player commits.

## 3. The 3D worker (`tools/`)

TripoSR / Hunyuan3D need PyTorch, so they live in a separate Python 3.11 virtualenv (`tools/.venv-3d`).
`recon_server.py` is a small resident HTTP server on `:7790` (≈1 s inference + ≈3 s meshing once warm);
`recon_worker.py` is the one-shot subprocess fallback. `rembg` handles matting, `torchmcubes` the
marching cubes (patched to run on Apple MPS).

## 4. Testing

`tests/` covers the engine, routing, terrain, placement, the object path and the service with pytest;
`test_closure.py` is a Hypothesis property test asserting the engine always produces a closed, valid grid
for arbitrary programs. `scripts/verify_bridge.py` smoke-tests the mod bridge against a running game.

## Ports and config

| What | Where |
| --- | --- |
| Mod bridge | `127.0.0.1:7777` |
| Python service | `127.0.0.1:7778` (`CRAFTPILOT_PORT`) |
| Reconstruction server | `127.0.0.1:7790` (`CRAFTPILOT_RECON_PORT`) |
| Secrets / tuning | `.env` (see `.env.example`): Azure endpoint, deployments, MC version, placement knobs |
