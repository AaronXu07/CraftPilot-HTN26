# CraftPilot

Procedural Minecraft buildings from natural language. A description is composed into a
`BuildProgram` by Azure OpenAI, a deterministic engine renders it as blocks using Minecraft
build techniques (palettes, gradients, texturing, depth, silhouette), and the result is either
placed straight into your open world through a small Fabric mod or written as a Litematica
schematic. See `PLAN.md` for the design.

## Repository map

Two subsystems share this repo and the same Azure resource:

| Directory | What | Entry point |
|---|---|---|
| `src/craftpilot/` | **Buildings** — LLM composes a `BuildProgram`, a deterministic engine renders it, litemapy exports a schematic (this README, `PLAN.md`) | `uv run craftpilot build "…"` / `uv run craftpilot serve` |
| `agent/` + `mod/` | **Live copilot** — Fabric mod `/cp …` chat, async jobs, live animated placement, CAD-style scene with undo/edits; the object (statue/vehicle/prop) path lives here | `agent/README.md`, `agent/plan.md` |

## Setup

```sh
cd ~/projects/craftpilot
cp .env.example .env        # fill in the Azure OpenAI values
uv sync
```

## Build something

```sh
uv run craftpilot build "a cozy two storey cottage with a stone chimney" --preview
```

This writes `<label>_<seed>_<id>.litematic` (plus a `.png` preview and the composed
`.program.json`) into `~/Library/Application Support/minecraft/schematics/craftpilot/`, which
Litematica reads directly.

Useful flags:

| Flag                   | Meaning                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| `--bounds 25x30x25`    | Force a bounding box (width x height x depth). Otherwise the program's own suggestion is used. |
| `--seed 7`             | Reproducible randomness. Same text + seed = same building.                                     |
| `--no-llm`             | Skip Azure and use the nearest exemplar, patched by keywords.                                  |
| `--exemplar castle`    | Render an exemplar program directly (see `exemplars/`).                                        |
| `--facing east`        | Which way the front door faces (default south).                                                |
| `--show-program`       | Print the composed program JSON.                                                               |
| `--out path.litematic` | Write somewhere else.                                                                          |

Render every exemplar to `out/exemplars/` for a quick gallery:

```sh
uv run craftpilot render
```

## Build in-game

The `mod/` folder is a client-side Fabric mod (Minecraft **26.2**) that lets the Python side place
blocks directly in the world you have open, bottom up, one layer per game tick. See `mod/README.md`.

1. Build and install the mod: `cd mod && ./gradlew build` (run Gradle with a JDK 21–25; it fetches a JDK 25 toolchain itself), then copy
   `mod/build/libs/copilot-<version>.jar` and the matching Fabric API into your `mods/` folder and open a
   singleplayer world. The mod listens on `127.0.0.1:7777`.
2. Start the service in a terminal: `uv run craftpilot serve` (listens on `127.0.0.1:7778`).
3. Either type in chat:

   ```
   /build a cozy two storey cottage with a stone chimney   # aim the box, G; a hologram appears, G builds it (H moves it)
   /build plan a timber church with a bell tower           # see the plan in chat first ...
   /build edit make it three floors                        # ... refine it ...
   /build go                                               # ... then aim and press G to build it
   /build again 7            # same program, new seed
   /build edit make the roof red
   /build preview <text>     # schematic only
   /build cancel             # abandon the pending build or stop a placement that is still streaming in
   ```

   or from the terminal (no service needed):

   ```sh
   uv run craftpilot build "a stone lighthouse" --place
   ```

The building lands two blocks in front of you, centred, with its door facing you. Placement flags:
`--gap`, `--sink`, `--no-clear`, `--delay-ms`, `--chunk`; `craftpilot place-status` and
`craftpilot place-cancel` talk to the mod directly. The `.litematic` is still written every time.

**Terrain.** On non-flat ground the service surveys the site first (the mod's `/heightmap`: per column
the ground under any tree or plant, water counting as ground, and the highest block standing on it) and
seats the plinth row a little below the median surface height under the building instead of at your
feet, sinking it further if the build would pass y=319. It then streams, with the building, the hill
cut out of the base — trees and plants removed whole, so nothing is left floating — a foundation down
to the terrain in the building's own foundation block (a solid plinth on gentle sites, a perimeter wall
+ pillars over a big drop), and the surrounding ground ramped one block per column and re-topped with
its own surface block. The chat summary reports the site (`terrain y 63..68 (range 5, fill); ground
y=67; fill 268, graded 453`). `CRAFTPILOT_PLACE_TERRAIN=0`
turns it off (plinth at your feet, bounding box cleared, as before); the hologram still previews at
your feet, the build settles onto the ground when you press G. Code: `src/craftpilot/terrain.py`.

The game version matters: the block catalog the engine draws from is `data/blocks_<version>.json`,
selected by `CRAFTPILOT_MC_VERSION` in `.env` (1.21.1 and 26.2 are shipped; generate others with
`scripts/gen_catalog.py`). If the game rejects a state the service reports it as `invalid`.
`uv run python scripts/verify_bridge.py` smoke-tests the bridge with the game open.

## Placing a schematic by hand

1. In game, open the Litematica menu (default key `M`), choose **Load Schematics**, open the
   `craftpilot` folder and load the file.
2. In creative, open **Schematic Placements**, select it, and use **Paste Schematic in World**
   (or the Litematica paste hotkey). The building faces south unless you passed `--facing`.

## Development

```sh
uv run pytest                       # unit tests + closure property test
uv run ruff check src tests
uv run python scripts/gen_catalog.py 26.2   # regenerate data/blocks_26.2.json from the game jars (block list, texture colours, noise)
uv run craftpilot catalog           # print the block families the LLM sees
```

Exemplars live in `exemplars/*.yaml` as complete programs. They are the few-shot examples for the
LLM, the offline fallback, and the gallery: cottage, mansion, castle, modern house, church, barn,
lighthouse, pagoda, watchtower, townhouse.

## Layout

- `src/craftpilot/program/` the BuildProgram vocabulary, validation, palette rules, and exemplar library
- `src/craftpilot/engine/` layout, massing, roof, attachments, facade, interior, depth, detail, materials, postprocess
- `src/craftpilot/blocks/` block catalog built from `data/blocks_<version>.json`: families by naming pattern, colour and noise measured from textures, a use class per block (structural, decorative, thematic, precious, functional, natural), style tags, and colour-first shape lookup; `palette_data.py` holds hand overrides and the curated palette library
- `src/craftpilot/llm/` Azure OpenAI compose/edit, strict schema, offline fallback
- `src/craftpilot/export/` litemapy export
- `src/craftpilot/terrain.py` seating a build on real terrain: ground level from the mod's survey, cut / fill / grading streamed with the building
- `src/craftpilot/place/` in-game placement: bridge client, anchoring in front of the player, layer chunking
- `mod/` the Fabric bridge mod (`/build` command, `/setblocks` HTTP endpoint)
- `src/craftpilot/preview/` isometric PNG renderer
- `exemplars/` complete programs used as few-shot examples, fallback, and goldens
