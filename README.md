# CraftPilot

Procedural Minecraft buildings from natural language. A description is composed into a
`BuildProgram` by Azure OpenAI, a deterministic engine renders it as blocks using Minecraft
build techniques (palettes, gradients, texturing, depth, silhouette), and the result is either
placed straight into your open world through a small Fabric mod or written as a Litematica
schematic. See `PLAN.md` for the design.

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

The `mod/` folder is a client-side Fabric mod (Minecraft **1.21.1**) that lets the Python side place
blocks directly in the world you have open, bottom up, one layer per game tick. See `mod/README.md`.

1. Build and install the mod: `cd mod && ./gradlew build` (Java 21), then copy
   `mod/build/libs/copilot-<version>.jar` and the matching Fabric API into your `mods/` folder and open a
   singleplayer world. The mod listens on `127.0.0.1:7777`.
2. Start the service in a terminal: `uv run craftpilot serve` (listens on `127.0.0.1:7778`).
3. Either type in chat:

   ```
   /build a cozy two storey cottage with a stone chimney
   /build again 7            # same program, new seed
   /build edit make the roof red
   /build preview <text>     # schematic only
   /build cancel             # stop a placement that is still streaming in
   ```

   or from the terminal (no service needed):

   ```sh
   uv run craftpilot build "a stone lighthouse" --place
   ```

The building lands two blocks in front of you, centred, with its door facing you and its plinth at your
feet. Terrain inside the bounding box is cleared. Placement flags: `--gap`, `--sink`, `--no-clear`,
`--delay-ms`, `--chunk`; `craftpilot place-status` and `craftpilot place-cancel` talk to the mod directly.
The `.litematic` is still written every time.

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
uv run python scripts/gen_catalog.py 26.2   # regenerate data/blocks_26.2.json from the game jar
uv run craftpilot catalog           # print the block families the LLM sees
```

Exemplars live in `exemplars/*.yaml` as complete programs. They are the few-shot examples for the
LLM, the offline fallback, and the gallery: cottage, mansion, castle, modern house, church, barn,
lighthouse, pagoda, watchtower, townhouse.

## Layout

- `src/craftpilot/program/` the BuildProgram vocabulary, validation, palette rules, and exemplar library
- `src/craftpilot/engine/` layout, massing, roof, attachments, facade, interior, depth, detail, materials, postprocess
- `src/craftpilot/blocks/` block family catalog (filtered by `data/blocks_<version>.json`); each family carries its colour, texture noise, material and style tags from `palette_data.py`, which also holds the curated palette library
- `src/craftpilot/llm/` Azure OpenAI compose/edit, strict schema, offline fallback
- `src/craftpilot/export/` litemapy export
- `src/craftpilot/place/` in-game placement: bridge client, anchoring in front of the player, layer chunking
- `mod/` the Fabric bridge mod (`/build` command, `/setblocks` HTTP endpoint)
- `src/craftpilot/preview/` isometric PNG renderer
- `exemplars/` complete programs used as few-shot examples, fallback, and goldens
