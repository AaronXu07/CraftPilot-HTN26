# CraftPilot

Procedural Minecraft buildings from natural language. A description is composed into a
`BuildProgram` by Azure OpenAI, a deterministic engine renders it as blocks using Minecraft
build techniques (palettes, gradients, texturing, depth, silhouette), and litemapy writes a
Litematica schematic. See `PLAN.md` for the design.

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

| Flag | Meaning |
|---|---|
| `--bounds 25x30x25` | Force a bounding box (width x height x depth). Otherwise the program's own suggestion is used. |
| `--seed 7` | Reproducible randomness. Same text + seed = same building. |
| `--no-llm` | Skip Azure and use the nearest exemplar, patched by keywords. |
| `--exemplar castle` | Render an exemplar program directly (see `exemplars/`). |
| `--facing east` | Which way the front door faces (default south). |
| `--show-program` | Print the composed program JSON. |
| `--out path.litematic` | Write somewhere else. |

Render every exemplar to `out/exemplars/` for a quick gallery:

```sh
uv run craftpilot render
```

## Placing it in Minecraft

1. In game, open the Litematica menu (default key `M`), choose **Load Schematics**, open the
   `craftpilot` folder and load the file.
2. In creative, open **Schematic Placements**, select it, and use **Paste Schematic in World**
   (or the Litematica paste hotkey). The building faces south unless you passed `--facing`.

The Fabric mod that does this automatically from a `/build` command is milestone 4 in `PLAN.md`.

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
- `src/craftpilot/blocks/` block family catalog (filtered by `data/blocks_<version>.json`), per-family colour, noise and style data with the curated palette library, and circle templates
- `src/craftpilot/llm/` Azure OpenAI compose/edit, strict schema, offline fallback
- `src/craftpilot/export/` litemapy export
- `src/craftpilot/preview/` isometric PNG renderer
- `exemplars/` complete programs used as few-shot examples, fallback, and goldens
