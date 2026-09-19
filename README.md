# CraftPilot — Minecraft Copilot (HTN26)

A CAD-style agentic build engine for Minecraft. You chat in-game (`/cp build a castle with four towers`),
and an agent designs the build the way a Blender user would: parametric solids, booleans, modifier stacks,
material rules, and a vision critic that reviews rendered contact sheets between stages.
No templates, no retrieval, no external 3D APIs. Every block in Minecraft is a material.

The full design is in [plan.md](plan.md); the interfaces every track builds against are in
[CONTRACTS.md](CONTRACTS.md).

```
Player chat  ──/cp──▶  Fabric mod (thin bridge, 7 HTTP endpoints)   mod/
                             ▲            │
                    setblocks│            │ scan / player / blocks / say / camera
                             │            ▼
                      Python agent  ◀──────────▶  Azure OpenAI (chat + vision)   agent/
                       ├─ session (scene document, history, undo stack)
                       ├─ pipeline (interpret → block → detail → materials → decorate → critic)
                       ├─ tools (scene ops, scripts, render, place, export)
                       └─ engine (scene · SDF solids · modifiers · raster · sub-voxel fit ·
                                  materials · resolver · registry · render · lint · diff · schematic)
```

## What is here

| Path | What |
|---|---|
| `agent/copilot/engine/` | Pure-Python geometry + materials engine (numpy/PIL). No game, no LLM needed. 16 SDF primitives, 8 modifiers, CSG rasterizer with per-object cache, sub-voxel stairs/slab fitting, material rules (palettes, gradients, per-face, coherent noise), registry-validated block states, software renderer, design linter, diff engine, `.litematic` export. |
| `agent/copilot/pipeline/` | Interpret → blocking → detailing → materials → decoration → critic; router for conversational edits; orchestrator (`handle_chat`). |
| `agent/copilot/tools/` | 48 LLM tools (strict JSON schemas) + dispatch, `run_script` sandbox. |
| `agent/copilot/prompts/` | System, stage, critic and router prompts (design principles P1–P8, worked op sequences). |
| `agent/copilot/server.py` | FastAPI `POST /chat` that the mod talks to. |
| `agent/mock_mod/` | In-memory fake of the mod; `python -m mock_mod` serves the same 7 endpoints over HTTP. |
| `agent/bench/` | 10 prompts + headless build/score harness (`--every 4` re-runs every 4 h). |
| `agent/tests/` | 213 pytest tests: engine goldens, tools, placement, bridge, mock-LLM pipeline. |
| `mod/` | Fabric client mod (Java 21, Minecraft 1.21.1, Fabric API 0.116): `/cp` + 7 endpoints. Prebuilt jar in `mod/build/libs/` after `./gradlew build`. |
| `scripts/` | `demo.py` (headless castle), `verify_bridge.py`, `calibrate_facing.py`, `gen_block_fallback.py`. |

## Quick start (no game, no LLM)

```bash
python3 -m venv .venv && .venv/bin/pip install -e "agent[dev]"
cd agent
../.venv/bin/pytest -q                       # ~225 tests, ~4 s
../.venv/bin/python ../scripts/demo.py       # scripted castle → out/demo_contact.png, diff placement into the mock
COPILOT_LLM=mock ../.venv/bin/python -m copilot.server --port 8000 --bridge mock   # whole pipeline, scripted LLM
curl -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"player":"me","text":"build a hall"}'   # → {"job_id":1,...}; reply via /say, or curl localhost:8000/jobs/1
../.venv/bin/python -m bench.run --mock --fast     # bench smoke run → bench/out/<ts>/report.md
../.venv/bin/python -m bench.run --quick --profile  # 5 prompts against Azure + per-stage latency table
```

## Running the real thing

1. Copy `.env.example` to `.env` and fill in the Azure OpenAI endpoint, key and deployments.
   A vision-capable deployment is needed for the critic; without it the critic falls back to ASCII slices + lint.
2. Build/install the mod (see `mod/README.md`; needs JDK 21), start Minecraft 1.21.1 with Fabric + Fabric API, open a world with cheats on.
3. `cd agent && ../.venv/bin/python -m copilot.server --port 8000 --bridge http`
4. Once per world: `../.venv/bin/python ../scripts/verify_bridge.py` and `../.venv/bin/python ../scripts/calibrate_facing.py`
   (confirms the stairs/slab facing table against the game).
5. In game:
   * `/cp build a castle with an L-shaped keep, four round towers and a gatehouse facing me`
   * `/cp make the northeast tower 8 blocks taller and give it a copper roof`
   * `/cp cut an arched bridge between the two north towers`
   * `/cp swap the walls to deepslate with a mossy base`
   * `/cp add lanterns along the walls and a banner over the gate`, then `/cp export for survival`
   * `/cp undo`, `/cp preview off` (live preview after blocking/detailing is on by default), `/cp verbose on` (one line per op), `/cp help`

## How a build happens

1. **Interpret** turns the request into a brief (type, style, footprint, height, silhouette plan) and says it in chat
   (`[cp·plan 0:04] L-shaped keep 20×14, four round towers…`; every later stage, critic round and the final
   placement print one tagged line like that — `[cp·block 0:31] added 7 solids`, `[cp·critic 1:41] 7/10 — fixing: …`,
   `[cp·build 50%]`).
2. **Blocking** places primary solids (`add`, `extrude`, `stack`, `mirror_copy`, `array`) and the critic checks silhouette on iso + front renders.
3. **Detailing** carves windows/doors/arches (`subtract`), adds battlements, buttresses, roofs (`wedge`/`pyramid`/`cone`).
4. **Materials** defines palettes/gradients/per-face rules and sets `fit` so roofs and curves come out as stairs and slabs.
5. **Decoration** adds props (`block` solids: lanterns, banners, doors) and a final critique with up to two fix rounds.
6. **Place** sends only the diff to the game in bottom-up animated chunks; every later edit re-sends only what changed.

## Coordinates and conventions

x east, y up, z south; units are blocks; the scene origin is ground level at the build anchor and the
build's front faces +z (rotated toward the player at placement). Stairs `facing` is the direction of the
full (high) side. Centred anchors snap to whole voxels so a 9-wide tower at x=34 fills exactly x 30..38.

## Status / known gaps

* The Azure path is exercised only by unit tests (no credentials on this machine); the mock LLM builds the
  same scripted hall for every prompt.
* The mod compiled against the real 1.21.1 API but has not been run in a live game here; `/camera` orbit is best effort.
* The block registry fallback is pattern-generated (1,060 blocks); a live `/blocks` dump replaces it automatically.
* Sweep profiles are circular only; non-uniform scale distances are approximate (signs exact).
