# Contracts (locked at hour 1)

Every track codes against this file. If you must change a contract, change it here
first and grep for consumers. All coordinates: **x east, y up, z south**, units = blocks.
Facing: north = -z, south = +z, east = +x, west = -x.

## 1. Repository layout

```
agent/copilot/
  server.py           FastAPI  POST /chat {player, text} -> {job_id, status, reply?}; GET/POST /jobs/{id}[/cancel]
  llm.py              LLM protocol: AzureLLM + MockLLM (scripted), tool loop
  session.py          Session: scene, history, undo/redo, snapshots, brief, chat, world state
  bridge.py           Bridge protocol + HttpBridge (talks to the Fabric mod)
  placement.py        scene -> world: anchor, rotation snap + state remap, diff placement, animation, world undo
  pipeline/           interpret / blocking / detailing / materials / decoration / critic / router / orchestrator
  tools/schemas.py    TOOL_SCHEMAS (OpenAI function-calling JSON), one per op in §2.3 + §7.3 of plan.md
  tools/dispatch.py   dispatch(ctx, name, args) -> ToolResult
  prompts/*.md        system_core, stage_*, critic, router
  engine/
    coretypes.py      Bbox, Facing, RasterResult, FitResult, BlockMap, LintFinding  (WRITTEN)
    transform.py      rotation matrices, anchors, world<->local                     (WRITTEN)
    scene.py          Scene, SceneObject, ops, select, describe                     (WRITTEN)
    solids.py         make_solid(shape) -> Solid; SDF primitives                     Track 3
    modifiers.py      build_object_sdf(scene, obj) -> world-space sdf callable       Track 3
    raster.py         rasterize(scene) -> RasterResult; scene_sdf                    Track 3
    fit.py            fit_surface(raster, fit_modes) -> FitResult                    Track 3
    render.py         render_blocks(...), contact_sheet(...), ascii_slice(...)       Track 3
    diff.py           diff_block_maps, changed_bbox                                  Track 3
    registry.py       Registry: blocks, families, colors, search, nearest, catalog   Track 4
    materials.py      MaterialSpec, PRESETS, palette/gradient/noise evaluation       Track 4
    resolver.py       resolve(raster, fit, scene, registry, seed) -> BlockMap        Track 4
    lint.py           lint(scene, raster, fit, block_map, registry) -> findings      Track 4
    schematic.py      export_litematic(block_map, name, path); materials_list        Track 4
agent/mock_mod/       MockBridge: in-memory world implementing the Bridge protocol    Track 2
agent/tests/          pytest; pure engine tests + mock-LLM agent tests
agent/bench/          prompts.json + run.py (headless build + vision scoring)
mod/                  Fabric client mod (Java): 7 HTTP endpoints + /cp command        Track 1
scripts/              verify_bridge.py, calibrate_facing.py, gen_block_fallback.py, demo.py
```

Python: `agent/` is the working dir; run `cd agent && ../.venv/bin/pytest`.

## 2. Scene JSON  (engine/scene.py is the source of truth)

```json
{
  "name": "castle_v3",
  "materials": {"stone_wall": {"base": "stone_bricks", "palette": [["stone_bricks", 0.7], ["cracked_stone_bricks", 0.3]], "fit": "stairs+slab"}},
  "objects": [
    {"id": "keep", "shape": {"type": "box", "size": [14, 18, 14]},
     "transform": {"pos": [17, 0, 13], "rot": [0, 0, 0], "scale": [1, 1, 1], "anchor": "bottom_center"},
     "op": "add", "material": "stone_wall",
     "modifiers": [{"type": "shell", "thickness": 1}],
     "tags": ["keep"], "group": "outer_wall", "visible": true}
  ],
  "groups": {"outer_wall": {"parent": null}},
  "meta": {"style": "medieval", "brief": {}}
}
```

* `op` ∈ add | subtract | intersect | paint.  Objects are evaluated **in list order** (CSG stack).
* `anchor` ∈ bottom_center (default) | center | bottom_min | top_center | bottom_max.
* `rot` = Euler degrees [rx, ry, rz]; applied X then Y then Z (R = Rz·Ry·Rx). Any angle.
* Group transforms are **baked into members** (groups carry no live transform). `groups[id] = {"parent": id|null}`.
* `material` is a name in `scene.materials` (or a preset name; presets are resolved by materials.py) or an inline dict.
* `shape.type` is one of the 16 primitives in §3 below; params are validated by solids.py.

## 3. Shape params (solids.py)

| type | params (defaults) |
|---|---|
| box | size=[sx,sy,sz] |
| cylinder | radius, height, axis="y", radius_top=None |
| sphere | radius, half=False (half=True keeps y>=0 hemisphere) |
| ellipsoid | radii=[rx,ry,rz] |
| cone | radius, height, radius_top=0 |
| pyramid | base=[sx,sz], height, top=[0,0] |
| wedge | size=[sx,sy,sz], slope_axis="x" (height rises from -axis side to +axis side) |
| prism | sides, radius, height |
| torus | major, minor, axis="y" |
| capsule | radius, height (total incl. caps) |
| extrude | profile=[[x,z],...], height |
| revolve | profile=[[r,y],...] (closed polygon in r>=0) |
| sweep | profile radius `radius` (v1: circular), path=[[x,y,z],...], closed=False |
| plane_cut | normal=[nx,ny,nz], offset (keeps n·p <= offset; use op=intersect) |
| block | state="minecraft:lantern[hanging=true]" (single explicit block; 1x1x1 bbox) |
| line | from=[x,y,z], to=[x,y,z], thickness=1 |

Local space: every solid's `local_bbox()` returns (lo, hi) as np arrays. Solids sit on y=0
and are centred in x/z where that makes sense (box, cylinder, cone, pyramid, prism, wedge, extrude
uses profile coords as given, height along +y from 0). `sphere`, `ellipsoid`, `torus`, `capsule` are
centred at the local origin (anchor `bottom_center` still places their bottom at pos.y).

## 4. Modifiers (modifiers.py) — ordered stack per object

| type | params |
|---|---|
| shell | thickness (inward; f' = max(f, -(f+t))) |
| round | radius (shrink primitive by r then f - r) |
| array | count, offset=[dx,dy,dz] (world axes; instances 0..count-1) |
| mirror | axis ("x"/"y"/"z"), plane (world coord), keep_original=true |
| taper | top_scale (xz scale at local top relative to bottom) |
| twist | deg_per_block (rotation about local y as a function of local y) |
| noise_displace | amplitude, scale=4, seed=0 |
| boolean | target (object id), op ("union"/"subtract"/"intersect") |

`build_object_sdf(scene, obj) -> Callable[[np.ndarray[N,3] world], np.ndarray[N]]`.
Evaluation order: array/mirror expand the query points (world), then world->local, then
taper/twist warp local points, then base SDF (scaled by min(scale)), then shell/round/noise/boolean
in stack order.

## 5. Engine data types (engine/coretypes.py)

```python
Bbox(lo: np.ndarray[3], hi: np.ndarray[3])         # world, float, inclusive-exclusive when voxelised
Facing: NORTH=0 EAST=1 SOUTH=2 WEST=3               # stairs facing = direction of the FULL (high) side
Half:   BOTTOM=0 TOP=1
RasterResult:
  origin: (x0, y0, z0) ints           # world coord of voxel index [0,0,0]
  material: np.int16[X,Y,Z]           # 0 = air, i>0 -> material_names[i]
  material_names: list[str]           # [""] + names in first-seen order
  owner: np.int32[X,Y,Z]              # index into scene.objects (-1 = air)
  sdf: np.float32[X,Y,Z]              # CSG sdf at voxel centre (<= 0 occupied; +inf air)
  props: dict[(x,y,z), str]           # explicit block states from `block` solids (world coords)
  scene_sdf: Callable[[pts[N,3]], (f[N], owner[N])]   # full CSG evaluation at arbitrary world pts
  def world_to_index(x,y,z) / index_to_world(i,j,k)
FitResult:
  kind: np.uint8[X,Y,Z]               # 0 air, 1 full, 2 slab, 3 stairs, 4 wall
  facing: np.uint8[X,Y,Z]             # Facing for stairs
  half: np.uint8[X,Y,Z]               # Half for stairs / slab type (0 bottom, 1 top)
BlockMap = dict[tuple[int,int,int], str]     # world coords -> "minecraft:id[prop=val,...]" (no air)
LintFinding(rule: str, severity: "warn"|"error", message: str, objects: list[str])
```

Sub-voxel fitting rule (fit.py): sample `scene_sdf` at the 8 sub-centres (±0.25) of every exposed
occupied voxel; pattern table from plan §5.2. Material fit permission passed as
`fit_modes: dict[material_name -> "none"|"slab"|"stairs"|"stairs+slab"|"walls"]`.

## 6. Material spec (materials.py)

```json
{"base": "stone_bricks",
 "palette": [["stone_bricks", 0.72], ["cracked_stone_bricks", 0.18], ["mossy_stone_bricks", 0.10]],
 "gradient": {"axis": "y", "from": 0, "to": 6, "palette": [["cobblestone", 0.6], ["mossy_cobblestone", 0.4]]},
 "faces": {"top": "stone_brick_slab", "side": null, "bottom": null},
 "fit": "stairs+slab",
 "noise": {"scale": 3, "seed": 7}}
```
`base` is required (or derived from the first palette entry). Block ids may omit `minecraft:`.
`MaterialSpec.from_any(name_or_dict, scene_materials)` resolves presets and inline dicts.
Family forms come from `registry.family(base)`: full, stairs, slab, wall, fence, ... .

## 7. Bridge protocol (mod <-> agent), JSON over HTTP, mod listens on 127.0.0.1:7777

| endpoint | request | response |
|---|---|---|
| GET /health | – | `{"ok":true,"mod_version":"0.1.0","mc_version":"1.21.1","client_jar":"/abs/path/1.21.1.jar","world_loaded":true,"player":"Steve"}` |
| GET /player | – | `{"name":"Steve","pos":[x,y,z] (floats),"yaw":f,"pitch":f,"facing":"north|east|south|west","looking_at":{"pos":[x,y,z],"block":"minecraft:stone"}|null,"dimension":"minecraft:overworld"}` |
| POST /scan | `{"min":[x,y,z],"max":[x,y,z]}` (inclusive) | `{"palette":["minecraft:air","minecraft:stone[...]"],"blocks":[[x,y,z,paletteIndex],...],"count":n}` air included via palette index 0 |
| POST /setblocks | `{"chunks":[{"blocks":[[x,y,z,"minecraft:stone"],...],"delay_ms":60}],"flags":3}` or `{"blocks":[...]}` | `{"queued":n,"chunks":k}` (placement runs asynchronously on the server thread, chunks spaced by delay_ms) |
| POST /say | `{"text":"..."}` | `{"ok":true}` (shows in the player's chat, prefixed `[copilot]`) |
| GET /blocks | – | `{"blocks":[{"id":"minecraft:oak_stairs","properties":{"facing":["north","south","west","east"],"half":["top","bottom"],"shape":[...],"waterlogged":["true","false"]},"default":"minecraft:oak_stairs[facing=north,half=bottom,shape=straight,waterlogged=false]"},...]}` |
| POST /camera | `{"mode":"orbit","center":[x,y,z],"radius":r,"seconds":n}` or `{"mode":"return"}` | `{"ok":true}` |

Chat: `/cp <text>` in game → mod POSTs `http://127.0.0.1:8000/chat {"player":name,"text":text}` →
agent answers **within ~1 s** with `{"job_id": n, "status": "queued|running|done|failed|cancelled|timeout",
"reply": str|null, "brief": ...}`. The turn runs as a background job: if it finished inside the inline window
(short commands such as `undo`, `help`, `status`) `reply` is set and the mod prints it; otherwise `reply` is
null, the mod prints `[cp] working (job n)…` and the final answer (or `[cp] stopped: <reason>`) arrives via
`/say`. `GET /jobs/{id}` → `{job_id, status, stage, elapsed_s, remaining_s, reply, error, line}`;
`POST /jobs/{id}/cancel` asks the job to stop after its current LLM/tool call. One active job per player:
a second `/chat` while busy returns `{busy: true, job_id, reply: "[cp] still working…"}`; the texts `cancel`
and `status` sent while busy act on the running job. Hard budget `COPILOT_HARD_BUDGET_S` (300 s) — a job
over budget is closed with `[cp] stopped: over time budget`. Mod HTTP requests time out after 15 s.

The python `Bridge` protocol (bridge.py) mirrors this 1:1:
`health()`, `player()`, `scan(lo, hi) -> BlockMap (includes air as "minecraft:air")`,
`setblocks(chunks: list[tuple[list[tuple[x,y,z,state]], delay_ms]], flags=3) -> int`,
`say(text)`, `blocks() -> list[dict]`, `camera(**kw)`. `MockBridge` (mock_mod) keeps a dict world,
a fake player at (0, 64, 0) facing north, and returns a fallback block dump.

## 8. Tools (LLM-facing) — names are fixed

Scene ops: add, delete, duplicate, rename, move, move_to, rotate, scale, align, stack, mirror_copy,
set_shape, set_op, set_material, set_anchor, set_visible, reorder, add_modifier, remove_modifier, set_modifier,
group, ungroup, select, describe, bbox, measure, top_of, side_of, define_material, list_materials,
paint, undo, redo, snapshot, restore.
Agent tools: run_script, render, lint, place, undo_world, export_schematic, materials_list,
get_player, say, search_blocks, nearest_block, set_brief, finish.

`dispatch(ctx, name, args) -> ToolResult(text: str, images: list[PIL.Image.Image] = [], data: dict = {})`.
`ToolContext(session, bridge, registry, log)`; engine outputs (raster/fit/block_map) are cached on the
session keyed by scene hash. Each op result text is one line + lint warnings (if any).

## 9. Outline format (describe) — see scene.py `describe()`

```
scene castle_v3  bbox 0..48 x 0..30 y 0..40 z  (11 objects, 3 groups, 5 materials)
[outer_wall]
  keep          box 14x18x14 @ (17,0,13) mat stone_wall  mods: shell(1)
  tower_ne      cylinder r4.5 h22 @ (34,0,4) mat stone_wall  mods: shell(1) round(0.5)
  gate_cut      box 4x6x3 @ (24,0,-1) SUBTRACT
```

## 10. Session API (session.py)

```python
s = Session(player)
s.apply(op_name, **kwargs) -> str      # applies scene op, records history, returns one-line message
s.undo(n=1) / s.redo(n=1) / s.snapshot(label) / s.restore(label)
s.scene, s.brief, s.chat (list of {"role","content"}), s.world (WorldState), s.cache (dict)
s.scene_hash() -> str
```

## 11. Integration notes (deviations accepted at integration time)

* **fit mutates raster.** `fit_surface` updates `raster.material/owner/sdf` in place so that afterwards
  `(raster.material > 0) == (fit.kind > 0)`; voxels that fitting adds (bottom slabs/stairs on gentle slopes)
  inherit the material of the neighbouring occupied voxel. Air candidates only ever gain slabs/stairs.
* **Stair facing** = direction of the FULL (high) side = up-slope. On a cone roof the stairs face the axis,
  as hand-built roofs do. `engine/fit.py: FACING_CALIBRATION` documents it; `scripts/calibrate_facing.py`
  verifies it against a live world (or the mock).
* **Shapes accept aliases**: `width/height/depth(length)` on box/wedge map onto `size`, `diameter` → radius,
  `size` on pyramid/ellipsoid. Unknown shape params raise (no silent no-ops).
* **Registry**: `validate_state(state, fill_defaults=True)`; the resolver emits minimal states (only the
  properties it sets) to keep `/setblocks` payloads small. `len(registry)` = block count.
* **Materials**: `validate_material_spec(spec, registry=None)`; a material may be `{"preset": name, ...overrides}`
  or a bare block id. Gradients blend outward from `[from, to]` so `from=0` always yields a full ground row.
* **run_script** runs in a subprocess (10 s, restricted builtins, no I/O); the parent replays the op log through
  `Session.apply`, so each scripted op is individually undoable.
* **Placement**: scene +z is the front and is rotated to face the player; anchor = 2 blocks in front of the
  player, laterally centred, ground = floor(player.y). `undo_world` restores only touched positions from the
  pre-placement scan (`session.world.touched`).
* **Mod**: after each `/setblocks` chunk the mod runs `Block.postProcessState` on every placed position so the
  placed block's own stair `shape` / fence / wall / pane connections are recomputed (flag 3 alone only updates
  neighbours).
* **Mock LLM**: `COPILOT_LLM=mock` uses `bench/mock_builder.ScriptedBuilderLLM` (role-aware) so the whole
  pipeline runs without credentials; it always builds the same scripted hall regardless of the prompt.
