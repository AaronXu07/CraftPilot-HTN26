# CraftPilot Bridge (Fabric mod)

A thin client-side Fabric mod that lets `craftpilot serve` place generated buildings straight into the
world you have open. It does two things:

1. Hosts a small HTTP server on `http://127.0.0.1:7777` that the Python service pushes blocks through.
   Blocks are queued and placed **one chunk per game tick**, which is what makes a build rise layer by
   layer.
2. Adds the `/build` chat command, which forwards to the Python service on `http://127.0.0.1:7778`.

No build logic lives here. All world access happens on the integrated server thread; the mod needs a
singleplayer world (it uses the integrated server).

## Build and install

Requires Java 21. Versions in `gradle.properties` must match the game you run (currently **1.21.1**).

```sh
cd mod
./gradlew build            # build/libs/copilot-<version>.jar
```

Copy the jar plus the matching [Fabric API](https://modrinth.com/mod/fabric-api) into your `mods/`
folder, launch with the Fabric loader, open a singleplayer world. The log shows
`[craftpilot] bridge listening on http://127.0.0.1:7777`.

To target another Minecraft version: update `minecraft_version`, `yarn_mappings`, `fabric_version` from
https://fabricmc.net/develop, bump `depends.minecraft` in `src/main/resources/fabric.mod.json`, rebuild,
and regenerate the Python block catalog for that version (`uv run python scripts/gen_catalog.py <version>`
plus `CRAFTPILOT_MC_VERSION` in `.env`). Otherwise the game rejects blocks it does not know and the
service reports them as `invalid`.

## Chat commands

| Command | Does |
| --- | --- |
| `/build <text>` | A box follows you; walk or turn to aim it, press **G** to lock. The service composes and sends back a **hologram** of the actual building, shown where you locked. **G** builds it there (streamed in bottom-up, replacing the hologram row by row); **H** lets it follow you again first. Keys are rebindable under Controls > CraftPilot. |
| `/build plan <text>` | Compose only and print the plan in chat (size, parts, floors, roofs, materials, attachments, what could not be expressed). Nothing is built yet. |
| `/build edit <text>` | With a plan pending: refine it ("make it three floors") and reprint it. Otherwise: patch the last program and show the edited hologram (G / H as above). |
| `/build go` | Show the pending plan as a hologram at your position; G builds it, H moves it. |
| `/build again [seed]` | Regenerate the last program (new or given seed) as a hologram; G builds it, H moves it. |
| `/build preview <text>` | Generate and write the `.litematic` only. |
| `/build cancel` | Drop the pending build or plan and everything still queued for placement. |
| `/build status` | Queue counters and the configured URLs. |

The service URL can be overridden with the JVM property `-Dcraftpilot.service=http://host:port`.

## HTTP bridge

All coordinates are absolute world coordinates; block states use command syntax
(`minecraft:oak_stairs[facing=north,half=bottom]`). Bodies are JSON; the method is not checked.

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /health` | – | `{ok, mod_version, mc_version, world_loaded, player, pending_chunks}` |
| `GET /player` | – | `{name, pos:[x,y,z], yaw, pitch, facing, looking_at, dimension}` |
| `POST /setblocks` | `{chunks:[{blocks:[[x,y,z,"state"],...], delay_ms}], flags?, postprocess?}` or `{blocks:[...]}` | `{queued, chunks, invalid, invalid_samples?}` (async) |
| `GET /setblocks/status` | – | `{pending_chunks, pending_blocks, placed_total, skipped_total, postprocessed}` |
| `POST /setblocks/cancel` | – | `{ok, cleared_chunks}` |
| `POST /scan` | `{min:[x,y,z], max:[x,y,z]}` inclusive | `{palette:[...], blocks:[[x,y,z,paletteIdx],...], count}` |
| `POST /heightmap` | `{min:[x,z], max:[x,z]}` inclusive, ≤ 1M columns | `{min, max, palette:[...], heights:[y,...], tops:[paletteIdx,...]}` row-major (z outer, x inner): the y of the top motion-blocking non-leaf block per column (water surfaces count) and that block's state; `null` for an empty column. The service surveys the build site with this before placing |
| `POST /say` | `{text}` | `{ok}` |
| `GET /blocks` | – | every registered block with its properties and default state |
| `POST /camera` | `{mode:"orbit", center, radius, seconds}` / `{mode:"return"}` | `{ok}` |
| `POST /outline` | `{min:[x,y,z], max:[x,y,z], phase:"generating"\|"placing"}` / `{clear:true}` | `{ok}` |

`/outline` draws the in-progress build box (`BuildOutline`): `/build` shows a placeholder guess at once, the
service replaces it with the exact box, a scan line tracks the placed rows, and the mod clears it when the
placement queue drains, on `/build cancel`, or on a service error.

The hologram (`GhostRender`) is not a bridge endpoint: the service's `/build`, `/edit` and `/regenerate`
accept `ghost: true` and answer with `{ghost: {width, height, depth, blocks: [x, y, z, rgb, ...]}, seed}`
instead of placing. The mod draws the cloud as translucent coloured cubes (outer faces only), rotated to
face the player with the engine's own quarter-turn rule, and commits with `/regenerate {seed, place}` so
the build is the preview block for block.

`/setblocks` semantics:

- Each chunk is placed in a single tick, then the placer idles `ceil(delay_ms / 50)` ticks. Chunks from
  later requests append to the same queue. A block whose state already matches is skipped (no lighting
  update, packet or chunk rebuild), which is most of the air around a building; `skipped_total` counts them.
  The service sizes chunks so a build lands in about `CRAFTPILOT_PLACE_SECONDS` (default 6), between
  `CRAFTPILOT_PLACE_CHUNK` and `CRAFTPILOT_PLACE_CHUNK_MAX` blocks per tick.
- `flags` is passed to `setBlockState`. Default `3` (`NOTIFY_ALL`). CraftPilot sends `18`
  (`NOTIFY_LISTENERS | FORCE_STATE`): the state is written verbatim with no neighbour updates, because
  the engine already bakes stair shapes, door hinges, fence connections and lantern support into every
  state.
- `postprocess` (default `true`) runs `Block.postProcessState` on every placed block after the chunk so
  connectivity is recomputed for callers that send bare states. CraftPilot sends `false`.
- Unparseable states are dropped and counted in `invalid`; the request never fails because of them.

Smoke test from the repo root with the game open: `uv run python scripts/verify_bridge.py`.
