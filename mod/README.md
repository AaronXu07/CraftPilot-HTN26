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
| `/build <text>` | Compose, generate, and place the building 2 blocks in front of you, facing you. |
| `/build again [seed]` | Regenerate the last program (new or given seed) and place it. |
| `/build edit <text>` | Patch the last program ("make it taller", "red roof") and place the result. |
| `/build preview <text>` | Generate and write the `.litematic` only. |
| `/build cancel` | Drop everything still queued for placement. |
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
| `GET /setblocks/status` | – | `{pending_chunks, pending_blocks, placed_total, postprocessed}` |
| `POST /setblocks/cancel` | – | `{ok, cleared_chunks}` |
| `POST /scan` | `{min:[x,y,z], max:[x,y,z]}` inclusive | `{palette:[...], blocks:[[x,y,z,paletteIdx],...], count}` |
| `POST /say` | `{text}` | `{ok}` |
| `GET /blocks` | – | every registered block with its properties and default state |
| `POST /camera` | `{mode:"orbit", center, radius, seconds}` / `{mode:"return"}` | `{ok}` |

`/setblocks` semantics:

- Each chunk is placed in a single tick, then the placer idles `ceil(delay_ms / 50)` ticks. Chunks from
  later requests append to the same queue.
- `flags` is passed to `setBlockState`. Default `3` (`NOTIFY_ALL`). CraftPilot sends `18`
  (`NOTIFY_LISTENERS | FORCE_STATE`): the state is written verbatim with no neighbour updates, because
  the engine already bakes stair shapes, door hinges, fence connections and lantern support into every
  state.
- `postprocess` (default `true`) runs `Block.postProcessState` on every placed block after the chunk so
  connectivity is recomputed for callers that send bare states. CraftPilot sends `false`.
- Unparseable states are dropped and counted in `invalid`; the request never fails because of them.

Smoke test from the repo root with the game open: `uv run python scripts/verify_bridge.py`.
