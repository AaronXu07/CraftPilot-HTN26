# Minecraft Copilot — Fabric bridge mod (Track 1)

A thin **client-side** Fabric mod for Minecraft **1.21.1**. It exposes the running game to the
Python agent over HTTP on `127.0.0.1:7777` (7 endpoints) and adds the `/cp <text>` chat command
that forwards requests to the agent. No build logic lives in Java (plan.md §10).

| | |
|---|---|
| Mod id | `copilot` (package `dev.craftpilot.copilot`) |
| Minecraft | 1.21.1 (yarn `1.21.1+build.3`) |
| Fabric Loader | 0.16.14 |
| Fabric API | 0.116.17+1.21.1 |
| Java | 21 (compile target and runtime) |
| Build | Gradle 8.10 wrapper, fabric-loom 1.8-SNAPSHOT |

## Build

The wrapper downloads Gradle, Loom, Minecraft and the mappings on first run (needs network).
**A JDK 21 is required** — this machine currently ships only JDK 27 and no system Gradle, so
point `JAVA_HOME` at a JDK 21 (e.g. a Temurin tarball extracted anywhere; no install needed):

```bash
cd mod
export JAVA_HOME=/path/to/jdk-21/Contents/Home     # macOS layout; on Linux: /path/to/jdk-21
./gradlew build --no-daemon
# -> build/libs/copilot-0.1.0.jar
```

Useful variations:

```bash
./gradlew runClient --no-daemon        # launch a dev client with the mod loaded
./gradlew genSources --no-daemon       # decompiled, mapped Minecraft sources for browsing
```

## Run

1. Install [Fabric Loader](https://fabricmc.net/use/installer/) for 1.21.1 in the launcher.
2. Drop `build/libs/copilot-0.1.0.jar` **and** the matching `fabric-api-0.116.x+1.21.1.jar`
   into `.minecraft/mods/`.
3. Start the game, open (or create) a **single-player world with cheats enabled** (needed for
   `/gamemode` during camera orbits; block placement itself does not need cheats).
4. Start the agent: `cd agent && ../.venv/bin/uvicorn copilot.server:app --port 8000`.
5. In chat: `/cp build a castle with four towers`.

The bridge logs `[copilot] bridge listening on http://127.0.0.1:7777` at startup. The agent URL
can be overridden with the JVM property `-Dcopilot.agent=http://host:port/chat`.

## `/cp` command

```
/cp                      prints usage
/cp <free text>          POSTs {"player": "<name>", "text": "<free text>"} to http://127.0.0.1:8000/chat
/cp status               GET  /jobs/{id} for the current job → "job 3: running, detailing, 1:32 elapsed"
/cp cancel               POST /jobs/{id}/cancel → the agent stops after its current step
```

The agent answers `/chat` within about a second with a job id. Short turns (`undo`, `help`, `status`
with no job running) carry the reply inline and it is printed at once; long builds print
`[cp] working (job 3)…` and the agent pushes progress and the final answer through the mod's `/say`
endpoint. Every request the mod makes times out after 15 s, so the client never blocks on a long
turn; if the agent is not running you get a one-line hint with the command to start it. When no
job is known, `status`/`cancel` are forwarded to the agent as ordinary chat.

## Endpoints (CONTRACTS.md §7)

All on `http://127.0.0.1:7777`, JSON in / JSON out. Errors are `{"ok": false, "error": "..."}`
with status 400 (bad request), 409 (no world/player loaded) or 500 (unexpected).

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | – | `{"ok":true,"mod_version":"0.1.0","mc_version":"1.21.1","client_jar":"/abs/path/1.21.1.jar"\|null,"world_loaded":bool,"player":"Steve"\|null,"pending_chunks":n}` |
| `GET /player` | – | `{"name","pos":[x,y,z],"yaw","pitch","facing":"north\|east\|south\|west","looking_at":{"pos":[x,y,z],"block":"minecraft:stone","side":"up"}\|null,"dimension":"minecraft:overworld"}` |
| `POST /scan` | `{"min":[x,y,z],"max":[x,y,z]}` (inclusive, ≤ 2M blocks) | `{"palette":["minecraft:air", ...],"blocks":[[x,y,z,paletteIndex],...],"count":n}` — index 0 is always air |
| `POST /setblocks` | `{"chunks":[{"blocks":[[x,y,z,"minecraft:stone"],...],"delay_ms":60}],"flags":3}` or `{"blocks":[...]}` | `{"queued":n,"chunks":k,"invalid":m,"invalid_samples":[...]}` — placement is asynchronous |
| `GET /setblocks/status` | – | `{"pending_chunks","pending_blocks","placed_total","postprocessed"}` — poll until `pending_chunks == 0` |
| `POST /say` | `{"text":"..."}` | `{"ok":true}` — printed as `[copilot] text` in the player's chat |
| `GET /blocks` | – | `{"blocks":[{"id":"minecraft:oak_stairs","properties":{"facing":["north",...],...},"default":"minecraft:oak_stairs[facing=north,...]"}],"count":n}` |
| `POST /camera` | `{"mode":"orbit","center":[x,y,z],"radius":30,"seconds":10}` or `{"mode":"return"}` | `{"ok":true,"seconds":10}` |

Notes:

- **Placement** runs on the integrated server thread. Each chunk is placed in one tick; chunks are
  spaced by `delay_ms` (rounded up to 50 ms ticks). Blocks are set with flags `3`
  (`Block.NOTIFY_ALL`), so the game recomputes stairs `shape`, fence/wall/pane connections and
  redstone itself — the agent never sets those properties.
- **Connectivity post-process**: after each chunk is set, every placed block is re-evaluated with
  `Block.postProcessState` and rewritten (`NOTIFY_LISTENERS | FORCE_STATE`, no cascade) if its own
  connectivity changed, because `setBlockState` only updates the *neighbours*; this fixes the last
  stair in a row (`shape`) and fences/walls/panes placed after their neighbours. The running count
  is `postprocessed` in `/setblocks/status`.
- **State strings** are parsed with the game's own `BlockArgumentParser`, so any id/property the
  game rejects is counted in `invalid` (with up to 5 samples) instead of failing the request.
  Ids without a namespace get `minecraft:` prepended.
- **Scan** returns the full inclusive volume including air, in a palette-compressed form.
- **Camera** is best effort: it sends `/gamemode spectator`, orbits the centre at
  `radius` (height ≈ 0.55·radius) while looking at it, then teleports the player back and
  restores the previous game mode. Needs cheats.

## Facing calibration (do this once, hour 6–12)

The fitting engine assumes the vanilla convention: a stair's `facing` is the direction of its
**full-height side** (a player facing north places `facing=north`, and the stairs rise toward
north); `half=top` is an upside-down stair; slabs use `type=bottom|top`. Never trust memory:
run `python scripts/calibrate_facing.py` with a world open. It places one stair per facing and
one slab per type via `/setblocks`, scans them back, and prints the lookup table the engine
uses. If the printed table differs from the assumption above, fix the table in
`agent/copilot/engine/fit.py` rather than the mod.

## Layout

```
mod/
  build.gradle, settings.gradle, gradle.properties, gradlew*, gradle/wrapper/
  src/main/resources/fabric.mod.json, assets/copilot/icon.png
  src/main/java/dev/craftpilot/copilot/
    CopilotClientMod.java   entrypoint: starts the bridge, registers /cp, hooks tick helpers
    HttpBridgeServer.java   the 7 endpoints (JDK HttpServer + Gson)
    WorldOps.java           state string round-trips, region scan, block registry dump
    BlockPlacer.java        tick-driven chunked placement queue (server thread)
    CameraOrbit.java        spectator orbit for demos (client thread)
    AgentChatClient.java    POST /chat to the agent from /cp
```
