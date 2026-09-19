# Overnight work order — Minecraft Copilot

You are running unattended, in a fresh session, on branch `overnight/*`. Nobody will answer questions. Your job each session: read `OVERNIGHT_PROGRESS.md`, pick the **first task below that is not marked `[done]` or `[blocked]`**, complete it, verify it, commit it, mark it, and stop. One task per session. If every task is `[done]` or `[blocked]`, append the line `ALL_TASKS_DONE` to `OVERNIGHT_PROGRESS.md` and stop.

## Rules (non-negotiable)

- Never run the Minecraft client or Gradle `runClient`. `./gradlew build` for the mod is fine. Anything that needs the game goes into `MORNING_CHECKLIST.md` as a precise "type X, expect Y" line.
- Never edit `.env`, never print secrets, never `git push`, never touch `main`, never rewrite history.
- Before committing: `cd agent && pytest -q` must be green, `ruff check .` clean. If you cannot make them green within your session, revert your changes (`git checkout -- .`), record what you tried under the task in `OVERNIGHT_PROGRESS.md`, and mark it `[blocked: reason]`.
- Commit message format: `overnight: <task id> — <what changed>`. One commit per task (squash your own intermediate commits if you made any).
- Update `OVERNIGHT_PROGRESS.md` under the task with: what you changed (files), how you verified it, numbers before/after (latency, bench score), and anything the morning person should know.
- The bench costs real API credits. Use `python bench/run.py --quick` (5 prompts) while iterating; run the full 20 only once, when you believe a quality change is ready, and record the mean score.
- Do not start a task with fewer than ~40 turns of budget left; instead write notes and stop.

## Tasks (priority order)

### T1 — Async chat; fix `HttpTimeoutException: request timed out`
Root cause: the mod's `/chat` HTTP request waits for the whole agent run and the Java `HttpClient` times out. Make chat asynchronous:
- Agent: `POST /chat` returns `{"job_id": ...}` within 1 s and runs the pipeline in a background task. Progress and the final answer are pushed to the mod via `POST /say`. Add `GET /jobs/{id}` (status, elapsed, stage) and `POST /jobs/{id}/cancel`.
- Mod: `/cp` posts and prints `[cp] working (job 3)…` immediately; request timeout 15 s; `/cp cancel` and `/cp status` commands. Never block on a long request again.
- Agent-side timeouts: every Azure call has a 120 s timeout with one retry; every bridge call 30 s; a stuck job is killed at the hard budget (T2) and reports `[cp] stopped: over time budget`.
- Tests: FastAPI TestClient covers job lifecycle with a fake pipeline. Mod: `./gradlew build` passes; add the in-game check to `MORNING_CHECKLIST.md`.
Done when: tests green, `./gradlew build` green, progress notes written.

### T2 — Latency: hard cap 5 min, target ≤ 2 min for a medium build
Instrument first, then cut. Add per-stage timing (`runs/<session>/<turn>.jsonl` already exists; add a summary line: stage, tool calls, LLM ms, engine ms, wall ms) and a `bench/run.py --profile` mode that prints a latency table.
Then implement, in this order, measuring after each:
1. **Wall-clock budget** passed through the pipeline: total 300 s hard; default plan 150 s: interpret 10, blocking 35, detailing 45, materials 30, decoration 20, critic 10. A stage that exceeds its budget ends after its current tool call; critic rounds are skipped when over budget; decoration is skipped entirely when the remaining budget < 40 s.
2. **Bulk ops via `run_script`**: the stage prompts must instruct the model to create geometry in one script per stage rather than dozens of single tool calls; cap individual op calls at 12 per stage.
3. **Parallel tool calls** enabled and honored by the dispatcher (execute independent calls concurrently).
4. **Cheaper calls where quality doesn't need Opus-class output**: router, describe, and lint-fix passes use `MODEL_FAST` from `.env` if set (fall back to `MODEL`).
5. **Smaller vision payloads**: one 640 px contact sheet per critique, JPEG, never more than 2 images per call.
6. **Engine**: incremental re-raster on edits (only objects whose bbox intersects the change); cache renders by scene hash; profile `fit.py` and vectorize any per-voxel Python loop.
Done when: `bench/run.py --profile --quick` shows median wall time ≤ 150 s and max ≤ 300 s across the 5 quick prompts; numbers recorded before/after.

### T3 — Chat visibility, live preview, animated final build
- Progress lines with stage tags and elapsed time: `[cp·plan 0:04] L-shaped keep 20×14, four round towers r4 h24, gatehouse south`, `[cp·block 0:31] added 7 solids`, `[cp·detail 1:02] carved 14 windows, 2 arches; battlements on 4 walls`, `[cp·materials 1:30] stone_wall/slate_roof/spruce_trim`, `[cp·critic 1:41] 7/10 — fixing: blank east façade (rule 3)`. Max one line per event; no JSON in chat.
- **Live preview**: when `LIVE_PREVIEW=true` (default), place the scene in-game after blocking and after detailing using diff placement, so viewers watch it evolve. Materials and decoration stages place at the end.
- **Final placement animation**: bottom-up layer chunks with `animate=true` and a short chat line per 25% (`[cp·build] 50%`). Ensure the mod's chunked `setblocks` honors delays without blocking the server thread (schedule chunks as successive server-tick tasks, not `Thread.sleep`).
- `/cp verbose on|off` toggles per-op lines (off by default keeps the tagged summaries only).
- The final reply: 2–4 lines: what was built, dimensions, block count, time taken, and "say `undo` / `export` / an edit".
Done when: mock-bridge integration test asserts the sequence of `/say` messages and placement calls for a scripted pipeline run; in-game items added to `MORNING_CHECKLIST.md`.

### T4 — Build quality via the bench (A/B every change)
- Expand `bench/prompts.json` to 20 prompts covering: castle, cathedral, pagoda, lighthouse, bridge, modern villa, desert temple, treehouse, windmill, gothic tower, viking longhouse, greenhouse, aqueduct, watchtower on a wall, round library, market hall, ship, chapel with bell tower, hexagonal keep, ruined tower. Store a baseline full-bench run (`bench/results/baseline.json`) before changing anything.
- Then iterate on the highest-leverage levers, one change per commit, keeping only changes whose `--quick` mean score does not drop and whose full-bench mean improves:
  1. Interpret stage must output a silhouette plan with explicit proportions; reject briefs with towers wider than tall, roofs without overhang, footprints without a stated entrance.
  2. Detailing checklist: every façade > 8 blocks gets depth (pilasters or recessed windows) and rhythm; corners get quoins or a trim material; roofs get overhang ≥ 1 and a ridge/eave trim.
  3. Materials: enforce ≥ 3 materials and a ground gradient; add 10 presets with palettes tuned by eye in the renderer (render each preset on a test wall and check the PNGs yourself).
  4. Fitting: verify cone, dome, wedge and carved-arch goldens still hold; fix any facing/half errors; make sure `fit` runs on subtract surfaces.
  5. Critic: must name object ids and one concrete op per finding; cap 3; second round only if first round score < 7.
- Record baseline vs final mean and per-prompt scores in `OVERNIGHT_PROGRESS.md`; save the contact sheets of the 3 most-improved and 3 worst prompts to `bench/results/<stamp>/` for the morning.
Done when: full-bench mean improves over baseline with median latency still ≤ 150 s, or you've documented exactly which changes didn't help.

### T5 — Fix all bugs
- Run `pytest -q`, `ruff check .`, `mypy copilot --ignore-missing-imports`; fix everything.
- Grep for `TODO|FIXME|XXX|HACK|pass  # stub|NotImplementedError`; resolve or file each in `OVERNIGHT_PROGRESS.md` under "known gaps".
- Harden the tool dispatcher: a tool exception never crashes a job; it returns `{"error": "..."}` to the model and logs the traceback. Malformed tool arguments → clear error message naming the bad field.
- Bridge client: timeouts, retries (2× with backoff) on connection errors, and a clear `[cp] mod not connected` chat message if the mod is unreachable before starting a job.
- Session: `undo`/`redo`/`undo_world` robust when there is nothing to undo; `reset` clears world-diff state.
- Replay the last 10 logs in `runs/` (if present): for each failure or error line, reproduce with a test and fix.
- Mod: `./gradlew build`; review `WorldOps` for any world access off the server thread; review the HTTP server for unclosed streams and missing 503 when no world is loaded.
Done when: all three checkers clean, every reproduced bug has a test, notes list any bug you could not reproduce.

### T6 — More tests
Add: fitting goldens for cone/dome/wedge/arch/torus; resolver on every wood and stone family from the registry cache (`agent/.cache/blocks.json`); edit flows (`set_shape` → diff re-emits only the changed bbox; `move` of a group; `mirror_copy` symmetry); router mapping for 15 edit phrasings; job lifecycle; animation chunking. Target: `pytest -q` under 60 s.
Done when: all added tests pass and are listed in the notes.

### T7 — Nice-to-haves (only after T1–T6)
- `/cp variants 3` (parallel pipelines, placed at offsets, `keep N`).
- Cutaway render on request ("show me the inside").
- Materials list formatting with stacks (64s) for survival players.
- A 30-second demo script in `docs/demo.md` derived from the bench's best prompts.

## Morning handoff
Keep `MORNING_CHECKLIST.md` current: every item is "In-game: type `…`; expect `…`". Add a top section "Merge readiness: yes/no and why" in your final session.