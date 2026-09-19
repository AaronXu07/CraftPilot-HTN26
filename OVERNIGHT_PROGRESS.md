# Overnight progress

Started 20260919-0329 on branch overnight/20260919-0329

Note: iteration 1 ran on `overnight/20260919-0337`; a second `overnight.sh` start at 03:39 created
`overnight/20260919-0339` on top of it and swept the in-progress ruff config into its "start" commit.
All later work continues on `overnight/20260919-0339` (history was not rewritten).

## T1 — Async chat [done]

**What changed**
- `agent/copilot/jobs.py` (new): `Job` / `JobManager` — one chat turn per job on a 4-thread pool, one active
  job per player, cooperative cancellation (`check_cancel(ctx)` raises `JobCancelled`), watchdog that closes a
  job at the hard budget (`COPILOT_HARD_BUDGET_S`, default 300 s) and says `[cp] stopped: over time budget`
  even if the worker is stuck in a slow call; late results from a closed job are dropped.
- `agent/copilot/server.py`: `POST /chat` submits a job and waits ≤ 0.8 s (`COPILOT_CHAT_INLINE_WAIT_S`);
  short turns (undo/help/status) come back inline as `{job_id, status, reply}`, long ones return
  `reply: null` and the answer is pushed via bridge `/say`. `GET /jobs`, `GET /jobs/{id}` (status, stage,
  elapsed, remaining, `line`), `POST /jobs/{id}/cancel`. While a job runs, `/chat` with `cancel|stop`
  cancels it, `status` reports it, anything else is refused with `busy: true`. Fixed the turn/chat
  double-counting between server and orchestrator; fixed `session.save()` never running through the
  server (orchestrator used `log.log()` on the server's plain-callable logger, exception swallowed).
- Cancel checks: `llm.run_tool_loop` (before each LLM call and each tool call), `pipeline/common.call_tool`,
  `interpret`, `route`, `critique`, between stages in `run_build`; `tools/dispatch.dispatch` and
  `handle_chat` re-raise `JobCancelled` instead of turning it into an error reply. `ToolContext.job` added.
- Timeouts: `AzureLLM` timeout 120 s (`AZURE_OPENAI_TIMEOUT_S`), attempts 2 = one retry
  (`AZURE_OPENAI_MAX_ATTEMPTS`; was 6), SDK-internal retries disabled (`max_retries=0`), no sleep after the
  last attempt. `HttpBridge`: scan/setblocks/blocks capped at 30 s (`BRIDGE_TIMEOUT_S`; scan was 60 s).
- Mod: `AgentChatClient` rewritten — 15 s request timeout, parses `{job_id, status, reply}`, prints
  `[cp] working (job N)… /cp status · /cp cancel`, remembers the current job; `/cp status` → `GET /jobs/{id}`,
  `/cp cancel` (or `stop`) → `POST /jobs/{id}/cancel` (fall through to chat when no job is known);
  `CopilotClientMod.AGENT_BASE_URL`. `HttpTimeoutException` now prints a hint instead of a stack trace.
- Docs: CONTRACTS.md §7 chat protocol, README, mod/README `/cp` section, help text.
- Tooling: `[tool.ruff]` in `agent/pyproject.toml` (E/F/W/I/B, line-length 200) — ruff was not installed
  and the user-level config flagged 1015 style issues; installed ruff into `.venv`, auto-fixed 29 unused
  imports / import order.

**Verification**
- `cd agent && ../.venv/bin/pytest -q` → 222 passed (was 213), 4.2 s. `../.venv/bin/ruff check .` clean.
- New tests: `tests/test_server.py` (9: inline short turn, long turn → job id + `/say` push, busy refusal,
  `status`/`cancel` via chat text, cancel endpoint, hard-budget timeout of a stuck job with late result
  dropped, cooperative budget), `tests/test_pipeline.py` (3: real `handle_chat` + scripted LLM unwinds on
  cancel between tool calls / over budget; tool loop checks before the LLM call), `tests/test_llm.py`
  (Azure 120 s / 2 attempts / no SDK retries / flaky-then-ok).
- `cd mod && ./gradlew build --offline` → BUILD SUCCESSFUL, `build/libs/copilot-0.1.0.jar` rebuilt.
- Smoke: `COPILOT_LLM=mock python -m copilot.server --bridge mock`, `POST /chat` returned in 0.58 s for a full
  scripted build (inline, job done); with `COPILOT_CHAT_INLINE_WAIT_S=0.01` it returned `status: running,
  stage: blocking` immediately and `/jobs/1` reached `done` with the reply; `session.json`/`turn_001.jsonl`
  now written with the correct turn count.

**Numbers**: /chat latency before = whole pipeline (minutes, mod timed out); after ≤ 0.8 s + inline window.

**For the morning**: see MORNING_CHECKLIST.md "T1". The mod's chat prefix is still `[copilot]`; T3 will
restyle the progress lines. Cancel is cooperative: a job stops after its current LLM/tool call
(≤ 120 s worst case on a slow Azure call); the watchdog is the hard stop.

## T2 — Latency: 5 min hard cap, ≤ 2 min target [done]

**Where the time went (instrumented first)**: the one real full build in `runs/` before T1 took 2025 s:
94 stage LLM calls (blocking 30, detailing 24, materials 13, decoration 27) averaging 6–23 s each, two Azure
hangs of 830 s and 235 s (now bounded by T1's timeouts), and *2.6 s* of engine time per stage. The engine is
not the problem: a full rasterize→fit→resolve of the 37-object / 12.8k-block bench castle is 0.13 s
(raster 4 ms after an edit thanks to the existing per-object cache; fit is numpy-vectorised), renders
0.17–0.27 s and are cached by scene hash. Latency = number of LLM round trips × per-call latency, plus
stages that ran until their 40-call cap.

**What changed**
- `copilot/pipeline/budget.py` (new): `Budget` — hard cap `COPILOT_HARD_BUDGET_S` (300 s, aligned with the
  job's deadline when running under `/chat`) and a `COPILOT_PLAN_BUDGET_S` (150 s) schedule: interpret 10,
  blocking 35, detailing 45, materials 30, decoration 20, critic 10 (scaled if the plan changes). Deadlines are
  cumulative, so time saved early rolls forward; a stage that starts late still gets half its allotment. A
  stage that reaches its deadline ends after its current tool call (`stopped_reason: "budget"`, the model
  gets an "about N s left" nudge first). Critic and fix rounds only run while the turn is on schedule
  (`allow_critic()/allow_fix()`); fix rounds are 25 s; decoration is skipped when < 40 s of hard budget
  remain (the reply says `skipped decoration: out of time`). `ProfileRow`s per interpret/stage/critic/
  fix/route/describe/place feed the run log (`__profile__` lines and one `__summary__` line per turn with
  wall/LLM/engine ms and calls) and `ChatResult.data["profile"]`.
- `copilot/llm.py`: `run_tool_loop(deadline=, op_cap=)`; individual scene-mutating calls capped at
  `COPILOT_OP_CAP` (12) per stage — the 13th returns `ERROR: op cap reached … batch the rest in ONE
  run_script`; runs of read-only calls in one response (`render`, `lint`, `describe`, `select`, `search_blocks`,
  …) execute concurrently on a small thread pool (results stay in order; `Session.build_lock` makes the
  rasterize/fit/resolve build once). `LoopResult` now carries `llm_ms`, `tool_ms`, `ops`. `llm_call()` caps
  each call's timeout to the remaining hard budget (`AzureLLM.chat(timeout=)` → SDK per-request timeout;
  `single_call(ctx=)` does the same for interpret/critic/router). `__llm__` log records carry `model`.
  Images: `image_content` is now 640 px JPEG (q82) by default, `image_parts()` keeps the last 2 images;
  the tool loop shows at most 2 renders per call and the critic sends one contact sheet.
- `AzureLLM`: `MODEL` is accepted as an alias of `AZURE_OPENAI_DEPLOYMENT`; `MODEL_FAST` (or
  `AZURE_OPENAI_FAST_DEPLOYMENT`) gives `llm.fast()`, a copy sharing the HTTP client and usage counters that
  uses the cheaper deployment. `pipeline.common.fast_llm()` applies it to the router, `answer_question`
  (describe) and critic fix rounds (`fix_round=True`). Unset → same model, no behaviour change.
  429s: `AZURE_OPENAI_RATE_LIMIT_ATTEMPTS` (4) short attempts with Retry-After honoured and the sleep capped
  by the call timeout (timeouts keep T1's one retry). The baseline bench hit 5–6 429s per prompt on the
  `gpt-5.4-mini` deployment, each ending a stage early with "error".
- `pipeline/stages.py`: stage user message asks for ONE `run_script` per stage + `render`+`lint` in one
  response and states the stage's time budget; profile row per stage. `orchestrator.run_build/run_edit`
  drive the budget (`can_start`, `start_stage`, `start_critic`, `fix_deadline`); `handle_chat` creates a
  fresh budget per turn and emits the `__summary__` line (`profile_line()`).
- Prompts: `system_core.md` rules 3/7 ("build in bulk": one script per stage, 12-op cap, stage time budget,
  3–4 responses per stage); each `stage_*.md` "Method" rewritten as 2–3 responses with the whole checklist
  inside one script; `run_script` tool description says it is the main way to build. Prompt sizes stayed
  under the `test_prompts` cap (core 10.2k chars).
- Bench: `bench/run.py --quick` (first 5 prompts), `--profile` (per-stage latency table + median/max/mean
  line), `--profile --from DIR` (table for an old run, rebuilt from its jsonl logs). Rows carry `profile`.
- Docs: CONTRACTS.md §7 (wall-clock plan), `.env.example` (`MODEL_FAST`, `COPILOT_PLAN_BUDGET_S`,
  `COPILOT_OP_CAP`), README bench line, `MORNING_CHECKLIST.md` "T2".
- Tests: `tests/test_budget.py` (17: schedule math, late stages, hard cap & decoration skip, job alignment,
  loop deadline after the current call, time nudge, op cap → run_script, concurrent read-only calls with
  ordered results, timeout capping, profile rows & `__summary__`, behind-schedule skips critics, router/fix
  rounds on the fast model, `AzureLLM.fast()`, JPEG/640 px/2-image payloads, critic single sheet, bench
  table incl. legacy logs); `tests/test_llm.py` +1 (429 retries + timeout forwarding).

**Iteration 3 (this session) — verification of the iteration-2 work above.** The code was in place but
unverified (iteration 2 hit its turn cap and was auto-committed as `b3de611`; it had also left a baseline
bench of the T1 code running in `/tmp/cp-baseline`, which finished at 04:12). This iteration let that
baseline finish, ran the same 5 prompts on the current code, and recorded both. Also: `agent/bench/out/`
is now gitignored (raw runs); curated copies live in `agent/bench/results/`.

**Verification**
- `cd agent && ../.venv/bin/pytest -q` → 240 passed, 5.3 s. `../.venv/bin/ruff check .` clean.
- `python -m bench.run --mock --profile --quick` (scripted builder) → 5/5 in ~0.4 s each, table prints.
- Real bench, `python -m bench.run --profile --quick` on `gpt-5.4-mini` (Azure), same 5 prompts as the
  baseline. Zero `__llm_error__` records in the 5 run logs (the baseline hit 429s on every prompt).
- Engine re-checked: the 37-object baseline castle rasterizes in 9 ms cold / 3 ms warm (37 per-object
  cache hits); renders are keyed by scene hash. Engine time is 1–4 s of a 150 s turn — nothing to cut.

**Numbers (quick bench, 5 prompts, wall seconds)** — `agent/bench/results/t2_{before,after}_quick/`
(report.json/md, profile.md, contact-sheet PNGs):

| build | before (T1 code) | after | LLM calls before → after | tool calls before → after | score before → after |
|---|---|---|---|---|---|
| medieval_castle | 309 | 142 | 36 → 16 | 70 → 29 | (429 on scoring) → 6.5 |
| modern_villa | 257 | 150 | 26 → 18 | 78 → 22 | 7.5 → 7.5 |
| japanese_pagoda | 301 | 151 | 33 → 17 | 82 → 21 | 6.0 → 7.5 |
| lighthouse | 153 | 164 | 18 → 18 | 54 → 28 | 6.5 → 6.0 |
| stone_bridge | 230 | 147 | 24 → 17 | 56 → 21 | 6.25 → 6.25 |
| **median / max / mean** | **257 / 309 / 250** | **149.6 / 164 / 151** | | | **6.56 → 6.75** |

Target was median ≤ 150 s and max ≤ 300 s: met (median 149.6 s, printed as 150 in the table). The
150 s plan is now what sets the latency: every stage runs to its deadline (`stops` column shows
`detailing:budget materials:budget decoration:budget` on all five) and ends after its current call, so
builds land at plan + one LLM call (~5–15 s on this deployment). The lighthouse is the outlier because
its blocking critic ran (the turn was on schedule): one critic vision call took **35 s** on
`gpt-5.4-mini`, which then squeezed detailing to 23 s. LLM time is 97–99 % of wall in every run.

**For the morning**
- To trade quality for speed, `COPILOT_PLAN_BUDGET_S` is the single knob (stage allotments scale with
  it); `COPILOT_HARD_BUDGET_S` is the safety stop. The mod's job watchdog and the pipeline budget are
  aligned (`new_budget` clamps to the job deadline).
- T4 should look at the critic cost first: a single critic call ≈ 35 s here, more than the 10 s
  allotment. Options: run the critic on `MODEL_FAST`, or drop the blocking-stage critic and keep only the
  final one when `plan_s` ≤ 150.
- The bench's median sits right at the target; if the morning run comes in at 152–155 s that is the
  "+ one call" overshoot, not a regression. `COPILOT_PLAN_BUDGET_S=135` gives ~15 s of margin.
- `MORNING_CHECKLIST.md` "T2" is unchanged except the expected LLM call count (16–18 per build measured).
- T2 spans two commits on this branch: `b3de611` (iteration 2 auto safety commit, the code) and this
  iteration's commit (results, notes, gitignore). History was not rewritten.

## T3 — Chat visibility, live preview, animated final build [done]

**What changed**
- `agent/copilot/pipeline/progress.py` (new): `Progress` (one per turn on `ctx.progress`, clock = the budget's
  `t0`) prints `[cp·<tag> m:ss] <text>` through the `say` tool; tags `plan/block/detail/materials/decor/fix/
  critic/edit`, `[cp·build N%]` for placement. `scene_snapshot` + `summarize_delta` derive stage lines from the
  scene delta instead of the model's prose: `added 7 solids`, `carved 14 windows, 1 arch; added 12 battlements`
  (array/mirror multiplicity counted, kind words from object ids), `stone_wall/slate_roof; 9 objects repainted`,
  `added 6 lanterns`; fallback = the model's finish summary; `(time)` appended when the stage was cut by budget.
  Critic lines: `7/10 — fixing: <op_suggestion> on <ids> (rule P8)` / `— noted: …` / `— <summary>`. Lines are
  whitespace-collapsed and capped at 200 chars, so never JSON or multi-line. `Progress.op` = verbose per-op line.
- `pipeline/orchestrator.py`: `run_build`/`run_edit` emit the lines above (also `skipped: out of time` for a
  skipped stage); live preview places diff after **blocking and detailing only** (was: after every stage when
  on); the final placement is `_place(report=True)`; the reply is now 2–4 lines: `Built <name>: <brief>`,
  `<W×D footprint, H tall>, N objects, B blocks, in m:ss; critic S/10`, optional `Skipped …`, and
  ``Say `undo`, `export`, or an edit (…)`` (the job runner already says one `/say` per line). New meta
  command `/cp verbose on|off`; `/cp preview on|off` now overrides the `LIVE_PREVIEW` env default (true);
  `/cp place full` re-animates with the % lines. `handle_chat` creates the `Progress` per turn.
- `pipeline/stages.py`: passes `progress.op` as the tool loop's `on_tool` when verbose.
- `tools/dispatch.py`: `place` accepts `report` (progress callback → `[cp·build N%]`); a model-originated
  `say()` is tagged with the running stage so all chat lines look alike.
- `placement.py`: `place_scene(on_progress=)` sends the bottom-up layer chunks in four batches and, after
  each, `wait_for_placement()` polls `bridge.setblocks_status()` until the mod's queue drains (bridges
  without it sleep the estimate), then reports the cumulative percentage — the chat tracks the world.
  Animated chunk size is now 400 blocks (was 1500: a 12k-block castle animated in 0.5 s, i.e. not at all);
  silent placements keep 1500. `HttpBridge.setblocks_status()` (GET `/setblocks/status`, 5 s) added; the mock
  HTTP server got the route.
- `session.py`: `live_preview: Optional[bool] = None` (None → env), `verbose: bool = False`.
- Prompts: `system_core.md` rule 10 and the `say` tool description now say the runtime prints progress and
  `say()` is only for a decision the player must know (kept the core prompt under the size cap: +21 chars).
- Mod: `CopilotClientMod.chat()` → `formatLine()`: lines starting with `[cp` are printed as-is with the tag
  coloured (gold; `[cp·build` green, `[cp·critic` yellow) and **no** `[copilot]` prefix; everything else (the
  final reply, errors) keeps `[copilot] `. `AgentChatClient` job lines (`[cp] working (job N)…`, status,
  cancel) dropped their `§7` so they are tagged too. `BlockPlacer` was already tick-driven (one chunk per
  `END_SERVER_TICK`, `delay_ms` → ticks, no `Thread.sleep`, nothing off the server thread) — verified, unchanged.
- Docs: CONTRACTS.md §7 (line format, live preview, animation/status polling, `/setblocks/status` row),
  README, mod/README, `.env.example` (`LIVE_PREVIEW`), MORNING_CHECKLIST "T3" (+ T1 lines updated for the
  new prefixes).

**Verification**
- `cd agent && ../.venv/bin/pytest -q` → 248 passed (was 240), 8.6 s (the integration test loads the real
  registry once). `../.venv/bin/ruff check .` clean. `cd mod && ./gradlew build --offline` → BUILD SUCCESSFUL,
  `build/libs/copilot-0.1.0.jar` rebuilt 04:58.
- New `tests/test_progress.py` (8): line format/elapsed/percent/truncation; `summarize_delta` per stage
  (multiplicity, kind words, materials list); `LIVE_PREVIEW` default + `preview`/`verbose` meta toggles;
  `place_scene` batches with a status-polling bridge — exact timeline `set:200, [cp·build 25%], set:200,
  [cp·build 50%] …` and bottom-up chunk order; `wait_for_placement` sleeps the estimate without a status
  endpoint; **mock-bridge integration** (real dispatch/engine/registry + `MockBridge` + scripted LLM with
  scripted critics): asserts the exact `/say` tag sequence `plan, block, critic, fix, detail, critic,
  materials, critic, decor, critic, critic, build, build`, the line texts, no JSON, three placement groups
  (blocking preview > detailing preview, then the batched final), and the 2–4 line reply; preview-off →
  one (batched) placement; verbose → per-op lines tagged with the stage; model `say()` tagging.
  `tests/test_http_bridge.py` covers `setblocks_status()`. Updated `test_pipeline`/`test_budget` for the new
  line format, the preview placements and the `Skipped …` reply line.
- Smoke: FastAPI `TestClient` + `COPILOT_LLM=mock` + `MockBridge`: `/chat` returned `running` at once; the
  bridge received the 12 tagged lines, `[cp·build 58%]`, `[cp·build 100%]`, then the 3 reply lines, and
  `/jobs/1` → done. `python -m bench.run --mock --quick` → 5/5, mean 9.0 (scripted critic).

**Numbers**: none latency-relevant — progress `say` calls are ~1 ms each on the mock bridge (5 ms HTTP on the
real mod, ≈ 12 per build); the final animation adds ≈ 0.15 s per 400-block chunk (a 12k-block build ≈ 4–5 s
of visible bottom-up placement, awaited by the agent, well inside the 300 s hard budget after a 150 s plan).
Previews place diffs so blocking+detailing previews cost one extra `/setblocks` each, not a rebuild.

**For the morning**
- The `[cp·build N%]` lines rely on `GET /setblocks/status` draining; if the mod is lagging badly the wait
  caps at `WAIT_MAX_S` (20 s) per batch and the percentage runs ahead of the world — cosmetic.
- Stage lines are built from object ids: models that name cuts `win_*`/`door_*`/`arch_*` and props
  `lantern_*`/`merlon_*` get the nicest text; otherwise `carved 5 openings` / `added 3 details`.
- Live preview is on by default (`LIVE_PREVIEW=false` or `/cp preview off` to disable). The preview shows the
  build in the blocking stage's single material — expected; the palette lands with the final placement.

## T4 — Build quality via the bench [in progress]

**Bench**: `agent/bench/prompts.json` expanded 10 → 20 (added windmill, viking_longhouse, aqueduct,
wall_watchtower, round_library, market_hall, ship, chapel_bell_tower, hexagonal_keep, ruined_tower; the
first 5 are still the `--quick` set). `bench/run.py --jobs N` builds N prompts concurrently (per-prompt
session/bridge/registry; the Azure client is shared, as under `/chat`). 4 jobs hit 429s on every prompt
(scoring failed, stages cut short) — restarted at 2 jobs, which sees ~1 short 429 retry per prompt.
One broken prompt no longer kills the run (`run_one` catches). A full run at 2 jobs ≈ 30 min.

**Levers (all in this commit; A/B'd as one full run vs the baseline — see "Numbers")**
1. Interpret (`pipeline/interpret.py: validate_brief`): the silhouette plan is parsed for towers/turrets/
   spires wider than tall (`r=4 h=24`, `6x6 h=20`, `… 8 tall`), roofs with no stated overhang (or "flat
   roof"/parapet), and no stated entrance. A failing brief is sent back once with the reasons (only when
   ≥ 20 s of plan remain); if still failing, the rules ride along as `design rule: …` constraints. Prompt
   examples/checklist updated so they pass the validator (tested).
2. Detailing prompt: checklist rewritten — façades > 8 get depth (pilasters/reveals) + rhythm, quoins on
   every corner (worked example with `mirror` on both axes), roofs get overhang ≥ 1 + ridge line + eave
   trim ring, framed entrance. Trimmed elsewhere to stay under the prompt size cap.
3. Materials: lint rule **R9** (`engine/lint.py`, active once the scene defines a material): ≥ 3 distinct
   materials on solids, and a `gradient` on the main wall material (largest grounded volume, so a big roof
   does not count as "the wall"). Ten presets added and checked on a rendered test wall
   (`limestone_pale`, `tudor_plaster`, `dark_slate_wall`, `red_brick_victorian`, `weathered_wood`,
   `turf_roof`, `spruce_shingle_roof`, `stone_trim_light`, `quartz_trim`, `sandstone_trim`); every preset
   validates against the registry. `/cp lint` meta command added (also in help).
4. Fitting: cone/dome/wedge/arch goldens re-run (green); the arch golden is a *subtracted* cylinder, so
   fit on carved surfaces is covered (facing/half asserted). Torus golden left for T6.
5. Critic (`critic.vet_fixes`): a fix must carry a concrete op call and name ids that exist in the scene
   (or `add`/`run_script` something new); prose fixes and invented ids are dropped; cap 3. Second final
   critic/fix round only when the first final score < 7 (`SECOND_ROUND_BELOW`). Critic prompt states the
   contract. Chat lines show an op gist (`fixing: add hall_win on hall`) — the raw op leaked braces.
