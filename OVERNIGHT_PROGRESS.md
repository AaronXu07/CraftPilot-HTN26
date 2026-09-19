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

## T2 — Latency: 5 min hard cap, ≤ 2 min target [in progress]

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
