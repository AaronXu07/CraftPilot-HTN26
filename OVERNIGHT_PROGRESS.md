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
