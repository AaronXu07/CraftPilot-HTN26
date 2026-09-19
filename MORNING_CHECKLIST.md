# Morning checklist (needs the game)

Start the agent first: `cd agent && ../.venv/bin/python -m copilot.server --port 8000` (real LLM needs `.env`;
`COPILOT_LLM=mock` runs the scripted builder). Install `mod/build/libs/copilot-0.1.0.jar` + Fabric API in 1.21.1.

## T1 — async chat
- In-game: type `/cp build a small stone hall`; expect `[copilot] thinking...` then, within ~1 s,
  `[copilot] [cp] working (job 1)… /cp status · /cp cancel`. No `HttpTimeoutException` in chat or the log,
  even if the build takes minutes.
- In-game: while it runs, type `/cp status`; expect one line like `[copilot] [cp] job 1: running, blocking, 0:12 elapsed`.
- In-game: while it runs, type `/cp build a tower`; expect `[cp] still working on job 1 (…) — say `cancel` to stop it.`
- In-game: type `/cp cancel` during a build; expect `[cp] job 1: cancelling (stops after the current step)` then,
  within one LLM/tool call (≤ 2 min worst case), `[copilot] [cp] stopped: cancelled`. Nothing more is placed.
- In-game: let a build finish; expect the progress lines (`Plan: …`, `blocking: …`, …) and the final
  `Built …` reply to arrive via `/say`, printed once (not twice).
- In-game: type `/cp help` or `/cp undo`; expect the reply immediately (inline, no "working" line).
- In-game: with the agent stopped, type `/cp hi`; expect `agent not running at http://127.0.0.1:8000 (start it with: …)`.
- Agent env `COPILOT_HARD_BUDGET_S=20` and `/cp build a castle` with the real LLM; expect
  `[cp] stopped: over time budget` after ~20 s and `/cp status` → `job N: timeout, …`.
