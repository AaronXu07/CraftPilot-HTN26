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

## T2 — latency budget
- In-game: type `/cp build a medieval castle with four round towers and a gatehouse`; expect the final `Built …`
  reply within about 2–3 minutes (hard stop at 5 min), and `/cp status` mid-run to show the stage advancing
  roughly every 30–45 s (`blocking` → `detailing` → `materials` → `decoration`).
- Agent log (`runs/<player>_<stamp>/turn_NNN.jsonl`, last `__summary__` line): expect `wall …s llm …s (N calls)`
  with N in the 15–20 range for a build (bench measured 16–18; was 26–36 on the same prompts, 90+ on the first real build), and per-stage entries like `blocking 31.0[budget]` when a
  stage was cut at its deadline.
- In-game: if a build reply ends with `(skipped decoration: out of time — say "add decoration" to continue)`,
  type `/cp add lanterns at the entrance and along the walls`; expect an edit turn that places lanterns.
- Agent env `MODEL_FAST=<cheaper deployment>` then `/cp how tall is the keep?`; expect an answer within ~10 s
  (the router and the answer run on the fast deployment; `__llm__` records in the run log carry `"model"`, so
  build stages should still show the main deployment).
