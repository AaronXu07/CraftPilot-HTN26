# Morning checklist (needs the game)

Start the agent first: `cd agent && ../.venv/bin/python -m copilot.server --port 8000` (real LLM needs `.env`;
`COPILOT_LLM=mock` runs the scripted builder). Install `mod/build/libs/copilot-0.1.0.jar` + Fabric API in 1.21.1.

## T1 — async chat
- In-game: type `/cp build a small stone hall`; expect `[copilot] thinking...` then, within ~1 s,
  `[cp] working (job 1)… /cp status · /cp cancel` (gold tag, no `[copilot]` prefix since T3). No `HttpTimeoutException` in chat or the log,
  even if the build takes minutes.
- In-game: while it runs, type `/cp status`; expect one line like `[cp] job 1: running, blocking, 0:12 elapsed`.
- In-game: while it runs, type `/cp build a tower`; expect `[cp] still working on job 1 (…) — say `cancel` to stop it.`
- In-game: type `/cp cancel` during a build; expect `[cp] job 1: cancelling (stops after the current step)` then,
  within one LLM/tool call (≤ 2 min worst case), `[cp] stopped: cancelled`. Nothing more is placed.
- In-game: let a build finish; expect the progress lines (`[cp·plan 0:04] …`, `[cp·block 0:31] …`, …; see T3) and
  the final `[copilot] Built …` reply (2–4 lines) to arrive via `/say`, printed once (not twice).
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

## T3 — chat visibility, live preview, animated build
- In-game: type `/cp build a medieval castle with four round towers and a gatehouse`; expect, in order, one line per
  event with a gold tag and the elapsed time, e.g. `[cp·plan 0:06] L-shaped keep 20×14, four round towers r4 h24, gatehouse south`,
  `[cp·block 0:38] added 7 solids`, `[cp·critic 0:55] 6/10 — fixing: … on tower_ne (rule P1)` (yellow tag),
  `[cp·fix 1:10] …`, `[cp·detail 1:25] carved 14 windows, 2 arches; added 40 battlements`,
  `[cp·materials 1:55] stone_wall/slate_roof/spruce_trim; 9 objects repainted`, `[cp·decor 2:15] added 6 lanterns`.
  No `[copilot]` prefix on tagged lines, no JSON, no line longer than the chat width (≤ 200 chars).
- In-game: while that build runs, look at the spot in front of you; expect the massing (walls/towers, all one
  material) to appear right after the `[cp·block …]` line (live preview, diff placement), the openings to be cut
  after `[cp·detail …]`, and materials + props to land only with the final placement.
- In-game: the final placement; expect it to rise bottom-up over a second or two (400-block layer chunks, one per
  server tick every 60 ms — no server freeze, other players/mobs keep moving) with green
  `[cp·build 25%]`, `[cp·build 50%]`, `[cp·build 75%]`, `[cp·build 100%]` lines that track the rising build
  (the agent polls `GET /setblocks/status` between batches). Small diffs (< 800 blocks) print fewer steps but always end with `100%`.
- In-game: after `100%`, expect the reply as 2–4 `[copilot]` lines: `Built castle_v1: castle (medieval stone), 40x32, 26 tall, facing south: …`,
  `44×36 footprint, 26 tall, 37 objects, 12,800 blocks, in 2:31; critic 7/10`, then
  `Say `undo`, `export`, or an edit (e.g. "make the towers taller", "copper roofs").`
- In-game: type `/cp preview off` (expect `Live preview off: …`), then `/cp build a lighthouse`; expect no blocks
  in the world until the `[cp·build …]` lines; `/cp preview on` restores the default.
- In-game: type `/cp verbose on` (expect `Verbose on: one chat line per tool call`), then `/cp add lanterns along the walls`;
  expect one line per op such as `[cp·decor 0:14] add lantern_row: added lantern_row …` (or `run_script: …`)
  in addition to the stage summary; `/cp verbose off` returns to summaries only.
- In-game: type `/cp place full`; expect the whole build to re-animate bottom-up with the `[cp·build N%]` lines.
- Agent log: `runs/<player>_<stamp>/turn_NNN.jsonl` shows one `say` record per progress line (`name: "say"`), and
  `place` rows in the `__summary__` profile: two previews (`note: diff`) plus the final one (`note: diff final`).

## T4 — build quality (bench levers)
- In-game: type `/cp build a squat watchtower 10 wide and 6 tall with a flat roof`; expect the `[cp·plan 0:0N]` line
  to describe a tower taller than wide with a stated entrance and parapet (the interpret stage rejects
  towers wider than tall / roofs without overhang / no entrance and asks the model once for a corrected brief;
  the run log's `interpret` profile row shows `llm_calls: 2` and a `note` naming the rule when that happened).
- In-game: after any build, type `/cp lint`; expect no `R9` finding (≥ 3 materials on solids, gradient on the main
  wall material). If the materials stage was cut short by the budget you may see `R9 … no ground gradient` —
  say `add a cobblestone gradient to the walls` and expect it to clear.
- In-game: watch the `[cp·critic …]` lines; expect `fixing: <op> <id> on <id> (rule Pn)` gists such as
  `6/10 — fixing: add wall_n_win on wall_n (rule P8)` — never a raw `add(id=…, shape={…})` string, never `{`.
- In-game: type `/cp define material limestone_pale` … (or in chat: `use the limestone_pale preset on the walls and
  quartz_trim on the trim`); expect the walls to re-paint pale sandstone/calcite with a darker cut-sandstone base
  band. New presets: `limestone_pale`, `tudor_plaster`, `dark_slate_wall`, `red_brick_victorian`, `weathered_wood`,
  `turf_roof`, `spruce_shingle_roof`, `stone_trim_light`, `quartz_trim`, `sandstone_trim`.
- Bench: `cd agent && ../.venv/bin/python -m bench.run --quick --profile --jobs 2`; expect 5 prompts in ~8 min,
  mean score in `bench/out/<ts>/report.md`; compare with `agent/bench/results/t4_baseline/report.md`.
