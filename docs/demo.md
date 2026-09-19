# 30-second demo script

Derived from the full 20-prompt bench (`agent/bench/results/t4_after/report.json`, mean 6.66/10):
the prompts below scored highest on the vision critic *and* finished fastest, so they are the safest
things to type on stage. A build takes 2½–3½ minutes end to end (hard cap 5), so **the 30-second
demo is an edit on a build made just before the demo** — the build itself is the warm-up.

| bench prompt | critic | wall | blocks |
|---|---|---|---|
| lighthouse (`build a tall tapered lighthouse with red and white bands, a glass lantern room on top and a small keeper's cottage attached`) | 7.25 | 2:39 | 1 228 |
| stone_bridge (`build a stone arch bridge 40 blocks long with three arches, parapets and lanterns on posts`) | 7.5 | 3:18 | 3 229 |
| market_hall (`build an open-sided medieval market hall: a timber frame on stone posts, a big pitched roof with a clock gable, stalls underneath`) | 7.5 | 3:08 | 4 336 |
| round_library (`build a round library with a domed roof, a ring of tall arched windows, a columned entrance porch and bookshelves lining the inside`) | 7.5 | 2:46 | 10 273 |

The lighthouse is the pick: smallest, fastest, and its silhouette (striped taper + lantern room + cottage)
reads instantly from across a field. Keep the bridge as a spare — it looks good from the side and
"add a fourth arch" is a satisfying edit.

## Before the demo (≈ 4 min, off camera)

1. Start the agent: `cd agent && python -m copilot.server --port 8000 --bridge auto`.
2. Load the world, stand on flat ground with ~30 blocks of clear space in front of you, face it.
3. `/cp preview on` (default on; say it anyway so the audience sees the massing appear during the build).
4. `/cp build a tall tapered lighthouse with red and white bands, a glass lantern room on top and a small keeper's cottage attached`
   — expect `[cp] working (job 1)…`, then `[cp·plan 0:0x] …`, `[cp·block …]`, `[cp·detail …]`, `[cp·materials …]`,
   `[cp·critic …]`, `[cp·build 25% … 100%]` and a 2–4 line `Built …` reply. If it ends with
   `(skipped decoration: out of time …)` that's fine for the demo.
5. Walk to a spot where the whole tower is in frame with the cottage on the right.

## The 30 seconds (on camera)

| t | you say / type | what the audience sees |
|---|---|---|
| 0:00 | "I described this lighthouse in one sentence; the copilot designed it like a CAD model — solids, booleans, a vision critic." | the finished lighthouse |
| 0:06 | `/cp make the tower 8 blocks taller and give the lantern room a copper roof` | `[cp] working (job 2)…` at once; the router maps this to `set_shape` + `set_material`; the diff re-places only the changed bbox (10–25 s) |
| 0:20 | `/cp materials` | `1228 blocks (20 stacks), N block types:` then `stone_bricks  …  (8 stacks + 12)` — the survival shopping list |
| 0:25 | `/cp undo` | the tower drops back to its original height in-world (diff placement) |
| 0:29 | `/cp export lighthouse` | `exported 1228 blocks to …/lighthouse.litematic` |

Fallbacks:
- If the edit job is slower than 20 s, type `/cp status` (stage + elapsed) and talk over it; anything else
  you type while a job runs is answered with `[cp] still working on job 2 (…) — say \`cancel\` to stop it.`,
  so wait for the `Built …`/edit reply before `/cp materials`.
- If the mod is not connected, `/cp` answers `[cp] mod not connected` — restart the agent with
  `--bridge auto` and check the mod log for `copilot bridge listening`.
- `/cp cancel` stops any job after its current tool call.

## Spare edits that read well on the lighthouse

- `/cp swap the tower to deepslate with a mossy base` (materials stage, ~20 s)
- `/cp add lanterns along the cottage roof` (decoration stage, ~30 s)
- `/cp mirror the cottage to the other side of the tower` (`mirror_copy`, instant)
