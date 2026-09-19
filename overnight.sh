#!/usr/bin/env bash
# overnight.sh — run Claude Code headless in fresh sessions until the task list
# is done or the time budget expires. Each iteration: read OVERNIGHT.md, do ONE
# task, run tests, commit, mark it done, exit.
#
# Usage:  caffeinate -dims ./overnight.sh 4        # 4 hours
# Requires: claude CLI logged in, repo with passing tests, OVERNIGHT.md present.

set -u
HOURS="${1:-4}"
MODEL="${MODEL:-opus}"             # override: MODEL=sonnet ./overnight.sh 4
MAX_TURNS="${MAX_TURNS:-80}"       # per-iteration cap; the outer loop bounds time
END=$(( $(date +%s) + HOURS * 3600 ))
STAMP="$(date +%Y%m%d-%H%M)"
BRANCH="overnight/$STAMP"
LOGDIR="runs/overnight/$STAMP"

mkdir -p "$LOGDIR"
git checkout -b "$BRANCH" || { echo "could not create branch"; exit 1; }
[ -f OVERNIGHT_PROGRESS.md ] || printf "# Overnight progress\n\nStarted %s on branch %s\n\n" "$STAMP" "$BRANCH" > OVERNIGHT_PROGRESS.md
[ -f MORNING_CHECKLIST.md ] || printf "# Morning checklist (needs the game)\n\n" > MORNING_CHECKLIST.md
git add -A && git commit -qm "overnight: start $STAMP" || true

i=0
while [ "$(date +%s)" -lt "$END" ]; do
  i=$((i + 1))
  if grep -q "ALL_TASKS_DONE" OVERNIGHT_PROGRESS.md; then
    echo "all tasks done after $((i - 1)) iterations"; break
  fi
  LOG="$LOGDIR/iter-$(printf '%03d' "$i").log"
  echo "=== iteration $i  $(date '+%H:%M')  (ends $(date -r "$END" '+%H:%M')) ==="

  REMAINING_MIN=$(( (END - $(date +%s)) / 60 ))
  claude -p "$(cat OVERNIGHT.md)

Time remaining in this overnight run: about ${REMAINING_MIN} minutes. Iteration ${i}." \
    --model "$MODEL" \
    --dangerously-skip-permissions \
    --max-turns "$MAX_TURNS" \
    --output-format text \
    > "$LOG" 2>&1
  STATUS=$?

  # Safety-net commit in case the session ended before committing.
  if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    git add -A
    git commit -qm "overnight: iteration $i (auto safety commit)" || true
  fi

  # Back off on usage/rate limits instead of burning iterations.
  if grep -qiE "rate.?limit|usage limit|limit reached|overloaded|429" "$LOG"; then
    echo "limit detected; sleeping 20 min"; sleep 1200; continue
  fi
  if [ "$STATUS" -ne 0 ]; then
    echo "claude exited with $STATUS; see $LOG"; sleep 60
  fi
  sleep 5
done

echo "finished at $(date '+%H:%M'); branch $BRANCH; logs in $LOGDIR"
git log --oneline "$BRANCH" ^main | head -50