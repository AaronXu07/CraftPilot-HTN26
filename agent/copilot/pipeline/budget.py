"""Wall-clock budget for one chat turn (T2): a 300 s hard cap and a 150 s plan split across stages.

The plan is a *schedule*, not a set of independent timers: each stage's deadline is the planned
cumulative end of that stage, so time saved early rolls into later stages and a stage that starts
late still gets at least half its allotment (never past the hard deadline). The orchestrator asks
`allow_critic()` / `allow_fix()` before optional rounds and skips decoration when the remaining hard
budget is under `DECORATION_MIN_S`. Every stage/critic/fix records a profile row so the run log and
the bench can print a latency table.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_TOTAL_S = 300.0
DEFAULT_PLAN_S = 150.0
DEFAULT_PLAN: Dict[str, float] = {"interpret": 10, "blocking": 35, "detailing": 45, "materials": 30, "decoration": 20, "critic": 10}
DECORATION_MIN_S = 40.0  # skip decoration entirely when less than this remains of the hard budget
STAGE_MIN_S = 15.0  # a stage that cannot get this much time is skipped
FIX_ROUND_S = 25.0  # a critic fix round gets at most this long
FIX_MIN_S = 30.0  # ...and only runs when at least this much hard budget remains
CRITIC_MIN_S = 20.0  # a critic call only runs with at least this much hard budget left
MIN_LLM_TIMEOUT_S = 8.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class ProfileRow:
    """One timed unit of a turn: a stage, a critic call, a fix round, interpret, route or place."""

    stage: str
    wall_ms: int = 0
    llm_ms: int = 0
    engine_ms: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    stopped: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "wall_ms": self.wall_ms, "llm_ms": self.llm_ms, "engine_ms": self.engine_ms, "llm_calls": self.llm_calls, "tool_calls": self.tool_calls, "stopped": self.stopped, "note": self.note}


class Budget:
    """Wall-clock plan for one turn. `total_s` is the hard cap, `plan_s` the target the stage plan is scaled to."""

    def __init__(self, total_s: Optional[float] = None, plan_s: Optional[float] = None, plan: Optional[Dict[str, float]] = None, t0: Optional[float] = None):
        self.total_s = float(total_s if total_s is not None else _env_float("COPILOT_HARD_BUDGET_S", DEFAULT_TOTAL_S))
        self.plan_s = float(plan_s if plan_s is not None else _env_float("COPILOT_PLAN_BUDGET_S", DEFAULT_PLAN_S))
        base = dict(plan or DEFAULT_PLAN)
        scale = self.plan_s / max(1.0, sum(base.values()))
        self.plan: Dict[str, float] = {k: v * scale for k, v in base.items()}
        self.t0 = t0 if t0 is not None else time.time()
        self.hard_deadline = self.t0 + self.total_s
        self.sched = self.t0  # planned cumulative end of the stages started so far
        self.stage: Optional[str] = None
        self.stage_deadline: Optional[float] = None
        self.stage_started: Optional[float] = None
        self.rows: List[ProfileRow] = []

    # -- time -----------------------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    @property
    def remaining(self) -> float:
        """Seconds left of the hard budget."""
        return max(0.0, self.hard_deadline - time.time())

    @property
    def plan_remaining(self) -> float:
        return max(0.0, self.t0 + self.plan_s - time.time())

    def over(self) -> bool:
        return time.time() >= self.hard_deadline

    def behind(self) -> bool:
        """True when the turn is running later than the plan for the stages started so far."""
        return time.time() > self.sched

    def stage_over(self) -> bool:
        return self.stage_deadline is not None and time.time() >= self.stage_deadline

    def llm_timeout(self, default: float) -> float:
        """Per-call LLM timeout: never longer than what is left of the hard budget (with a small floor)."""
        return max(MIN_LLM_TIMEOUT_S, min(float(default), self.remaining))

    # -- stages ---------------------------------------------------------------------------
    def allot(self, name: str) -> float:
        return float(self.plan.get(name, self.plan.get("critic", 10.0)))

    def can_start(self, name: str) -> bool:
        """A stage needs at least STAGE_MIN_S of hard budget; decoration needs DECORATION_MIN_S."""
        need = DECORATION_MIN_S if name == "decoration" else STAGE_MIN_S
        return self.remaining >= need

    def start_stage(self, name: str) -> float:
        """Advance the schedule by the stage's allotment and return the stage deadline (absolute time)."""
        now = time.time()
        allot = self.allot(name)
        self.sched += allot
        deadline = max(self.sched, now + allot * 0.5)
        self.stage = name
        self.stage_started = now
        self.stage_deadline = min(deadline, self.hard_deadline)
        return self.stage_deadline

    def end_stage(self) -> None:
        self.stage = None
        self.stage_deadline = None
        self.stage_started = None

    def allow_critic(self) -> bool:
        """Critic rounds are skipped when the turn is behind schedule or nearly out of hard budget."""
        return not self.behind() and self.remaining >= CRITIC_MIN_S

    def start_critic(self) -> None:
        """A critic call is planned work: it extends the schedule by the critic allotment."""
        self.sched += self.allot("critic")

    def allow_fix(self) -> bool:
        return not self.behind() and self.remaining >= FIX_MIN_S

    def fix_deadline(self) -> float:
        d = time.time() + FIX_ROUND_S
        self.sched += FIX_ROUND_S  # a fix round is planned work, not lateness
        self.stage_deadline = min(d, self.hard_deadline)
        return self.stage_deadline

    # -- profile --------------------------------------------------------------------------
    def record(self, row: ProfileRow) -> ProfileRow:
        self.rows.append(row)
        return row

    def summary(self) -> Dict[str, Any]:
        rows = [r.to_dict() for r in self.rows]
        return {
            "wall_ms": int(self.elapsed * 1000),
            "llm_ms": sum(r.llm_ms for r in self.rows),
            "engine_ms": sum(r.engine_ms for r in self.rows),
            "llm_calls": sum(r.llm_calls for r in self.rows),
            "tool_calls": sum(r.tool_calls for r in self.rows),
            "plan_s": self.plan_s,
            "total_s": self.total_s,
            "stages": rows,
        }


def get_budget(ctx: Any) -> Optional[Budget]:
    b = getattr(ctx, "budget", None)
    return b if isinstance(b, Budget) else None


def new_budget(ctx: Any, **kw: Any) -> Budget:
    """A fresh budget for this turn, stored on the context (and, for the job watchdog, aligned with its deadline)."""
    b = Budget(**kw)
    job = getattr(ctx, "job", None)
    if job is not None and getattr(job, "deadline", None):
        b.hard_deadline = min(b.hard_deadline, float(job.deadline))
        b.total_s = b.hard_deadline - b.t0
    try:
        ctx.budget = b
    except Exception:  # noqa: BLE001
        pass
    return b


def profile_row(ctx: Any, stage: str, **fields: Any) -> Optional[ProfileRow]:
    """Record a profile row on the turn's budget (no-op without one) and mirror it to the run log."""
    b = get_budget(ctx)
    row = ProfileRow(stage=stage, **fields)
    if b is not None:
        b.record(row)
    log = getattr(ctx, "log", None)
    if log is not None:
        try:
            rec = {"stage": stage, "name": "__profile__", "args": {}, "result_preview": "", "ms": row.wall_ms, "profile": row.to_dict()}
            if hasattr(log, "log"):
                log.log(**rec)
            elif callable(log):
                log(rec)
        except Exception:  # noqa: BLE001
            pass
    return row
