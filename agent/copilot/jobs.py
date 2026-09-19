"""Background chat jobs: `POST /chat` returns a job id at once, the pipeline runs on a worker thread.

A `Job` is one chat turn. The worker runs the (synchronous) pipeline; the final reply is pushed to the
player through the bridge's `/say` unless the HTTP caller was still waiting when the job finished (short
turns such as `undo` or `help` are answered inline). Cancellation is cooperative: pipeline code calls
`check_cancel(ctx)` between LLM/tool calls and a `JobCancelled` unwinds the run. A watchdog marks a job
that outlives the hard budget as `timeout` and tells the player immediately, even if the worker is stuck
inside a slow call (the call itself is bounded by the LLM/bridge timeouts).
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

DEFAULT_HARD_BUDGET_S = 300.0
STOPPED_PREFIX = "[cp] stopped: "
TERMINAL = ("done", "failed", "cancelled", "timeout")


class JobCancelled(Exception):
    """Raised inside the pipeline when the job was cancelled or ran over its hard budget."""

    def __init__(self, reason: str = "cancelled"):
        super().__init__(reason)
        self.reason = reason


def hard_budget_s() -> float:
    try:
        return float(os.environ.get("COPILOT_HARD_BUDGET_S", DEFAULT_HARD_BUDGET_S))
    except ValueError:
        return DEFAULT_HARD_BUDGET_S


@dataclass
class Job:
    id: int
    player: str
    text: str
    budget_s: float
    status: str = "queued"  # queued | running | done | failed | cancelled | timeout
    reply: Optional[str] = None
    error: Optional[str] = None
    images: List[str] = field(default_factory=list)
    brief: Optional[Dict[str, Any]] = None
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    cancel_reason: Optional[str] = None
    waiting: bool = False  # an HTTP handler is blocked on this job and can return the reply itself
    inline: bool = False  # reply handed back in the /chat response (no /say)
    stage_fn: Optional[Callable[[], Optional[str]]] = None
    done_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def deadline(self) -> float:
        return (self.started or self.created) + self.budget_s

    @property
    def elapsed(self) -> float:
        end = self.finished or time.time()
        return max(0.0, end - (self.started or self.created))

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def stage(self) -> Optional[str]:
        if self.is_terminal or self.stage_fn is None:
            return None
        try:
            return self.stage_fn()
        except Exception:  # noqa: BLE001
            return None

    def cancel_requested(self) -> Optional[str]:
        """The reason the running pipeline should stop now, or None to keep going."""
        if self.cancel_reason:
            return self.cancel_reason
        if time.time() > self.deadline:
            return "over time budget"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "player": self.player,
            "text": self.text,
            "status": self.status,
            "stage": self.stage,
            "elapsed_s": round(self.elapsed, 1),
            "budget_s": self.budget_s,
            "remaining_s": round(max(0.0, self.deadline - time.time()), 1) if not self.is_terminal else 0.0,
            "cancelling": bool(self.cancel_reason) and not self.is_terminal,
            "reply": self.reply,
            "error": self.error,
            "images": list(self.images),
            "brief": self.brief,
        }


def check_cancel(ctx: Any) -> None:
    """Raise JobCancelled if the job on `ctx` (if any) was cancelled or is over budget."""
    job = getattr(ctx, "job", None)
    if job is None:
        return
    reason = job.cancel_requested()
    if reason:
        raise JobCancelled(reason)


class JobManager:
    """Runs one chat job per player at a time on a small thread pool and tracks their lifecycle.

    `runner(job) -> result` does the work; `result` needs `.reply` (str) and may have `.images`
    (list of saved paths) and `.brief`. `say(player, text)` pushes chat lines to the player.
    """

    def __init__(self, runner: Callable[[Job], Any], say: Callable[[str, str], None], max_workers: int = 4, budget_s: Optional[float] = None):
        self._runner = runner
        self._say = say
        self._budget_s = budget_s
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="copilot-job")
        self._jobs: Dict[int, Job] = {}
        self._next = 1
        self._lock = threading.Lock()

    # -- queries ----------------------------------------------------------------------------
    def get(self, job_id: int) -> Optional[Job]:
        return self._jobs.get(job_id)

    def all(self) -> List[Job]:
        return list(self._jobs.values())

    def active_for(self, player: str) -> Optional[Job]:
        for job in self._jobs.values():
            if job.player == player and not job.is_terminal:
                return job
        return None

    def last_for(self, player: str) -> Optional[Job]:
        jobs = [j for j in self._jobs.values() if j.player == player]
        return jobs[-1] if jobs else None

    # -- lifecycle --------------------------------------------------------------------------
    def submit(self, player: str, text: str, stage_fn: Optional[Callable[[], Optional[str]]] = None, budget_s: Optional[float] = None) -> Job:
        with self._lock:
            job = Job(id=self._next, player=player, text=text, budget_s=budget_s or self._budget_s or hard_budget_s(), stage_fn=stage_fn)
            self._next += 1
            self._jobs[job.id] = job
        self._pool.submit(self._run, job)
        timer = threading.Timer(job.budget_s + 0.05, self._watchdog, args=(job,))
        timer.daemon = True
        timer.start()
        return job

    def cancel(self, job_id: int, reason: str = "cancelled") -> Optional[Job]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        with job.lock:
            if job.is_terminal:
                return job
            job.cancel_reason = job.cancel_reason or reason
        return job

    def wait_inline(self, job: Job, timeout: float) -> bool:
        """Block up to `timeout` s for the job. True if it finished and the caller must deliver
        `job.reply` itself (the worker did not push it); False if the caller should return the job id."""
        with job.lock:
            job.waiting = True
        job.done_event.wait(timeout)
        with job.lock:
            job.waiting = False
            return job.inline

    # -- internals --------------------------------------------------------------------------
    def _run(self, job: Job) -> None:
        with job.lock:
            if job.is_terminal:
                return
            job.status = "running"
            job.started = time.time()
        reason = job.cancel_requested()
        if reason:  # cancelled while queued
            self._finish(job, "cancelled", reply=STOPPED_PREFIX + reason)
            return
        try:
            result = self._runner(job)
            reply = str(getattr(result, "reply", result) or "")
            images = list(getattr(result, "images", None) or [])
            brief = getattr(result, "brief", None)
            self._finish(job, "done", reply=reply, images=images, brief=brief)
        except JobCancelled as e:
            status = "timeout" if e.reason == "over time budget" else "cancelled"
            self._finish(job, status, reply=STOPPED_PREFIX + e.reason)
        except Exception as e:  # noqa: BLE001
            self._finish(job, "failed", reply=f"[cp] failed: {type(e).__name__}: {str(e)[:200]}", error=traceback.format_exc(limit=4))

    def _watchdog(self, job: Job) -> None:
        if job.is_terminal:
            return
        job.cancel_reason = job.cancel_reason or "over time budget"
        self._finish(job, "timeout", reply=STOPPED_PREFIX + "over time budget")

    def _finish(self, job: Job, status: str, reply: str = "", images: Optional[List[str]] = None, brief: Any = None, error: Optional[str] = None) -> None:
        with job.lock:
            if job.is_terminal:
                return  # the watchdog (or a cancel) already closed this job; drop the late result
            job.status = status
            job.reply = reply
            job.images = list(images or [])
            job.brief = brief
            job.error = error
            job.finished = time.time()
            # A /chat handler still waiting on this job returns the reply itself (short turns feel
            # synchronous); otherwise it goes to the player's chat through /say.
            job.inline = job.waiting
            job.done_event.set()
            push = not job.inline
        if push:
            self._push(job)

    def _push(self, job: Job) -> None:
        text = job.reply or ""
        for line in [ln for ln in text.split("\n") if ln.strip()] or [text]:
            try:
                self._say(job.player, line)
            except Exception:  # noqa: BLE001
                pass
