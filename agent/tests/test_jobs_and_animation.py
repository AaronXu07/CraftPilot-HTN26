"""Job lifecycle (JobManager unit level) and animation chunking (T6)."""
import threading
import time

import pytest

from copilot.engine.diff import layer_chunks
from copilot.jobs import STOPPED_PREFIX, Job, JobCancelled, JobManager, check_cancel
from copilot.placement import ANIMATE_CHUNK, estimate_seconds, place_scene
from copilot.session import Session


class Result:
    def __init__(self, reply, images=None, brief=None):
        self.reply, self.images, self.brief = reply, images or [], brief


def _manager(runner, budget_s=None, max_workers=2):
    said = []
    mgr = JobManager(runner, lambda player, text: said.append((player, text)), max_workers=max_workers, budget_s=budget_s)
    return mgr, said


def _wait(job, timeout=5.0):
    assert job.done_event.wait(timeout), "job did not finish"
    return job


# -- lifecycle ---------------------------------------------------------------------------------
def test_job_runs_to_done_and_pushes_each_reply_line():
    gate = threading.Event()
    mgr, said = _manager(lambda job: (gate.wait(5), Result("line one\n\nline two", images=["a.png"], brief={"name": "x"}))[1])
    job = mgr.submit("steve", "build a hut")
    assert job.id == 1 and job.status in ("queued", "running") and job.to_dict()["remaining_s"] > 0
    gate.set()
    _wait(job)
    assert job.status == "done" and job.reply == "line one\n\nline two" and job.images == ["a.png"] and job.brief == {"name": "x"}
    assert said == [("steve", "line one"), ("steve", "line two")]
    d = job.to_dict()
    assert d["job_id"] == 1 and d["status"] == "done" and d["remaining_s"] == 0.0 and d["stage"] is None and not d["cancelling"]
    assert mgr.get(1) is job and mgr.get(99) is None and mgr.last_for("steve") is job and mgr.active_for("steve") is None


def test_job_ids_increase_and_active_for_tracks_running_job():
    gate = threading.Event()

    def runner(job):
        gate.wait(5)
        return Result("ok")

    mgr, said = _manager(runner)
    j1 = mgr.submit("alex", "one")
    j2 = mgr.submit("alex", "two")
    assert (j1.id, j2.id) == (1, 2) and mgr.all() == [j1, j2]
    time.sleep(0.05)
    assert mgr.active_for("alex") in (j1, j2)
    gate.set()
    _wait(j1)
    _wait(j2)
    assert mgr.active_for("alex") is None


def test_cancel_running_job_is_cooperative_and_reported():
    seen = {}

    def runner(job):
        class Ctx:
            pass

        ctx = Ctx()
        ctx.job = job
        for _ in range(200):
            try:
                check_cancel(ctx)
            except JobCancelled as e:
                seen["reason"] = e.reason
                raise
            time.sleep(0.01)
        return Result("finished anyway")

    mgr, said = _manager(runner)
    job = mgr.submit("steve", "long build")
    time.sleep(0.05)
    assert mgr.cancel(job.id, "player asked") is job and job.to_dict()["cancelling"]
    _wait(job)
    assert job.status == "cancelled" and seen["reason"] == "player asked"
    assert said == [("steve", STOPPED_PREFIX + "player asked")]
    # cancelling a finished job is a no-op that returns the job; unknown ids return None
    assert mgr.cancel(job.id) is job and job.status == "cancelled" and mgr.cancel(42) is None


def test_cancel_while_queued_never_runs():
    ran = []
    gate = threading.Event()

    def runner(job):
        ran.append(job.id)
        gate.wait(5)
        return Result("ok")

    mgr, said = _manager(runner, max_workers=1)  # the second job queues behind the first
    j1 = mgr.submit("steve", "first")
    time.sleep(0.05)
    j2 = mgr.submit("steve", "second")
    mgr.cancel(j2.id)
    gate.set()
    _wait(j1)
    _wait(j2)
    assert j2.status == "cancelled" and j2.reply == STOPPED_PREFIX + "cancelled" and ran == [1]


def test_runner_exception_marks_failed_with_traceback():
    def runner(job):
        raise ValueError("bad brief")

    mgr, said = _manager(runner)
    job = _wait(mgr.submit("steve", "x"))
    assert job.status == "failed" and job.reply.startswith("[cp] failed: ValueError: bad brief")
    assert "ValueError" in job.error and said[0][1].startswith("[cp] failed")


def test_watchdog_times_out_a_stuck_job_and_drops_its_late_result():
    gate = threading.Event()

    def runner(job):
        gate.wait(5)
        return Result("late")

    mgr, said = _manager(runner, budget_s=0.2)
    job = mgr.submit("steve", "stuck")
    _wait(job, 3.0)
    assert job.status == "timeout" and job.reply == STOPPED_PREFIX + "over time budget"
    assert job.cancel_requested() == "over time budget"
    gate.set()
    time.sleep(0.1)
    assert job.status == "timeout" and job.reply == STOPPED_PREFIX + "over time budget" and said == [("steve", job.reply)]


def test_wait_inline_hands_reply_back_without_say():
    gate = threading.Event()
    mgr, said = _manager(lambda job: (gate.wait(5), Result("quick answer"))[1])
    job = mgr.submit("steve", "how tall?")
    threading.Timer(0.05, gate.set).start()
    assert mgr.wait_inline(job, 3.0) is True
    assert job.inline and job.reply == "quick answer" and said == []


def test_wait_inline_timeout_falls_back_to_push():
    gate = threading.Event()
    mgr, said = _manager(lambda job: (gate.wait(5), Result("slow"))[1])
    job = mgr.submit("steve", "slow one")
    assert mgr.wait_inline(job, 0.05) is False
    gate.set()
    _wait(job)
    assert not job.inline and said == [("steve", "slow")]


def test_stage_fn_is_reported_only_while_running():
    gate = threading.Event()
    mgr, _ = _manager(lambda job: (gate.wait(5), Result("ok"))[1])
    job = mgr.submit("steve", "x", stage_fn=lambda: "blocking")
    time.sleep(0.05)
    assert job.stage == "blocking" and job.to_dict()["stage"] == "blocking"
    gate.set()
    _wait(job)
    assert job.stage is None


def test_job_deadline_and_elapsed():
    j = Job(id=1, player="p", text="t", budget_s=10)
    assert j.cancel_requested() is None and 0 <= j.elapsed < 1
    j.started = time.time() - 11
    assert j.cancel_requested() == "over time budget"


# -- animation chunking --------------------------------------------------------------------------
class ChunkBridge:
    def __init__(self):
        self.calls = []
        self.status_polls = 0

    def player(self):
        return {"name": "t", "pos": [0.5, 64.0, 0.5], "yaw": 180.0, "pitch": 0.0, "facing": "north"}

    def scan(self, lo, hi):
        return {}

    def setblocks(self, chunks, flags=3):
        self.calls.append([(list(c), d) for c, d in chunks])
        return sum(len(c) for c, _ in chunks)

    def setblocks_status(self):
        self.status_polls += 1
        return {"pending_blocks": 0, "pending_chunks": 0}


def test_layer_chunks_never_split_a_lower_layer_after_a_higher_one():
    blocks = [(x, y, z, "s") for y in (3, 0, 2, 1) for x in range(3) for z in range(3)]
    chunks = layer_chunks(blocks, chunk_size=7)
    assert sum(len(c) for c in chunks) == 36 and all(len(c) <= 7 for c in chunks)
    ys = [b[1] for c in chunks for b in c]
    assert ys == sorted(ys)
    assert layer_chunks([], chunk_size=5) == []


@pytest.mark.parametrize("n,chunk,delay,expect_chunks", [(1, 4, 60, 1), (12, 4, 60, 3), (13, 4, 60, 4)])
def test_animated_placement_chunk_count_and_delay(n, chunk, delay, expect_chunks):
    b = ChunkBridge()
    s = Session("p")
    bm = {(i % 4, i // 4, 0): "minecraft:stone" for i in range(n)}
    place_scene(s, b, bm, mode="diff", animate=True, chunk_size=chunk, delay_ms=delay)
    sent = [c for call in b.calls for c in call]
    assert len(sent) == expect_chunks and all(d == delay for _, d in sent)
    assert sum(len(c) for c, _ in sent) == n
    assert estimate_seconds(n, chunk, delay) >= 0


def test_progress_callback_hits_quarters_bottom_up():
    b = ChunkBridge()
    s = Session("p")
    bm = {(x, y, z): "minecraft:stone" for x in range(4) for y in range(8) for z in range(2)}  # 64 blocks, 8 layers
    pct = []
    msg = place_scene(s, b, bm, mode="diff", animate=True, chunk_size=8, delay_ms=1, on_progress=pct.append)
    assert pct == [25, 50, 75, 100] and len(b.calls) == 4 and b.status_polls >= 4
    flat = [blk for call in b.calls for c, _ in call for blk in c]
    ys = [blk[1] for blk in flat]
    assert ys == sorted(ys) and "8 chunks" in msg


def test_silent_placement_uses_big_chunks_and_no_delay():
    b = ChunkBridge()
    s = Session("p")
    bm = {(x, y, 0): "minecraft:stone" for x in range(50) for y in range(40)}  # 2000 blocks
    place_scene(s, b, bm, mode="diff", animate=False)
    sent = [c for call in b.calls for c in call]
    assert len(sent) == 2 and all(d == 0 for _, d in sent) and max(len(c) for c, _ in sent) == 1500
    assert ANIMATE_CHUNK < 1500
