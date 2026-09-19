"""Image -> mesh with TripoSR in tools/.venv-3d (torch stays out of this env).

Two transports: the persistent worker (`tools/recon_server.py`, started here on first use; ~1 s
inference + ~3 s meshing once warm) and, if it cannot be started, a one-shot subprocess
(`tools/recon_worker.py`, pays 5-18 s of model load every call).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from craftpilot.config import PROJECT_ROOT

TOOLS = PROJECT_ROOT / "tools"
WORKER = TOOLS / "recon_worker.py"
SERVER = TOOLS / "recon_server.py"
DEFAULT_PYTHON = TOOLS / ".venv-3d" / "bin" / "python"
SERVER_START_TIMEOUT_S = 90.0  # model load (5-18 s) + MPS warm-up (~10 s), with margin for a cold disk cache


class ReconUnavailable(RuntimeError):
    pass


def worker_python() -> Path:
    return Path(os.environ.get("CRAFTPILOT_RECON_PYTHON", str(DEFAULT_PYTHON))).expanduser()


def server_url() -> str:
    return f"http://127.0.0.1:{os.environ.get('CRAFTPILOT_RECON_PORT', '7790')}"


def available() -> bool:
    return worker_python().exists() and WORKER.exists()


def _health(timeout: float = 1.0) -> dict | None:
    try:
        with urllib.request.urlopen(server_url() + "/health", timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def ensure_server(wait: bool = True) -> bool:
    """Start the persistent worker if it is not running; with `wait`, block until it is warm."""
    if os.environ.get("CRAFTPILOT_RECON_SERVER", "1").lower() in ("0", "false", "off", "no"):
        return False
    h = _health()
    if h is None:
        if not (worker_python().exists() and SERVER.exists()):
            return False
        log = TOOLS / "recon_server.log"
        with open(log, "ab") as f:
            subprocess.Popen([str(worker_python()), str(SERVER)], stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    if not wait:
        return True
    deadline = time.time() + SERVER_START_TIMEOUT_S
    while time.time() < deadline:
        h = _health()
        if h and h.get("ok") and h.get("warm"):
            return True
        time.sleep(0.5)
    return bool(h and h.get("ok"))


def _via_server(image_path: Path, out_ply: Path, resolution: int, timeout: float) -> dict:
    body = json.dumps({"image": str(image_path), "out": str(out_ply), "resolution": resolution}).encode()
    req = urllib.request.Request(server_url() + "/reconstruct", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        report = json.loads(r.read().decode())
    if not report.get("ok"):
        raise ReconUnavailable(f"3D worker failed: {report.get('error')}")
    report["transport"] = "server"
    return report


def _via_subprocess(image_path: Path, out_ply: Path, resolution: int, timeout: float) -> dict:
    cmd = [str(worker_python()), str(WORKER), str(image_path), str(out_ply), "--resolution", str(resolution)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ReconUnavailable(f"3D worker timed out after {timeout:.0f}s") from exc
    report: dict | None = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                report = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if report is None or not report.get("ok"):
        err = (report or {}).get("error") or proc.stderr.strip().splitlines()[-1:] or f"exit {proc.returncode}"
        raise ReconUnavailable(f"3D worker failed: {err}")
    report["transport"] = "subprocess"
    return report


def reconstruct(image_path: Path, out_ply: Path, resolution: int = 256, timeout: float = 300.0) -> dict:
    """Run the reconstruction; returns the worker's report ({ok, vertices, faces, seconds, infer_s, mesh_s, …})."""
    if not available():
        raise ReconUnavailable(
            f"3D worker not installed: expected {worker_python()} and {WORKER}. See tools/README.md."
        )
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict
    if ensure_server():
        try:
            report = _via_server(image_path, out_ply, resolution, timeout)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            report = _via_subprocess(image_path, out_ply, resolution, timeout)
    else:
        report = _via_subprocess(image_path, out_ply, resolution, timeout)
    report["wall_seconds"] = round(time.time() - t0, 1)
    return report
