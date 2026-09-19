"""Image -> mesh through the TripoSR worker in tools/.venv-3d (a subprocess: torch stays out of this env)."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from craftpilot.config import PROJECT_ROOT

WORKER = PROJECT_ROOT / "tools" / "recon_worker.py"
DEFAULT_PYTHON = PROJECT_ROOT / "tools" / ".venv-3d" / "bin" / "python"


class ReconUnavailable(RuntimeError):
    pass


def worker_python() -> Path:
    p = Path(os.environ.get("CRAFTPILOT_RECON_PYTHON", str(DEFAULT_PYTHON))).expanduser()
    return p


def available() -> bool:
    return worker_python().exists() and WORKER.exists()


def reconstruct(image_path: Path, out_ply: Path, resolution: int = 256, timeout: float = 300.0) -> dict:
    """Run the worker; returns its JSON report ({ok, vertices, faces, seconds, ...}). Raises on failure."""
    if not available():
        raise ReconUnavailable(
            f"3D worker not installed: expected {worker_python()} and {WORKER}. See tools/README.md (uv venv "
            "tools/.venv-3d, torch, torchmcubes, TripoSR)."
        )
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(worker_python()), str(WORKER), str(image_path), str(out_ply), "--resolution", str(resolution)]
    t0 = time.time()
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
    report["wall_seconds"] = round(time.time() - t0, 1)
    return report
