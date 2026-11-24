"""Run records: every experiment saves config, seed, hardware, git SHA, metrics."""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _gpu() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL).decode().strip().replace("\n", "; ")
    except Exception:  # noqa: BLE001
        return "none"


def write_run_record(out_dir: str | Path, config: dict, metrics: dict | None = None,
                     extra: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "machine": platform.node(),
        "gpu": _gpu(),
        "config": config,
        "metrics": metrics or {},
        **(extra or {}),
    }
    path = out / "run_record.json"
    path.write_text(json.dumps(rec, indent=2))
    if metrics:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return path
