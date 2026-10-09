"""Append-only CSV ledger. The single source of truth for every bits claim.

No number from an experiment script is meaningful unless it's in here --
this file is what a report or a plot is allowed to read from. It is
intentionally dumb (no aggregation, no derived columns) so it can't
quietly become a second, inconsistent source of truth.
"""
from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime, timezone

FIELDS = [
    "timestamp", "experiment", "task_id", "condition", "seed",
    "bits_total", "bits_recon", "bits_weights",
    "steps", "seconds", "peak_vram_mb", "commit", "solved",
]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def append(path: str, row: dict) -> None:
    row = {**row}
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    row.setdefault("commit", git_commit())
    exists = os.path.exists(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
