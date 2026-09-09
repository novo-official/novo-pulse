"""Filesystem locations used across the ML stack.

Every path is derived from the repository root so the package behaves the same
whether it is imported by Django, by pytest or by a bare script.
"""
from __future__ import annotations

import os
from pathlib import Path

# backend/ml/paths.py -> backend/ml -> backend -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(env_var: str, default: str) -> Path:
    raw = os.getenv(env_var, default)
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path)


DATA_DIR = _resolve("DATA_DIR", "data")
RAW_DATA_DIR = DATA_DIR / "raw"
UPLOAD_DIR = DATA_DIR / "uploads"
RUNS_DIR = _resolve("RUNS_DIR", "runs")
REPORTS_DIR = _resolve("REPORTS_DIR", "reports")
CONFIG_DIR = REPO_ROOT / "config"
ARTIFACTS_DIR = _resolve("ARTIFACTS_DIR", "artifacts")

PROFILES_FILE = CONFIG_DIR / "profiles.yaml"
ACTIVE_CONTRACT_FILE = CONFIG_DIR / "data_contract.active.yaml"


def ensure_dirs() -> None:
    """Create every writable directory the pipeline expects."""
    for directory in (
        DATA_DIR,
        RAW_DATA_DIR,
        UPLOAD_DIR,
        RUNS_DIR,
        REPORTS_DIR,
        ARTIFACTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
