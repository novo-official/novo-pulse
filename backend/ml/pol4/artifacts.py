"""Reproducible, checksum-protected artefacts for the Pol 4 pipeline.

The competition pipeline has exactly four allowed raw inputs.  This module
records their identity and provides small atomic-write/checksum primitives used
by the materialised train set and the fitted model bundle.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Pol4Config

ARTEFACT_FORMAT_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a large CSV at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(payload: Any, path: str | Path) -> Path:
    """Write JSON completely before replacing the public path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
    """Atomically materialise a dataframe with a portable compression codec."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=False, compression="zstd")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def verify_checksums(directory: str | Path, files: dict[str, str]) -> None:
    """Refuse a missing or modified artefact before deserialising any state."""
    directory = Path(directory)
    for relative, expected in files.items():
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"artefact is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {path}: expected {expected}, got {actual}"
            )


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import lightgbm

        versions["lightgbm"] = lightgbm.__version__
    except ImportError:  # pragma: no cover - fitting already gives a clearer error
        versions["lightgbm"] = None
    try:
        import pyarrow

        versions["pyarrow"] = pyarrow.__version__
    except ImportError:  # pragma: no cover
        versions["pyarrow"] = None
    return versions


def _csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def build_input_manifest(config: Pol4Config, data: Any) -> dict[str, Any]:
    """Describe the only raw files allowed to influence a Pol 4 run."""
    declarations = (
        ("search_data.csv", config.search_path, data.search, "training history"),
        ("evaluation.csv", config.evaluation_path, data.evaluation, "target observations"),
        ("cities.csv", config.cities_path, data.cities, "city features"),
        ("city_code_mapping.csv", config.city_names_path, None, "display labels"),
    )

    sources: list[dict[str, Any]] = []
    digest_lines: list[str] = []
    for name, path, frame, role in declarations:
        path = Path(path)
        if not path.is_file():
            # The name mapping is optional in the competition contract, but its
            # absence is still explicit in provenance rather than silently hidden.
            sources.append(
                {
                    "name": name,
                    "role": role,
                    "path": str(path.resolve()),
                    "exists": False,
                }
            )
            digest_lines.append(f"{name}:MISSING")
            continue

        checksum = sha256_file(path)
        if frame is not None:
            rows = int(len(frame))
        else:
            # This is a 321-row label table; reading one column is cheap and
            # avoids trusting newline counts in a CSV.
            first_column = _csv_header(path)[0]
            rows = int(len(pd.read_csv(path, usecols=[first_column])))
        sources.append(
            {
                "name": name,
                "role": role,
                "path": str(path.resolve()),
                "exists": True,
                "bytes": path.stat().st_size,
                "rows": rows,
                "columns": _csv_header(path),
                "sha256": checksum,
            }
        )
        digest_lines.append(f"{name}:{checksum}")

    input_digest = hashlib.sha256("\n".join(digest_lines).encode("utf-8")).hexdigest()
    return {
        "format_version": ARTEFACT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(Path(config.raw_dir).resolve()),
        "allowed_sources": [item[0] for item in declarations],
        "input_digest": input_digest,
        "sources": sources,
        "runtime": _package_versions(),
    }
