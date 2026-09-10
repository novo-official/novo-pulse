"""Where the clustered arm writes, kept out of the city-level artefact tree.

`artifacts/pol4/` is the submitted city-level run and is referenced by the model
card, the traceability doc and the API. The clustered arm is an experiment about
that run, so it gets its own directory and cannot overwrite a single file the
submission depends on.
"""
from __future__ import annotations

from pathlib import Path

from .config import Pol4Config


def cluster_artifacts_dir(config: Pol4Config) -> Path:
    return config.artifacts_dir.parent / "pol4_cluster"
