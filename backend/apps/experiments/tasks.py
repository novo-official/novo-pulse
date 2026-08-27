"""Optional Celery task.

The project never *requires* Celery: with `SYNC_TASKS=true` training runs in a
background thread. This module only defines a task when Celery is importable,
so importing it is always safe.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when Celery is installed
    from celery import shared_task
except ImportError:  # pragma: no cover
    shared_task = None


if shared_task is not None:  # pragma: no cover

    @shared_task(name="novopulse.train")
    def run_training_task(run_pk, contract_dict, profile_name, horizon, seed):
        from ml.contract import DataContract

        from .services import _execute

        _execute(run_pk, DataContract.from_dict(contract_dict), profile_name, horizon, seed)
