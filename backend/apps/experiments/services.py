"""Training orchestration.

`SYNC_TASKS=true` (the default) runs training inline in a background thread so
the demo needs no Redis or Celery. If Celery *is* configured the same entry
point is used, so nothing else in the codebase has to know which mode is active.
"""
from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime, timezone
from typing import Any

from django.conf import settings
from django.utils.text import slugify

from ml.contract import DataContract, load_profile
from ml.models.registry import REGISTRY, is_baseline
from ml.paths import REPO_ROOT
from ml.pipelines.training import (
    TrainingConfig,
    TrainingPipeline,
    load_future_overrides,
    sanitise_json,
)

from .models import Experiment, ModelResult, TrainingRun

log = logging.getLogger(__name__)


def start_training(
    contract: DataContract | None = None,
    profile_name: str | None = None,
    horizon: int | None = None,
    metric: str | None = None,
    seed: int | None = None,
    future_covariates: str | None = None,
    experiment_name: str | None = None,
    blocking: bool = False,
) -> TrainingRun:
    """Create the run record and kick off the pipeline."""
    contract = contract or DataContract.load()
    if metric:
        contract.evaluation.primary_metric = metric
    profile_name = profile_name or settings.TRAINING_PROFILE
    horizon = int(horizon or contract.evaluation.max_horizon)
    seed = int(seed if seed is not None else settings.RANDOM_SEED)

    # A count-aggregation contract has no target column: demand is the number
    # of rows. The field is display-only, so it carries that instead of a blank.
    target_label = contract.target or (
        "count of rows" if contract.aggregation == "count" else ""
    )

    experiment = None
    if experiment_name:
        experiment, _ = Experiment.objects.get_or_create(
            slug=slugify(experiment_name)[:200],
            defaults={
                "name": experiment_name,
                "dataset_name": contract.name,
                "target": target_label,
            },
        )

    run = TrainingRun.objects.create(
        experiment=experiment,
        run_id=_reserve_run_id(),
        status=TrainingRun.Status.PENDING,
        profile=profile_name,
        target=target_label,
        primary_metric=contract.evaluation.primary_metric,
        horizon=horizon,
        frequency=contract.frequency or "D",
        dataset_name=contract.name,
        dataset_path=contract.path or "",
        config=contract.to_dict(),
    )

    if blocking or settings.SYNC_TASKS:
        if blocking:
            _execute(run.pk, contract, profile_name, horizon, seed, future_covariates)
        else:
            thread = threading.Thread(
                target=_execute,
                args=(run.pk, contract, profile_name, horizon, seed, future_covariates),
                daemon=True,
                name=f"train-{run.run_id}",
            )
            thread.start()
    else:  # pragma: no cover - only when a broker is configured
        from .tasks import run_training_task

        run_training_task.delay(run.pk, contract.to_dict(), profile_name, horizon, seed)

    run.refresh_from_db()
    return run


def _reserve_run_id() -> str:
    """A run id that is unique in the database as well as on disk."""
    stamp = datetime.now().strftime("%Y-%m-%d")
    taken = set(
        TrainingRun.objects.filter(run_id__startswith=stamp).values_list("run_id", flat=True)
    )
    from ml.paths import RUNS_DIR

    if RUNS_DIR.exists():
        taken |= {p.name for p in RUNS_DIR.iterdir() if p.is_dir()}
    index = 1
    while f"{stamp}_{index:03d}" in taken:
        index += 1
    return f"{stamp}_{index:03d}"


def _execute(
    run_pk: int,
    contract: DataContract,
    profile_name: str,
    horizon: int,
    seed: int,
    future_covariates: str | None = None,
) -> None:
    from django.db import connection

    run = TrainingRun.objects.get(pk=run_pk)
    run.status = TrainingRun.Status.RUNNING
    run.stage = "starting"
    run.save(update_fields=["status", "stage"])

    def progress(stage: str, fraction: float) -> None:
        TrainingRun.objects.filter(pk=run_pk).update(stage=stage, progress=round(fraction, 3))

    try:
        config = TrainingConfig(
            contract=contract,
            profile_name=profile_name,
            horizon=horizon,
            seed=seed,
            run_id=run.run_id,
            future_overrides=load_future_overrides(future_covariates),
            progress=progress,
        )
        result = TrainingPipeline(config).run()
        _persist(run_pk, result)
    except Exception as exc:  # noqa: BLE001 - the failure belongs in the record
        log.exception("Training run %s failed", run.run_id)
        TrainingRun.objects.filter(pk=run_pk).update(
            status=TrainingRun.Status.FAILED,
            stage="failed",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
            finished_at=datetime.now(timezone.utc),
        )
    finally:
        connection.close()


def _persist(run_pk: int, result) -> None:
    from apps.forecasting.store import clear_cache
    from apps.scenarios.services import clear_cache as clear_scenario_cache

    metrics = result.metrics
    leaderboard = result.leaderboard
    champion = next((row for row in leaderboard if row["model"] == result.champion), None)
    baseline = next((row for row in leaderboard if row.get("is_baseline")), None)
    panel = metrics.get("panel") or {}

    run = TrainingRun.objects.get(pk=run_pk)
    run.status = TrainingRun.Status.SUCCEEDED
    run.stage = "done"
    run.progress = 1.0
    try:
        run.run_dir = str(result.run_dir.relative_to(REPO_ROOT))
    except ValueError:  # a custom RUNS_DIR outside the repository
        run.run_dir = str(result.run_dir)
    run.champion_model = result.champion
    run.champion_score = champion.get("primary_value") if champion else None
    run.baseline_model = baseline.get("model") if baseline else ""
    run.baseline_score = baseline.get("primary_value") if baseline else None
    run.improvement = champion.get("improvement_vs_baseline") if champion else None
    run.data_health_score = (metrics.get("data_quality") or {}).get("health_score")
    run.dataset_rows = panel.get("rows", 0)
    run.n_entities = panel.get("entities", 0)
    run.n_features = (result.artefacts and metrics.get("panel", {}).get("rows") and 0) or 0
    run.frequency = panel.get("frequency", "D")
    # SQLite enforces JSON_VALID on JSONFields, and NaN is not valid JSON.
    run.metrics = sanitise_json(
        {
            "leaderboard": leaderboard,
            "intervals": metrics.get("intervals"),
            "coverage_by_horizon": metrics.get("coverage_by_horizon"),
            "ensemble_weights": metrics.get("ensemble_weights"),
            "folds": metrics.get("folds"),
        }
    )
    run.warnings = sanitise_json(result.warnings)
    run.training_seconds = metrics.get("training_seconds")
    run.finished_at = datetime.now(timezone.utc)

    metadata_path = result.run_dir / "metadata.json"
    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run.n_features = int(metadata.get("n_features") or 0)
    run.save()

    ModelResult.objects.filter(run=run).delete()
    weights = metrics.get("ensemble_weights") or {}
    ModelResult.objects.bulk_create(
        [
            ModelResult(
                run=run,
                model_name=row["model"],
                label=_label_for(row["model"]),
                kind=_kind_for(row["model"]),
                is_baseline=bool(row.get("is_baseline")),
                is_champion=row["model"] == result.champion,
                rank=row.get("rank", 0),
                primary_metric=row.get("primary_metric", run.primary_metric),
                primary_value=row.get("primary_value"),
                improvement_vs_baseline=row.get("improvement_vs_baseline"),
                metrics=sanitise_json(row.get("metrics") or {}),
                by_horizon=sanitise_json(row.get("by_horizon") or []),
                ensemble_weight=weights.get(row["model"]),
                fit_seconds=row.get("fit_seconds"),
            )
            for row in leaderboard
        ]
    )

    clear_cache()
    clear_scenario_cache()
    _write_report(run, result)


def _label_for(name: str) -> str:
    entry = REGISTRY.get(name)
    return entry.label if entry else ("Ensemble" if name == "ensemble" else name)


def _kind_for(name: str) -> str:
    entry = REGISTRY.get(name)
    if entry:
        return entry.kind
    return "meta" if name == "ensemble" else ("baseline" if is_baseline(name) else "ml")


def _write_report(run: TrainingRun, result) -> None:
    try:
        from .reporting import write_model_report

        write_model_report(run, result)
    except Exception as exc:  # noqa: BLE001 - a report failure is not a run failure
        log.warning("Could not write the model report: %s", exc)


def profile_options() -> dict[str, Any]:
    from ml.contract import list_profiles

    return {
        name: {"description": description, **_profile_summary(name)}
        for name, description in list_profiles().items()
    }


def _profile_summary(name: str) -> dict[str, Any]:
    profile = load_profile(name)
    return {
        "models": profile.get("models", []),
        "cv_folds": profile.get("cv_folds"),
        "max_train_rows": profile.get("max_train_rows"),
        "tuning": profile.get("tuning", {}),
    }
