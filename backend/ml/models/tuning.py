"""Optional hyper-parameter search.

Optuna is only worth running when there is time to spare, so it is:

* off by default (`tuning.enabled: false` in every profile except `full`),
* strictly time-boxed by `tuning.max_minutes`,
* validated the same way everything else is - on a held-out *time* window, not
  a random split,
* completely skippable: if Optuna is not installed, or the search fails, the
  profile's default hyper-parameters are used and a warning is recorded.

Training must never depend on this module succeeding.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..evaluation.metrics import get_metric
from .base import FitContext, ForecastModel, PredictContext

log = logging.getLogger(__name__)

# Below this many out-of-sample points the search would tune to noise.
MIN_VALIDATION_POINTS = 50


def optuna_available() -> tuple[bool, str]:
    try:
        import optuna  # noqa: F401
    except ImportError:
        return False, "optuna is not installed (pip install -r requirements-optional.txt)"
    return True, "ready"


@dataclass
class TuningResult:
    model_name: str
    best_params: dict[str, Any]
    best_score: float
    baseline_score: float
    n_trials: int
    seconds: float
    improved: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "best_params": self.best_params,
            "best_score": round(float(self.best_score), 6),
            "default_score": round(float(self.baseline_score), 6),
            "n_trials": self.n_trials,
            "seconds": round(self.seconds, 1),
            "improved": self.improved,
            "note": self.note,
        }


# Search spaces. Deliberately narrow: a hackathon has minutes, not hours, and a
# wide space with 20 trials is worse than a narrow one with 20 trials.
SEARCH_SPACES: dict[str, Callable[[Any], dict[str, Any]]] = {
    "lightgbm": lambda trial: {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
    },
    "catboost": lambda trial: {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
    },
    "nhits": lambda trial: {
        "max_steps": trial.suggest_int("max_steps", 100, 800, step=100),
        "input_size_multiplier": trial.suggest_int("input_size_multiplier", 2, 5),
    },
}


def tune_model(
    model_name: str,
    factory: Callable[..., ForecastModel],
    base_params: dict[str, Any],
    fit_context: FitContext,
    predict_context: PredictContext,
    metric: str = "wape",
    max_minutes: float = 20.0,
    max_trials: int = 40,
    seed: int = 42,
) -> TuningResult | None:
    """Search hyper-parameters against a held-out time window.

    Always reports *why* nothing happened rather than failing silently:
    `None` means this model has no search space at all; otherwise a
    `TuningResult` comes back carrying the defaults and an accurate note.
    """
    space = SEARCH_SPACES.get(model_name)
    if space is None:
        return None

    def skipped(note: str) -> TuningResult:
        return TuningResult(
            model_name=model_name,
            best_params=dict(base_params),
            best_score=float("nan"),
            baseline_score=float("nan"),
            n_trials=0,
            seconds=0.0,
            improved=False,
            note=note,
        )

    available, reason = optuna_available()
    if not available:
        log.info("Skipping hyper-parameter search: %s", reason)
        return skipped(reason)

    actual = predict_context.frame.y
    if actual is None:
        return skipped("the validation window has no observed values to score against")
    mask = np.isfinite(actual)
    if mask.sum() < MIN_VALIDATION_POINTS:
        return skipped(
            f"only {int(mask.sum())} validation points (need {MIN_VALIDATION_POINTS}); "
            "tuning on this little data would overfit the search itself"
        )

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    spec = get_metric(metric)
    started = time.perf_counter()
    deadline = started + max_minutes * 60.0

    def score(params: dict[str, Any]) -> float:
        model = factory(**params)
        model.fit(fit_context)
        prediction = model.predict(predict_context)
        value = spec(actual[mask], prediction[mask])
        if not np.isfinite(value):
            return float("inf")
        # Optuna minimises, so flip metrics where higher is better.
        return -value if spec.greater_is_better else value

    try:
        baseline = score(base_params)
    except Exception as exc:  # noqa: BLE001 - a failed baseline means no tuning
        log.warning("Could not score default parameters for %s: %s", model_name, exc)
        return skipped(f"could not score the default parameters: {exc}")

    def objective(trial):
        if time.perf_counter() > deadline:
            trial.study.stop()
            raise optuna.TrialPruned()
        params = {**base_params, **space(trial)}
        try:
            return score(params)
        except Exception as exc:  # noqa: BLE001 - one bad trial is not fatal
            log.debug("trial failed: %s", exc)
            raise optuna.TrialPruned() from exc

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    try:
        study.optimize(
            objective,
            n_trials=max_trials,
            timeout=max(1.0, deadline - time.perf_counter()),
            catch=(Exception,),
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001 - never let the search abort a run
        log.warning("Hyper-parameter search for %s failed: %s", model_name, exc)
        return skipped(f"the search failed: {exc}")

    completed = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    if not completed:
        return TuningResult(
            model_name=model_name,
            best_params=dict(base_params),
            best_score=baseline,
            baseline_score=baseline,
            n_trials=0,
            seconds=time.perf_counter() - started,
            improved=False,
            note="no trial completed inside the time budget; keeping the defaults",
        )

    best = min(completed, key=lambda t: t.value)
    improved = best.value < baseline
    return TuningResult(
        model_name=model_name,
        best_params={**base_params, **best.params} if improved else dict(base_params),
        # Report scores in the metric's natural orientation.
        best_score=-best.value if spec.greater_is_better else best.value,
        baseline_score=-baseline if spec.greater_is_better else baseline,
        n_trials=len(completed),
        seconds=time.perf_counter() - started,
        improved=improved,
        note="" if improved else "the search did not beat the defaults; defaults kept",
    )
