"""The Phase 2 experiment harness.

Everything here answers one question: does this change beat the champion on
leakage-safe rolling WAPE? A model is not accepted for being more sophisticated,
and a feature group is not kept for sounding useful - each has to survive the
ablation, and the ones that do not are recorded as rejected rather than quietly
dropped.

Each fold is one simulated competition. The expensive parts - the pickup
baseline, the calibration fitted on *earlier* windows, the feature tensors - are
built once per fold and shared by every experiment, so an eight-stage ablation
costs eight model fits per fold rather than eight full rebuilds.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .backtest import annotate, fold_frame, score, _score_by
from .baseline import PickupBaseline
from .calibration import Calibrator, calibration_cutoffs
from .config import Pol4Config
from .dataset import build_training_frame, build_inference_frame, SupervisedFrame
from .features import GROUP_ORDER, feature_names
from .loader import CHECKIN, CITY, Pol4Data

log = logging.getLogger(__name__)

Predictor = Callable[["FoldContext"], np.ndarray]


@dataclass
class FoldContext:
    """Everything an experiment needs for one simulated competition."""

    data: Pol4Data
    config: Pol4Config
    cutoff: pd.Timestamp
    frame: pd.DataFrame               # grid + observed + actual + baseline prediction
    baseline: PickupBaseline
    calibrators: dict[str, Calibrator] = field(default_factory=dict)
    models: dict[Any, Any] = field(default_factory=dict, repr=False)
    diagnostics: dict[Any, dict[str, float]] = field(default_factory=dict, repr=False)
    _cache: dict[Any, Any] = field(default_factory=dict, repr=False)

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(
            self.cutoff + pd.Timedelta(days=1), periods=self.config.target_days
        )

    # Feature groups are additive column subsets, so the tensors are built once
    # per fold with every group and each ablation stage selects its columns.
    # Rebuilding per stage would repeatedly materialise a multi-million-row
    # frame for exactly the same numbers.
    def training_frame(self) -> SupervisedFrame:
        if "train" not in self._cache:
            started = time.perf_counter()
            self._cache["train"] = build_training_frame(
                self.data, self.cutoff, GROUP_ORDER, self.baseline, self.config
            )
            log.info(
                "  fold %s: %s training rows in %.1fs",
                self.cutoff.date(),
                f"{len(self._cache['train']):,}",
                time.perf_counter() - started,
            )
        return self._cache["train"]

    def inference_frame(self) -> SupervisedFrame:
        if "infer" not in self._cache:
            self._cache["infer"] = build_inference_frame(
                self.data, self.cutoff, self.target_dates, GROUP_ORDER, self.baseline, self.config
            )
        return self._cache["infer"]

    def release(self) -> None:
        """Drop the cached tensors once a fold is scored - they are large."""
        self._cache.clear()


def build_context(
    data: Pol4Data, cutoff: pd.Timestamp, config: Pol4Config
) -> FoldContext:
    """One fold, with the baseline scored and the calibrators fitted."""
    cutoff = pd.Timestamp(cutoff)
    baseline = PickupBaseline.fit(data, cutoff, config)
    frame = baseline.predict(fold_frame(data, cutoff, config))
    frame = annotate(frame, baseline.curves.province_of, config)

    # Calibration is fitted on earlier 30-day windows whose outcomes had all
    # closed by this fold's cutoff. Nothing from the fold itself is used.
    history: list[pd.DataFrame] = []
    for earlier in calibration_cutoffs(cutoff, config):
        try:
            earlier_model = PickupBaseline.fit(data, earlier, config)
            history.append(earlier_model.predict(fold_frame(data, earlier, config)))
        except ValueError:  # not enough history that far back
            continue
    calibration_history = (
        pd.concat(history, ignore_index=True) if history else frame.iloc[0:0]
    )

    # Every method is always present, even with no history to fit on: an empty
    # frame yields alpha = 1.0, which is the honest answer ("no evidence to
    # calibrate on"), and a missing key would be a crash at predict time.
    calibrators = {
        method: Calibrator.fit(calibration_history, method, config)
        for method in Calibrator.METHODS
    }
    return FoldContext(
        data=data,
        config=config,
        cutoff=cutoff,
        frame=frame,
        baseline=baseline,
        calibrators=calibrators,
    )


# ------------------------------------------------------------------ scoring
def breakdowns(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    return {
        "by_horizon": _score_by(frame, "horizon", column),
        "by_horizon_bucket": _score_by(frame, "horizon_bucket", column),
        "by_province": _score_by(frame, "province_code", column),
        "by_weekday": _score_by(frame, "weekday", column),
        "by_demand_bucket": _score_by(frame, "demand_bucket", column),
        "by_observation_state": _score_by(frame, "observation_state", column),
    }


def high_demand_scores(frame: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    """Error on the pairs that actually move WAPE.

    The demand distribution is extreme - the top 1% of pairs carry a quarter of
    all demand - so an improvement that only touches the long tail is worth
    almost nothing, and this makes that visible.
    """
    out: dict[str, dict[str, float]] = {}
    actual = frame["actual"].to_numpy(dtype=np.float64)
    for share in (0.01, 0.05, 0.10):
        threshold = np.quantile(actual, 1.0 - share)
        mask = actual >= threshold
        out[f"top_{int(share * 100)}pct"] = score(
            frame.loc[mask, "actual"], frame.loc[mask, column]
        )
    return out


@dataclass
class ExperimentResult:
    name: str
    stage: str
    pooled: dict[str, float]
    folds: dict[str, dict[str, float]]
    horizon: list[dict]
    detail: dict[str, Any]
    runtime_seconds: float
    meta: dict[str, Any] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    training: dict[str, float] = field(default_factory=dict)
    training_folds: dict[str, dict[str, float]] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        """One line for experiments.csv."""
        horizon = {f"wape_h{item['key']}": item["wape"] for item in self.horizon}
        folds = {f"wape_{cutoff}": s["wape"] for cutoff, s in self.folds.items()}
        return {
            "experiment": self.name,
            "stage": self.stage,
            "wape": self.pooled["wape"],
            "normalised_bias": self.pooled["normalised_bias"],
            "mae": self.pooled["mae"],
            "runtime_seconds": round(self.runtime_seconds, 2),
            "train_wape": self.training.get("wape"),
            "validation_minus_train_wape": (
                round(self.pooled["wape"] - self.training["wape"], 6)
                if "wape" in self.training
                else None
            ),
            "n_features": self.meta.get("n_features"),
            "model": self.meta.get("model", ""),
            "feature_groups": ",".join(self.meta.get("groups", ())),
            **folds,
            **horizon,
        }


def run_experiment(
    name: str,
    stage: str,
    contexts: list[FoldContext],
    predictor: Predictor,
    meta: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Score one predictor across every fold, then pool the predictions."""
    started = time.perf_counter()
    frames: list[pd.DataFrame] = []
    folds: dict[str, dict[str, float]] = {}

    for context in contexts:
        # Every experiment inherits the Phase 1 floor: a pair cannot finish
        # below what has already been counted.
        label = context.cutoff.date().isoformat()
        scored = _score_fold(context, predictor).assign(fold_cutoff=label)
        folds[context.cutoff.date().isoformat()] = score(scored["actual"], scored["prediction"])
        frames.append(scored)

    pooled_frame = pd.concat(frames, ignore_index=True)
    detail = breakdowns(pooled_frame, "prediction")
    detail["high_demand"] = high_demand_scores(pooled_frame, "prediction")
    return ExperimentResult(
        name=name,
        stage=stage,
        pooled=score(pooled_frame["actual"], pooled_frame["prediction"]),
        folds=folds,
        horizon=detail["by_horizon_bucket"],
        detail=detail,
        runtime_seconds=time.perf_counter() - started,
        meta=meta or {},
        predictions=pooled_frame.copy(),
    )


# -------------------------------------------------------------- predictors
def baseline_predictor(context: FoldContext) -> np.ndarray:
    return context.frame["predicted_demand"].to_numpy()


def calibrated_predictor(method: str) -> Predictor:
    def predict(context: FoldContext) -> np.ndarray:
        return context.calibrators[method].apply(context.frame)

    return predict


def gbdt_predictor(
    kind: str,
    groups: tuple[str, ...],
    log1p: bool = False,
    params: dict[str, Any] | None = None,
    calibration: str | None = None,
    blend: float | None = None,
    bands: tuple[tuple[int, int], ...] = ((1, 30),),
    measure_training: bool = False,
) -> Predictor:
    """Fit a remaining-demand model per fold and predict that fold's grid.

    `bands` splits the horizon: one model per band, fitted only on rows inside
    it. A single band is one global model with `days_to_checkin` as a feature.
    `blend` mixes the result with the pickup baseline named by `calibration`.
    """
    from .models import RemainingDemandModel

    columns = feature_names(groups)

    diagnostic_key = object()

    def predict(context: FoldContext) -> np.ndarray:
        train = context.training_frame()
        infer = context.inference_frame()
        train_horizon = train.meta["horizon"].to_numpy()
        infer_horizon = infer.meta["horizon"].to_numpy()

        predicted = np.full(len(infer.meta), np.nan)
        train_predicted = (
            np.full(len(train.meta), np.nan) if measure_training else None
        )
        for low, high in bands:
            fit_rows = (train_horizon >= low) & (train_horizon <= high)
            predict_rows = (infer_horizon >= low) & (infer_horizon <= high)
            if not fit_rows.any() or not predict_rows.any():
                continue
            model = RemainingDemandModel(
                kind=kind, params=params or {}, log1p=log1p, seed=context.config.seed
            ).fit(train.X.loc[fit_rows, columns], train.y[fit_rows])
            context.models[(kind, groups, log1p, (low, high))] = model
            if train_predicted is not None:
                train_predicted[fit_rows] = model.predict_final(
                    train.X.loc[fit_rows, columns],
                    train.meta.loc[fit_rows, "observed"].to_numpy(),
                )
            predicted[predict_rows] = model.predict_final(
                infer.X.loc[predict_rows, columns],
                infer.meta.loc[predict_rows, "observed"].to_numpy(),
            )
        aligned = _align(infer.meta, predicted, context.frame)

        covered = np.isfinite(train_predicted) if train_predicted is not None else None
        if covered is not None and covered.any() and train.y is not None:
            train_actual = (
                train.meta.loc[covered, "observed"].to_numpy(dtype=np.float64)
                + train.y[covered]
            )
            context.diagnostics[diagnostic_key] = score(
                train_actual, train_predicted[covered]
            )

        if blend is not None:
            other = context.calibrators[calibration or "none"].apply(context.frame)
            aligned = blend * aligned + (1.0 - blend) * other
        return aligned

    predict.diagnostic_key = diagnostic_key  # type: ignore[attr-defined]
    return predict


def _align(meta: pd.DataFrame, values: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Reorder model output onto the scoring frame's row order."""
    lookup = pd.Series(
        values, index=pd.MultiIndex.from_frame(meta[[CITY, CHECKIN]])
    )
    return lookup.reindex(pd.MultiIndex.from_frame(frame[[CITY, CHECKIN]])).to_numpy()


@dataclass
class ExperimentSpec:
    """One predictor to score, named for the experiments table."""

    name: str
    stage: str
    predictor: Predictor
    meta: dict[str, Any] = field(default_factory=dict)


def run_suite(
    data: Pol4Data, config: Pol4Config, specs: list[ExperimentSpec]
) -> list[ExperimentResult]:
    """Score every spec on every fold, one fold at a time.

    The loop is fold-outer rather than experiment-outer so a fold's feature
    tensors - roughly half a gigabyte each - are released before the next fold
    is built. Scoring experiment-outer would need all five resident at once,
    which is the difference between running on a laptop and not.
    """
    per_spec: dict[str, list[pd.DataFrame]] = {spec.name: [] for spec in specs}
    per_spec_folds: dict[str, dict[str, dict[str, float]]] = {spec.name: {} for spec in specs}
    runtimes: dict[str, float] = {spec.name: 0.0 for spec in specs}
    per_spec_training: dict[str, dict[str, dict[str, float]]] = {
        spec.name: {} for spec in specs
    }

    for cutoff in config.backtest_cutoffs:
        log.info("fold %s: building context", cutoff)
        context = build_context(data, pd.Timestamp(cutoff), config)
        label = context.cutoff.date().isoformat()
        for spec in specs:
            started = time.perf_counter()
            scored = _score_fold(context, spec.predictor).assign(fold_cutoff=label)
            runtimes[spec.name] += time.perf_counter() - started
            per_spec_folds[spec.name][label] = score(scored["actual"], scored["prediction"])
            per_spec[spec.name].append(scored)
            diagnostic_key = getattr(spec.predictor, "diagnostic_key", None)
            if diagnostic_key in context.diagnostics:
                per_spec_training[spec.name][label] = context.diagnostics[diagnostic_key]
            log.info(
                "  %-42s wape=%.4f", spec.name, per_spec_folds[spec.name][label]["wape"]
            )
        context.release()

    results = []
    for spec in specs:
        pooled_frame = pd.concat(per_spec[spec.name], ignore_index=True)
        detail = breakdowns(pooled_frame, "prediction")
        detail["high_demand"] = high_demand_scores(pooled_frame, "prediction")
        results.append(
            ExperimentResult(
                name=spec.name,
                stage=spec.stage,
                pooled=score(pooled_frame["actual"], pooled_frame["prediction"]),
                folds=per_spec_folds[spec.name],
                horizon=detail["by_horizon_bucket"],
                detail=detail,
                runtime_seconds=runtimes[spec.name],
                meta=spec.meta,
                predictions=pooled_frame.copy(),
                training=_pool_scores(per_spec_training[spec.name].values()),
                training_folds=per_spec_training[spec.name],
            )
        )
    return results


def _pool_scores(scores) -> dict[str, float]:
    """Pool already-scored disjoint frames without retaining their rows."""
    scores = list(scores)
    if not scores:
        return {}
    n = sum(int(item["n"]) for item in scores)
    actual = sum(float(item["actual_total"]) for item in scores)
    predicted = sum(float(item["predicted_total"]) for item in scores)
    absolute_error = sum(float(item["wape"]) * float(item["actual_total"]) for item in scores)
    return {
        "n": n,
        "actual_total": round(actual, 2),
        "predicted_total": round(predicted, 2),
        "wape": round(absolute_error / actual, 6) if actual else float("nan"),
        "mae": round(absolute_error / n, 4) if n else float("nan"),
        "normalised_bias": round((predicted - actual) / actual, 6)
        if actual
        else float("nan"),
    }


def fit_oof_calibrator(
    predictions: pd.DataFrame,
    config: Pol4Config,
    *,
    method: str = "bias_horizon_shrunk",
    outcomes_available_by: pd.Timestamp | None = None,
) -> Calibrator:
    """Fit a final calibrator on completed out-of-fold predictions only."""
    history = predictions.copy()
    if outcomes_available_by is not None:
        available = pd.to_datetime(history["fold_cutoff"]) + pd.Timedelta(
            days=config.target_days
        )
        history = history.loc[available <= pd.Timestamp(outcomes_available_by)]
    frame = history[["actual", "observed", "horizon"]].assign(
        predicted_demand=history["prediction"].to_numpy()
    )
    return Calibrator.fit(frame, method, config)


def cross_fit_calibration(
    raw: ExperimentResult,
    config: Pol4Config,
    *,
    method: str = "bias_horizon_shrunk",
) -> tuple[ExperimentResult, Calibrator]:
    """Calibrate each fold using only earlier, fully-observed OOF folds.

    The returned calibrator is fitted on all OOF predictions and is therefore
    valid for the later competition cutoff. It is never used to score the same
    rows that fitted it; the reported backtest remains strictly cross-fitted.
    """
    calibrated_frames: list[pd.DataFrame] = []
    fold_scores: dict[str, dict[str, float]] = {}
    predictions = raw.predictions.copy()

    for cutoff_value in config.backtest_cutoffs:
        cutoff = pd.Timestamp(cutoff_value)
        label = cutoff.date().isoformat()
        current = predictions.loc[predictions["fold_cutoff"] == label].copy()
        if current.empty:
            continue
        outcome_dates = pd.to_datetime(predictions["fold_cutoff"]) + pd.Timedelta(
            days=config.target_days
        )
        history = predictions.loc[outcome_dates <= cutoff]
        calibration_frame = history[["actual", "observed", "horizon"]].assign(
            predicted_demand=history["prediction"].to_numpy()
        )
        calibrator = Calibrator.fit(calibration_frame, method, config)
        current["prediction"] = calibrator.apply(current, column="prediction")
        fold_scores[label] = score(current["actual"], current["prediction"])
        calibrated_frames.append(current)

    if not calibrated_frames:
        raise ValueError("cross-fitted calibration received no matching folds")

    pooled = pd.concat(calibrated_frames, ignore_index=True)
    detail = breakdowns(pooled, "prediction")
    detail["high_demand"] = high_demand_scores(pooled, "prediction")
    result = ExperimentResult(
        name=f"CHAMPION calibrated ({method})",
        stage="champion",
        pooled=score(pooled["actual"], pooled["prediction"]),
        folds=fold_scores,
        horizon=detail["by_horizon_bucket"],
        detail=detail,
        runtime_seconds=raw.runtime_seconds,
        meta={**raw.meta, "calibration": method, "source": raw.name},
        predictions=pooled,
        training=raw.training,
        training_folds=raw.training_folds,
    )
    return result, fit_oof_calibrator(predictions, config, method=method)


def _score_fold(context: FoldContext, predictor: Predictor) -> pd.DataFrame:
    """Run one predictor on one fold, imposing the observed floor."""
    predicted = np.asarray(predictor(context), dtype=np.float64)
    observed = context.frame["observed"].to_numpy(dtype=np.float64)
    predicted = np.maximum(np.nan_to_num(predicted, nan=0.0), observed)
    return context.frame.assign(prediction=predicted)


def staged_groups() -> list[tuple[str, tuple[str, ...]]]:
    """The ablation ladder: each stage adds exactly one feature group."""
    stages: list[tuple[str, tuple[str, ...]]] = []
    for index, group in enumerate(GROUP_ORDER):
        stages.append((group, tuple(GROUP_ORDER[: index + 1])))
    return stages
