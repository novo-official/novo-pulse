"""Rolling-origin backtesting and the model leaderboard.

For every fold the models are refitted on data strictly before the cutoff and
scored on the horizon that follows. Scores are reported overall *and* per
horizon bucket, which is what shows whether a model degrades over a 90-day
horizon or holds up.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..contract import ENTITY, DataContract
from ..features.engineering import FeatureEngine
from ..models.base import FitContext, ForecastModel, ModelUnavailable, PredictContext
from ..models.registry import is_baseline
from .metrics import evaluate, get_metric
from .splitters import Fold, RollingOriginSplitter

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame                       # long: model, entity, ds, horizon, ...
    fold_scores: pd.DataFrame                       # model x fold x metric
    model_scores: dict[str, dict[str, float]]       # model -> metric -> value
    horizon_scores: dict[str, list[dict[str, Any]]] # model -> per-bucket metrics
    folds: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


class Backtester:
    def __init__(
        self,
        contract: DataContract,
        engine: FeatureEngine,
        profile: dict[str, Any],
        seed: int = 42,
        log1p: bool = False,
        horizon: int | None = None,
    ):
        self.contract = contract
        self.engine = engine
        self.profile = profile
        self.seed = seed
        self.log1p = log1p
        self.metric = contract.evaluation.primary_metric
        # Evaluate at the horizon we will actually serve. Scoring 90 days when
        # the deployed forecast is 30 would report a degradation the product
        # never exposes.
        self.horizon = int(horizon or contract.evaluation.max_horizon)

    # ------------------------------------------------------------------ run
    def run(
        self,
        model_factory,
        n_folds: int | None = None,
        step: int | None = None,
    ) -> BacktestResult:
        tensor = self.engine.tensor
        splitter = RollingOriginSplitter(
            horizon=self.horizon,
            n_folds=n_folds or self.contract.evaluation.n_folds,
            step=step or self.contract.evaluation.step or self.horizon,
            freq=tensor.freq,
            min_train_periods=max(self.horizon * 2, 60),
        )
        observed_dates = tensor.dates[: tensor.n_observed]
        folds = splitter.split(pd.Series(observed_dates))
        if not folds:
            raise ValueError("Not enough history for even one backtest fold")

        date_to_idx = {date: i for i, date in enumerate(tensor.dates)}
        rows: list[pd.DataFrame] = []
        failures: list[dict[str, str]] = []
        timings: dict[str, float] = {}

        for fold in folds:
            origin_idx = date_to_idx[fold.train_end]
            log.info("Backtest fold %s | origin=%s", fold.index, fold.train_end.date())

            train_frame = self.engine.build_training(
                train_end_idx=origin_idx,
                samples_per_target=int(self.profile.get("samples_per_target", 3)),
                max_rows=int(self.profile.get("max_train_rows", 500_000)),
                seed=self.seed + fold.index,
            )
            inference_frame = self.engine.build_inference(origin_idx, self.horizon)
            if inference_frame.y is None:
                continue

            fit_context = FitContext(
                engine=self.engine,
                train_end_idx=origin_idx,
                frame=train_frame,
                seed=self.seed,
                quantiles=tuple(self.contract.evaluation.quantiles),
                non_negative=self.contract.target_options.non_negative,
                log1p=self.log1p,
            )
            predict_context = PredictContext(
                engine=self.engine,
                origin_idx=origin_idx,
                horizon=self.horizon,
                frame=inference_frame,
            )

            models = model_factory()
            for name, model in models.items():
                started = time.perf_counter()
                try:
                    model.fit(fit_context)
                    point = model.predict(predict_context)
                    quantiles = model.predict_quantiles(
                        predict_context, tuple(self.contract.evaluation.quantiles)
                    )
                except ModelUnavailable as exc:
                    failures.append({"model": name, "fold": str(fold.index), "reason": str(exc)})
                    continue
                except Exception as exc:  # noqa: BLE001 - one model must not kill the run
                    failures.append(
                        {"model": name, "fold": str(fold.index), "reason": f"{type(exc).__name__}: {exc}"}
                    )
                    log.warning("Model %s failed on fold %s: %s", name, fold.index, exc)
                    continue
                timings[name] = timings.get(name, 0.0) + (time.perf_counter() - started)

                piece = inference_frame.meta[[ENTITY, "ds", "horizon"]].copy()
                piece["model"] = name
                piece["fold"] = fold.index
                piece["actual"] = inference_frame.y
                piece["prediction"] = point
                lower_q, upper_q = _tail_quantiles(self.contract.evaluation.quantiles)
                piece["lower"] = quantiles.get(lower_q, np.full(len(piece), np.nan))
                piece["upper"] = quantiles.get(upper_q, np.full(len(piece), np.nan))
                rows.append(piece)

        if not rows:
            raise RuntimeError(
                "Every model failed during backtesting. Failures: "
                + "; ".join(f"{f['model']}: {f['reason']}" for f in failures[:5])
            )

        predictions = pd.concat(rows, ignore_index=True)
        predictions = predictions[np.isfinite(predictions["actual"])]

        fold_scores = self._fold_scores(predictions)
        model_scores = self._model_scores(predictions)
        horizon_scores = self._horizon_scores(predictions)

        return BacktestResult(
            predictions=predictions,
            fold_scores=fold_scores,
            model_scores=model_scores,
            horizon_scores=horizon_scores,
            folds=[f.as_dict() for f in folds],
            failures=failures,
            timings={k: round(v, 2) for k, v in timings.items()},
        )

    # -------------------------------------------------------------- scoring
    def _metric_names(self) -> list[str]:
        names = [self.metric, *self.contract.evaluation.secondary_metrics]
        return list(dict.fromkeys(names))

    def _fold_scores(self, predictions: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, fold), group in predictions.groupby(["model", "fold"]):
            scores = evaluate(
                group["actual"],
                group["prediction"],
                self._metric_names(),
                self.contract.target_options.non_negative,
            )
            rows.append({"model": model, "fold": int(fold), "n": len(group), **scores})
        return pd.DataFrame(rows)

    def _model_scores(self, predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for model, group in predictions.groupby("model"):
            scores = evaluate(
                group["actual"],
                group["prediction"],
                self._metric_names(),
                self.contract.target_options.non_negative,
            )
            scores["n_predictions"] = int(len(group))
            out[str(model)] = scores
        return out

    def _horizon_scores(self, predictions: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        buckets = [b for b in self.contract.evaluation.horizon_buckets if b[0] <= self.horizon]
        for model, group in predictions.groupby("model"):
            entries = []
            for low, high in buckets:
                window = group[(group["horizon"] >= low) & (group["horizon"] <= high)]
                if window.empty:
                    continue
                scores = evaluate(
                    window["actual"],
                    window["prediction"],
                    self._metric_names(),
                    self.contract.target_options.non_negative,
                )
                entries.append(
                    {"bucket": f"{low}-{high}", "horizon_min": low, "horizon_max": high,
                     "n": int(len(window)), **scores}
                )
            out[str(model)] = entries
        return out

    # ---------------------------------------------------------- leaderboard
    def leaderboard(self, result: BacktestResult) -> list[dict[str, Any]]:
        spec = get_metric(self.metric)
        entries = []
        for model, scores in result.model_scores.items():
            primary = scores.get(self.metric)
            entries.append(
                {
                    "model": model,
                    "is_baseline": is_baseline(model),
                    "primary_metric": self.metric,
                    "primary_value": primary,
                    "metrics": scores,
                    "by_horizon": result.horizon_scores.get(model, []),
                    "fit_seconds": result.timings.get(model),
                }
            )
        entries.sort(
            key=lambda item: (
                item["primary_value"] is None,
                -(item["primary_value"] or 0) if spec.greater_is_better else (item["primary_value"] or 0),
            )
        )
        for rank, entry in enumerate(entries, start=1):
            entry["rank"] = rank

        baselines = [e for e in entries if e["is_baseline"] and e["primary_value"] is not None]
        best_baseline = baselines[0] if baselines else None
        for entry in entries:
            if best_baseline and entry["primary_value"] is not None:
                base = best_baseline["primary_value"]
                if base and base != 0:
                    improvement = (
                        (entry["primary_value"] - base) / abs(base)
                        if spec.greater_is_better
                        else (base - entry["primary_value"]) / abs(base)
                    )
                    entry["improvement_vs_baseline"] = round(float(improvement), 4)
                    entry["baseline_model"] = best_baseline["model"]
        return entries


def _tail_quantiles(quantiles: list[float]) -> tuple[float, float]:
    ordered = sorted(float(q) for q in quantiles)
    return ordered[0], ordered[-1]
