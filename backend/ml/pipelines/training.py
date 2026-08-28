"""The end-to-end training pipeline.

    load -> validate -> normalise -> feature engineer -> rolling backtest
         -> compare -> ensemble -> calibrate -> refit on all data
         -> forecast -> explain -> detect anomalies -> save artefacts

Every stage is defensive: an optional model that fails is recorded and skipped,
never allowed to abort the run.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..anomaly.detector import (
    AnomalyConfig,
    detect_forecast_anomalies,
    detect_peak_periods,
    detect_residual_anomalies,
)
from ..contract import ENTITY, TARGET, TS, DataContract, load_profile
from ..data.adapter import DataAdapter, Panel, read_tabular
from ..data.censoring import analyse as analyse_censoring
from ..data.validator import DataValidator
from ..evaluation.backtest import Backtester
from ..evaluation.metrics import evaluate_intervals, get_metric
from ..evaluation.uncertainty import (
    ConformalCalibrator,
    buckets_from_contract,
    coverage_by_horizon,
)
from ..explainability.shap_explainer import ShapExplainer, dependence_curve
from ..features.engineering import FeatureConfig, FeatureEngine
from ..features.tensor import build_tensor
from ..hierarchy import aggregate_all_levels, aggregate_bottom_up, coherence_check, entity_map
from ..insights.engine import build_decision_opportunities, build_insights
from ..models.base import FitContext, ModelUnavailable, PredictContext
from ..models.ensemble import EnsembleModel, compute_weights
from ..models.registry import REGISTRY, build_models, is_baseline
from ..models.tuning import tune_model
from ..paths import REPO_ROOT, RUNS_DIR, ensure_dirs
from ..resources import detect_resources

log = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Reproducibility: seed every RNG we can reach."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class TrainingConfig:
    contract: DataContract
    profile_name: str = "demo"
    horizon: int | None = None
    seed: int = 42
    run_id: str | None = None
    runs_dir: Path | None = None
    future_overrides: pd.DataFrame | None = None
    progress: Callable[[str, float], None] | None = None
    profile: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile:
            self.profile = load_profile(self.profile_name)
        if self.horizon:
            self.contract.evaluation.horizons = sorted(
                {*self.contract.evaluation.horizons, int(self.horizon)}
            )


@dataclass
class TrainingResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    champion: str
    forecast: pd.DataFrame
    backtest: pd.DataFrame
    insights: dict[str, Any]
    artefacts: dict[str, str]
    warnings: list[str] = field(default_factory=list)


class TrainingPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.contract = config.contract
        self.profile = config.profile
        self.warnings: list[str] = []
        self.tuning_results: list[dict[str, Any]] = []
        self.run_id = config.run_id or _new_run_id(config.runs_dir or RUNS_DIR)
        self.run_dir = (config.runs_dir or RUNS_DIR) / self.run_id
        self.started = time.perf_counter()

    # ------------------------------------------------------------------ run
    def run(self) -> TrainingResult:
        ensure_dirs()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        set_global_seed(self.config.seed)
        self._progress("loading dataset", 0.02)

        panel = DataAdapter(self.contract).build()
        self._progress("validating data", 0.08)
        quality = DataValidator(self.contract).validate(panel.frame)
        for finding in quality["findings"]:
            if finding["severity"] in {"critical", "high"}:
                self.warnings.append(f"{finding['title']}: {finding['detail']}")

        horizon = int(self.config.horizon or self.contract.evaluation.max_horizon)
        log1p = self._decide_log1p(panel)
        engine = self._build_engine(panel, horizon)

        # Demand censoring: how much of the history was capped by supply?
        censoring = analyse_censoring(panel.frame, self.contract, engine.config.season)
        if censoring.enabled and censoring.censored_share > 0.15:
            self.warnings.append(
                f"{censoring.censored_share:.0%} of observed periods sit at capacity - "
                "the target understates true market demand where supply binds"
            )

        self._progress("rolling backtest", 0.15)
        backtester = Backtester(
            contract=self.contract,
            engine=engine,
            profile=self.profile,
            seed=self.config.seed,
            log1p=log1p,
            horizon=horizon,
        )
        season = engine.config.season
        model_names = list(self.profile.get("models") or ["seasonal_naive_7", "lightgbm"])

        def factory():
            models, skipped = build_models(model_names, self.profile, season)
            for entry in skipped:
                message = f"model '{entry['model']}' skipped: {entry['reason']}"
                if message not in self.warnings:
                    self.warnings.append(message)
            return models

        backtest_result = backtester.run(
            factory, n_folds=int(self.profile.get("cv_folds", self.contract.evaluation.n_folds))
        )
        for failure in backtest_result.failures:
            message = f"model '{failure['model']}' failed on fold {failure['fold']}: {failure['reason']}"
            if message not in self.warnings:
                self.warnings.append(message)

        leaderboard = backtester.leaderboard(backtest_result)
        self._progress("selecting champion", 0.55)

        # ---- ensemble weights from the backtest, then refit on everything ---
        metric = self.contract.evaluation.primary_metric
        cv_scores = _ensemble_candidates(backtest_result.model_scores, metric)
        weights = compute_weights(cv_scores, metric)

        tuned_params = self._tune(engine, model_names, log1p, backtest_result)

        self._progress("refitting on full history", 0.6)
        origin_idx = engine.tensor.origin_index
        final_train = engine.build_training(
            train_end_idx=origin_idx,
            samples_per_target=int(self.profile.get("samples_per_target", 3)),
            max_rows=int(self.profile.get("max_train_rows", 500_000)),
            seed=self.config.seed,
        )
        fit_context = FitContext(
            engine=engine,
            train_end_idx=origin_idx,
            frame=final_train,
            seed=self.config.seed,
            quantiles=tuple(self.contract.evaluation.quantiles),
            non_negative=self.contract.target_options.non_negative,
            log1p=log1p,
        )
        final_models, skipped = build_models(
            model_names, self._profile_with(tuned_params), season
        )
        for entry in skipped:
            message = f"model '{entry['model']}' skipped: {entry['reason']}"
            if message not in self.warnings:
                self.warnings.append(message)

        fitted: dict[str, Any] = {}
        for name, model in final_models.items():
            try:
                model.fit(fit_context)
                fitted[name] = model
            except ModelUnavailable as exc:
                self.warnings.append(f"model '{name}' unavailable at refit: {exc}")
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"model '{name}' failed at refit: {type(exc).__name__}: {exc}")

        if not fitted:
            raise RuntimeError("No model could be fitted on the full history")

        ensemble_members = {k: v for k, v in fitted.items() if k in weights}
        ensemble = None
        if len(ensemble_members) >= 2:
            ensemble = EnsembleModel(members=ensemble_members, weights=weights)
            leaderboard = self._score_ensemble(
                backtester, backtest_result, leaderboard, weights, metric
            )

        champion_name, champion_model = self._pick_champion(leaderboard, fitted, ensemble)
        self._progress(f"forecasting with {champion_name}", 0.72)

        # ---- future forecast ------------------------------------------------
        inference_frame = engine.build_inference(origin_idx, horizon)
        predict_context = PredictContext(
            engine=engine, origin_idx=origin_idx, horizon=horizon, frame=inference_frame
        )
        point = champion_model.predict(predict_context)
        quantiles = champion_model.predict_quantiles(
            predict_context, tuple(self.contract.evaluation.quantiles)
        )

        calibrator = self._calibrate(backtest_result, champion_name, quantiles)
        lower_q, upper_q = min(self.contract.evaluation.quantiles), max(
            self.contract.evaluation.quantiles
        )
        horizons = inference_frame.meta["horizon"].to_numpy()
        if lower_q in quantiles and upper_q in quantiles:
            lower, upper = quantiles[lower_q], quantiles[upper_q]
            uncertainty_method = "native model quantiles"
        else:
            conformal = calibrator.apply(
                point, horizons, self.contract.target_options.non_negative
            )
            lower = conformal.get(lower_q, point * 0.8)
            upper = conformal.get(upper_q, point * 1.2)
            uncertainty_method = calibrator.describe()["method"]

        forecast = inference_frame.meta[[ENTITY, "ds", "horizon"]].copy()
        forecast["forecast"] = point
        forecast["lower"] = np.minimum(lower, point)
        forecast["upper"] = np.maximum(upper, point)
        forecast["model"] = champion_name
        if self.contract.target_options.integer:
            # A count target cannot be fractional. Bounds round outwards so the
            # interval never narrows as a side effect of rounding.
            forecast["forecast"] = np.round(forecast["forecast"])
            forecast["lower"] = np.floor(forecast["lower"])
            forecast["upper"] = np.ceil(forecast["upper"])

        # ---- champion backtest slice ---------------------------------------
        champion_backtest = backtest_result.predictions[
            backtest_result.predictions["model"] == champion_name
        ].copy()
        if champion_backtest.empty:
            champion_backtest = backtest_result.predictions.copy()

        self._progress("explaining drivers", 0.82)
        explanation, dependence = self._explain(engine, champion_model, fitted, inference_frame)

        self._progress("detecting anomalies", 0.88)
        history = panel.frame[[ENTITY, TS, TARGET]].rename(columns={TS: "ds", TARGET: "y"})
        mapping = entity_map(panel.frame)
        labels = _entity_labels(panel, mapping)
        # Anomalies and peaks are detected on the coarsest hierarchy level the
        # dataset offers. A single listing swinging from 1 to 2 bookings is
        # noise; a destination moving 30% is the signal a planner acts on.
        anomaly_level = "destination" if "destination" in panel.available_levels() else "listing"
        anomalies, peaks = self._anomalies(
            champion_backtest, forecast, history, season, mapping, anomaly_level
        )

        self._progress("building insights", 0.92)
        # Insights are reported at the same level as the anomalies, which is
        # the level a planner actually thinks in.
        insight_forecast = (
            forecast if anomaly_level == "listing"
            else aggregate_bottom_up(forecast, mapping, anomaly_level)
        )
        insight_history = (
            history if anomaly_level == "listing"
            else aggregate_bottom_up(history, mapping, anomaly_level)
        )
        insights = build_insights(
            forecast=insight_forecast,
            history=insight_history,
            leaderboard=leaderboard,
            peaks=peaks,
            anomalies=anomalies,
            primary_metric=metric,
            horizon=horizon,
            labels=labels,
        )
        insights["level"] = anomaly_level
        insights["decision_opportunities"] = build_decision_opportunities(insights, labels)
        insights["data_quality"] = quality

        stacked_forecast = aggregate_all_levels(forecast, mapping, panel.available_levels())
        # Rolling-origin folds overlap, so one (entity, date) can be predicted by
        # several folds. Metrics keep every fold - each is an independent
        # out-of-sample observation - but the chart artefact must not sum them,
        # so folds are averaged before the hierarchy roll-up.
        stacked_backtest = aggregate_all_levels(
            _collapse_folds(champion_backtest), mapping, panel.available_levels()
        )
        stacked_history = aggregate_all_levels(
            history.rename(columns={"y": "y"}), mapping, panel.available_levels()
        )

        interval_metrics = evaluate_intervals(
            champion_backtest["actual"],
            champion_backtest["lower"].fillna(champion_backtest["prediction"]),
            champion_backtest["upper"].fillna(champion_backtest["prediction"]),
            nominal=round(upper_q - lower_q, 4),
            y_median=champion_backtest["prediction"],
        )

        metrics = {
            "run_id": self.run_id,
            "champion": champion_name,
            "primary_metric": metric,
            "horizon": horizon,
            "leaderboard": leaderboard,
            "model_scores": backtest_result.model_scores,
            "horizon_scores": backtest_result.horizon_scores,
            "fold_scores": backtest_result.fold_scores.to_dict(orient="records"),
            "folds": backtest_result.folds,
            "intervals": interval_metrics,
            "coverage_by_horizon": coverage_by_horizon(
                champion_backtest, buckets_from_contract(self.contract.evaluation.horizon_buckets),
                nominal=round(upper_q - lower_q, 4),
            ),
            "ensemble_weights": weights,
            "uncertainty_method": uncertainty_method,
            "tuning": self.tuning_results,
            "censoring": censoring.as_dict(),
            "segment_scores": _segment_scores(champion_backtest, mapping, metric),
            "hierarchy_coherence": coherence_check(stacked_forecast),
            "data_quality": quality,
            "panel": panel.summary(),
            "training_seconds": round(time.perf_counter() - self.started, 2),
            "warnings": self.warnings,
        }

        self._progress("saving artefacts", 0.96)
        artefacts = self._save(
            panel=panel,
            engine=engine,
            metrics=metrics,
            forecast=stacked_forecast,
            backtest=stacked_backtest,
            history=stacked_history,
            anomalies=anomalies,
            peaks=peaks,
            insights=insights,
            explanation=explanation,
            dependence=dependence,
            models=fitted,
            ensemble=ensemble,
            champion=champion_name,
            calibrator=calibrator,
            labels=labels,
            mapping=mapping,
        )
        self._progress("done", 1.0)

        return TrainingResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            metrics=metrics,
            leaderboard=leaderboard,
            champion=champion_name,
            forecast=stacked_forecast,
            backtest=stacked_backtest,
            insights=insights,
            artefacts=artefacts,
            warnings=self.warnings,
        )

    # -------------------------------------------------------------- helpers
    def _progress(self, stage: str, fraction: float) -> None:
        log.info("[%s] %s (%.0f%%)", self.run_id, stage, fraction * 100)
        if self.config.progress:
            try:
                self.config.progress(stage, fraction)
            except Exception:  # noqa: BLE001 - progress reporting is best effort
                pass

    def _decide_log1p(self, panel: Panel) -> bool:
        option = self.contract.target_options.log1p_transform
        if isinstance(option, bool):
            return option
        target = panel.frame[TARGET]
        if not self.contract.target_options.non_negative:
            return False
        # Heavily skewed or zero-heavy counts benefit from a log1p target.
        zero_ratio = float((target == 0).mean())
        mean, std = float(target.mean()), float(target.std())
        skewed = std > 1.5 * max(mean, 1e-9)
        decision = bool(zero_ratio > 0.4 or skewed)
        if decision:
            self.warnings.append(
                f"target is skewed (zero ratio {zero_ratio:.2f}) - training on log1p(target)"
            )
        return decision

    def _build_engine(self, panel: Panel, horizon: int) -> FeatureEngine:
        overrides = self.config.future_overrides
        tensor = build_tensor(
            panel.frame,
            freq=panel.frequency,
            horizon=horizon,
            future_features=panel.future_features,
            past_features=panel.historical_features,
            static_features=panel.static_features,
            future_overrides=overrides,
        )
        config = FeatureConfig.for_frequency(panel.frequency, max_horizon=horizon)
        return FeatureEngine(tensor, config).prepare()

    def _tune(self, engine, model_names, log1p, backtest_result) -> dict[str, dict[str, Any]]:
        """Optional Optuna search, validated on a held-out time window.

        Runs only when the profile asks for it, is hard-capped by
        `tuning.max_minutes`, and never blocks a run: any failure falls through
        to the profile defaults with a warning.
        """
        settings = self.profile.get("tuning") or {}
        if not settings.get("enabled"):
            return {}
        budget = float(settings.get("max_minutes") or 0)
        if budget <= 0:
            return {}

        tunable = [n for n in model_names if n in {"lightgbm", "catboost"} and not is_baseline(n)]
        if not tunable:
            return {}

        self._progress("hyper-parameter search", 0.56)
        horizon = engine.config.max_horizon
        # Validate on the last window, exactly as the final fold does.
        origin = engine.tensor.origin_index - horizon
        if origin <= engine.config.min_context:
            self.warnings.append("history too short for hyper-parameter search; defaults kept")
            return {}

        train = engine.build_training(
            train_end_idx=origin,
            samples_per_target=int(self.profile.get("samples_per_target", 3)),
            max_rows=int(self.profile.get("max_train_rows", 500_000)),
            seed=self.config.seed,
        )
        fit_context = FitContext(
            engine=engine,
            train_end_idx=origin,
            frame=train,
            seed=self.config.seed,
            quantiles=tuple(self.contract.evaluation.quantiles),
            non_negative=self.contract.target_options.non_negative,
            log1p=log1p,
        )
        predict_context = PredictContext(
            engine=engine,
            origin_idx=origin,
            horizon=horizon,
            frame=engine.build_inference(origin, horizon),
        )

        per_model = budget / len(tunable)
        results: dict[str, dict[str, Any]] = {}
        for name in tunable:
            entry = REGISTRY.get(name)
            if entry is None:
                continue
            outcome = tune_model(
                model_name=name,
                factory=entry.build,
                base_params=dict(self.profile.get(name) or {}),
                fit_context=fit_context,
                predict_context=predict_context,
                metric=self.contract.evaluation.primary_metric,
                max_minutes=per_model,
                seed=self.config.seed,
            )
            if outcome is None:
                continue
            self.tuning_results.append(outcome.as_dict())
            if outcome.n_trials == 0 and outcome.note:
                self.warnings.append(
                    f"hyper-parameter search skipped for '{name}': {outcome.note}"
                )
            elif outcome.improved:
                results[name] = outcome.best_params
                log.info(
                    "Tuned %s: %s -> %s over %d trials",
                    name, outcome.baseline_score, outcome.best_score, outcome.n_trials,
                )
            elif outcome.note:
                self.warnings.append(f"'{name}': {outcome.note}")
        return results

    def _profile_with(self, tuned: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """The profile with any tuned hyper-parameters merged in."""
        if not tuned:
            return self.profile
        merged = dict(self.profile)
        for name, params in tuned.items():
            merged[name] = {**(merged.get(name) or {}), **params}
        return merged

    def _score_ensemble(
        self, backtester, backtest_result, leaderboard, weights, metric
    ) -> list[dict[str, Any]]:
        """Score the ensemble on the *same* backtest predictions as its members."""
        predictions = backtest_result.predictions
        pivot = predictions.pivot_table(
            index=[ENTITY, "ds", "horizon", "fold"], columns="model", values="prediction"
        )
        available = [m for m in weights if m in pivot.columns]
        if len(available) < 2:
            return leaderboard
        normaliser = sum(weights[m] for m in available)
        blended = sum(pivot[m].fillna(0) * weights[m] for m in available) / normaliser
        actual = predictions.pivot_table(
            index=[ENTITY, "ds", "horizon", "fold"], columns="model", values="actual"
        ).mean(axis=1)

        frame = pd.DataFrame(
            {"actual": actual, "prediction": blended}
        ).reset_index().dropna(subset=["actual", "prediction"])
        frame["model"] = "ensemble"
        frame["lower"] = np.nan
        frame["upper"] = np.nan

        combined = pd.concat([predictions, frame], ignore_index=True)
        backtest_result.predictions = combined
        backtest_result.model_scores = backtester._model_scores(combined)
        backtest_result.horizon_scores = backtester._horizon_scores(combined)
        backtest_result.fold_scores = backtester._fold_scores(combined)
        return backtester.leaderboard(backtest_result)

    def _pick_champion(self, leaderboard, fitted, ensemble):
        """Champion = best primary metric among models we can actually serve."""
        serveable = dict(fitted)
        if ensemble is not None:
            serveable["ensemble"] = ensemble
        for entry in leaderboard:
            name = entry["model"]
            if name in serveable and entry.get("primary_value") is not None:
                return name, serveable[name]
        name = next(iter(serveable))
        self.warnings.append(f"no leaderboard entry was serveable; falling back to '{name}'")
        return name, serveable[name]

    def _calibrate(self, backtest_result, champion_name, native_quantiles) -> ConformalCalibrator:
        buckets = buckets_from_contract(self.contract.evaluation.horizon_buckets)
        quantiles = (
            min(self.contract.evaluation.quantiles),
            max(self.contract.evaluation.quantiles),
        )
        calibrator = ConformalCalibrator(quantiles=quantiles, buckets=buckets)
        slice_ = backtest_result.predictions[
            backtest_result.predictions["model"] == champion_name
        ]
        if slice_.empty:
            slice_ = backtest_result.predictions
        calibrator.fit(
            slice_["actual"].to_numpy(),
            slice_["prediction"].to_numpy(),
            slice_["horizon"].to_numpy(),
        )
        return calibrator

    def _explain(self, engine, champion_model, fitted, inference_frame):
        """SHAP on a tree model - the champion when possible, else any tree."""
        tree_model = champion_model if getattr(champion_model, "family", "") == "tabular" else None
        if tree_model is None:
            tree_model = next(
                (m for m in fitted.values() if getattr(m, "family", "") == "tabular"), None
            )
        if tree_model is None:
            self.warnings.append("no tree model available - explainability is limited")
            return {"global_importance": [], "group_importance": [], "method": "unavailable"}, []

        explainer = ShapExplainer(tree_model, engine.feature_groups()).prepare(inference_frame.X)
        result = explainer.explain_global(inference_frame.X)
        dependence: list[dict[str, Any]] = []
        price_features = [c for c in inference_frame.X.columns if c.startswith("fut_") and "price" in c]
        if price_features:
            sample = inference_frame.X.sample(
                min(len(inference_frame.X), 4000), random_state=42
            )
            values = explainer.shap_values(sample)
            if values is not None:
                for feature in price_features[:2]:
                    curve = dependence_curve(sample, values, feature)
                    if curve:
                        dependence.append({"feature": feature, "curve": curve})
        return result.as_dict(), dependence

    def _anomalies(self, champion_backtest, forecast, history, season, mapping, level):
        config = AnomalyConfig()
        if level != "listing":
            champion_backtest = aggregate_bottom_up(_collapse_folds(champion_backtest), mapping, level)
            forecast = aggregate_bottom_up(forecast, mapping, level)
            history = aggregate_bottom_up(history, mapping, level)
        residual = detect_residual_anomalies(champion_backtest, config)
        forward = detect_forecast_anomalies(forecast, history, season, config)
        columns = [ENTITY, "ds", "score", "type", "severity", "deviation", "source"]
        pieces = []
        for frame in (residual, forward):
            if len(frame):
                present = [c for c in columns if c in frame.columns]
                piece = frame.loc[:, present].copy()
                for extra in ("actual", "expected", "forecast"):
                    if extra in frame.columns:
                        piece[extra] = frame[extra]
                pieces.append(piece)
        anomalies = (
            pd.concat(pieces, ignore_index=True)
            if pieces
            else pd.DataFrame(columns=[*columns, "actual", "expected", "forecast"])
        )
        if len(anomalies):
            anomalies["level"] = level
        peaks = detect_peak_periods(forecast, history, season)
        for peak in peaks:
            peak["level"] = level
        return anomalies, peaks

    # ------------------------------------------------------------ artefacts
    def _save(self, **kwargs) -> dict[str, str]:
        run_dir = self.run_dir
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        artefacts: dict[str, str] = {}

        def write_parquet(name: str, frame: pd.DataFrame) -> None:
            path = run_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            artefacts[name] = _repo_relative(path)

        def write_json(name: str, payload: Any) -> None:
            path = run_dir / f"{name}.json"
            path.write_text(
                json.dumps(
                    sanitise_json(payload),
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            artefacts[name] = _repo_relative(path)

        write_parquet("forecast", kwargs["forecast"])
        write_parquet("backtest", kwargs["backtest"])
        write_parquet("history", kwargs["history"])
        write_parquet("anomalies", _ensure_frame(kwargs["anomalies"]))
        write_parquet("entity_map", kwargs["mapping"])

        write_json("metrics", kwargs["metrics"])
        write_json("insights", kwargs["insights"])
        write_json("feature_importance", kwargs["explanation"])
        write_json("price_dependence", kwargs["dependence"])
        write_json("peaks", kwargs["peaks"])
        write_json("labels", kwargs["labels"])
        write_json("uncertainty", kwargs["calibrator"].describe())

        self.contract.save(run_dir / "config.yaml")
        artefacts["config"] = _repo_relative(run_dir / "config.yaml")

        # Model binaries: the champion always, others best-effort.
        saved_models: dict[str, str] = {}
        for name, model in kwargs["models"].items():
            try:
                path = model.save(models_dir)
                saved_models[name] = _repo_relative(path)
            except Exception as exc:  # noqa: BLE001 - artefacts are not critical
                self.warnings.append(f"could not persist model '{name}': {exc}")

        engine = kwargs["engine"]
        metadata = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "champion": kwargs["champion"],
            "profile": self.config.profile_name,
            "profile_config": {
                k: v for k, v in self.profile.items() if isinstance(v, (str, int, float, list))
            },
            "seed": self.config.seed,
            "dataset": self.contract.name,
            "dataset_path": self.contract.path,
            "target": self.contract.target,
            "primary_metric": self.contract.evaluation.primary_metric,
            "horizon": kwargs["metrics"]["horizon"],
            "levels": kwargs["panel"].available_levels(),
            "frequency": kwargs["panel"].frequency,
            "n_features": len(engine.feature_names),
            "feature_names": engine.feature_names,
            "categorical_features": engine.categorical_features,
            "training_window": {
                "start": str(kwargs["panel"].start.date()),
                "end": str(kwargs["panel"].end.date()),
            },
            "models": saved_models,
            "ensemble_weights": kwargs["metrics"]["ensemble_weights"],
            "uncertainty_method": kwargs["metrics"]["uncertainty_method"],
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                **detect_resources(),
            },
            "warnings": self.warnings,
        }
        write_json("metadata", metadata)
        return artefacts


# --------------------------------------------------------------------------
def _repo_relative(path: Path) -> str:
    """Repo-relative path when possible, absolute otherwise.

    Artefacts normally live under `runs/` inside the repository, but tests and
    custom `RUNS_DIR` values can put them anywhere.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _collapse_folds(backtest: pd.DataFrame) -> pd.DataFrame:
    """Average overlapping fold predictions for one (entity, date).

    Only for display and hierarchy aggregation - scoring always uses every
    fold row so each fold stays an independent evaluation.
    """
    if backtest.empty or "fold" not in backtest.columns:
        return backtest
    numeric = ["actual", "prediction", "lower", "upper"]
    present = [c for c in numeric if c in backtest.columns]
    aggregation: dict[str, str] = {c: "mean" for c in present}
    if "horizon" in backtest.columns:
        aggregation["horizon"] = "min"
    collapsed = backtest.groupby([ENTITY, "ds"], as_index=False).agg(aggregation)
    if "model" in backtest.columns:
        collapsed["model"] = backtest["model"].iloc[0]
    return collapsed


def _ensemble_candidates(model_scores: dict[str, dict], metric: str) -> dict[str, float]:
    """Which models are allowed into the blend.

    Baselines are excluded: they exist to be beaten, and letting them into the
    blend reliably drags it below the best single learner. The exception is a
    run with fewer than two learned models, where the strongest baseline is
    admitted so an ensemble still means something.
    """
    learned = {
        name: scores.get(metric)
        for name, scores in model_scores.items()
        if not is_baseline(name) and name != "ensemble" and scores.get(metric) is not None
    }
    if len(learned) >= 2:
        return learned

    baselines = {
        name: scores.get(metric)
        for name, scores in model_scores.items()
        if is_baseline(name) and scores.get(metric) is not None
    }
    if not baselines:
        return learned
    spec = get_metric(metric)
    best = (max if spec.greater_is_better else min)(baselines, key=baselines.get)
    return {**learned, best: baselines[best]}


def _ensure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return pd.DataFrame(
            {
                ENTITY: pd.Series(dtype="object"),
                "ds": pd.Series(dtype="datetime64[ns]"),
                "score": pd.Series(dtype="float64"),
                "type": pd.Series(dtype="object"),
                "severity": pd.Series(dtype="object"),
                "deviation": pd.Series(dtype="float64"),
                "source": pd.Series(dtype="object"),
            }
        )
    out = frame.copy()
    for column in ("actual", "expected", "forecast", "deviation", "score"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out[ENTITY] = out[ENTITY].astype(str)
    return out


def _entity_labels(panel: Panel, mapping: pd.DataFrame) -> dict[str, str]:
    """Human-facing names for every key at every level.

    Driven by the contract's `labels:` section (key column -> label column), so
    a Persian dashboard works for any dataset that ships display names. Keys
    without a label simply map to themselves.
    """
    from ..contract import CATEGORY, DESTINATION

    labels: dict[str, str] = {}
    frame = panel.frame
    canonical = {
        panel.contract.destination: DESTINATION,
        panel.contract.category: CATEGORY,
        panel.contract.entity_id: ENTITY,
    }

    for key_column, label_column in (panel.contract.labels or {}).items():
        if label_column not in frame.columns:
            continue
        internal = canonical.get(key_column, key_column)
        if internal not in frame.columns:
            continue
        pairs = frame.loc[:, [internal, label_column]].dropna().drop_duplicates(subset=[internal])
        labels.update({str(k): str(v) for k, v in pairs.itertuples(index=False)})

    for column in mapping.columns:
        if column == ENTITY:
            continue
        for value in mapping[column].dropna().unique():
            labels.setdefault(str(value), str(value))
    for value in frame[ENTITY].unique():
        labels.setdefault(str(value), str(value))
    return labels


def _segment_scores(backtest: pd.DataFrame, mapping: pd.DataFrame, metric: str) -> dict[str, Any]:
    """Error broken down by destination, weekday and month."""
    from ..contract import DESTINATION

    spec = get_metric(metric)
    out: dict[str, Any] = {}
    if backtest.empty:
        return out

    frame = backtest.merge(
        mapping[[ENTITY, DESTINATION]] if DESTINATION in mapping.columns else mapping[[ENTITY]],
        on=ENTITY,
        how="left",
    )
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame["weekday"] = frame["ds"].dt.dayofweek
    frame["month"] = frame["ds"].dt.month

    def score(group: pd.DataFrame) -> float | None:
        value = spec(group["actual"], group["prediction"])
        return None if not np.isfinite(value) else round(float(value), 6)

    if DESTINATION in frame.columns:
        out["by_destination"] = [
            {"key": str(key), "n": int(len(group)), metric: score(group)}
            for key, group in frame.groupby(DESTINATION)
            if len(group) > 10
        ]
    out["by_weekday"] = [
        {"key": int(key), "n": int(len(group)), metric: score(group)}
        for key, group in frame.groupby("weekday")
    ]
    out["by_month"] = [
        {"key": int(key), "n": int(len(group)), metric: score(group)}
        for key, group in frame.groupby("month")
    ]
    out["by_horizon"] = [
        {"key": int(key), "n": int(len(group)), metric: score(group)}
        for key, group in frame.groupby("horizon")
    ]
    return out


def sanitise_json(value: Any) -> Any:
    """Recursively replace NaN/Infinity with None.

    `json.dumps` happily writes bare `NaN` and `Infinity`, which are not valid
    JSON: SQLite's JSON_VALID rejects them and - far worse for a demo -
    JavaScript's `JSON.parse` throws, so a single NaN anywhere in a run's
    metrics would blank the entire dashboard.
    """
    if isinstance(value, dict):
        return {k: sanitise_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitise_json(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _new_run_id(runs_dir: Path) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d")
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith(stamp)]
    return f"{stamp}_{len(existing) + 1:03d}"


def load_future_overrides(path: str | Path | None) -> pd.DataFrame | None:
    """Load a known-future covariate file (holiday calendar, planned prices)."""
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.exists():
        log.warning("future covariate file not found: %s", resolved)
        return None
    frame = read_tabular(resolved)
    for candidate in ("ds", "date", "timestamp", "day"):
        if candidate in frame.columns:
            frame = frame.rename(columns={candidate: TS})
            break
    if TS not in frame.columns:
        log.warning("future covariate file has no date column: %s", resolved)
        return None
    frame[TS] = pd.to_datetime(frame[TS], errors="coerce")
    return frame.dropna(subset=[TS])
