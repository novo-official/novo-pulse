"""Conservative validation diagnostics for the Pol 4 temporal backtest.

The ordinary pooled backtest is useful for model comparison, but its score is
slightly optimistic after choosing a calibration method on those same folds.
This module reports a second, prequential estimate: at each cutoff the method
is selected using only earlier folds whose outcomes are already known.
"""
from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from .config import Pol4Config

if TYPE_CHECKING:
    from .experiments import ExperimentResult


def _pool_fold_scores(scores: Iterable[dict[str, float]]) -> dict[str, float]:
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


def _safe_on_history(
    raw: "ExperimentResult",
    candidate: "ExperimentResult",
    labels: list[str],
    fold_tolerance: float,
) -> bool:
    if not labels:
        return False
    raw_score = _pool_fold_scores(raw.folds[label] for label in labels)
    candidate_score = _pool_fold_scores(candidate.folds[label] for label in labels)
    return bool(
        candidate_score["wape"] <= raw_score["wape"]
        and abs(candidate_score["normalised_bias"]) < abs(raw_score["normalised_bias"])
        and all(
            candidate.folds[label]["wape"]
            <= raw.folds[label]["wape"] + fold_tolerance
            for label in labels
        )
    )


def _choose_from_history(
    raw: "ExperimentResult",
    candidates: list["ExperimentResult"],
    labels: list[str],
    fold_tolerance: float,
) -> "ExperimentResult":
    eligible = [
        candidate
        for candidate in candidates
        if _safe_on_history(raw, candidate, labels, fold_tolerance)
    ]
    if not eligible:
        return raw
    return min(
        eligible,
        key=lambda item: abs(
            _pool_fold_scores(item.folds[label] for label in labels)["normalised_bias"]
        ),
    )


def _fold_cluster_interval(
    folds: dict[str, dict[str, float]], confidence: float = 0.95
) -> dict[str, float | str]:
    """Fold-cluster bootstrap CI; whole temporal windows are resampled."""
    labels = list(folds)
    n_folds = len(labels)
    if n_folds < 2:
        return {"method": "unavailable", "confidence": confidence}

    actual = np.asarray([folds[label]["actual_total"] for label in labels], dtype=float)
    errors = np.asarray(
        [folds[label]["wape"] * folds[label]["actual_total"] for label in labels],
        dtype=float,
    )
    if n_folds**n_folds <= 100_000:
        samples = product(range(n_folds), repeat=n_folds)
        values = np.fromiter(
            (errors[list(index)].sum() / actual[list(index)].sum() for index in samples),
            dtype=float,
        )
        method = "exact fold-cluster bootstrap"
    else:  # pragma: no cover - Pol 4 currently has five folds
        rng = np.random.default_rng(42)
        indices = rng.integers(0, n_folds, size=(20_000, n_folds))
        values = errors[indices].sum(axis=1) / actual[indices].sum(axis=1)
        method = "20000-sample fold-cluster bootstrap"
    tail = (1.0 - confidence) / 2.0
    return {
        "method": method,
        "confidence": confidence,
        "lower": round(float(np.quantile(values, tail)), 6),
        "upper": round(float(np.quantile(values, 1.0 - tail)), 6),
    }


def build_validation_audit(
    raw: "ExperimentResult",
    selected: "ExperimentResult",
    candidates: list["ExperimentResult"],
    baseline: "ExperimentResult",
    config: Pol4Config,
    *,
    fold_tolerance: float = 0.01,
) -> dict:
    """Build honest-selection, uncertainty, and overfit diagnostics."""
    decisions: list[dict] = []
    prequential_scores: dict[str, dict[str, float]] = {}
    ordered_labels = [
        pd.Timestamp(value).date().isoformat()
        for value in config.backtest_cutoffs
        if pd.Timestamp(value).date().isoformat() in raw.folds
    ]

    for label in ordered_labels:
        cutoff = pd.Timestamp(label)
        history = [
            earlier
            for earlier in ordered_labels
            if pd.Timestamp(earlier) + pd.Timedelta(days=config.target_days) <= cutoff
        ]
        choice = _choose_from_history(raw, candidates, history, fold_tolerance)
        prequential_scores[label] = choice.folds[label]
        decisions.append(
            {
                "cutoff": label,
                "history_folds": history,
                "selected": choice.meta.get("calibration", "none"),
                "wape": choice.folds[label]["wape"],
                "normalised_bias": choice.folds[label]["normalised_bias"],
            }
        )

    selected_fold_wapes = np.asarray(
        [selected.folds[label]["wape"] for label in ordered_labels], dtype=float
    )
    final_label = ordered_labels[-1]
    final_decision = decisions[-1]
    final_baseline = baseline.folds[final_label]
    prequential = _pool_fold_scores(prequential_scores.values())
    selected_ci = _fold_cluster_interval(selected.folds)

    train = raw.training
    generalisation_gap = (
        round(raw.pooled["wape"] - train["wape"], 6) if train else None
    )
    return {
        "protocol": {
            "kind": "rolling-origin pseudo-competition with prequential calibration selection",
            "folds": len(ordered_labels),
            "rows": int(sum(raw.folds[label]["n"] for label in ordered_labels)),
            "selection_rule": (
                "each fold selects calibration only from earlier fully-closed folds; "
                "raw is used when no candidate is safe"
            ),
            "model_spec_status": "preselected; not nested inside this run",
        },
        "reported_selected": {
            **selected.pooled,
            "confidence_interval": selected_ci,
            "fold_wape_median": round(float(np.median(selected_fold_wapes)), 6),
            "fold_wape_std": round(float(np.std(selected_fold_wapes, ddof=1)), 6),
            "best_fold_wape": round(float(selected_fold_wapes.min()), 6),
            "worst_fold_wape": round(float(selected_fold_wapes.max()), 6),
            "beats_baseline_folds": int(
                sum(
                    selected.folds[label]["wape"] < baseline.folds[label]["wape"]
                    for label in ordered_labels
                )
            ),
        },
        "prequential_selection": {
            "pooled": prequential,
            "decisions": decisions,
        },
        "last_fold_holdout": {
            "cutoff": final_label,
            "selected_from_prior_folds": final_decision["selected"],
            "wape": final_decision["wape"],
            "normalised_bias": final_decision["normalised_bias"],
            "baseline_wape": final_baseline["wape"],
            "relative_improvement_vs_baseline": round(
                1.0 - final_decision["wape"] / final_baseline["wape"], 6
            ),
        },
        "overfit_diagnostics": {
            "raw_train": train or None,
            "raw_validation": raw.pooled,
            "validation_minus_train_wape": generalisation_gap,
            "interpretation": (
                "large positive gaps are an overfit warning; this is diagnostic, not a proof, "
                "because train and future-window distributions differ"
            ),
        },
        "limitations": [
            "only five temporal folds are available, so uncertainty remains wide",
            "the model specification was chosen before this audit and is not nested per outer fold",
            "the competition target has no labels yet and remains the only truly unseen test",
        ],
    }
