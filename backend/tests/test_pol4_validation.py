"""Tests for the conservative temporal validation audit."""
from __future__ import annotations

from dataclasses import replace

from ml.pol4.config import Pol4Config
from ml.pol4.experiments import ExperimentResult
from ml.pol4.validation import build_validation_audit


def _score(wape: float, bias: float) -> dict[str, float]:
    actual = 1000.0
    return {
        "n": 100,
        "actual_total": actual,
        "predicted_total": actual * (1.0 + bias),
        "wape": wape,
        "mae": wape * actual / 100,
        "normalised_bias": bias,
    }


def _result(name: str, method: str, fold_values: list[tuple[float, float]]) -> ExperimentResult:
    labels = ["2024-01-01", "2024-02-01", "2024-03-01"]
    folds = {
        label: _score(wape, bias)
        for label, (wape, bias) in zip(labels, fold_values, strict=True)
    }
    return ExperimentResult(
        name=name,
        stage="candidate",
        pooled=_score(
            sum(item[0] for item in fold_values) / len(fold_values),
            sum(item[1] for item in fold_values) / len(fold_values),
        ),
        folds=folds,
        horizon=[],
        detail={},
        runtime_seconds=0.0,
        meta={"calibration": method},
    )


def test_prequential_selection_never_uses_current_or_future_fold():
    config = replace(
        Pol4Config(),
        target_days=10,
        backtest_cutoffs=["2024-01-01", "2024-02-01", "2024-03-01"],
    )
    raw = _result("raw", "none", [(0.20, -0.10)] * 3)
    # Great on fold one, catastrophic on fold two. It may be selected for fold
    # two, but that failure must make it ineligible for fold three.
    candidate = _result(
        "candidate", "guarded", [(0.10, -0.02), (0.90, 0.50), (0.01, 0.0)]
    )
    baseline = _result("baseline", "none", [(0.30, -0.20)] * 3)

    audit = build_validation_audit(raw, raw, [candidate], baseline, config)
    decisions = audit["prequential_selection"]["decisions"]

    assert decisions[0]["selected"] == "none"
    assert decisions[1]["selected"] == "guarded"
    assert decisions[2]["selected"] == "none"
    assert decisions[2]["history_folds"] == ["2024-01-01", "2024-02-01"]


def test_validation_audit_exposes_training_gap_and_fold_uncertainty():
    config = replace(
        Pol4Config(),
        target_days=10,
        backtest_cutoffs=["2024-01-01", "2024-02-01", "2024-03-01"],
    )
    raw = _result("raw", "none", [(0.12, -0.05), (0.20, -0.08), (0.16, -0.06)])
    raw.training = _score(0.06, -0.01)
    baseline = _result("baseline", "none", [(0.30, -0.20)] * 3)

    audit = build_validation_audit(raw, raw, [], baseline, config)

    assert audit["overfit_diagnostics"]["validation_minus_train_wape"] == 0.10
    interval = audit["reported_selected"]["confidence_interval"]
    assert interval["lower"] <= raw.pooled["wape"] <= interval["upper"]
    assert audit["reported_selected"]["beats_baseline_folds"] == 3
