"""Forecast stability: does the prediction hold still as the check-in approaches?

Final WAPE is not the whole story. A forecast that reads 40, then 190, then 80
for the same night is unusable even if it happens to land close in the end,
because nobody can plan against it. So for a set of historical check-ins the
same pair is predicted repeatedly - at D-30, D-21, D-14, D-7, D-3 and D-1 - and
the revisions are measured.

Why this is cheap: a feature at horizon `h` reads only the searches logged at or
before `h` days out, so ONE feature tensor over the window yields every snapshot.
The genuinely cutoff-dependent parts - pickup curves, city history, the zero
prior, and the model itself - are all fitted at the EARLIEST snapshot cutoff and
reused for the later ones. That is conservative rather than optimistic: a later
snapshot is entitled to more information than it is given here, so no snapshot
can see anything it should not.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .baseline import PickupBaseline
from .config import Pol4Config
from .dataset import _complete_grid, build_training_frame
from .features import GROUP_ORDER, FeatureBuilder, feature_names
from .loader import CHECKIN, CITY, Pol4Data

log = logging.getLogger(__name__)

SNAPSHOT_HORIZONS = (30, 21, 14, 7, 3, 1)


@dataclass
class StabilityReport:
    snapshots: pd.DataFrame          # city_code, checkin, horizon, prediction, actual
    revisions: pd.DataFrame          # consecutive-snapshot deltas
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.summary


def _revision_table(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Change between consecutive snapshots of the same pair."""
    ordered = snapshots.sort_values([CITY, CHECKIN, "horizon"], ascending=[True, True, False])
    grouped = ordered.groupby([CITY, CHECKIN], sort=False)["prediction"]
    ordered = ordered.assign(
        previous=grouped.shift(1),
        previous_horizon=ordered.groupby([CITY, CHECKIN], sort=False)["horizon"].shift(1),
    )
    revisions = ordered.dropna(subset=["previous"]).copy()
    revisions["absolute_revision"] = (revisions["prediction"] - revisions["previous"]).abs()
    revisions["relative_revision"] = revisions["absolute_revision"] / (
        revisions["previous"].abs() + 1.0
    )
    revisions["step"] = (
        revisions["previous_horizon"].astype(int).astype(str)
        + "->"
        + revisions["horizon"].astype(int).astype(str)
    )
    revisions["absolute_error"] = (revisions["prediction"] - revisions["actual"]).abs()
    revisions["previous_error"] = (revisions["previous"] - revisions["actual"]).abs()
    revisions["converged"] = revisions["absolute_error"] <= revisions["previous_error"]
    return revisions


def analyse(
    data: Pol4Data,
    target_start: pd.Timestamp,
    target_days: int,
    predict: Callable[[FeatureBuilder, int, np.ndarray], np.ndarray],
    config: Pol4Config | None = None,
) -> StabilityReport:
    """Snapshot a predictor across the horizon ladder for one historical window."""
    config = config or Pol4Config()
    target_start = pd.Timestamp(target_start)
    target_dates = pd.date_range(target_start, periods=target_days, freq="D")

    # The earliest snapshot decides what every snapshot is allowed to know.
    anchor = target_start - pd.Timedelta(days=max(SNAPSHOT_HORIZONS))
    keys = _complete_grid(data.city_codes, target_dates)
    events = data.search[data.search[CHECKIN].isin(target_dates)]
    builder = FeatureBuilder.build(data, anchor, keys, events, config)
    actual = builder.tensor.final()

    rows: list[pd.DataFrame] = []
    for horizon in SNAPSHOT_HORIZONS:
        observed = builder.tensor.observed(horizon)
        rows.append(
            keys.assign(
                horizon=horizon,
                observed=observed,
                actual=actual,
                prediction=predict(builder, horizon, observed),
            )
        )
    snapshots = data.label(pd.concat(rows, ignore_index=True))
    revisions = _revision_table(snapshots)

    by_step = (
        revisions.groupby("step", sort=False)
        .agg(
            mean_absolute_revision=("absolute_revision", "mean"),
            mean_relative_revision=("relative_revision", "mean"),
            convergence_rate=("converged", "mean"),
            n=("absolute_revision", "size"),
        )
        .round(4)
        .reset_index()
    )
    by_horizon = (
        snapshots.assign(error=(snapshots["prediction"] - snapshots["actual"]).abs())
        .groupby("horizon")
        .agg(
            wape=("error", "sum"),
            actual=("actual", "sum"),
            mean_prediction=("prediction", "mean"),
        )
        .assign(wape=lambda f: (f["wape"] / f["actual"]).round(4))
        .drop(columns="actual")
        .round(2)
        .reset_index()
    )

    # One number for the dashboard: 1.0 means the forecast never moved.
    stability_score = float(
        np.clip(1.0 - revisions["relative_revision"].mean(), 0.0, 1.0)
    )
    summary = {
        "target_start": target_start.date().isoformat(),
        "target_days": int(target_days),
        "anchor_cutoff": anchor.date().isoformat(),
        "snapshot_horizons": list(SNAPSHOT_HORIZONS),
        "pairs": int(len(keys)),
        "stability_score": round(stability_score, 4),
        "mean_absolute_revision": round(float(revisions["absolute_revision"].mean()), 3),
        "mean_relative_revision": round(float(revisions["relative_revision"].mean()), 4),
        "convergence_rate": round(float(revisions["converged"].mean()), 4),
        "by_step": by_step.to_dict(orient="records"),
        "by_horizon": by_horizon.to_dict(orient="records"),
    }
    return StabilityReport(snapshots=snapshots, revisions=revisions, summary=summary)


# ------------------------------------------------------------------ predictors
def baseline_snapshot(baseline: PickupBaseline) -> Callable:
    """The pickup baseline, evaluated at an arbitrary horizon."""

    def predict(builder: FeatureBuilder, horizon: int, observed: np.ndarray) -> np.ndarray:
        cities = builder.tensor.keys[CITY].to_numpy()
        fraction = baseline.curves.fraction(cities, np.full(len(cities), horizon))
        predicted = np.maximum(observed / fraction, observed)
        nothing = observed <= 0
        if nothing.any():
            predicted[nothing] = baseline.zero_prior.level(
                cities[nothing],
                np.full(int(nothing.sum()), horizon),
                baseline.curves.province_of,
            )
        return predicted

    return predict


def model_snapshot(
    model: Any,
    groups: tuple[str, ...],
    baseline: PickupBaseline,
    calibrator: Any | None = None,
) -> Callable:
    """A fitted remaining-demand model, evaluated at an arbitrary horizon."""
    columns = feature_names(groups)

    def predict(builder: FeatureBuilder, horizon: int, observed: np.ndarray) -> np.ndarray:
        features = builder.at_horizon(horizon, groups, baseline)
        predicted = model.predict_final(features[columns], observed)
        if calibrator is None:
            return predicted
        frame = pd.DataFrame(
            {
                "observed": observed,
                "horizon": np.full(len(observed), horizon),
                "predicted_demand": predicted,
            }
        )
        return calibrator.apply(frame)

    return predict


def fit_snapshot_model(
    data: Pol4Data,
    anchor: pd.Timestamp,
    groups: tuple[str, ...],
    kind: str,
    params: dict[str, Any] | None,
    log1p: bool,
    config: Pol4Config,
) -> tuple[Any, PickupBaseline]:
    """Fit the model at the anchor cutoff, so every snapshot stays honest."""
    from .models import RemainingDemandModel

    baseline = PickupBaseline.fit(data, anchor, config)
    train = build_training_frame(data, anchor, GROUP_ORDER, baseline, config)
    model = RemainingDemandModel(
        kind=kind, params=params or {}, log1p=log1p, seed=config.seed
    ).fit(train.X[feature_names(groups)], train.y)
    return model, baseline
