"""The selected model, and the code that runs it end to end.

Phase 2 chose this configuration by pooled walk-forward WAPE across five
simulated competitions - not by sophistication. The ablation that produced it is
in `artifacts/pol4/experiments.csv`, and `--ablation` regenerates it.

Everything here obeys the Phase 1 guarantees: features read only pre-cutoff
searches, the prediction can never fall below the demand already observed, and
the same inputs produce the same output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .baseline import PickupBaseline
from .config import Pol4Config
from .dataset import build_inference_frame, build_training_frame, export_training_frame
from .features import GROUP_ORDER, feature_names
from .loader import CHECKIN, CITY, Pol4Data
from .models import RemainingDemandModel

log = logging.getLogger(__name__)


@dataclass
class ChampionSpec:
    """What the champion is. Changing this changes the submission."""

    #: Horizon bands, each fitted as its own model. A single band is one global
    #: model with `days_to_checkin` as a feature; more bands buy specialisation
    #: and pay for it in data per model. Two won on the evidence:
    #:
    #:   bands   pooled WAPE   2024-11-21 fold (the Azar window, a year earlier)
    #:     x1       0.1618       0.1193
    #:     x2       0.1581       0.1222   (+2.5%)
    #:     x3       0.1574       0.1281   (+7.4%)
    #:     x4       0.1563       0.1318   (+10.5%)
    #:
    #: Four bands score best pooled, but the damage to the seasonal analogue of
    #: the competition window rises monotonically with the split - a coherent
    #: signal that thinner bands generalise worse on a low-season window, which
    #: is exactly what Azar is. Two bands take 2.3 of the 3.4 percentage points
    #: of pooled gain for a quarter of that risk.
    bands: tuple[tuple[int, int], ...] = ((1, 14), (15, 30))
    kind: str = "lightgbm"
    groups: tuple[str, ...] = GROUP_ORDER
    #: ON, and measured rather than assumed. The Phase 1 audit found the generic
    #: platform's automatic log1p under-predicting, so this was A/B'd: with an L1
    #: objective on *remaining* demand it improves WAPE (0.1692 -> 0.1618), the
    #: top-1% slice (0.1681 -> 0.1544) and peak bias (-0.119 -> -0.100). An L1
    #: objective in log space targets the conditional median, which is what WAPE
    #: rewards; the platform's L2-on-log1p targeted a mean and then under-shot.
    log1p: bool = True
    params: dict[str, Any] = field(default_factory=lambda: {"n_estimators": 600})
    #: Weight on the model when blending with the pickup baseline. Set to 1.0 by
    #: the ensemble experiment: a leave-one-fold-out alpha search picked 1.0 on
    #: four folds and 0.95 on the fifth, and blending scored 0.1623 against the
    #: model's own 0.1618. No ensemble.
    blend: float = 1.0

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.kind,
            "horizon_bands": [list(b) for b in self.bands],
            "feature_groups": list(self.groups),
            "n_features": len(feature_names(self.groups)),
            "log1p_target": self.log1p,
            "params": dict(self.params),
            "blend_weight_on_model": self.blend,
        }


@dataclass
class FittedChampion:
    spec: ChampionSpec
    models: dict[tuple[int, int], RemainingDemandModel]
    baseline: PickupBaseline
    cutoff: pd.Timestamp
    config: Pol4Config
    training_rows: int
    training_frame: Any = field(default=None, repr=False)

    @classmethod
    def fit(
        cls,
        data: Pol4Data,
        cutoff: pd.Timestamp,
        spec: ChampionSpec | None = None,
        config: Pol4Config | None = None,
    ) -> "FittedChampion":
        spec = spec or ChampionSpec()
        config = config or Pol4Config()
        cutoff = pd.Timestamp(cutoff)
        columns = feature_names(spec.groups)

        baseline = PickupBaseline.fit(data, cutoff, config)
        train = build_training_frame(data, cutoff, spec.groups, baseline, config)
        horizons = train.meta["horizon"].to_numpy()

        models: dict[tuple[int, int], RemainingDemandModel] = {}
        for band in spec.bands:
            low, high = band
            rows = (horizons >= low) & (horizons <= high)
            if not rows.any():
                continue
            models[band] = RemainingDemandModel(
                kind=spec.kind, params=spec.params, log1p=spec.log1p, seed=config.seed
            ).fit(train.X.loc[rows, columns], train.y[rows])
            log.info(
                "champion %s band %d-%d fitted on %s rows at cutoff %s",
                spec.kind,
                low,
                high,
                f"{int(rows.sum()):,}",
                cutoff.date(),
            )
        if not models:
            raise RuntimeError("no horizon band produced any training rows")
        return cls(
            spec=spec,
            models=models,
            baseline=baseline,
            cutoff=cutoff,
            config=config,
            training_rows=len(train),
            training_frame=train,
        )

    @property
    def model(self) -> RemainingDemandModel:
        """The single model, when there is only one band."""
        return next(iter(self.models.values()))

    def predict(self, target_dates: pd.DatetimeIndex, data: Pol4Data) -> pd.DataFrame:
        """Predict the full (city x check-in) grid for `target_dates`."""
        columns = feature_names(self.spec.groups)
        infer = build_inference_frame(
            data, self.cutoff, target_dates, self.spec.groups, self.baseline, self.config
        )
        observed = infer.meta["observed"].to_numpy(dtype=np.float64)
        horizons = infer.meta["horizon"].to_numpy()

        predicted = np.full(len(infer.meta), np.nan)
        for (low, high), model in self.models.items():
            rows = (horizons >= low) & (horizons <= high)
            if rows.any():
                predicted[rows] = model.predict_final(
                    infer.X.loc[rows, columns], observed[rows]
                )
        if np.isnan(predicted).any():
            missing = sorted(set(horizons[np.isnan(predicted)]))
            raise RuntimeError(f"no horizon band covers horizon(s) {missing}")

        if self.spec.blend < 1.0:
            grid = infer.meta[[CITY, CHECKIN]].assign(observed=observed)
            projection = self.baseline.predict(grid)["predicted_demand"].to_numpy()
            predicted = self.spec.blend * predicted + (1.0 - self.spec.blend) * projection

        # The Phase 1 floor holds for the champion too.
        predicted = np.maximum(np.nan_to_num(predicted, nan=0.0), observed)
        return infer.meta.assign(
            predicted_demand=predicted,
            predicted_remaining=predicted - observed,
        )

    def export_training_data(self, path, sample: int | None = None):
        """Write the exact rows this champion was fitted on."""
        if self.training_frame is None:
            raise RuntimeError("training frame was not retained")
        return export_training_frame(
            self.training_frame, path, sample=sample, seed=self.config.seed
        )

    def importance(self) -> pd.DataFrame:
        """Gain importance, averaged across bands and weighted by their share.

        Each band is trained on its own slice, so a single band's table would
        describe a quarter of the forecast. Averaging keeps the answer about the
        champion rather than about one of its parts.
        """
        frames = [model.importance().assign(band=f"{low}-{high}")
                  for (low, high), model in self.models.items()]
        if len(frames) == 1:
            return frames[0].drop(columns="band")
        stacked = pd.concat(frames, ignore_index=True)
        pooled = (
            stacked.groupby("feature", as_index=False)["share"]
            .mean()
            .rename(columns={"share": "share"})
        )
        pooled["importance"] = pooled["share"]
        pooled["share"] = pooled["share"] / pooled["share"].sum()
        return pooled.sort_values("share", ascending=False, ignore_index=True)[
            ["feature", "importance", "share"]
        ]
