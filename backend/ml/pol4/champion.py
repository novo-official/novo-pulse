"""The selected model, and the code that runs it end to end.

Phase 2 chose this configuration by pooled walk-forward WAPE across five
simulated competitions - not by sophistication. The ablation that produced it is
in `artifacts/pol4/experiments.csv`, and `--ablation` regenerates it.

Everything here obeys the Phase 1 guarantees: features read only pre-cutoff
searches, the prediction can never fall below the demand already observed, and
the same inputs produce the same output.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .baseline import PickupBaseline
from .calibration import Calibrator
from .config import Pol4Config
from .dataset import SupervisedFrame, build_inference_frame, build_training_frame
from .features import GROUP_ORDER, feature_names
from .loader import CHECKIN, CITY, Pol4Data
from .models import RemainingDemandModel
from .artifacts import (
    ARTEFACT_FORMAT_VERSION,
    sha256_file,
    verify_checksums,
    write_json_atomic,
)

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
    #: Learned from strictly out-of-fold predictions and accepted only when it
    #: reduces absolute bias without worsening pooled WAPE.
    calibration: str = "guarded_bias_horizon"
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
            "calibration_method": self.calibration,
            "blend_weight_on_model": self.blend,
        }


@dataclass
class FittedChampion:
    spec: ChampionSpec
    models: dict[tuple[int, int], RemainingDemandModel]
    baseline: PickupBaseline
    calibrator: Calibrator
    cutoff: pd.Timestamp
    config: Pol4Config
    training_rows: int

    @classmethod
    def fit(
        cls,
        data: Pol4Data,
        cutoff: pd.Timestamp,
        spec: ChampionSpec | None = None,
        config: Pol4Config | None = None,
        *,
        baseline: PickupBaseline | None = None,
        train: SupervisedFrame | None = None,
        calibrator: Calibrator | None = None,
    ) -> "FittedChampion":
        spec = spec or ChampionSpec()
        config = config or Pol4Config()
        cutoff = pd.Timestamp(cutoff)
        columns = feature_names(spec.groups)

        baseline = baseline or PickupBaseline.fit(data, cutoff, config)
        train = train or build_training_frame(data, cutoff, spec.groups, baseline, config)
        if train.y is None:
            raise ValueError("champion training data does not carry a target")
        if list(train.X.columns) != columns:
            raise ValueError("champion training features do not match the selected feature schema")
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
            calibrator=calibrator or Calibrator(method="none", alpha_global=1.0),
            cutoff=cutoff,
            config=config,
            training_rows=len(train),
        )

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path, *, input_digest: str) -> dict[str, Any]:
        """Save both native boosters and all state needed by a fresh process."""
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        model_entries: list[dict[str, Any]] = []
        for (low, high), model in sorted(self.models.items()):
            extension = "txt" if model.kind == "lightgbm" else "cbm"
            filename = f"{model.kind}_h{low}_{high}.{extension}"
            model.save_native(directory / filename)
            model_entries.append(
                {
                    "band": [low, high],
                    "file": filename,
                    "kind": model.kind,
                    "params": model.params,
                    "log1p": model.log1p,
                    "seed": model.seed,
                    "feature_names": model.feature_names,
                    "categorical": model.categorical,
                }
            )

        # The baseline is only loaded after its checksum has been verified.
        # joblib is used here for trusted, locally-built state containing nested
        # numpy arrays and tuple-key dictionaries; native LightGBM files remain
        # independently inspectable and portable.
        baseline_path = directory / "baseline_state.joblib"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{baseline_path.name}.", suffix=".tmp", dir=directory
        )
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            joblib.dump(self.baseline, temporary_path)
            temporary_path.replace(baseline_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        calibrator_path = directory / "calibrator_state.joblib"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{calibrator_path.name}.", suffix=".tmp", dir=directory
        )
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            joblib.dump(self.calibrator, temporary_path)
            temporary_path.replace(calibrator_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        spec_payload = {
            "format_version": ARTEFACT_FORMAT_VERSION,
            "cutoff": self.cutoff.isoformat(),
            "training_rows": self.training_rows,
            "input_digest": input_digest,
            "spec": {
                "bands": [list(band) for band in self.spec.bands],
                "kind": self.spec.kind,
                "groups": list(self.spec.groups),
                "log1p": self.spec.log1p,
                "params": self.spec.params,
                "calibration": self.spec.calibration,
                "blend": self.spec.blend,
            },
            "models": model_entries,
            "baseline_file": baseline_path.name,
            "calibrator_file": calibrator_path.name,
            "calibration": self.calibrator.as_dict(),
        }
        write_json_atomic(spec_payload, directory / "model_spec.json")

        protected = [entry["file"] for entry in model_entries]
        protected += ["baseline_state.joblib", "calibrator_state.joblib", "model_spec.json"]
        checksums = {name: sha256_file(directory / name) for name in protected}
        bundle_digest = hashlib.sha256(
            "\n".join(f"{name}:{checksums[name]}" for name in sorted(checksums)).encode("utf-8")
        ).hexdigest()
        manifest = {
            "format_version": ARTEFACT_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_digest": input_digest,
            "bundle_digest": bundle_digest,
            "files": checksums,
        }
        write_json_atomic(manifest, directory / "manifest.json")
        return manifest

    @classmethod
    def load(cls, directory: str | Path) -> "FittedChampion":
        """Verify and restore a fitted champion without training."""
        import joblib

        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"model manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != ARTEFACT_FORMAT_VERSION:
            raise ValueError(f"unsupported model artefact format: {manifest.get('format_version')}")
        verify_checksums(directory, manifest["files"])

        payload = json.loads((directory / "model_spec.json").read_text(encoding="utf-8"))
        if payload.get("input_digest") != manifest.get("input_digest"):
            raise ValueError("model spec and manifest refer to different input data")
        raw_spec = payload["spec"]
        spec = ChampionSpec(
            bands=tuple(tuple(int(value) for value in band) for band in raw_spec["bands"]),
            kind=raw_spec["kind"],
            groups=tuple(raw_spec["groups"]),
            log1p=bool(raw_spec["log1p"]),
            params=dict(raw_spec["params"]),
            calibration=raw_spec.get("calibration", "guarded_bias_horizon"),
            blend=float(raw_spec["blend"]),
        )
        expected_features = feature_names(spec.groups)

        models: dict[tuple[int, int], RemainingDemandModel] = {}
        for entry in payload["models"]:
            if entry["feature_names"] != expected_features:
                raise ValueError("saved model feature schema does not match its champion spec")
            band = tuple(int(value) for value in entry["band"])
            models[band] = RemainingDemandModel.load_native(
                directory / entry["file"],
                kind=entry["kind"],
                params=entry["params"],
                log1p=bool(entry["log1p"]),
                seed=int(entry["seed"]),
                feature_names=entry["feature_names"],
                categorical=entry["categorical"],
            )
        if set(models) != set(spec.bands):
            raise ValueError("saved model bundle does not contain every configured horizon band")

        baseline = joblib.load(directory / payload["baseline_file"])
        if not isinstance(baseline, PickupBaseline):
            raise ValueError("saved baseline state has an unexpected type")
        calibrator_file = payload.get("calibrator_file")
        calibrator = (
            joblib.load(directory / calibrator_file)
            if calibrator_file
            else Calibrator(method="none", alpha_global=1.0)
        )
        if not isinstance(calibrator, Calibrator):
            raise ValueError("saved calibrator state has an unexpected type")
        cutoff = pd.Timestamp(payload["cutoff"])
        if pd.Timestamp(baseline.cutoff) != cutoff:
            raise ValueError("saved baseline and champion use different cutoffs")
        return cls(
            spec=spec,
            models=models,
            baseline=baseline,
            calibrator=calibrator,
            cutoff=cutoff,
            config=baseline.config,
            training_rows=int(payload["training_rows"]),
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

        predicted = self.calibrator.apply(
            infer.meta.assign(predicted_demand=predicted)
        )

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
