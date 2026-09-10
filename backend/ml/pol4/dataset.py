"""Training and inference frames for the remaining-demand model.

One training row is `(city c, check-in T, horizon h)`:

    features  read only from searches logged at or before  T - h
    target    remaining = final_demand(c, T) - observed(c, T, T - h)

which is the same shape as the real task: stand h days before a check-in,
knowing only what has been searched so far, and predict what is still to come.
Horizons are sampled rather than enumerated - 30 per pair is seven million rows
carrying no extra signal.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .config import Pol4Config
from .features import FeatureBuilder
from .loader import CHECKIN, CITY, Pol4Data
from .artifacts import ARTEFACT_FORMAT_VERSION, sha256_file, verify_checksums, write_json_atomic


def _complete_grid(cities: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Every (city, check-in) pair, including the ones that saw no searches.

    A pair absent from the log is a real observation - zero demand - and
    dropping it would teach the model that demand is never zero.
    """
    return pd.MultiIndex.from_product(
        [np.sort(cities), pd.DatetimeIndex(dates)], names=[CITY, CHECKIN]
    ).to_frame(index=False)


@dataclass
class SupervisedFrame:
    X: pd.DataFrame
    y: np.ndarray | None
    meta: pd.DataFrame          # city_code, checkin, horizon, observed, (final)

    def __len__(self) -> int:
        return len(self.X)


def _schema(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [{"name": name, "dtype": str(dtype)} for name, dtype in frame.dtypes.items()]


_INTENTIONALLY_MISSING = {
    "days_since_first_search",
    "days_since_last_search",
    "search_span",
}


def _feature_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit missingness and reject values LightGBM should never receive."""
    missing = {
        name: int(count)
        for name, count in frame.isna().sum().items()
        if int(count) > 0
    }
    unexpected_missing = sorted(set(missing) - _INTENTIONALLY_MISSING)
    if unexpected_missing:
        raise ValueError(
            "training features contain unexpected missing values: "
            + ", ".join(unexpected_missing)
        )

    infinity_count = 0
    for name in frame.select_dtypes(include=[np.number]).columns:
        infinity_count += int(np.isinf(frame[name].to_numpy()).sum())
    if infinity_count:
        raise ValueError(f"training features contain {infinity_count} infinite values")
    return {
        "null_counts": missing,
        "intentionally_nullable": sorted(_INTENTIONALLY_MISSING),
        "unexpected_null_columns": unexpected_missing,
        "infinite_values": infinity_count,
    }


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write one train-set component atomically without duplicating the full frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_parquet(temporary_path, index=False, compression="zstd")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def materialize_training_frame(
    train: SupervisedFrame,
    directory: str | Path,
    groups: tuple[str, ...],
    config: Pol4Config,
    *,
    input_digest: str,
) -> dict[str, Any]:
    """Persist X/y/audit metadata as a checksum-protected reusable train set."""
    if train.y is None:
        raise ValueError("a materialised training frame must carry a target")
    if not (len(train.X) == len(train.y) == len(train.meta)):
        raise ValueError("training features, target and metadata have different row counts")
    required_meta = {CITY, CHECKIN, "horizon", "observed", "final"}
    missing = required_meta - set(train.meta.columns)
    if missing:
        raise ValueError(f"training metadata is missing column(s): {sorted(missing)}")

    expected = np.maximum(
        train.meta["final"].to_numpy(dtype=np.float64)
        - train.meta["observed"].to_numpy(dtype=np.float64),
        0.0,
    )
    if not np.allclose(expected, np.asarray(train.y, dtype=np.float64)):
        raise ValueError("training target does not equal max(final - observed, 0)")
    feature_quality = _feature_quality(train.X)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "features.parquet": directory / "features.parquet",
        "target.parquet": directory / "target.parquet",
        "meta.parquet": directory / "meta.parquet",
    }
    _write_parquet(train.X, paths["features.parquet"])
    _write_parquet(
        pd.DataFrame({"remaining_demand": np.asarray(train.y, dtype=np.float64)}),
        paths["target.parquet"],
    )
    _write_parquet(train.meta, paths["meta.parquet"])

    files = {name: sha256_file(path) for name, path in paths.items()}
    dates = pd.DatetimeIndex(train.meta[CHECKIN])
    horizons = sorted(int(value) for value in train.meta["horizon"].unique())
    candidate_rows = int(
        train.meta[CITY].nunique() * dates.nunique() * len(config.train_horizons)
    )
    manifest = {
        "format_version": ARTEFACT_FORMAT_VERSION,
        "input_digest": input_digest,
        "rows": len(train),
        "candidate_rows_before_sampling": candidate_rows,
        "sampled": len(train) < candidate_rows,
        "seed": config.seed,
        "cutoff": pd.Timestamp(config.cutoff).isoformat(),
        "train_window_days": config.train_window_days,
        "city_history_days": config.city_history_days,
        "max_train_rows": config.max_train_rows,
        "checkin_min": dates.min().isoformat(),
        "checkin_max": dates.max().isoformat(),
        "horizons": horizons,
        "feature_groups": list(groups),
        "feature_count": len(train.X.columns),
        "feature_schema": _schema(train.X),
        "feature_quality": feature_quality,
        "meta_schema": _schema(train.meta),
        "target": "remaining_demand",
        "target_summary": {
            "minimum": float(np.min(train.y)),
            "median": float(np.quantile(train.y, 0.5)),
            "p90": float(np.quantile(train.y, 0.9)),
            "p99": float(np.quantile(train.y, 0.99)),
            "maximum": float(np.max(train.y)),
            "zero_rows": int((np.asarray(train.y) == 0).sum()),
        },
        "rows_by_horizon": {
            str(key): int(value)
            for key, value in train.meta.groupby("horizon").size().items()
        },
        "files": files,
    }
    write_json_atomic(manifest, directory / "manifest.json")
    return manifest


def load_materialized_training_frame(directory: str | Path) -> SupervisedFrame:
    """Load a train set only after its files and declared schema are verified."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"training manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != ARTEFACT_FORMAT_VERSION:
        raise ValueError(f"unsupported training artefact format: {manifest.get('format_version')}")
    verify_checksums(directory, manifest["files"])

    X = pd.read_parquet(directory / "features.parquet")
    target = pd.read_parquet(directory / "target.parquet")
    meta = pd.read_parquet(directory / "meta.parquet")
    expected_columns = [item["name"] for item in manifest["feature_schema"]]
    if list(X.columns) != expected_columns:
        raise ValueError("materialised feature columns do not match the manifest")
    if _schema(X) != manifest["feature_schema"]:
        raise ValueError("materialised feature dtypes do not match the manifest")
    if _schema(meta) != manifest["meta_schema"]:
        raise ValueError("materialised metadata schema does not match the manifest")
    quality = _feature_quality(X)
    if manifest.get("feature_quality") not in (None, quality):
        raise ValueError("materialised feature-quality audit does not match the manifest")
    if list(target.columns) != [manifest["target"]]:
        raise ValueError("materialised target column does not match the manifest")
    if not (len(X) == len(target) == len(meta) == int(manifest["rows"])):
        raise ValueError("materialised train-set components have different row counts")

    y = target[manifest["target"]].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("materialised target contains invalid values")
    expected_target = np.maximum(
        meta["final"].to_numpy(dtype=np.float64)
        - meta["observed"].to_numpy(dtype=np.float64),
        0.0,
    )
    if not np.array_equal(y, expected_target):
        raise ValueError("materialised target does not match its audit metadata")

    return SupervisedFrame(
        X=X,
        y=y,
        meta=meta,
    )


def build_training_frame(
    data: Pol4Data,
    cutoff: pd.Timestamp,
    groups: tuple[str, ...],
    baseline: Any,
    config: Pol4Config | None = None,
) -> SupervisedFrame:
    """Rows for every completed check-in in the training window."""
    config = config or Pol4Config()
    cutoff = pd.Timestamp(cutoff)

    history = data.history_before(cutoff)
    start = max(cutoff - pd.Timedelta(days=config.train_window_days - 1), history[CHECKIN].min())
    dates = pd.date_range(start, cutoff, freq="D")
    keys = _complete_grid(data.city_codes, dates)
    events = history[history[CHECKIN].isin(dates)]

    builder = FeatureBuilder.build(data, cutoff, keys, events, config)
    final = builder.tensor.final()

    blocks: list[pd.DataFrame] = []
    metas: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for horizon in config.train_horizons:
        frame = builder.at_horizon(horizon, groups, baseline)
        observed = frame["observed_total"].to_numpy()
        blocks.append(frame)
        metas.append(
            keys.assign(horizon=horizon, observed=observed, final=final)
        )
        targets.append(final - observed)

    X = pd.concat(blocks, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    y = np.concatenate(targets)

    if config.max_train_rows is not None and len(X) > config.max_train_rows:
        rng = np.random.default_rng(config.seed)
        picked = np.sort(rng.choice(len(X), size=config.max_train_rows, replace=False))
        X = X.iloc[picked].reset_index(drop=True)
        meta = meta.iloc[picked].reset_index(drop=True)
        y = y[picked]

    return SupervisedFrame(X=X, y=np.maximum(y, 0.0), meta=meta)


def build_inference_frame(
    data: Pol4Data,
    cutoff: pd.Timestamp,
    target_dates: pd.DatetimeIndex,
    groups: tuple[str, ...],
    baseline: Any,
    config: Pol4Config | None = None,
) -> SupervisedFrame:
    """Rows for the 30 target check-ins, each at its own horizon.

    Every city shares a check-in's horizon, so the features are built once per
    horizon and the matching pairs are selected out - 30 passes, not 9,630.
    """
    config = config or Pol4Config()
    cutoff = pd.Timestamp(cutoff)
    target_dates = pd.DatetimeIndex(target_dates)

    keys = _complete_grid(data.city_codes, target_dates)
    rows = pd.concat([data.search, data.evaluation], ignore_index=True)
    # Belt and braces: the tensor would ignore these columns anyway (a feature
    # at horizon h never reads below h), but filtering here makes the guarantee
    # visible and gives the leakage test something to bite on.
    events = rows[
        (rows["log_date"] <= cutoff)
        & (rows[CHECKIN] >= target_dates.min())
        & (rows[CHECKIN] <= target_dates.max())
    ]

    builder = FeatureBuilder.build(data, cutoff, keys, events, config)
    horizons = (pd.DatetimeIndex(keys[CHECKIN]) - cutoff).days.to_numpy()

    blocks: list[pd.DataFrame] = []
    metas: list[pd.DataFrame] = []
    for horizon in np.unique(horizons):
        mask = horizons == horizon
        frame = builder.at_horizon(int(horizon), groups, baseline)
        blocks.append(frame[mask])
        metas.append(
            keys[mask].assign(
                horizon=int(horizon), observed=frame.loc[mask, "observed_total"].to_numpy()
            )
        )

    X = pd.concat(blocks, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    order = np.lexsort((meta[CHECKIN].to_numpy(), meta[CITY].to_numpy()))
    return SupervisedFrame(
        X=X.iloc[order].reset_index(drop=True),
        y=None,
        meta=meta.iloc[order].reset_index(drop=True),
    )
