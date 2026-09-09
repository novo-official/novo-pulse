"""Constants and knobs for the Pol 4 competition run.

Everything the brief fixes lives here as a default, and everything the brief
leaves open is a parameter. No number in this package is written twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import pandas as pd

from ..paths import RAW_DATA_DIR, REPO_ROOT

# ---------------------------------------------------------------- the brief
#: Data as of Aban 30, 1404. No information after this may reach inference.
COMPETITION_CUTOFF = pd.Timestamp("2025-11-21")
#: Azar 1404 - the 30 check-in dates to forecast.
TARGET_START = pd.Timestamp("2025-11-22")
TARGET_END = pd.Timestamp("2025-12-21")
TARGET_DAYS = 30
N_CITIES = 321
N_SUBMISSION_ROWS = N_CITIES * TARGET_DAYS  # 9,630

#: The booking window observed in this dataset. Searches never occur more than
#: this many days before the check-in, which is why so much of a near-term
#: check-in is already visible at the cutoff.
MAX_LEAD_TIME = 59

SUBMISSION_COLUMNS = ("cluster_code", "checkin", "predicted_demand")

# ------------------------------------------------------------------- paths
POL4_RAW_DIR = RAW_DATA_DIR / "pol4"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "pol4"


@dataclass
class Pol4Config:
    """Configuration for one Pol 4 run."""

    raw_dir: Path = POL4_RAW_DIR
    artifacts_dir: Path = ARTIFACTS_DIR

    cutoff: pd.Timestamp = COMPETITION_CUTOFF
    target_start: pd.Timestamp = TARGET_START
    target_days: int = TARGET_DAYS

    # -- pickup curve --------------------------------------------------------
    #: A city gets its own completion curve only once its history carries this
    #: much total demand. Below it the curve is noise and the province (then the
    #: global) curve is the better estimator. Tuned in Phase 2 (experiment E1).
    min_city_support: int = 20_000
    #: A province needs less support than a city because it pools 18-78 cities,
    #: but an empty province must still fall through to the global curve.
    min_province_support: int = 50_000
    #: Never divide by a completion fraction below this. At h=30 the global
    #: fraction is ~0.087; anything an order of magnitude smaller is an
    #: estimation artefact and would explode the projection.
    min_fraction: float = 0.005

    # -- baseline behaviour --------------------------------------------------
    #: What to predict when a (city, check-in) pair has zero observed searches
    #: at the cutoff. "prior" uses the city's own historical demand level;
    #: "zero" predicts nothing. Zero-so-far does NOT mean zero final demand -
    #: 82 of 321 cities have no observed Azar searches at all - so "prior" is
    #: the default and "zero" exists to measure what the prior is worth.
    zero_observation_policy: str = "prior"
    #: Trailing window (in check-in days before the cutoff) used for the prior.
    prior_window_days: int = 120

    # -- backtest ------------------------------------------------------------
    #: Simulated competition cutoffs. Each is followed by a 30-day target
    #: window, exactly like the real one. The first is the seasonal analogue of
    #: the competition window one year earlier.
    backtest_cutoffs: list[str] = field(
        default_factory=lambda: [
            "2024-11-21",
            "2025-05-21",
            "2025-08-21",
            "2025-09-22",
            "2025-10-22",
        ]
    )
    horizon_buckets: tuple[tuple[int, int], ...] = ((1, 3), (4, 7), (8, 14), (15, 21), (22, 30))

    # -- calibration ---------------------------------------------------------
    #: How many earlier 30-day windows a fold's calibration is fitted on. Each
    #: one closes before the fold's own cutoff, so none of them can see it.
    calibration_windows: int = 4
    #: Shrinkage weight, in units of demand: a horizon bucket supported by this
    #: much demand gets half its own factor and half the global one. Keeps a thin
    #: bucket from swinging the forecast on noise.
    calibration_shrinkage: float = 500_000.0
    calibration_method: str = "horizon_shrunk"

    # -- features / models ---------------------------------------------------
    #: Trailing window for the per-city demand statistics.
    city_history_days: int = 365
    #: Horizons sampled when building training rows. Stratified rather than
    #: exhaustive: 30 horizons per pair is 7M rows for no extra signal.
    train_horizons: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 14, 18, 21, 25, 30)
    #: Check-ins before the cutoff used for training. Longer is not always
    #: better - the regime drifts - so this is an experiment knob.
    train_window_days: int = 540
    max_train_rows: int = 900_000
    #: Trees for the ablation ladder. Fewer than the champion uses: the ladder
    #: compares feature sets, and nine full-size fits per fold buys nothing.
    ablation_params: dict[str, Any] = field(default_factory=lambda: {"n_estimators": 300})
    #: Check-in days in the historical window used for the stability snapshots.
    stability_window_days: int = 30
    #: Rows in the committed training-data sample, stratified by horizon.
    training_sample_rows: int = 5_500
    #: Quantile edges for the demand-bucket error breakdown.
    demand_bucket_quantiles: tuple[float, ...] = (0.0, 0.5, 0.75, 0.9, 0.99, 1.0)

    seed: int = 42

    @property
    def target_end(self) -> pd.Timestamp:
        return self.target_start + pd.Timedelta(days=self.target_days - 1)

    def target_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.target_start, periods=self.target_days, freq="D")

    @property
    def search_path(self) -> Path:
        return self.raw_dir / "search_data.csv"

    @property
    def evaluation_path(self) -> Path:
        return self.raw_dir / "evaluation.csv"

    @property
    def cities_path(self) -> Path:
        return self.raw_dir / "cities.csv"

    @property
    def city_names_path(self) -> Path:
        """Optional city_code -> name mapping. Reports fall back to codes."""
        return self.raw_dir / "city_code_mapping.csv"
