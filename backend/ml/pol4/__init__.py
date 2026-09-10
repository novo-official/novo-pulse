"""Pol 4 competition domain.

The generic platform in `backend/ml/` models ONE time axis: a value observed at
a timestamp. The Pol 4 problem has TWO, and collapsing them loses the signal
that decides the competition:

    log_date  - when somebody searched
    checkin   - the night they want to stay

    days_to_checkin = checkin - log_date          (0 .. 59 in this dataset)

Demand for a (city, checkin) pair accumulates over the ~60 days before the
check-in. At any cutoff C the pair is therefore only *partially* observed:

    final_demand   = observed_demand(C) + remaining_pickup(C)

This package owns that formulation end to end - loading, cutoff-safe pickup
curves, the baseline, features, the remaining-demand model, the backtest,
forecast stability and the submission file. It deliberately does not go through
`ml/contract.py`, because a `DataContract` has exactly one timestamp field and
no way to express a forecast that is already half-observed.

    loader      the three CSVs, validated, events intact
    pickup      completion curves, city -> province -> global, cutoff-safe
    baseline    observed / expected_completion_fraction  (the Phase 1 champion)
    features    (pair x days_to_checkin) matrices; a feature at horizon h can
                only read columns >= h, which makes cutoff safety structural
    dataset     supervised frames: target is remaining = final - observed
    models      LightGBM / CatBoost on remaining demand
    champion    the selected configuration, fitted and applied
    calibration bias correction - built, measured, and rejected on WAPE
    backtest    pseudo-competition folds
    experiments the ablation harness that decided all of the above
    stability   D-30 -> D-1 forecast snapshots
    submission  the 9,630-row grid, results.csv, and a loud validator

The clustered arm sits alongside it and reuses all of the above:

    clustering  train-only demand-shape dendrograms; membership only
    aggregate   a mapping -> a Pol4Data of virtual cities, so every module
                above runs on a clustered panel without being modified
    cluster_experiment  the aggregation level as a swept parameter
    cluster_pipeline    sweep, select, and write a clustered submission
"""
from .aggregate import aggregate_data
from .baseline import PickupBaseline
from .calibration import Calibrator
from .champion import ChampionSpec, FittedChampion
from .clustering import ClusterAssignment, ClusterPlan, identity_assignment
from .config import Pol4Config
from .loader import Pol4Data, load_pol4
from .pickup import PickupCurves
from .submission import SubmissionError, build_submission, validate_submission

__all__ = [
    "Calibrator",
    "ChampionSpec",
    "ClusterAssignment",
    "ClusterPlan",
    "FittedChampion",
    "Pol4Config",
    "Pol4Data",
    "PickupBaseline",
    "PickupCurves",
    "SubmissionError",
    "aggregate_data",
    "build_submission",
    "identity_assignment",
    "load_pol4",
    "validate_submission",
]
