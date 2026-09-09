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
curves, the baseline, the backtest, and the submission file. It deliberately
does not go through `ml/contract.py`, because a `DataContract` has exactly one
timestamp field and no way to express a forecast that is already half-observed.
"""
from .config import Pol4Config
from .loader import Pol4Data, load_pol4
from .pickup import PickupCurves
from .baseline import PickupBaseline
from .submission import build_submission, validate_submission, SubmissionError

__all__ = [
    "Pol4Config",
    "Pol4Data",
    "load_pol4",
    "PickupCurves",
    "PickupBaseline",
    "build_submission",
    "validate_submission",
    "SubmissionError",
]
