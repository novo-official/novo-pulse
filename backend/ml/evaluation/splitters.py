"""Time-series splitting. Random splits are never used anywhere in this project.

Rolling-origin (walk-forward) evaluation:

    Fold 1:  TRAIN ================|  VALID ----
    Fold 2:  TRAIN ======================|  VALID ----
    Fold 3:  TRAIN ============================|  VALID ----

Each fold trains only on data strictly before its cutoff and validates on the
`horizon` periods that follow it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..contract import TS


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    def as_dict(self) -> dict[str, str | int]:
        return {
            "fold": self.index,
            "train_end": str(self.train_end.date()),
            "valid_start": str(self.valid_start.date()),
            "valid_end": str(self.valid_end.date()),
        }


class RollingOriginSplitter:
    """Generate walk-forward folds over a regularly-spaced timeline."""

    def __init__(
        self,
        horizon: int,
        n_folds: int = 3,
        step: int | None = None,
        freq: str = "D",
        min_train_periods: int = 90,
    ):
        self.horizon = int(horizon)
        self.n_folds = max(1, int(n_folds))
        self.step = int(step) if step else self.horizon
        self.freq = freq
        self.min_train_periods = int(min_train_periods)

    def split(self, timestamps: pd.Series | pd.DatetimeIndex) -> list[Fold]:
        stamps = pd.DatetimeIndex(pd.Series(timestamps).dropna().unique()).sort_values()
        if len(stamps) < self.min_train_periods + self.horizon:
            # Not enough history for the requested layout - shrink gracefully
            # rather than refusing to evaluate at all.
            usable_folds = 1
            min_train = max(len(stamps) - self.horizon, max(1, len(stamps) // 2))
        else:
            usable_folds = self.n_folds
            min_train = self.min_train_periods

        folds: list[Fold] = []
        last_index = len(stamps) - 1
        for i in range(usable_folds):
            valid_end_idx = last_index - i * self.step
            valid_start_idx = valid_end_idx - self.horizon + 1
            train_end_idx = valid_start_idx - 1
            if train_end_idx < min_train - 1 or valid_start_idx < 0:
                break
            folds.append(
                Fold(
                    index=len(folds) + 1,
                    train_end=stamps[train_end_idx],
                    valid_start=stamps[valid_start_idx],
                    valid_end=stamps[valid_end_idx],
                )
            )
        folds.reverse()
        for position, fold in enumerate(folds, start=1):
            folds[position - 1] = Fold(
                index=position,
                train_end=fold.train_end,
                valid_start=fold.valid_start,
                valid_end=fold.valid_end,
            )
        return folds

    def split_frame(self, frame: pd.DataFrame) -> list[tuple[Fold, pd.Index, pd.Index]]:
        """Return (fold, train_index, valid_index) triples for a panel frame."""
        out = []
        for fold in self.split(frame[TS]):
            train_mask = frame[TS] <= fold.train_end
            valid_mask = (frame[TS] >= fold.valid_start) & (frame[TS] <= fold.valid_end)
            out.append((fold, frame.index[train_mask], frame.index[valid_mask]))
        return out


def holdout_split(timestamps: pd.Series, horizon: int) -> Fold:
    """A single final holdout: the last `horizon` periods."""
    stamps = pd.DatetimeIndex(pd.Series(timestamps).dropna().unique()).sort_values()
    horizon = min(horizon, max(1, len(stamps) - 1))
    valid_start_idx = len(stamps) - horizon
    return Fold(
        index=1,
        train_end=stamps[valid_start_idx - 1],
        valid_start=stamps[valid_start_idx],
        valid_end=stamps[-1],
    )
