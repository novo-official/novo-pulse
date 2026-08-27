"""SHAP-based explainability for the tree models.

Two views, both computed from real numbers:

* **global** - mean |SHAP| per feature over a sample of the forecast horizon,
  rolled up into human-readable driver groups (holiday, price, season, ...).
* **local**  - for one destination and date window, the *signed* contribution
  of each driver group relative to the model's base value.

Percentages are only reported where they are defensible: a group's share of the
total absolute contribution. Signed effects are reported in target units and as
a share of the base value, never dressed up as a causal elasticity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MAX_SHAP_ROWS = 4000

# Persian labels for the driver groups surfaced in the dashboard.
GROUP_LABELS_FA = {
    "holiday": "تعطیلات و مناسبت‌ها",
    "weekday": "روز هفته و آخر هفته",
    "season": "فصل و الگوی سالانه",
    "calendar": "تقویم",
    "price": "قیمت",
    "availability": "ظرفیت و موجودی",
    "promotion": "تخفیف و کمپین",
    "planned_covariates": "متغیرهای برنامه‌ریزی‌شده",
    "search_activity": "جست‌وجو و بازدید کاربران",
    "recent_behaviour": "رفتار اخیر کاربران",
    "demand_history": "روند تاریخی تقاضا",
    "entity_profile": "مشخصات اقامتگاه",
    "horizon": "افق پیش‌بینی",
    "other": "سایر",
}


@dataclass
class ExplanationResult:
    global_importance: list[dict[str, Any]]
    group_importance: list[dict[str, Any]]
    base_value: float
    method: str
    n_samples: int
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_importance": self.global_importance,
            "group_importance": self.group_importance,
            "base_value": self.base_value,
            "method": self.method,
            "n_samples": self.n_samples,
            "note": self.note,
        }


class ShapExplainer:
    """Wraps `shap.TreeExplainer`, degrading to gain importance when absent."""

    def __init__(self, model, feature_groups: dict[str, str]):
        self.model = model
        self.feature_groups = feature_groups
        self.explainer = None
        self.base_value = 0.0
        self.method = "unavailable"

    def _underlying(self):
        """The raw booster SHAP can read."""
        return getattr(self.model, "model", None)

    def prepare(self, background: pd.DataFrame) -> "ShapExplainer":
        """Pick the fastest contribution backend this model supports.

        LightGBM and CatBoost compute exact tree SHAP values natively and -
        unlike the generic `shap` wrapper - they keep the categorical encoding
        the model was fitted with, so we prefer them.
        """
        booster = self._underlying()
        if booster is None:
            return self
        if hasattr(booster, "booster_"):
            self.method = "lightgbm.pred_contrib"
            return self
        if hasattr(booster, "get_feature_importance"):
            self.method = "catboost.ShapValues"
            return self
        try:
            import shap

            self.explainer = shap.TreeExplainer(booster)
            self.method = "shap.TreeExplainer"
        except Exception as exc:  # noqa: BLE001 - SHAP is a nice-to-have
            log.warning("SHAP unavailable, falling back to gain importance: %s", exc)
            self.explainer = None
            self.method = "gain_importance"
        return self

    # ------------------------------------------------------------- compute
    def shap_values(self, X: pd.DataFrame) -> np.ndarray | None:
        """Per-row, per-feature contributions. `None` means "not available"."""
        try:
            if self.method == "lightgbm.pred_contrib":
                return self._lightgbm_contributions(X)
            if self.method == "catboost.ShapValues":
                return self._catboost_contributions(X)
            if self.explainer is not None:
                frame = self._numeric(X)
                values = self.explainer.shap_values(frame, check_additivity=False)
                expected = self.explainer.expected_value
                self.base_value = float(
                    np.mean(expected) if isinstance(expected, (list, np.ndarray)) else expected
                )
                return np.asarray(values, dtype=float)
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP computation failed (%s): %s", self.method, exc)
        return None

    def _lightgbm_contributions(self, X: pd.DataFrame) -> np.ndarray:
        booster = self._underlying().booster_
        aligned = X.reindex(columns=getattr(self.model, "feature_names", list(X.columns)))
        raw = np.asarray(booster.predict(aligned, pred_contrib=True), dtype=float)
        # The final column is the expected (base) value.
        self.base_value = float(np.mean(raw[:, -1]))
        return raw[:, :-1]

    def _catboost_contributions(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        booster = self._underlying()
        prepared = self.model._catboost_frame(X)  # noqa: SLF001 - same package
        pool = Pool(prepared, cat_features=getattr(self.model, "categorical", []))
        raw = np.asarray(
            booster.get_feature_importance(pool, type="ShapValues"), dtype=float
        )
        self.base_value = float(np.mean(raw[:, -1]))
        return raw[:, :-1]

    def _numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP needs numeric input; categorical codes preserve the split logic."""
        frame = X.copy()
        for column in frame.columns:
            if isinstance(frame[column].dtype, pd.CategoricalDtype):
                frame[column] = frame[column].cat.codes.astype(np.float64)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float64)
        return frame

    # -------------------------------------------------------------- global
    def explain_global(self, X: pd.DataFrame, top_n: int = 20) -> ExplanationResult:
        sample = X.sample(min(len(X), MAX_SHAP_ROWS), random_state=42) if len(X) else X
        values = self.shap_values(sample)

        if values is None:
            importance = self.model.get_feature_importance()
            rows = [
                {"feature": name, "importance": round(float(score), 6), "direction": None}
                for name, score in list(importance.items())[:top_n]
            ]
            groups = self._group_rollup(
                {k: float(v) for k, v in importance.items()}, signed={}
            )
            return ExplanationResult(
                global_importance=rows,
                group_importance=groups,
                base_value=0.0,
                method="gain_importance",
                n_samples=int(len(sample)),
                note="SHAP was not available; reporting split-gain importance instead.",
            )

        magnitude = np.abs(values).mean(axis=0)
        signed = values.mean(axis=0)
        total = magnitude.sum()
        names = list(sample.columns)

        rows = [
            {
                "feature": name,
                "importance": round(float(mag / total), 6) if total > 0 else 0.0,
                "mean_abs_shap": round(float(mag), 6),
                "mean_shap": round(float(sig), 6),
                "direction": "positive" if sig > 0 else ("negative" if sig < 0 else "neutral"),
                "group": self.feature_groups.get(name, "other"),
            }
            for name, mag, sig in zip(names, magnitude, signed)
        ]
        rows.sort(key=lambda item: -item["mean_abs_shap"])

        groups = self._group_rollup(
            dict(zip(names, magnitude.astype(float))),
            signed=dict(zip(names, signed.astype(float))),
        )
        return ExplanationResult(
            global_importance=rows[:top_n],
            group_importance=groups,
            base_value=round(self.base_value, 6),
            method=self.method,
            n_samples=int(len(sample)),
        )

    # --------------------------------------------------------------- local
    def explain_local(
        self, X: pd.DataFrame, top_n: int = 8
    ) -> dict[str, Any]:
        """Signed driver contributions for a specific slice of the forecast."""
        if len(X) == 0:
            return {"drivers": [], "base_value": 0.0, "method": "empty", "n_samples": 0}
        sample = X.sample(min(len(X), MAX_SHAP_ROWS), random_state=42)
        values = self.shap_values(sample)
        if values is None:
            importance = self.model.get_feature_importance()
            groups = self._group_rollup({k: float(v) for k, v in importance.items()}, signed={})
            return {
                "drivers": groups[:top_n],
                "base_value": 0.0,
                "method": "gain_importance",
                "n_samples": int(len(sample)),
                "note": "SHAP unavailable - magnitudes only, no direction.",
            }

        names = list(sample.columns)
        signed = values.mean(axis=0)
        magnitude = np.abs(values).mean(axis=0)
        groups = self._group_rollup(
            dict(zip(names, magnitude.astype(float))),
            signed=dict(zip(names, signed.astype(float))),
        )
        return {
            "drivers": groups[:top_n],
            "base_value": round(self.base_value, 6),
            "prediction_mean": round(float(self.base_value + signed.sum()), 6),
            "method": self.method,
            "n_samples": int(len(sample)),
        }

    # -------------------------------------------------------------- rollup
    def _group_rollup(
        self, magnitude: dict[str, float], signed: dict[str, float]
    ) -> list[dict[str, Any]]:
        totals: dict[str, float] = {}
        signed_totals: dict[str, float] = {}
        for feature, value in magnitude.items():
            group = self.feature_groups.get(feature, "other")
            totals[group] = totals.get(group, 0.0) + float(value)
            signed_totals[group] = signed_totals.get(group, 0.0) + float(signed.get(feature, 0.0))

        grand_total = sum(totals.values())
        rows = []
        for group, value in totals.items():
            share = value / grand_total if grand_total > 0 else 0.0
            effect = signed_totals.get(group, 0.0)
            rows.append(
                {
                    "group": group,
                    "label_fa": GROUP_LABELS_FA.get(group, group),
                    "contribution_share": round(float(share), 6),
                    "contribution_score": round(float(value), 6),
                    "signed_effect": round(float(effect), 6),
                    "direction": "positive" if effect > 0 else ("negative" if effect < 0 else "neutral"),
                }
            )
        rows.sort(key=lambda item: -item["contribution_score"])
        return rows


def dependence_curve(
    X: pd.DataFrame, shap_values: np.ndarray, feature: str, bins: int = 12
) -> list[dict[str, float]]:
    """SHAP dependence: how the model's contribution varies with a feature.

    Used for the price/demand relationship. It describes the *model*, which was
    fitted to observational data - it is an association, not a causal effect.
    """
    if feature not in X.columns:
        return []
    index = list(X.columns).index(feature)
    values = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
    contributions = shap_values[:, index]
    mask = np.isfinite(values) & np.isfinite(contributions)
    if mask.sum() < 20:
        return []
    values, contributions = values[mask], contributions[mask]

    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return []
    assignment = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        selected = assignment == b
        if selected.sum() < 5:
            continue
        out.append(
            {
                "bin": b,
                "feature_value": round(float(np.mean(values[selected])), 4),
                "feature_min": round(float(edges[b]), 4),
                "feature_max": round(float(edges[b + 1]), 4),
                "mean_contribution": round(float(np.mean(contributions[selected])), 6),
                "n": int(selected.sum()),
            }
        )
    return out
