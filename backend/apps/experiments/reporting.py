"""Automatic model comparison report (`reports/model_report.md`).

Written after every successful run. It is the document to hand a judge who asks
"how do you know the model is any good?".
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ml.paths import REPORTS_DIR


def write_model_report(run, result) -> Path:
    metrics = result.metrics
    metric = metrics.get("primary_metric", "wape")
    lines: list[str] = []

    add = lines.append
    add(f"# Model Report - {run.run_id}")
    add("")
    add(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    add("")

    # ---- dataset ---------------------------------------------------------
    panel = metrics.get("panel") or {}
    quality = metrics.get("data_quality") or {}
    add("## Dataset")
    add("")
    add("| Property | Value |")
    add("| --- | --- |")
    add(f"| Dataset | `{run.dataset_name}` |")
    add(f"| Target | `{run.target}` |")
    add(f"| Rows | {panel.get('rows', 0):,} |")
    add(f"| Entities | {panel.get('entities', 0):,} |")
    add(f"| Periods | {panel.get('periods', 0):,} |")
    add(f"| Range | {panel.get('start')} → {panel.get('end')} |")
    add(f"| Frequency | `{panel.get('frequency')}` |")
    add(f"| Zero ratio | {panel.get('zero_ratio', 0):.1%} |")
    add(f"| Data health | {quality.get('health_score', 'n/a')}/100 ({quality.get('grade', 'n/a')}) |")
    add("")

    # ---- validation setup ------------------------------------------------
    add("## Validation setup")
    add("")
    add(
        "Rolling-origin (walk-forward) cross-validation. No random splitting is used "
        "anywhere in this project."
    )
    add("")
    add("| Fold | Train ends | Validation window |")
    add("| --- | --- | --- |")
    for fold in metrics.get("folds") or []:
        add(f"| {fold['fold']} | {fold['train_end']} | {fold['valid_start']} → {fold['valid_end']} |")
    add("")
    add(f"- Primary metric: **{metric.upper()}**")
    add(f"- Forecast horizon: **{metrics.get('horizon')}** periods")
    add(f"- Training profile: `{run.profile}`")
    add("")

    # ---- leaderboard -----------------------------------------------------
    leaderboard = metrics.get("leaderboard") or []
    add("## Leaderboard")
    add("")
    add(f"| # | Model | Type | {metric.upper()} | MAE | RMSE | vs. baseline |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for row in leaderboard:
        scores = row.get("metrics") or {}
        improvement = row.get("improvement_vs_baseline")
        add(
            f"| {row.get('rank', '')} | `{row['model']}`"
            f"{' **(champion)**' if row['model'] == result.champion else ''} "
            f"| {'baseline' if row.get('is_baseline') else 'learned'} "
            f"| {_fmt(row.get('primary_value'))} | {_fmt(scores.get('mae'))} "
            f"| {_fmt(scores.get('rmse'))} "
            f"| {'—' if improvement is None else f'{improvement:+.1%}'} |"
        )
    add("")

    champion_row = next((r for r in leaderboard if r["model"] == result.champion), None)
    baseline_row = next((r for r in leaderboard if r.get("is_baseline")), None)
    if champion_row and baseline_row:
        add("### Headline")
        add("")
        add(f"- Champion `{result.champion}` {metric.upper()}: **{_fmt(champion_row.get('primary_value'))}**")
        add(f"- Best baseline `{baseline_row['model']}` {metric.upper()}: **{_fmt(baseline_row.get('primary_value'))}**")
        if champion_row.get("improvement_vs_baseline") is not None:
            add(f"- Improvement: **{champion_row['improvement_vs_baseline']:+.1%}**")
        add("")

    # ---- horizon degradation --------------------------------------------
    add("## Performance by horizon")
    add("")
    add(f"| Horizon | {metric.upper()} | MAE | n |")
    add("| --- | --- | --- | --- |")
    for bucket in (metrics.get("horizon_scores") or {}).get(result.champion, []):
        add(
            f"| {bucket['bucket']} | {_fmt(bucket.get(metric))} "
            f"| {_fmt(bucket.get('mae'))} | {bucket.get('n', 0):,} |"
        )
    add("")

    # ---- segments --------------------------------------------------------
    segments = metrics.get("segment_scores") or {}
    if segments.get("by_destination"):
        add("## Performance by segment (destination)")
        add("")
        add(f"| Destination | {metric.upper()} | n |")
        add("| --- | --- | --- |")
        for row in sorted(
            segments["by_destination"], key=lambda r: (r.get(metric) is None, r.get(metric) or 0)
        )[:15]:
            add(f"| {row['key']} | {_fmt(row.get(metric))} | {row['n']:,} |")
        add("")

    # ---- intervals -------------------------------------------------------
    intervals = metrics.get("intervals") or {}
    add("## Prediction interval coverage")
    add("")
    add(f"Method: `{metrics.get('uncertainty_method')}`")
    add("")
    add("| Horizon bucket | Nominal | Observed | Mean width |")
    add("| --- | --- | --- | --- |")
    for row in metrics.get("coverage_by_horizon") or []:
        add(
            f"| {row['bucket']} | {row['nominal_coverage']:.0%} "
            f"| {row['observed_coverage']:.1%} | {row['mean_width']:.2f} |"
        )
    add("")
    if intervals.get("observed_coverage") is not None:
        add(
            f"Overall: nominal {intervals['nominal_coverage']:.0%}, "
            f"observed **{intervals['observed_coverage']:.1%}** "
            f"(gap {intervals['coverage_gap']:+.1%})."
        )
        add("")

    # ---- ensemble --------------------------------------------------------
    weights = metrics.get("ensemble_weights") or {}
    if weights:
        add("## Ensemble weights")
        add("")
        add("Derived from rolling-CV scores on the primary metric - never hard-coded.")
        add("")
        add("| Model | Weight |")
        add("| --- | --- |")
        for name, weight in weights.items():
            add(f"| `{name}` | {weight:.1%} |")
        add("")

    # ---- hyper-parameter search -----------------------------------------
    tuning = metrics.get("tuning") or []
    if tuning:
        add("## Hyper-parameter search")
        add("")
        add(
            "Searched with Optuna against a held-out time window, time-boxed by "
            "`tuning.max_minutes`. Defaults are kept whenever the search does not beat them."
        )
        add("")
        add("| Model | Trials | Default | Best | Kept | Seconds |")
        add("| --- | --- | --- | --- | --- | --- |")
        for entry in tuning:
            add(
                f"| `{entry['model']}` | {entry['n_trials']} "
                f"| {_fmt(entry.get('default_score'))} | {_fmt(entry.get('best_score'))} "
                f"| {'tuned' if entry.get('improved') else 'defaults'} "
                f"| {entry.get('seconds', 0)} |"
            )
            if entry.get("note"):
                add(f"| | | | | | _{entry['note']}_ |")
        add("")

    # ---- demand censoring -------------------------------------------------
    censoring = metrics.get("censoring") or {}
    if censoring.get("enabled"):
        add("## Demand censoring")
        add("")
        add(
            f"{censoring['censored_share']:.1%} of observed periods sit at or above "
            f"available capacity (`{censoring['capacity_column']}`), across "
            f"{censoring['affected_entities']} entities."
        )
        add("")
        add(
            "Where supply binds, the recorded target is capacity rather than demand, so "
            "the forecast describes **bookable** demand rather than unconstrained market "
            f"demand. Estimated mean uplift on censored periods: "
            f"{censoring['mean_uplift']:+.1%}."
        )
        add("")
        add(f"Method: {censoring['method']}")
        add("")

    # ---- drivers ---------------------------------------------------------
    importance = _read_json(result.run_dir / "feature_importance.json")
    groups = importance.get("group_importance") or []
    if groups:
        add("## Top demand drivers")
        add("")
        add(f"Method: `{importance.get('method')}`")
        add("")
        add("| Driver group | Share of contribution | Direction |")
        add("| --- | --- | --- |")
        for group in groups[:10]:
            add(
                f"| {group['group']} | {group['contribution_share']:.1%} "
                f"| {group['direction']} |"
            )
        add("")

    features = importance.get("global_importance") or []
    if features:
        add("### Top individual features")
        add("")
        add("| Feature | Importance |")
        add("| --- | --- |")
        for feature in features[:15]:
            add(f"| `{feature['feature']}` | {feature.get('importance', 0):.2%} |")
        add("")

    # ---- limitations -----------------------------------------------------
    add("## Known limitations")
    add("")
    add(
        "- Aggregated prediction intervals are bottom-up sums of listing-level "
        "bounds, which assumes correlated errors and therefore overstates width."
    )
    add(
        "- Driver contributions describe what the model learned from observational "
        "data. They are associations, not causal effects."
    )
    add(
        "- Known-future covariates beyond the observed range are projected "
        "seasonally unless real planned values are supplied."
    )
    for warning in result.warnings:
        add(f"- {warning}")
    add("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "model_report.md"
    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")
    (REPORTS_DIR / f"model_report_{run.run_id}.md").write_text(content, encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) < 1:
        return f"{number:.4f}"
    return f"{number:,.3f}"


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
