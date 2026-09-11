"""Generate presentation-ready Pol 4 charts from recorded artefacts.

The chart copy is intentionally English: it keeps technical terminology exact
and avoids relying on Persian shaping support in the presentation environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/novo-pol4-presentation-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "pol4"
CLUSTER_ARTIFACTS = ROOT / "artifacts" / "pol4_cluster"
OUTPUT = ARTIFACTS / "presentation_charts"

NAVY = "#11283f"
BLUE = "#2f7ed8"
BLUE_LIGHT = "#b9d5f4"
ORANGE = "#f06f3c"
AMBER = "#d97706"
TEAL = "#0f9d8a"
SLATE = "#98a6b7"
GRID = "#dce5ef"
BG = "#f4f7fb"
RED = "#d94b5b"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact(value: float, _position: int | None = None) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def canvas(title: str, subtitle: str, *, figsize: tuple[float, float] = (13.333, 7.5)):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    fig.text(0.06, 0.94, title, fontsize=23, weight="bold", color=NAVY, va="top")
    fig.text(0.06, 0.895, subtitle, fontsize=11, color="#60758d", va="top")
    return fig


def style_axis(ax, *, y_grid: bool = True) -> None:
    ax.set_facecolor("white")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors="#60758d", labelsize=9, length=0)
    if y_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle=(0, (2, 4)))
        ax.set_axisbelow(True)


def source(fig, text: str) -> None:
    fig.text(0.06, 0.025, f"Source: {text}", fontsize=8, color="#8292a5")


def save(fig, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def demand_formation(results: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    daily = results.groupby("checkin", as_index=False).agg(
        observed=("observed_so_far", "sum"),
        remaining=("predicted_remaining", "sum"),
        final=("predicted_demand", "sum"),
    )
    daily["checkin"] = pd.to_datetime(daily["checkin"])
    peak = daily.loc[daily["final"].idxmax()]
    low = daily.loc[daily["final"].idxmin()]
    observed_share = daily["observed"].sum() / daily["final"].sum()

    fig = canvas(
        "Demand formation across the next 30 check-in nights",
        "Observed demand is a fact; the model forecasts only the remaining demand.",
    )
    ax = fig.add_axes([0.06, 0.15, 0.88, 0.64])
    style_axis(ax)
    ax.fill_between(daily["checkin"], 0, daily["observed"], color=ORANGE, alpha=0.72, label="Observed by cutoff")
    ax.fill_between(daily["checkin"], daily["observed"], daily["final"], color=BLUE_LIGHT, alpha=0.95, label="Forecast remaining")
    ax.plot(daily["checkin"], daily["final"], color=NAVY, linewidth=2.5, label="Final demand forecast")
    ax.axvline(peak["checkin"], color=RED, linestyle=(0, (4, 4)), linewidth=1.3)
    ax.axvline(low["checkin"], color=TEAL, linestyle=(0, (4, 4)), linewidth=1.3)
    ax.annotate(f"Peak\n{peak['checkin']:%b %d}\n{compact(peak['final'])}", (peak["checkin"], peak["final"]),
                xytext=(9, -12), textcoords="offset points", va="top", color=RED, fontsize=9, weight="bold")
    ax.annotate(f"Low\n{low['checkin']:%b %d}\n{compact(low['final'])}", (low["checkin"], low["final"]),
                xytext=(-42, 18), textcoords="offset points", color=TEAL, fontsize=9, weight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("Search demand per night", color="#60758d", labelpad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False, fontsize=9)
    ax.text(0.012, 0.95, f"30-night total  {daily['final'].sum():,.0f}", transform=ax.transAxes,
            fontsize=11, weight="bold", color=NAVY, bbox=dict(boxstyle="round,pad=.5", fc="white", ec=GRID))
    ax.text(0.012, 0.84, f"Observed share  {observed_share:.1%}", transform=ax.transAxes,
            fontsize=10, color="#60758d", bbox=dict(boxstyle="round,pad=.45", fc="white", ec=GRID))
    ax.text(0.985, 0.055, "Blue gap = forecast remaining, not model error", transform=ax.transAxes,
            ha="right", fontsize=9, color=BLUE, weight="bold",
            bbox=dict(boxstyle="round,pad=.35", fc="white", ec=GRID, alpha=0.9))
    source(fig, f"results_calibrated_named.csv · cutoff {cutoff.date()}")
    save(fig, "01_demand_formation.png")


def city_priority(results: pd.DataFrame) -> None:
    cities = results.groupby(["city_code", "city", "province"], as_index=False).agg(
        demand=("predicted_demand", "sum"),
        observed=("observed_so_far", "sum"),
        remaining=("predicted_remaining", "sum"),
    ).sort_values("demand", ascending=False)
    cities["share"] = cities["demand"] / cities["demand"].sum()
    cities["cumulative"] = cities["share"].cumsum()
    top = cities.head(15).copy()
    concentration = cities.head(10)["share"].sum()

    fig = canvas(
        "Destination priority: where is forecast demand concentrated?",
        "A ranked review queue turns the forecast into a finite set of destinations to investigate first.",
    )
    ax = fig.add_axes([0.07, 0.18, 0.86, 0.60])
    style_axis(ax)
    x = np.arange(len(top))
    bars = ax.bar(x, top["demand"], color=BLUE, alpha=0.87, width=0.72, label="Forecast demand")
    ax.yaxis.set_major_formatter(FuncFormatter(compact))
    ax.set_xticks(x, top["city"], rotation=35, ha="right")
    ax.set_ylabel("30-night search demand", color="#60758d", labelpad=10)
    for index, bar in enumerate(bars[:5]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top["demand"].max() * 0.02,
                f"{top.iloc[index]['share']:.1%}", ha="center", va="bottom", fontsize=8, color=NAVY, weight="bold")
    right = ax.twinx()
    right.plot(x, top["cumulative"] * 100, color=ORANGE, marker="o", markersize=4, linewidth=2.2,
               label="Cumulative share of total forecast demand")
    right.axhline(80, color=ORANGE, alpha=0.35, linewidth=1, linestyle=(0, (4, 4)))
    right.set_ylim(0, 105)
    right.yaxis.set_major_formatter(PercentFormatter(100))
    right.tick_params(axis="y", colors=ORANGE, labelsize=9, length=0)
    right.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.text(0.01, 0.93, f"Top 10 = {concentration:.1%} of forecast demand", transform=ax.transAxes,
            fontsize=11, weight="bold", color=NAVY, bbox=dict(boxstyle="round,pad=.5", fc="white", ec=GRID))
    right.text(10.4, 81.5, "80% threshold", ha="left", fontsize=8, color=ORANGE)
    right.annotate(f"Top 15 = {top.iloc[-1]['cumulative']:.1%}",
                   (14, top.iloc[-1]["cumulative"] * 100), xytext=(0, -18),
                   textcoords="offset points", ha="right", fontsize=8.5,
                   color=ORANGE, weight="bold")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right", frameon=False, ncol=2, fontsize=9)
    source(fig, "results_calibrated_named.csv · city aggregation")
    save(fig, "02_city_priority_pareto.png")


def opportunity_matrices(results: pd.DataFrame, momentum: pd.DataFrame) -> None:
    cities = results.groupby(["city_code", "city", "province"], as_index=False).agg(
        demand=("predicted_demand", "sum"),
        observed=("observed_so_far", "sum"),
        remaining=("predicted_remaining", "sum"),
    )
    cities["remaining_share"] = cities["remaining"] / cities["demand"].clip(lower=1)
    cities = cities.merge(momentum[["city_code", "pickup_ratio"]], on="city_code", how="left")
    cities = cities[cities["demand"] > 0].copy()
    sizes = 24 + 330 * np.sqrt(cities["remaining"] / max(cities["remaining"].max(), 1))

    fig = canvas(
        "Two business lenses on the same forecast",
        "Growth detects demand acceleration; Supply flags destinations that deserve an inventory review.",
    )
    growth = fig.add_axes([0.06, 0.16, 0.415, 0.65])
    supply = fig.add_axes([0.535, 0.16, 0.415, 0.65])

    valid = cities.dropna(subset=["pickup_ratio"]).copy()
    gx, gy = valid["demand"].median(), valid["pickup_ratio"].median()
    growth.scatter(valid["demand"], valid["pickup_ratio"], s=sizes.loc[valid.index], c=BLUE, alpha=0.45,
                   edgecolors="white", linewidths=0.5)
    growth.axvline(gx, color=SLATE, linestyle=(0, (4, 4)), linewidth=1)
    growth.axhline(gy, color=SLATE, linestyle=(0, (4, 4)), linewidth=1)
    growth.set_xscale("log")
    growth.set_title("Growth review", loc="left", color=NAVY, weight="bold", fontsize=14)
    growth.set_xlabel("Final demand forecast · log scale", color="#60758d")
    growth.set_ylabel("Pickup / historical expectation", color="#60758d")
    style_axis(growth)
    growth.text(0.98, 0.96, "HIGH DEMAND\n+ FAST PICKUP", transform=growth.transAxes, ha="right", va="top",
                fontsize=8, color=TEAL, weight="bold")
    growth_candidates = valid[(valid["demand"] >= gx) & (valid["pickup_ratio"] >= gy)].assign(
        score=lambda frame: frame["demand"] * frame["pickup_ratio"].clip(upper=3)
    ).nlargest(4, "score")
    for offset, (_, row) in zip((-30, -10, 10, 30), growth_candidates.iterrows()):
        growth.annotate(row["city"], (row["demand"], row["pickup_ratio"]), xytext=(-8, offset),
                        textcoords="offset points", ha="right", va="center", fontsize=7.5, color=NAVY,
                        arrowprops=dict(arrowstyle="-", color="#9cb0c4", linewidth=0.7))

    sx, sy = cities["demand"].median(), cities["remaining_share"].median()
    supply.scatter(cities["demand"], cities["remaining_share"] * 100, s=sizes, c=ORANGE, alpha=0.42,
                   edgecolors="white", linewidths=0.5)
    supply.axvline(sx, color=SLATE, linestyle=(0, (4, 4)), linewidth=1)
    supply.axhline(sy * 100, color=SLATE, linestyle=(0, (4, 4)), linewidth=1)
    supply.set_xscale("log")
    supply.set_title("Supply review", loc="left", color=NAVY, weight="bold", fontsize=14)
    supply.set_xlabel("Final demand forecast · log scale", color="#60758d")
    supply.set_ylabel("Remaining share of forecast", color="#60758d")
    supply.yaxis.set_major_formatter(PercentFormatter(100))
    style_axis(supply)
    supply.text(0.98, 0.96, "HIGH DEMAND\n+ MORE UNFORMED DEMAND", transform=supply.transAxes, ha="right", va="top",
                fontsize=8, color=AMBER, weight="bold")
    supply_candidates = cities[(cities["demand"] >= sx) & (cities["remaining_share"] >= sy)].assign(
        score=lambda frame: frame["demand"] * frame["remaining_share"]
    ).nlargest(4, "score")
    for offset, (_, row) in zip((-30, -10, 10, 30), supply_candidates.iterrows()):
        supply.annotate(row["city"], (row["demand"], row["remaining_share"] * 100), xytext=(-8, offset),
                        textcoords="offset points", ha="right", va="center", fontsize=7.5, color=NAVY,
                        arrowprops=dict(arrowstyle="-", color="#c7a58f", linewidth=0.7))

    fig.text(0.5, 0.075,
             "Bubble size = forecast remaining demand  ·  These are review signals, not automatic spend or shortage decisions.",
             ha="center", fontsize=9, color="#60758d")
    source(fig, "results_calibrated_named.csv + city_momentum.parquet")
    save(fig, "03_growth_supply_opportunities.png")


def lead_time_risk(results: pd.DataFrame, metrics: dict, cutoff: pd.Timestamp) -> None:
    rows = pd.DataFrame(metrics["champion"]["by_horizon"]).sort_values("key")
    target = results.copy()
    target["horizon"] = (pd.to_datetime(target["checkin"]) - cutoff).dt.days
    observed = target.groupby("horizon").agg(observed=("observed_so_far", "sum"), total=("predicted_demand", "sum"))
    observed["observed_share"] = observed["observed"] / observed["total"].clip(lower=1)
    rows["observed_share"] = rows["key"].map(observed["observed_share"])

    fig = canvas(
        "Lead-time risk: confidence changes every day",
        "Near-term forecasts are execution signals; far-term forecasts are prioritisation signals.",
    )
    ax = fig.add_axes([0.07, 0.17, 0.86, 0.62])
    style_axis(ax)
    x = rows["key"].to_numpy()
    ax.bar(x, rows["wape"] * 100, color=BLUE, alpha=0.84, width=0.72, label="Historical WAPE")
    ax.set_xlabel("Days to check-in", color="#60758d")
    ax.set_ylabel("Historical WAPE", color="#60758d")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xticks([1, 3, 7, 10, 14, 18, 21, 25, 30], ["D-1", "D-3", "D-7", "D-10", "D-14", "D-18", "D-21", "D-25", "D-30"])
    right = ax.twinx()
    right.plot(x, rows["observed_share"] * 100, color=ORANGE, marker="o", markersize=3.2, linewidth=2.2,
               label="Observed share on target grid")
    right.set_ylim(0, 105)
    right.yaxis.set_major_formatter(PercentFormatter(100))
    right.set_ylabel("Observed share", color=ORANGE)
    right.tick_params(axis="y", colors=ORANGE, labelsize=9, length=0)
    right.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for day in (1, 7, 14, 30):
        row = rows.loc[rows["key"] == day].iloc[0]
        ax.annotate(f"{row['wape']:.1%}", (day, row["wape"] * 100), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color=NAVY, weight="bold")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper left", frameon=False, ncol=2, fontsize=9)
    ax.axvspan(21.5, 30.5, color=AMBER, alpha=0.08)
    ax.text(29.8, ax.get_ylim()[1] * 0.88, "Higher-risk\nplanning zone", ha="right", color=AMBER, fontsize=9, weight="bold")
    source(fig, "backtest_metrics_phase2.json + results_calibrated_named.csv")
    save(fig, "04_lead_time_risk.png")


def demand_regimes(metrics: dict) -> None:
    frame = pd.DataFrame(metrics["champion"]["by_demand_bucket"])
    labels = ["Low\nbottom 50%", "Q50–75", "Q75–90", "High\nQ90–99", "Peak\ntop 1%"]
    x = np.arange(len(frame))

    fig = canvas(
        "Accuracy is not one number: Low vs High vs Peak demand",
        "Percentage error and absolute impact tell different stories across demand regimes.",
    )
    ax = fig.add_axes([0.07, 0.17, 0.86, 0.62])
    style_axis(ax)
    colors = [SLATE, "#7da9dc", "#568fce", BLUE, ORANGE]
    bars = ax.bar(x, frame["wape"] * 100, color=colors, alpha=0.92, width=0.65, label="WAPE")
    ax.set_xticks(x, labels)
    ax.set_ylabel("WAPE", color="#60758d")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    right = ax.twinx()
    right.plot(x, frame["normalised_bias"] * 100, color=RED, linewidth=2.4, marker="o", label="Normalised bias")
    right.axhline(0, color=GRID, linewidth=1)
    right.set_ylim(min(-12, frame["normalised_bias"].min() * 130), 3)
    right.yaxis.set_major_formatter(PercentFormatter(100))
    right.set_ylabel("Normalised bias", color=RED)
    right.tick_params(axis="y", colors=RED, labelsize=9, length=0)
    right.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for bar, (_, row) in zip(bars, frame.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"WAPE {row['wape']:.1%}\nMAE {compact(row['mae'])}", ha="center", va="bottom",
                fontsize=8.5, color=NAVY, weight="bold")
    ax.text(0.01, 0.91, "Low demand: high % error, low absolute cost", transform=ax.transAxes,
            fontsize=9, color="#60758d")
    ax.text(0.99, 0.91, "Peak demand: lower % error, high absolute exposure", transform=ax.transAxes,
            fontsize=9, color="#60758d", ha="right")
    handles, labels_ = ax.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    ax.legend(handles + handles2, labels_ + labels2, loc="upper center", frameon=False, ncol=2, fontsize=9)
    source(fig, "backtest_metrics_phase2.json · five-fold out-of-fold predictions")
    save(fig, "05_demand_regime_accuracy.png")


def temporal_validation(metrics: dict, evidence: dict) -> None:
    champion = metrics["champion"]["folds"]
    baseline = metrics["baseline"]["folds"]
    cutoffs = list(champion)
    x = np.arange(len(cutoffs))
    model_values = [champion[key]["wape"] * 100 for key in cutoffs]
    baseline_values = [baseline[key]["wape"] * 100 for key in cutoffs]
    model_colors = [AMBER if key == "2025-05-21" else BLUE for key in cutoffs]

    fig = canvas(
        "Temporal validation: the difficult regime stays in the score",
        "Five rolling-origin simulations compare the submitted model with the same Pickup baseline.",
    )
    ax = fig.add_axes([0.07, 0.18, 0.86, 0.59])
    style_axis(ax)
    width = 0.34
    ax.bar(x - width / 2, baseline_values, width, color=SLATE, alpha=0.62, label="Pickup baseline")
    bars = ax.bar(x + width / 2, model_values, width, color=model_colors, label="Submitted model")
    ax.set_xticks(x, [pd.Timestamp(key).strftime("%b %Y") for key in cutoffs])
    ax.set_ylabel("WAPE", color="#60758d")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    for bar, value in zip(bars, model_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}%", ha="center", fontsize=8.5,
                color=NAVY, weight="bold")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    shock = evidence["shock"]
    ax.annotate(
        f"Shock-overlap fold\n{shock['error_share']:.1%} of total absolute error",
        xy=(1 + width / 2, model_values[1]), xytext=(1.65, max(baseline_values) * 0.92),
        arrowprops=dict(arrowstyle="->", color=AMBER, linewidth=1.5), color=AMBER, fontsize=10, weight="bold",
        bbox=dict(boxstyle="round,pad=.45", fc="#fff8e8", ec="#f0d39a"),
    )
    ax.text(0.99, 0.06,
            f"Official all-fold WAPE  {metrics['champion']['pooled']['wape']:.2%}\n"
            f"Without shock fold  {shock['without_fold_wape']:.2%}  · diagnostic only",
            transform=ax.transAxes, ha="right", fontsize=10, color=NAVY,
            bbox=dict(boxstyle="round,pad=.55", fc="white", ec=GRID))
    source(fig, "backtest_metrics_phase2.json + jury_evidence.json · shock dates from UN DPPA")
    save(fig, "06_temporal_validation_shock.png")


def clustering_decision(evidence: dict) -> None:
    levels = pd.DataFrame(evidence["clustering"]["levels"])
    selected = levels.loc[levels["relative_total_gain"].idxmax()]
    city = selected["unclustered_wape"] * 100
    control = selected["control_wape"] * 100
    clustered = selected["clustered_wape"] * 100
    mechanical = selected["mechanical_gain"] * 100
    modelling = selected["modelling_gain"] * 100
    total = selected["total_gain"] * 100

    fig = canvas(
        "Clustering decision: most of the apparent gain was mechanical",
        "Aggregating errors can improve WAPE before the clustered model learns anything new.",
    )
    ax = fig.add_axes([0.07, 0.22, 0.54, 0.53])
    style_axis(ax)
    labels = ["City-level", "Aggregation\ncontrol", "Clustered"]
    values = [city, control, clustered]
    bars = ax.bar(np.arange(3), values, color=[BLUE, SLATE, ORANGE], width=0.62, alpha=0.9)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("WAPE · lower is better", color="#60758d")
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_ylim(0, max(values) * 1.24)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.22, f"{value:.4f}%",
                ha="center", fontsize=10, color=NAVY, weight="bold")

    detail = fig.add_axes([0.66, 0.22, 0.28, 0.53])
    detail.set_facecolor("white")
    detail.set_xlim(0, total * 1.05)
    detail.set_ylim(0, 1)
    detail.axis("off")
    detail.text(0.06, 0.88, "APPARENT WAPE GAIN", transform=detail.transAxes,
                color="#60758d", fontsize=9, weight="bold")
    detail.text(0.06, 0.76, f"{total:.4f} pp", transform=detail.transAxes,
                color=NAVY, fontsize=22, weight="bold")
    detail.barh([0.53], [mechanical], height=0.13, color=SLATE, label="Mechanical aggregation")
    detail.barh([0.53], [modelling], left=[mechanical], height=0.13, color=ORANGE, label="Actual modelling")
    detail.text(mechanical / 2, 0.53, f"{mechanical / total:.1%}", ha="center", va="center",
                fontsize=10, color="white", weight="bold")
    detail.text(mechanical + modelling / 2, 0.53, f"{modelling / total:.1%}", ha="center", va="center",
                fontsize=9, color="white", weight="bold")
    detail.text(0.06, 0.30,
                f"Relative modelling gain  {selected['relative_modelling_gain']:.2%}\n"
                "Acceptance gate             2.00%\n"
                "Decision                    KEEP CITY-LEVEL",
                transform=detail.transAxes, fontsize=10, color=NAVY, linespacing=1.75,
                bbox=dict(boxstyle="round,pad=.65", fc="#f7f9fc", ec=GRID))
    detail.legend(loc="lower left", bbox_to_anchor=(0.0, -0.14), frameon=False, fontsize=8)
    source(fig, "jury_evidence.json · three-fold uncalibrated cluster sweep; compare within this sweep only")
    save(fig, "07_clustering_gain_decomposition.png")


def main() -> None:
    results = pd.read_csv(ARTIFACTS / "results_calibrated_named.csv")
    momentum = pd.read_parquet(ARTIFACTS / "city_momentum.parquet")
    metrics = load_json(ARTIFACTS / "backtest_metrics_phase2.json")
    evidence = load_json(ARTIFACTS / "jury_evidence.json")
    summary = load_json(ARTIFACTS / "run_summary.json")
    cutoff = pd.Timestamp(summary["cutoff"])

    demand_formation(results, cutoff)
    city_priority(results)
    opportunity_matrices(results, momentum)
    lead_time_risk(results, metrics, cutoff)
    demand_regimes(metrics)
    temporal_validation(metrics, evidence)
    clustering_decision(evidence)
    print(f"Wrote 7 presentation charts to {OUTPUT}")


if __name__ == "__main__":
    main()
