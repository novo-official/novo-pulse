"""The Pol 4 dashboard API.

Two things these assert that matter more than the shapes: every number the API
serves matches the artefact it claims to come from, and nothing is fabricated
when an artefact is absent.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
from django.urls import reverse

from apps.pol4 import services
from ml.pol4.config import Pol4Config

CONFIG = Pol4Config()
pytestmark = pytest.mark.skipif(
    not (CONFIG.artifacts_dir / "results.csv").exists(),
    reason="Pol 4 artefacts not generated in this checkout",
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    services.clear_cache()
    yield
    services.clear_cache()


@pytest.fixture
def results() -> pd.DataFrame:
    return pd.read_csv(CONFIG.artifacts_dir / "results.csv")


# --------------------------------------------------------------- overview
def test_overview_totals_match_results_csv(client, results):
    body = client.get(reverse("pol4-overview")).json()
    assert body["available"] is True
    kpis = body["data"]["kpis"]
    assert kpis["total_predicted_demand"] == int(results["predicted_demand"].sum())
    assert kpis["cities"] == results["cluster_code"].nunique() == 321
    assert kpis["dates"] == results["checkin"].nunique() == 30


def test_overview_series_covers_every_target_date(client, results):
    series = client.get(reverse("pol4-overview")).json()["data"]["series"]
    assert len(series) == 30
    assert sum(row["predicted_demand"] for row in series) == int(
        results["predicted_demand"].sum()
    )
    # Observed and remaining must reconstruct the total, not merely accompany it.
    for row in series:
        assert row["observed_so_far"] + row["predicted_remaining"] == pytest.approx(
            row["predicted_demand"], abs=1
        )


def test_overview_wape_matches_the_backtest_artefact(client):
    kpis = client.get(reverse("pol4-overview")).json()["data"]["kpis"]
    payload = json.loads(
        (CONFIG.artifacts_dir / "backtest_metrics_phase2.json").read_text()
    )
    assert kpis["backtest_wape"] == payload["champion"]["pooled"]["wape"]
    assert kpis["baseline_wape"] == payload["baseline"]["pooled"]["wape"]


def test_overview_provinces_sum_to_the_national_total(client, results):
    provinces = client.get(reverse("pol4-overview")).json()["data"]["provinces"]
    assert len(provinces) == 7
    total = sum(row["predicted_demand"] for row in provinces)
    assert total == pytest.approx(float(results["predicted_demand"].sum()), rel=1e-6)


def test_overview_reports_a_real_peak_date(client, results):
    kpis = client.get(reverse("pol4-overview")).json()["data"]["kpis"]
    by_date = results.groupby("checkin")["predicted_demand"].sum()
    assert kpis["peak_date"] == by_date.idxmax()
    assert kpis["peak_demand"] == int(by_date.max())


def test_dashboard_switches_the_forecast_and_the_validation_metrics(client):
    raw = client.get(reverse("pol4-dashboard"), {"model": "raw"}).json()["data"]
    calibrated = client.get(
        reverse("pol4-dashboard"), {"model": "calibrated"}
    ).json()["data"]

    assert raw["selected_model"] == "raw"
    assert calibrated["selected_model"] == "calibrated"
    assert raw["summary"]["forecast_total"] != calibrated["summary"]["forecast_total"]
    assert raw["summary"]["wape"] != calibrated["summary"]["wape"]
    assert len(raw["national_series"]) == len(calibrated["national_series"]) == 30
    assert len(raw["heatmap"]["rows"]) == 321
    assert len(calibrated["lead_time_daily"]) == 30
    assert [row["key"] for row in calibrated["lead_time_daily"]] == list(range(1, 31))
    assert len(calibrated["demand_buckets"]) == 5
    assert len(calibrated["high_demand"]) == 3
    assert set(calibrated["evaluation_dimensions"]) == {
        "province", "weekday", "observation"
    }


def test_dashboard_rejects_an_unknown_model(client):
    response = client.get(reverse("pol4-dashboard"), {"model": "invented"})
    assert response.status_code == 404


# ------------------------------------------------------------------ cities
def test_every_city_is_listed_with_a_name(client):
    cities = client.get(reverse("pol4-cities")).json()["data"]["cities"]
    assert len(cities) == 321
    assert all(row["city"] and not str(row["city"]).isdigit() for row in cities)
    assert all(isinstance(row["city_code"], int) for row in cities)


def test_city_detail_resolves_by_code_and_by_name(client):
    by_code = client.get(reverse("pol4-city-detail", args=["1183"])).json()["data"]
    by_name = client.get(reverse("pol4-city-detail", args=["tehran"])).json()["data"]
    assert by_code["city"] == by_name["city"]
    assert by_code["city"]["city"] == "tehran"
    assert by_code["city"]["city_code"] == 1183


def test_city_detail_totals_match_the_forecast_rows(client, results):
    body = client.get(reverse("pol4-city-detail", args=["1183"])).json()["data"]
    rows = results[results["cluster_code"] == 1183]
    assert body["totals"]["predicted_demand"] == int(rows["predicted_demand"].sum())
    assert len(body["series"]) == 30


def test_an_unknown_city_is_a_404_not_a_guess(client):
    response = client.get(reverse("pol4-city-detail", args=["atlantis"]))
    assert response.status_code == 404
    assert "Unknown city" in response.json()["detail"]


# ------------------------------------------------------------------ pickup
def test_pickup_curve_carries_both_series(client):
    body = client.get(reverse("pol4-city-pickup", args=["1183"])).json()["data"]
    assert body["observed"], "a top city must have observed pickup"
    assert body["expected"], "the historical curve must always be present"
    assert body["predicted_demand"] >= body["observed_so_far"]
    assert len(body["available_checkins"]) == 30


def test_pickup_observed_is_cumulative_and_monotone(client):
    body = client.get(reverse("pol4-city-pickup", args=["1183"])).json()["data"]
    rows = sorted(body["observed"], key=lambda row: -row["days_to_checkin"])
    values = [row["observed_cumulative"] for row in rows]
    assert values == sorted(values), "cumulative demand cannot decrease"


def test_pickup_rejects_a_checkin_outside_the_window(client):
    response = client.get(
        reverse("pol4-city-pickup", args=["1183"]), {"checkin": "2024-01-01"}
    )
    assert response.status_code == 404


# --------------------------------------------------------------- stability
def test_stability_returns_the_full_snapshot_ladder(client):
    body = client.get(reverse("pol4-stability")).json()["data"]
    horizons = sorted(row["horizon"] for row in body["snapshots"])
    assert horizons == [1, 3, 7, 14, 21, 30]
    assert body["aggregate"]["stability_score"] is not None


def test_stability_accepts_a_city_and_checkin(client):
    default = client.get(reverse("pol4-stability")).json()["data"]
    target = default["available_checkins"][0]
    body = client.get(
        reverse("pol4-stability"),
        {"city_code": default["city"]["city_code"], "checkin": target},
    ).json()["data"]
    assert body["checkin"] == target


# ------------------------------------------------------------ performance
def test_model_performance_reports_every_fold(client):
    body = client.get(reverse("pol4-model-performance")).json()["data"]
    assert len(body["folds"]) == 5
    # The hardest fold must be surfaced, not quietly dropped.
    assert body["hardest_fold"] == max(
        body["folds"], key=lambda fold: fold["champion_wape"]
    )["cutoff"]
    assert body["champion"]["wape"] < body["baseline"]["wape"]
    assert body["improvement"] > 0


def test_model_performance_exposes_real_feature_importance(client):
    body = client.get(reverse("pol4-model-performance")).json()["data"]
    importance = body["feature_importance"]
    assert importance
    assert all(row["share"] >= 0 for row in importance)
    stored = json.loads((CONFIG.artifacts_dir / "feature_importance.json").read_text())
    assert importance[0]["feature"] == stored[0]["feature"]


# ---------------------------------------------------------------- reports
def test_report_csv_downloads_with_the_right_headers(client):
    response = client.get(reverse("pol4-report-download", args=["demand"]))
    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/csv")
    assert "pol4_demand.csv" in response["Content-Disposition"]
    frame = pd.read_csv(pd.io.common.StringIO(response.content.decode("utf-8")))
    assert len(frame) == 9630


def test_report_filters_narrow_rows_without_changing_numbers(client, results):
    response = client.get(reverse("pol4-report-download", args=["demand"]), {"city": "tehran"})
    frame = pd.read_csv(pd.io.common.StringIO(response.content.decode("utf-8")))
    assert len(frame) == 30
    assert set(frame["city"]) == {"tehran"}
    expected = results[results["cluster_code"] == 1183]["predicted_demand"].sum()
    assert frame["predicted_demand"].sum() == expected


def test_unknown_report_is_a_404(client):
    assert client.get(reverse("pol4-report-download", args=["nonsense"])).status_code == 404


def test_report_preview_matches_the_download(client):
    preview = client.get(reverse("pol4-report-preview", args=["provinces"])).json()["data"]
    response = client.get(reverse("pol4-report-download", args=["provinces"]))
    frame = pd.read_csv(pd.io.common.StringIO(response.content.decode("utf-8")))
    assert preview["rows"] == len(frame)
    assert preview["columns"] == list(frame.columns)


# ------------------------------------------------------------- honesty
def test_a_missing_artefact_reports_unavailable_rather_than_inventing_one(client, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "artifacts_dir", lambda: tmp_path)
    services.clear_cache()
    body = client.get(reverse("pol4-overview")).json()
    assert body["available"] is False
    assert body["data"] is None
    assert "make pol4" in body["detail_fa"]


def test_there_is_no_synthetic_or_demo_endpoint():
    from config import api_urls

    names = [pattern.name for pattern in api_urls.urlpatterns]
    assert not any("demo" in (name or "") or "synthetic" in (name or "") for name in names)
    assert not any("scenario" in (name or "") for name in names)
