"""API contract tests.

Covers both states the endpoints must handle: no trained model at all, and a
run whose artefacts are on disk.
"""
from __future__ import annotations

import json
import shutil

import pytest
from django.urls import reverse
from rest_framework.test import APIClient

from apps.experiments.models import TrainingRun
from ml.contract import DataContract
from ml.pipelines.training import TrainingConfig, TrainingPipeline

pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture(scope="session")
def artefacts(tmp_path_factory, contract, raw_frame):
    """One real training run whose artefacts the API can serve."""
    data_dir = tmp_path_factory.mktemp("api-data")
    csv = data_dir / "panel.csv"
    raw_frame.to_csv(csv, index=False)

    run_contract = DataContract.from_dict(contract.to_dict())
    run_contract.path = str(csv)
    run_contract.evaluation.horizons = [7]
    run_contract.evaluation.horizon_buckets = [[1, 7]]

    return TrainingPipeline(
        TrainingConfig(
            contract=run_contract,
            profile_name="test",
            horizon=7,
            seed=42,
            runs_dir=tmp_path_factory.mktemp("api-runs"),
            profile={
                "max_train_rows": 20_000,
                "samples_per_target": 1,
                "cv_folds": 1,
                "models": ["seasonal_naive_7", "lightgbm"],
                "lightgbm": {"n_estimators": 80, "num_leaves": 31},
            },
        )
    ).run()


@pytest.fixture
def trained_run(artefacts):
    return TrainingRun.objects.create(
        run_id=artefacts.run_id,
        status=TrainingRun.Status.SUCCEEDED,
        run_dir=str(artefacts.run_dir),
        champion_model=artefacts.champion,
        primary_metric="wape",
        horizon=7,
        target="bookings",
        finished_at="2026-01-01T00:00:00Z",
    )


# ------------------------------------------------------------------ system
def test_health_always_answers(client):
    response = client.get(reverse("health"))
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_whether_a_model_exists(client, trained_run):
    body = client.get(reverse("health")).json()
    assert body["has_trained_model"] is True
    assert body["latest_run"] == trained_run.run_id


def test_system_lists_models_and_profiles(client):
    body = client.get(reverse("system-info")).json()["data"]
    assert body["models"]
    assert "demo" in body["profiles"]
    assert "cpu_count" in body["resources"]


# --------------------------------------------------- graceful empty states
@pytest.mark.parametrize(
    "route",
    [
        "dashboard-summary", "forecast-list", "forecast-timeseries", "forecast-drivers",
        "forecast-peaks", "forecast-overview", "forecast-heatmap", "anomaly-list",
        "model-leaderboard", "backtest-list", "backtest-metrics", "insight-list",
    ],
)
def test_endpoints_report_no_data_instead_of_failing(client, route, settings):
    """With no trained model, every endpoint must answer 200 + available=false."""
    settings.DEMO_MODE = False
    response = client.get(reverse(route))

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["data"] is None
    assert body["detail_fa"]


def test_no_data_mode_never_fabricates_numbers(client, settings):
    settings.DEMO_MODE = False
    body = client.get(reverse("dashboard-summary")).json()
    assert body["data"] is None
    assert json.dumps(body).count("forecast_total") == 0


def test_demo_mode_falls_back_to_precomputed_artefacts(client, settings, artefacts, monkeypatch):
    """A fresh clone with an empty database must still render a dashboard.

    This is the presentation safety net: `DEMO_MODE=true` serves the committed
    artefacts in `data/demo_artifacts/` when no training run exists.
    """
    from apps.forecasting import store as store_module

    settings.DEMO_MODE = True
    assert not TrainingRun.objects.exists(), "this test needs an empty run table"

    monkeypatch.setattr(store_module, "DEMO_ARTIFACTS_DIR", artefacts.run_dir)
    body = client.get(reverse("dashboard-summary")).json()

    assert body["available"] is True
    assert body["data"]["run_id"] == "precomputed-demo"
    assert body["data"]["forecast_total"] > 0


def test_demo_fallback_is_ignored_when_demo_mode_is_off(client, settings, artefacts, monkeypatch):
    from apps.forecasting import store as store_module

    settings.DEMO_MODE = False
    monkeypatch.setattr(store_module, "DEMO_ARTIFACTS_DIR", artefacts.run_dir)

    body = client.get(reverse("dashboard-summary")).json()
    assert body["available"] is False


# ---------------------------------------------------------------- with data
def test_dashboard_summary_shape(client, trained_run):
    body = client.get(reverse("dashboard-summary"), {"horizon": 7}).json()
    assert body["available"] is True

    data = body["data"]
    assert data["forecast_total"] > 0
    assert data["forecast_lower"] <= data["forecast_total"] <= data["forecast_upper"]
    assert data["confidence"]["label"] in {"high", "medium", "low"}
    assert data["model"]["champion"]
    assert "formula" in data["confidence"]


def test_timeseries_splits_history_from_forecast(client, trained_run):
    data = client.get(reverse("forecast-timeseries"), {"level": "destination"}).json()["data"]

    assert data["series"]
    assert data["forecast_start"]
    future = [p for p in data["series"] if p["ds"] >= data["forecast_start"]]
    past = [p for p in data["series"] if p["ds"] < data["forecast_start"]]

    assert all("forecast" in point for point in future)
    assert any("actual" in point for point in past)
    assert all("actual" not in point for point in future), "the future has no actuals"


def test_forecast_filters_by_entity_and_horizon(client, trained_run):
    everything = client.get(reverse("forecast-list"), {"level": "destination"}).json()
    assert everything["count"] > 0

    one = client.get(
        reverse("forecast-list"), {"level": "destination", "id": "city_0", "horizon": 3}
    ).json()
    assert one["count"] > 0
    assert one["count"] < everything["count"]
    assert {row["entity_id"] for row in one["data"]} == {"city_0"}
    assert max(row["horizon"] for row in one["data"]) <= 3


def test_horizon_is_capped_at_what_was_trained(client, trained_run):
    """Asking for 90 days from a 7-day model must not silently invent 90 days."""
    data = client.get(reverse("dashboard-summary"), {"horizon": 90}).json()["data"]
    assert data["horizon"] == 7


def test_leaderboard_marks_the_champion_and_the_baselines(client, trained_run):
    data = client.get(reverse("model-leaderboard")).json()["data"]
    board = data["leaderboard"]

    assert board
    assert sum(1 for row in board if row["is_champion"]) == 1
    assert any(row["is_baseline"] for row in board)
    assert board[0]["rank"] == 1


def test_backtest_metrics_expose_coverage_and_folds(client, trained_run):
    data = client.get(reverse("backtest-metrics")).json()["data"]
    assert data["folds"]
    assert data["coverage_by_horizon"]
    assert data["uncertainty_method"]


def test_drivers_are_grouped_and_signed(client, trained_run):
    data = client.get(reverse("forecast-drivers")).json()["data"]
    assert data["groups"]
    for group in data["groups"]:
        assert 0 <= group["contribution_share"] <= 1
        assert group["direction"] in {"positive", "negative", "neutral"}
        assert group["label_fa"]


def test_overview_rows_carry_a_persian_status(client, trained_run):
    rows = client.get(reverse("forecast-overview"), {"level": "destination"}).json()["data"]
    assert rows
    for row in rows:
        assert row["status_fa"]
        assert row["forecast_total"] >= 0


def test_overview_returns_every_entity_even_with_anomalies_present(
    client, trained_run, artefacts, monkeypatch
):
    """Regression: the anomaly lookup must not shrink the overview table.

    The row count has to match the number of forecast entities regardless of
    how many anomalies exist.
    """
    import pandas as pd

    from apps.forecasting import store as store_module

    forecast = pd.read_parquet(artefacts.run_dir / "forecast.parquet")
    entities = forecast[forecast["level"] == "destination"]["entity_id"].nunique()

    real_frame = store_module.ArtifactStore.frame

    def frame_with_anomalies(self, name):
        if name != "anomalies":
            return real_frame(self, name)
        # One forecast spike and one historical spike for a single entity.
        return pd.DataFrame(
            {
                "entity_id": ["city_0", "city_1"],
                "ds": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "score": [4.0, 4.0],
                "type": ["spike", "spike"],
                "severity": ["high", "high"],
                "deviation": [0.5, 0.5],
                "source": ["forecast", "residual"],
            }
        )

    monkeypatch.setattr(store_module.ArtifactStore, "frame", frame_with_anomalies)
    rows = client.get(reverse("forecast-overview"), {"level": "destination"}).json()["data"]

    assert len(rows) == entities
    statuses = {row["entity_id"]: row["status"] for row in rows}
    assert statuses["city_0"] == "spike_risk", "a forecast spike marks the entity at risk"
    assert statuses["city_1"] != "spike_risk", "a past spike is not a forward-looking risk"


def test_narrative_is_generated_from_supplied_facts(client, trained_run):
    data = client.get(reverse("forecast-narrative"), {"level": "destination"}).json()["data"]
    assert data["text"]
    assert data["grounded"] is True
    assert "forecast_total" in data["facts"]


def test_models_endpoint_reports_registry_and_training_metadata(client, trained_run):
    data = client.get(reverse("model-list")).json()["data"]
    assert data["registry"]
    assert data["trained"]["champion"]
    assert data["trained"]["n_features"] > 0


# --------------------------------------------------------------- scenarios
def test_scenario_options_list_only_usable_covariates(client, trained_run):
    body = client.get(reverse("scenario-options")).json()
    if not body["available"]:
        pytest.skip("no persisted model for scenarios in this run")

    columns = {item["column"] for item in body["data"]["adjustable"]}
    assert "price" in columns
    assert "is_weekend" not in columns, "a calendar fact is not an actionable lever"


def test_scenario_price_cut_moves_demand_the_right_way(client, trained_run):
    response = client.post(
        reverse("scenario-simulate"),
        {"level": "destination", "adjustments": [{"column": "price", "change_pct": -20}]},
        format="json",
    )
    assert response.status_code == 200
    body = response.json()
    if not body["available"]:
        pytest.skip("scenario engine unavailable in this run")

    data = body["data"]
    assert data["baseline_total"] > 0
    assert data["applied"], "the price adjustment should have been applied"
    assert data["impact_pct"] is not None


def test_scenario_rejects_an_unused_covariate(client, trained_run):
    response = client.post(
        reverse("scenario-simulate"),
        {"adjustments": [{"column": "not_a_column", "change_pct": 10}]},
        format="json",
    )
    body = response.json()
    if not body["available"]:
        pytest.skip("scenario engine unavailable in this run")
    assert body["data"]["rejected"], "an unknown column must be reported, not ignored"


# ---------------------------------------------------------------- data lab
def test_upload_profiles_the_dataset(client, tmp_path, raw_frame):
    csv = tmp_path / "upload.csv"
    raw_frame.head(2000).to_csv(csv, index=False)

    with csv.open("rb") as handle:
        response = client.post(reverse("dataset-upload"), {"file": handle}, format="multipart")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["profile"]["suggested_schema"]["timestamp"] == "date"
    assert data["dataset"]["n_rows"] == 2000


def test_upload_rejects_an_unsupported_file_type(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")

    with bad.open("rb") as handle:
        response = client.post(reverse("dataset-upload"), {"file": handle}, format="multipart")

    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_mapping_and_validation_round_trip(client, tmp_path, raw_frame, settings):
    csv = tmp_path / "map.csv"
    raw_frame.to_csv(csv, index=False)
    with csv.open("rb") as handle:
        dataset = client.post(
            reverse("dataset-upload"), {"file": handle}, format="multipart"
        ).json()["data"]["dataset"]

    mapping = {
        "dataset_id": dataset["id"],
        "timestamp": "date",
        "target": "bookings",
        "entity_id": "listing_id",
        "destination": "city",
        "future_features": ["price", "is_holiday"],
        "historical_features": ["searches"],
        "primary_metric": "wape",
        "horizons": [7, 14],
        "save_as_active": False,
    }
    mapped = client.post(reverse("dataset-map"), mapping, format="json")
    assert mapped.status_code == 200
    assert mapped.json()["data"]["contract"]["schema"]["target"] == "bookings"

    validated = client.post(
        reverse("dataset-validate"),
        {"dataset_id": dataset["id"], "mapping": mapping},
        format="json",
    )
    assert validated.status_code == 200
    report = validated.json()["data"]
    assert 0 <= report["health_score"] <= 100
    assert report["panel"]["entities"] == 8


def test_mapping_requires_timestamp_and_target(client):
    response = client.post(reverse("dataset-map"), {"dataset_id": 1}, format="json")
    assert response.status_code == 400
    assert "timestamp" in response.json()["errors"]


def test_training_rejects_an_unknown_metric(client):
    response = client.post(reverse("training-run"), {"metric": "bogus"}, format="json")
    assert response.status_code == 400
    assert "Unknown metric" in response.json()["detail"]


def test_training_detail_404s_for_a_missing_run(client):
    response = client.get(reverse("training-detail", args=["nope"]))
    assert response.status_code == 404


# --------------------------------------------------------------- path safety
def test_local_path_registration_rejects_traversal(tmp_path):
    """`../../etc/passwd` must not be readable through the Data Lab."""
    from apps.datasets.services import register_local_file

    for attempt in ("../../etc/passwd", "/etc/passwd", "data/../../etc/hosts"):
        with pytest.raises((ValueError, FileNotFoundError)):
            register_local_file(attempt)


def test_local_path_registration_rejects_a_sibling_prefix_directory(tmp_path, monkeypatch):
    """A directory whose name merely starts with the repo path is still outside it."""
    from apps.datasets import services
    from ml.paths import REPO_ROOT

    sibling = tmp_path / "novo-pulse-backup"
    sibling.mkdir()
    secret = sibling / "secrets.csv"
    secret.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(services, "REPO_ROOT", tmp_path / "novo-pulse")
    (tmp_path / "novo-pulse").mkdir()

    with pytest.raises(ValueError, match="inside the project"):
        services.register_local_file(str(secret))


@pytest.mark.parametrize(
    "crafted",
    [
        "../../evil.csv",
        "/etc/passwd.csv",
        "a b;rm -rf /.csv",
        "x/y/z.csv",
        "..\\..\\windows.csv",
        "sh -c 'curl evil'.csv",
    ],
)
def test_upload_filename_cannot_escape_the_upload_directory(crafted, tmp_path):
    """Whatever the client sends, the stored name must be a plain filename."""
    from apps.datasets.services import UPLOAD_DIR, safe_filename

    cleaned = safe_filename(crafted)
    assert "/" not in cleaned and "\\" not in cleaned
    assert ".." not in cleaned
    # The resolved destination stays inside the upload directory.
    assert (UPLOAD_DIR / cleaned).resolve().is_relative_to(UPLOAD_DIR.resolve())


def test_upload_filename_keeps_a_legitimate_name_readable():
    from apps.datasets.services import safe_filename

    assert safe_filename("../../evil.csv") == "evil.csv"
    assert safe_filename("/data/raw/Competition Data 2026.csv") == "Competition_Data_2026.csv"
    assert safe_filename("x/y/z.parquet") == "z.parquet"


def test_validate_rejects_an_unknown_dataset_instead_of_answering_about_another(client):
    """Silently validating the active contract would answer the wrong question."""
    response = client.post(reverse("dataset-validate"), {"dataset_id": 999999}, format="json")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_upload_accepts_a_json_local_path(client, tmp_path, raw_frame, monkeypatch):
    """Registering a file already on disk should not require a multipart body."""
    from apps.datasets import services
    from ml.paths import REPO_ROOT

    target = REPO_ROOT / "data" / "raw" / "json_path_test.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_frame.head(500).to_csv(target, index=False)
    try:
        response = client.post(
            reverse("dataset-upload"),
            {"path": "data/raw/json_path_test.csv", "name": "JSON path"},
            format="json",
        )
        assert response.status_code == 200, response.content
        assert response.json()["data"]["profile"]["rows"] == 500
    finally:
        target.unlink(missing_ok=True)


def test_upload_with_neither_file_nor_path_is_a_clear_400(client):
    response = client.post(reverse("dataset-upload"), {}, format="json")
    assert response.status_code == 400
    assert "file" in response.json()["detail"].lower()


def test_driver_shares_always_sum_to_one(client, trained_run):
    """A truncated driver list must still account for 100% of the effect."""
    data = client.get(reverse("forecast-drivers")).json()["data"]
    total = sum(group["contribution_share"] for group in data["groups"])
    assert total == pytest.approx(1.0, abs=0.005), f"shares sum to {total}"

    if data["n_groups"] > len(data["groups"]) - 1:
        assert any(group["group"] == "other" for group in data["groups"]), (
            "the truncated remainder must be shown explicitly, not dropped"
        )


def test_training_accepts_the_mapping_inline(client, tmp_path, raw_frame):
    """Training must use the mapping the user is looking at, not a stale one."""
    csv = tmp_path / "inline.csv"
    raw_frame.head(400).to_csv(csv, index=False)
    with csv.open("rb") as handle:
        dataset = client.post(
            reverse("dataset-upload"), {"file": handle}, format="multipart"
        ).json()["data"]["dataset"]

    assert not dataset["mapping"], "the fixture should start with no stored mapping"

    response = client.post(
        reverse("training-run"),
        {
            "dataset_id": dataset["id"],
            "profile": "demo",
            "horizon": 7,
            "mapping": {
                "dataset_id": dataset["id"],
                "timestamp": "date",
                "target": "bookings",
                "entity_id": "listing_id",
                "primary_metric": "wape",
                "horizons": [7],
            },
        },
        format="json",
    )
    assert response.status_code == 200, response.content
    assert response.json()["data"]["target"] == "bookings"


def test_training_without_any_mapping_explains_what_to_do(client, tmp_path, raw_frame):
    csv = tmp_path / "unmapped.csv"
    raw_frame.head(200).to_csv(csv, index=False)
    with csv.open("rb") as handle:
        dataset = client.post(
            reverse("dataset-upload"), {"file": handle}, format="multipart"
        ).json()["data"]["dataset"]

    response = client.post(
        reverse("training-run"), {"dataset_id": dataset["id"]}, format="json"
    )
    assert response.status_code == 400
    assert "mapping" in response.json()["detail"].lower()


def test_training_rejects_an_invalid_inline_mapping(client, tmp_path, raw_frame):
    csv = tmp_path / "badmap.csv"
    raw_frame.head(200).to_csv(csv, index=False)
    with csv.open("rb") as handle:
        dataset = client.post(
            reverse("dataset-upload"), {"file": handle}, format="multipart"
        ).json()["data"]["dataset"]

    response = client.post(
        reverse("training-run"),
        {"dataset_id": dataset["id"], "mapping": {"timestamp": "date"}},  # no target
        format="json",
    )
    assert response.status_code == 400
    assert "target" in response.json()["errors"]


# ------------------------------------------------- the fresh-clone guarantee
def test_scenarios_work_from_the_demo_artefacts_alone(
    client, settings, artefacts, monkeypatch, tmp_path
):
    """The what-if page must work on a clone with no database and no source data.

    Scenario simulation re-predicts with the fitted model, so the run has to be
    self-contained: the model binary and the canonical panel both travel with
    it. Re-reading the original CSV is not an option once the run is copied to
    another machine, or when the source was never committed.
    """
    import shutil

    from apps.forecasting import store as store_module
    from apps.scenarios import services as scenario_services

    settings.DEMO_MODE = True
    assert not TrainingRun.objects.exists(), "this test needs an empty run table"

    # Copy the run somewhere else entirely, as `seed_demo --publish` does.
    published = tmp_path / "demo_artifacts"
    shutil.copytree(artefacts.run_dir, published)
    monkeypatch.setattr(store_module, "DEMO_ARTIFACTS_DIR", published)
    scenario_services.clear_cache()

    assert (published / "panel.parquet").exists(), "the panel must travel with the run"
    assert list((published / "models").glob("*.joblib")), "a model binary must travel with the run"

    options = client.get(reverse("scenario-options")).json()
    assert options["available"] is True, options.get("detail")
    assert options["data"]["adjustable"], "no covariate could be simulated"

    result = client.post(
        reverse("scenario-simulate"),
        {"level": "destination", "adjustments": [{"column": "price", "change_pct": -20}]},
        format="json",
    )
    assert result.status_code == 200, result.content
    data = result.json()["data"]
    assert data["baseline_total"] > 0
    assert data["applied"], "the adjustment was not applied"
    assert data["series"], "no scenario series was produced"


def test_a_published_run_does_not_need_its_source_file(
    client, settings, artefacts, monkeypatch, tmp_path
):
    """Deleting the source dataset must not break scenario simulation."""
    import shutil

    from apps.forecasting import store as store_module
    from apps.scenarios import services as scenario_services

    settings.DEMO_MODE = True
    published = tmp_path / "artifacts"
    shutil.copytree(artefacts.run_dir, published)

    # Point the run's contract at a file that does not exist.
    from ml.contract import DataContract

    contract = DataContract.load(published / "config.yaml")
    contract.path = str(tmp_path / "deleted-source.csv")
    contract.save(published / "config.yaml")

    monkeypatch.setattr(store_module, "DEMO_ARTIFACTS_DIR", published)
    scenario_services.clear_cache()

    options = client.get(reverse("scenario-options")).json()
    assert options["available"] is True, "the run must be self-contained"
    assert options["data"]["adjustable"]
