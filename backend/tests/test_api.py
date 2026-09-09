"""API contract tests.

Covers both states the endpoints must handle: no trained model at all, and a
run whose artefacts are on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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


# ------------------------------------------------- competition-day mapping
def _upload(client, path):
    with path.open("rb") as handle:
        response = client.post(
            reverse("dataset-upload"), {"file": handle}, format="multipart"
        )
    assert response.status_code == 200, response.json()
    return response.json()["data"]


def test_mapping_accepts_count_aggregation_without_a_target(client, tmp_path):
    """A raw booking table has no demand column: demand is the row count."""
    rows = []
    for day in pd.date_range("2024-01-01", periods=90):
        for listing in ("ACC-1", "ACC-2", "ACC-3"):
            for _ in range((day.dayofyear + len(listing)) % 4 + 1):
                rows.append({"booked_on": day.date().isoformat(), "unit_code": listing})
    csv = tmp_path / "bookings.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    dataset = _upload(client, csv)["dataset"]

    response = client.post(
        reverse("dataset-map"),
        {
            "dataset_id": dataset["id"],
            "timestamp": "booked_on",
            "entity_id": "unit_code",
            "aggregation": "count",
            "save_as_active": False,
        },
        format="json",
    )
    assert response.status_code == 200, response.json()
    schema = response.json()["data"]["contract"]["schema"]
    assert schema["aggregation"] == "count"
    assert schema["target"] is None


def test_mapping_without_a_target_is_rejected_unless_counting(client, tmp_path, raw_frame):
    csv = tmp_path / "needs_target.csv"
    raw_frame.to_csv(csv, index=False)
    dataset = _upload(client, csv)["dataset"]

    response = client.post(
        reverse("dataset-map"),
        {"dataset_id": dataset["id"], "timestamp": "date", "save_as_active": False},
        format="json",
    )
    assert response.status_code == 400
    assert "target" in response.json()["errors"]


def test_the_profiler_reports_a_jalali_calendar_through_the_api(client, tmp_path):
    jdatetime = pytest.importorskip("jdatetime")
    days = pd.date_range("2024-01-01", periods=60)
    frame = pd.DataFrame(
        {
            "تاریخ": [
                "{0.year:04d}/{0.month:02d}/{0.day:02d}".format(
                    jdatetime.date.fromgregorian(date=day.date())
                )
                for day in days
            ],
            "تعداد_رزرو": [(index % 7) + 1 for index in range(60)],
        }
    )
    csv = tmp_path / "jalali.csv"
    frame.to_csv(csv, index=False)

    profile = _upload(client, csv)["profile"]
    assert profile["calendar"]["calendar"] == "jalali"
    assert profile["suggested_schema"]["calendar"] == "jalali"


def test_a_side_table_is_joined_by_dataset_id(client, tmp_path, raw_frame):
    """The competition ships several files; the Data Lab stitches them together."""
    main = tmp_path / "main.csv"
    raw_frame.drop(columns=["capacity"]).to_csv(main, index=False)
    main_dataset = _upload(client, main)["dataset"]

    side = tmp_path / "listings.csv"
    pd.DataFrame(
        {
            "listing_id": sorted(raw_frame["listing_id"].unique()),
            "capacity": range(len(raw_frame["listing_id"].unique())),
        }
    ).to_csv(side, index=False)
    side_dataset = _upload(client, side)["dataset"]

    mapping = {
        "dataset_id": main_dataset["id"],
        "timestamp": "date",
        "target": "bookings",
        "entity_id": "listing_id",
        "static_features": ["capacity"],
        "joins": [{"dataset_id": side_dataset["id"], "on": ["listing_id"]}],
        "save_as_active": False,
    }
    mapped = client.post(reverse("dataset-map"), mapping, format="json")
    assert mapped.status_code == 200, mapped.json()
    joins = mapped.json()["data"]["contract"]["dataset"]["joins"]
    assert len(joins) == 1
    assert joins[0]["on"] == ["listing_id"]

    # The joined column must survive all the way into the validated panel.
    validated = client.post(
        reverse("dataset-validate"),
        {"dataset_id": main_dataset["id"], "mapping": mapping},
        format="json",
    )
    assert validated.status_code == 200, validated.json()
    notes = validated.json()["data"]["adapter_notes"]
    assert any("100% of rows matched" in note for note in notes)


def test_a_join_pointing_at_a_missing_dataset_is_a_clear_400(client, tmp_path, raw_frame):
    main = tmp_path / "main.csv"
    raw_frame.to_csv(main, index=False)
    dataset = _upload(client, main)["dataset"]

    response = client.post(
        reverse("dataset-map"),
        {
            "dataset_id": dataset["id"],
            "timestamp": "date",
            "target": "bookings",
            "joins": [{"dataset_id": 999_999, "on": ["listing_id"]}],
            "save_as_active": False,
        },
        format="json",
    )
    assert response.status_code == 400
    assert "999999" in response.json()["detail"]


def test_a_join_with_neither_id_nor_path_is_rejected(client, tmp_path, raw_frame):
    main = tmp_path / "main.csv"
    raw_frame.to_csv(main, index=False)
    dataset = _upload(client, main)["dataset"]

    response = client.post(
        reverse("dataset-map"),
        {
            "dataset_id": dataset["id"],
            "timestamp": "date",
            "target": "bookings",
            "joins": [{"on": ["listing_id"]}],
            "save_as_active": False,
        },
        format="json",
    )
    assert response.status_code == 400


def test_training_starts_for_a_count_contract_with_no_target(client, tmp_path):
    """Regression: `target` is non-null in the DB, so a count run used to 500.

    Demand is the row count, so there is no target column to store - and the
    field is display-only anyway.
    """
    rows = []
    for day in pd.date_range("2024-01-01", periods=150):
        for listing in ("ACC-1", "ACC-2"):
            for _ in range((day.dayofyear + len(listing)) % 3 + 1):
                rows.append({"booked_on": day.date().isoformat(), "unit_code": listing})
    csv = tmp_path / "bookings.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    dataset = _upload(client, csv)["dataset"]

    response = client.post(
        reverse("training-run"),
        {
            "dataset_id": dataset["id"],
            "mapping": {
                "dataset_id": dataset["id"],
                "timestamp": "booked_on",
                "entity_id": "unit_code",
                "aggregation": "count",
                "horizons": [7],
            },
            "horizon": 7,
        },
        format="json",
    )
    assert response.status_code == 200, response.json()
    assert response.json()["data"]["target"] == "count of rows"


def test_training_keeps_the_joins_it_was_given(client, tmp_path, raw_frame):
    """Regression: joins were resolved in the mapping view only.

    Training from an inline mapping therefore built a contract with its side
    tables silently dropped - a run that looks fine and quietly ignores two
    thirds of the competition data. Resolution now lives inside
    `contract_from_mapping`, so every caller gets it.
    """
    from apps.datasets.services import contract_from_mapping

    small = raw_frame.head(400)
    main = tmp_path / "main.csv"
    small.drop(columns=["capacity"]).to_csv(main, index=False)
    main_dataset = _upload(client, main)["dataset"]

    side = tmp_path / "listings.csv"
    pd.DataFrame(
        {
            "listing_id": sorted(small["listing_id"].unique()),
            "capacity": range(small["listing_id"].nunique()),
        }
    ).to_csv(side, index=False)
    side_dataset = _upload(client, side)["dataset"]

    mapping = {
        "dataset_id": main_dataset["id"],
        "timestamp": "date",
        "target": "bookings",
        "entity_id": "listing_id",
        "static_features": ["capacity"],
        "joins": [{"dataset_id": side_dataset["id"], "on": ["listing_id"]}],
        "horizons": [7],
    }

    contract = contract_from_mapping(mapping, main_dataset["path"], main_dataset["name"])
    assert len(contract.joins) == 1, "the side table must survive into the contract"
    assert contract.joins[0].keys == ["listing_id"]
    # Uploads are de-duplicated with a numeric suffix, so match the stem.
    assert "listings" in Path(contract.joins[0].path).stem


def test_driver_group_ids_are_unique(client, trained_run):
    """Regression: two groups could both be called "other".

    The explainer emits one for features matching no known family, and the
    truncation fold added a second. They collide as React keys, so the chart
    can drop or duplicate a row - and the same label appears twice.
    """
    data = client.get(reverse("forecast-drivers")).json()["data"]
    ids = [group["group"] for group in data["groups"]]
    assert len(ids) == len(set(ids)), f"duplicate driver group id in {ids}"
    labels = [group["label_fa"] for group in data["groups"]]
    assert len(labels) == len(set(labels)), f"duplicate driver label in {labels}"
