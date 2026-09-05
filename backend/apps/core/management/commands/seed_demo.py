"""`python manage.py seed_demo`

One command that takes a fresh clone to a fully populated dashboard:
generate synthetic data -> register it -> train -> persist demo artefacts.
"""
from __future__ import annotations

import datetime as dt
import shutil

from django.core.management.base import BaseCommand

from apps.datasets.models import Dataset
from apps.datasets.services import unique_slug
from apps.experiments import services as training_services
from apps.experiments.models import TrainingRun
from ml.contract import DataContract
from ml.data.holidays import build_calendar
from ml.data.synthetic import SyntheticConfig, generate_and_save
from ml.paths import DEMO_ARTIFACTS_DIR, REPO_ROOT, SYNTHETIC_DATA_DIR, ensure_dirs


class Command(BaseCommand):
    help = "Generate synthetic data, train a demo model and populate the dashboard."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=900, help="Days of history to generate.")
        parser.add_argument("--accommodations", type=int, default=140)
        parser.add_argument("--profile", default="demo", help="demo | competition | full")
        parser.add_argument("--horizon", type=int, default=90)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--skip-data", action="store_true", help="Reuse the existing synthetic dataset."
        )
        parser.add_argument(
            "--skip-training", action="store_true", help="Only (re)generate the dataset."
        )
        parser.add_argument(
            "--publish", action="store_true",
            help="Copy the resulting artefacts to data/demo_artifacts/ as the offline fallback.",
        )

    def handle(self, *args, **options):
        ensure_dirs()

        if not options["skip_data"]:
            self.stdout.write("Generating synthetic dataset...")
            written = generate_and_save(
                SyntheticConfig(
                    days=options["days"],
                    n_accommodations=options["accommodations"],
                    seed=options["seed"],
                )
            )
            for name, path in written.items():
                self.stdout.write(f"  {name:<16} {path.relative_to(REPO_ROOT)}")
            self._write_future_calendar(options["horizon"])
        else:
            self.stdout.write("Reusing the existing synthetic dataset.")

        contract = DataContract.load()
        dataset = self._register_dataset(contract)
        self.stdout.write(self.style.SUCCESS(f"Dataset registered: {dataset.name}"))

        if options["skip_training"]:
            return

        self.stdout.write(
            f"Training (profile={options['profile']}, horizon={options['horizon']})..."
        )
        run = training_services.start_training(
            contract=contract,
            profile_name=options["profile"],
            horizon=options["horizon"],
            seed=options["seed"],
            future_covariates=str(
                (SYNTHETIC_DATA_DIR / "future_covariates.csv").relative_to(REPO_ROOT)
            ),
            experiment_name="Demo - synthetic market",
            blocking=True,
        )
        run.refresh_from_db()

        if run.status != TrainingRun.Status.SUCCEEDED:
            self.stderr.write(self.style.ERROR(f"Training failed: {run.error[:600]}"))
            return

        self.stdout.write(
            self.style.SUCCESS(
                f"Run {run.run_id} | champion={run.champion_model} "
                f"| {run.primary_metric}={run.champion_score:.4f} "
                f"| improvement over baseline={_pct(run.improvement)}"
            )
        )
        for result in run.results.all()[:8]:
            self.stdout.write(
                f"   {result.rank:>2}. {result.model_name:<20} "
                f"{result.primary_metric}={_num(result.primary_value)}"
            )
        for warning in run.warnings:
            self.stdout.write(self.style.WARNING(f"   ! {warning}"))

        if options["publish"]:
            self._publish(run)

    # ------------------------------------------------------------------
    def _write_future_calendar(self, horizon: int) -> None:
        """Known-future covariates for the forecast window.

        Holidays and published festival dates are genuinely known months in
        advance, so supplying them is not leakage - it is exactly the
        information a revenue planner would have on the day. This file is the
        synthetic stand-in for the `--future-covariates` input the pipeline
        accepts on competition day.
        """
        demand_path = SYNTHETIC_DATA_DIR / "daily_demand.csv"
        accommodation_path = SYNTHETIC_DATA_DIR / "accommodations.csv"
        if not demand_path.exists():
            return
        import pandas as pd

        last = pd.read_csv(demand_path, usecols=["date"])["date"].max()
        start = pd.Timestamp(last).date() + dt.timedelta(days=1)
        end = start + dt.timedelta(days=max(horizon, 120))
        calendar = build_calendar(start, end)

        # Market-wide flags.
        market_columns = ["date", "is_holiday", "is_weekend", "holiday_name", "event_name", "season"]
        market_path = SYNTHETIC_DATA_DIR / "future_calendar.csv"
        calendar.loc[:, market_columns].to_csv(market_path, index=False)
        self.stdout.write(f"  future_calendar  {market_path.relative_to(REPO_ROOT)}")

        if not accommodation_path.exists():
            return
        # Events are destination-specific, so the per-entity file carries them.
        accommodations = pd.read_csv(
            accommodation_path, usecols=["accommodation_id", "destination_id"]
        )
        cross = accommodations.merge(
            calendar.loc[:, ["date", "is_holiday", "is_weekend", "event_destinations"]],
            how="cross",
        )
        cross["is_event"] = [
            int(dest in (spec.split("|") if isinstance(spec, str) and spec else []))
            for dest, spec in zip(cross["destination_id"], cross["event_destinations"])
        ]
        entity_path = SYNTHETIC_DATA_DIR / "future_covariates.csv"
        cross.loc[:, ["date", "accommodation_id", "is_holiday", "is_weekend", "is_event"]].rename(
            columns={"accommodation_id": "entity_id"}
        ).to_csv(entity_path, index=False)
        self.stdout.write(f"  future_covariates {entity_path.relative_to(REPO_ROOT)}")

    def _register_dataset(self, contract: DataContract) -> Dataset:
        path = REPO_ROOT / (contract.path or "")
        rows, columns = self._shape(path)
        existing = Dataset.objects.filter(path=contract.path).first()
        if existing:
            if existing.n_rows != rows or existing.n_columns != columns:
                existing.n_rows, existing.n_columns = rows, columns
                existing.save(update_fields=["n_rows", "n_columns", "updated_at"])
            return existing
        slugs = set(Dataset.objects.values_list("slug", flat=True))
        return Dataset.objects.create(
            name=contract.name,
            slug=unique_slug(contract.name, slugs),
            source=Dataset.Source.SYNTHETIC,
            path=contract.path or "",
            original_filename=path.name,
            size_bytes=path.stat().st_size if path.exists() else 0,
            n_rows=rows,
            n_columns=columns,
            mapping={
                "timestamp": contract.timestamp,
                "target": contract.target,
                "entity_id": contract.entity_id,
                "destination": contract.destination,
                "category": contract.category,
                "future_features": contract.future_features,
                "historical_features": contract.historical_features,
                "static_features": contract.static_features,
                "primary_metric": contract.evaluation.primary_metric,
                "horizons": contract.evaluation.horizons,
            },
        )

    def _shape(self, path) -> tuple[int, int]:
        """Row and column counts for the Data Lab listing."""
        if not path.exists():
            return 0, 0
        import pandas as pd

        header = pd.read_csv(path, nrows=0)
        with path.open("r", encoding="utf-8") as handle:
            rows = sum(1 for _ in handle) - 1
        return max(rows, 0), len(header.columns)

    def _publish(self, run) -> None:
        """Freeze this run as the offline presentation fallback."""
        source = run.directory
        if not source.exists():
            self.stderr.write(self.style.ERROR("Run directory is missing; nothing to publish."))
            return
        if DEMO_ARTIFACTS_DIR.exists():
            shutil.rmtree(DEMO_ARTIFACTS_DIR)
        # Model binaries are included: without them the scenario simulator
        # cannot re-predict, and "change the price, see the forecast move" is
        # the step of the demo that lands hardest.
        shutil.copytree(source, DEMO_ARTIFACTS_DIR)
        size_mb = sum(f.stat().st_size for f in DEMO_ARTIFACTS_DIR.rglob("*") if f.is_file()) / 1e6
        self.stdout.write(
            self.style.SUCCESS(
                f"Published precomputed demo artefacts to "
                f"{DEMO_ARTIFACTS_DIR.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)"
            )
        )


def _num(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def _pct(value) -> str:
    return "—" if value is None else f"{value:+.1%}"
