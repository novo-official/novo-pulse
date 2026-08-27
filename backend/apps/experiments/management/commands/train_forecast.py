"""`python manage.py train_forecast`

The competition-day entry point:

    python manage.py train_forecast \
        --dataset data/raw/competition.csv \
        --target booking_count \
        --horizon 90 \
        --profile competition \
        --metric wape
"""
from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from apps.experiments import services
from apps.experiments.models import TrainingRun
from ml.contract import DataContract, list_profiles
from ml.evaluation.metrics import available_metrics
from ml.paths import ACTIVE_CONTRACT_FILE, REPO_ROOT
from ml.resources import resource_summary


class Command(BaseCommand):
    help = "Train forecasting models on a dataset and publish the artefacts."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", help="Path to the dataset (overrides the contract).")
        parser.add_argument("--contract", help="Path to a data contract YAML file.")
        parser.add_argument("--target", help="Target column name.")
        parser.add_argument("--timestamp", help="Timestamp column name.")
        parser.add_argument("--entity", help="Entity id column name.")
        parser.add_argument("--horizon", type=int, help="Forecast horizon in periods.")
        parser.add_argument("--profile", default=None, help=f"One of: {', '.join(list_profiles())}")
        parser.add_argument("--metric", help=f"Primary metric. One of: {', '.join(available_metrics())}")
        parser.add_argument("--frequency", help="D | W | M | H (default: auto-detect).")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--future-covariates", help="CSV of known-future covariates.")
        parser.add_argument("--experiment", help="Group this run under a named experiment.")
        parser.add_argument(
            "--save-contract", action="store_true",
            help="Persist the resulting contract as the active one.",
        )

    def handle(self, *args, **options):
        self.stdout.write(resource_summary())

        contract = DataContract.load(options.get("contract"))
        overrides = {
            "path": options.get("dataset"),
            "target": options.get("target"),
            "timestamp": options.get("timestamp"),
            "entity_id": options.get("entity"),
            "frequency": options.get("frequency"),
            "primary_metric": options.get("metric"),
        }
        contract = contract.with_overrides(**overrides)

        if options.get("dataset") and not (REPO_ROOT / options["dataset"]).exists():
            raise CommandError(f"Dataset not found: {options['dataset']}")
        if options.get("metric") and options["metric"] not in available_metrics():
            raise CommandError(
                f"Unknown metric '{options['metric']}'. Available: {', '.join(available_metrics())}"
            )

        if options["save_contract"]:
            path = contract.save(ACTIVE_CONTRACT_FILE)
            self.stdout.write(f"Contract saved to {path.relative_to(REPO_ROOT)}")

        self.stdout.write(
            f"Training '{contract.name}' | target={contract.target} "
            f"| metric={contract.evaluation.primary_metric} "
            f"| horizon={options.get('horizon') or contract.evaluation.max_horizon}"
        )

        run = services.start_training(
            contract=contract,
            profile_name=options.get("profile"),
            horizon=options.get("horizon"),
            metric=options.get("metric"),
            seed=options.get("seed"),
            future_covariates=options.get("future_covariates"),
            experiment_name=options.get("experiment"),
            blocking=True,
        )
        run.refresh_from_db()

        if run.status != TrainingRun.Status.SUCCEEDED:
            raise CommandError(f"Training failed:\n{run.error}")

        self.stdout.write(self.style.SUCCESS(f"\nRun {run.run_id} complete."))
        self.stdout.write(f"  champion        : {run.champion_model}")
        self.stdout.write(f"  {run.primary_metric:<16}: {run.champion_score:.4f}")
        self.stdout.write(f"  best baseline   : {run.baseline_model} ({run.baseline_score:.4f})")
        if run.improvement is not None:
            self.stdout.write(f"  improvement     : {run.improvement:+.1%}")
        self.stdout.write(f"  data health     : {run.data_health_score}/100")
        self.stdout.write(f"  duration        : {run.training_seconds:.1f}s")
        self.stdout.write(f"  artefacts       : {run.run_dir}")
        self.stdout.write("  report          : reports/model_report.md")
        for warning in run.warnings:
            self.stdout.write(self.style.WARNING(f"  ! {warning}"))
