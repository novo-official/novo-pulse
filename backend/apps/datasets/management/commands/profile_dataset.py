"""`python manage.py profile_dataset --file data/raw/<file>`

Competition-day step 2: what is in this file, and which column plays which role?
"""
from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from apps.datasets.services import profile, register_local_file


class Command(BaseCommand):
    help = "Profile a dataset file and suggest a column mapping."

    def add_arguments(self, parser):
        parser.add_argument("--file", required=True, help="Path to the dataset.")
        parser.add_argument("--json", action="store_true", help="Emit the raw JSON profile.")
        parser.add_argument("--rows", type=int, default=5, help="Preview rows to print.")

    def handle(self, *args, **options):
        try:
            path = register_local_file(options["file"])
        except (ValueError, FileNotFoundError) as exc:
            raise CommandError(str(exc)) from exc

        try:
            report = profile(path)
        except Exception as exc:  # noqa: BLE001
            raise CommandError(f"Could not read the dataset: {exc}") from exc

        if options["json"]:
            self.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return

        self.stdout.write(self.style.SUCCESS(f"\n{report['file']}"))
        self.stdout.write(
            f"  {report['rows']:,} rows x {report['n_columns']} columns "
            f"({report['memory_mb']} MB in memory)"
        )

        frequency = report.get("frequency")
        if frequency:
            marker = self.style.SUCCESS("regular") if frequency["regular"] else self.style.WARNING("IRREGULAR")
            self.stdout.write(
                f"  frequency: {frequency['frequency']} ({marker}, "
                f"median spacing {frequency['median_delta_days']}d)"
            )

        self.stdout.write("\nColumns")
        header = f"  {'name':<28} {'kind':<14} {'null%':>7} {'unique':>10}  range"
        self.stdout.write(header)
        self.stdout.write("  " + "-" * (len(header) - 2))
        for column in report["columns"]:
            span = ""
            if column.get("min") is not None:
                span = f"{column['min']} .. {column['max']}"
            flag = self.style.ERROR("!") if column["null_pct"] > 30 else " "
            self.stdout.write(
                f" {flag}{column['name']:<28} {column['kind']:<14} "
                f"{column['null_pct']:>6.1f}% {column['unique_count']:>10,}  {span}"
            )

        suggested = report["suggested_schema"]
        self.stdout.write(self.style.SUCCESS("\nSuggested mapping"))
        for role in ("timestamp", "target", "entity_id", "destination", "category"):
            value = suggested.get(role)
            style = self.style.SUCCESS if value else self.style.WARNING
            self.stdout.write(f"  {role:<16} {style(str(value) or 'not detected')}")
        self.stdout.write(f"  {'future':<16} {suggested['future_features']}")
        self.stdout.write(f"  {'historical':<16} {suggested['historical_features']}")

        self.stdout.write("\nAlternative target candidates")
        for candidate in report["candidates"].get("target", [])[:5]:
            self.stdout.write(f"  {candidate['column']:<28} score={candidate['score']}")

        self.stdout.write(
            "\nNext: map the columns in the Data Lab (/data-lab), or copy "
            "config/data_contract.example.yaml to config/data_contract.active.yaml and edit it."
        )
