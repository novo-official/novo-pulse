"""`python manage.py validate_dataset`

Competition-day step 9: data quality and potential-leakage report for the
active data contract, before a single model is trained.
"""
from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from apps.datasets.services import validate_contract
from ml.contract import DataContract

SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]


class Command(BaseCommand):
    help = "Validate the active data contract and report data quality."

    def add_arguments(self, parser):
        parser.add_argument("--contract", help="Path to a contract YAML (default: the active one).")
        parser.add_argument("--dataset", help="Override the dataset path.")
        parser.add_argument("--target", help="Override the target column.")
        parser.add_argument("--json", action="store_true", help="Emit the raw JSON report.")

    def handle(self, *args, **options):
        contract = DataContract.load(options.get("contract")).with_overrides(
            path=options.get("dataset"), target=options.get("target")
        )
        try:
            report = validate_contract(contract)
        except Exception as exc:  # noqa: BLE001
            raise CommandError(f"Validation failed: {exc}") from exc

        if options["json"]:
            self.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, default=str))
            return

        score = report["health_score"]
        style = (
            self.style.SUCCESS if score >= 75 else
            self.style.WARNING if score >= 50 else self.style.ERROR
        )
        self.stdout.write(style(f"\nData Health Score: {score}/100 ({report['grade']})"))

        panel = report.get("panel") or {}
        self.stdout.write(
            f"  {panel.get('rows', 0):,} rows | {panel.get('entities', 0):,} entities | "
            f"{panel.get('periods', 0):,} periods | {panel.get('start')} -> {panel.get('end')}"
        )
        self.stdout.write(
            f"  target '{contract.target}': mean={panel.get('target_mean', 0):.3f} "
            f"zeros={panel.get('zero_ratio', 0):.1%}"
        )

        for note in report.get("adapter_notes") or []:
            self.stdout.write(self.style.WARNING(f"  note: {note}"))

        findings = report["findings"]
        if not findings:
            self.stdout.write(self.style.SUCCESS("\nNo issues found."))
        else:
            self.stdout.write(f"\n{len(findings)} finding(s):")
            for severity in SEVERITY_ORDER:
                for finding in [f for f in findings if f["severity"] == severity]:
                    marker = {
                        "critical": self.style.ERROR, "high": self.style.ERROR,
                        "medium": self.style.WARNING, "low": self.style.NOTICE,
                    }.get(severity, self.style.NOTICE)
                    self.stdout.write(f"\n  {marker(severity.upper())} {finding['title']}")
                    self.stdout.write(f"      {finding['detail']}")
                    if finding["recommendation"]:
                        self.stdout.write(f"      -> {finding['recommendation']}")

        leakage = [f for f in findings if f["code"] == "potential_leakage"]
        if leakage:
            self.stdout.write(
                self.style.WARNING(
                    "\nPotential Leakage Warnings: correlation alone is not proof. "
                    "Confirm each flagged column is genuinely available at prediction time; "
                    "if not, move it from `future` to `historical` or `ignored`."
                )
            )
        self.stdout.write("")
