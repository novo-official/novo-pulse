"""Local experiment tracking.

No external tracking service. The database stores metadata and pointers; the
heavy artefacts (predictions, backtests, model binaries) live on disk under
`runs/<run_id>/` and are read lazily with a cache.
"""
from __future__ import annotations

from pathlib import Path

from django.db import models

from ml.paths import REPO_ROOT


class Experiment(models.Model):
    """A named line of work - usually one dataset + one target."""

    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    dataset_name = models.CharField(max_length=200, blank=True)
    target = models.CharField(max_length=120, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name


class TrainingRun(models.Model):
    """One execution of the training pipeline."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"

    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, related_name="runs", null=True, blank=True
    )
    run_id = models.CharField(max_length=64, unique=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    stage = models.CharField(max_length=120, blank=True)
    progress = models.FloatField(default=0.0)

    profile = models.CharField(max_length=32, default="demo")
    target = models.CharField(max_length=120, blank=True)
    primary_metric = models.CharField(max_length=32, default="wape")
    horizon = models.IntegerField(default=30)
    frequency = models.CharField(max_length=8, default="D")
    dataset_name = models.CharField(max_length=200, blank=True)
    dataset_path = models.CharField(max_length=500, blank=True)
    dataset_rows = models.IntegerField(default=0)
    n_features = models.IntegerField(default=0)
    n_entities = models.IntegerField(default=0)

    champion_model = models.CharField(max_length=64, blank=True)
    champion_score = models.FloatField(null=True, blank=True)
    baseline_model = models.CharField(max_length=64, blank=True)
    baseline_score = models.FloatField(null=True, blank=True)
    improvement = models.FloatField(null=True, blank=True)
    data_health_score = models.IntegerField(null=True, blank=True)

    run_dir = models.CharField(max_length=500, blank=True)
    config = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    warnings = models.JSONField(default=list, blank=True)
    error = models.TextField(blank=True)

    training_seconds = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["status", "-created_at"])]

    def __str__(self) -> str:
        return f"{self.run_id} ({self.status})"

    @property
    def directory(self) -> Path:
        path = Path(self.run_dir)
        return path if path.is_absolute() else (REPO_ROOT / path)

    @property
    def is_ready(self) -> bool:
        return self.status == self.Status.SUCCEEDED and self.directory.exists()


class ModelResult(models.Model):
    """One model's score within a training run - the leaderboard row."""

    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE, related_name="results")
    model_name = models.CharField(max_length=64)
    label = models.CharField(max_length=120, blank=True)
    kind = models.CharField(max_length=32, blank=True)
    is_baseline = models.BooleanField(default=False)
    is_champion = models.BooleanField(default=False)
    rank = models.IntegerField(default=0)

    primary_metric = models.CharField(max_length=32, default="wape")
    primary_value = models.FloatField(null=True, blank=True)
    improvement_vs_baseline = models.FloatField(null=True, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    by_horizon = models.JSONField(default=list, blank=True)
    ensemble_weight = models.FloatField(null=True, blank=True)
    fit_seconds = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["rank"]
        unique_together = [("run", "model_name")]

    def __str__(self) -> str:
        return f"{self.model_name}@{self.run.run_id}"


class ForecastRun(models.Model):
    """A produced forecast - either the training run's own, or a scenario."""

    class Kind(models.TextChoices):
        BASELINE = "baseline", "Baseline forecast"
        SCENARIO = "scenario", "Scenario simulation"

    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE, related_name="forecasts")
    kind = models.CharField(max_length=16, choices=Kind.choices, default=Kind.BASELINE)
    level = models.CharField(max_length=32, default="listing")
    horizon = models.IntegerField(default=30)
    model_name = models.CharField(max_length=64, blank=True)
    parameters = models.JSONField(default=dict, blank=True)
    summary = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
