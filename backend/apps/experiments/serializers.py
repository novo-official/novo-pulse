from __future__ import annotations

from rest_framework import serializers

from .models import Experiment, ModelResult, TrainingRun


class ModelResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelResult
        fields = [
            "model_name", "label", "kind", "is_baseline", "is_champion", "rank",
            "primary_metric", "primary_value", "improvement_vs_baseline",
            "metrics", "by_horizon", "ensemble_weight", "fit_seconds",
        ]


class TrainingRunSerializer(serializers.ModelSerializer):
    results = ModelResultSerializer(many=True, read_only=True)

    class Meta:
        model = TrainingRun
        fields = [
            "run_id", "status", "stage", "progress", "profile", "target",
            "primary_metric", "horizon", "frequency", "dataset_name", "dataset_rows",
            "n_features", "n_entities", "champion_model", "champion_score",
            "baseline_model", "baseline_score", "improvement", "data_health_score",
            "training_seconds", "created_at", "finished_at", "warnings", "error",
            "results",
        ]


class TrainingRunListSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRun
        fields = [
            "run_id", "status", "stage", "progress", "profile", "target",
            "primary_metric", "horizon", "champion_model", "champion_score",
            "improvement", "training_seconds", "created_at", "finished_at",
        ]


class ExperimentSerializer(serializers.ModelSerializer):
    run_count = serializers.IntegerField(source="runs.count", read_only=True)

    class Meta:
        model = Experiment
        fields = ["name", "slug", "description", "dataset_name", "target", "created_at", "run_count"]
