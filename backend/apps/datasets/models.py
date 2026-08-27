"""Uploaded datasets and their column mappings (the Data Lab)."""
from __future__ import annotations

from pathlib import Path

from django.db import models

from ml.paths import REPO_ROOT


class Dataset(models.Model):
    class Source(models.TextChoices):
        UPLOAD = "upload", "Uploaded"
        SYNTHETIC = "synthetic", "Synthetic"
        LOCAL = "local", "Local file"

    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    source = models.CharField(max_length=16, choices=Source.choices, default=Source.UPLOAD)
    path = models.CharField(max_length=500)
    original_filename = models.CharField(max_length=300, blank=True)
    size_bytes = models.BigIntegerField(default=0)
    n_rows = models.IntegerField(default=0)
    n_columns = models.IntegerField(default=0)

    profile = models.JSONField(default=dict, blank=True)
    mapping = models.JSONField(default=dict, blank=True)
    validation = models.JSONField(default=dict, blank=True)
    contract_path = models.CharField(max_length=500, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    @property
    def absolute_path(self) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else (REPO_ROOT / path)
