from rest_framework import serializers

from .models import Dataset


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = [
            "id", "name", "slug", "source", "path", "original_filename", "size_bytes",
            "n_rows", "n_columns", "mapping", "contract_path", "created_at", "updated_at",
        ]


class ColumnMappingSerializer(serializers.Serializer):
    """The Data Lab form. Everything except timestamp/target is optional."""

    dataset_id = serializers.IntegerField(required=False, allow_null=True)
    timestamp = serializers.CharField()
    target = serializers.CharField()
    entity_id = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    destination = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    category = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    frequency = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    aggregation = serializers.ChoiceField(
        choices=["sum", "mean", "max", "min", "first"], required=False, default="sum"
    )
    future_features = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    historical_features = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    static_features = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    ignored = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    primary_metric = serializers.CharField(required=False, default="wape")
    horizons = serializers.ListField(child=serializers.IntegerField(), required=False, default=list)
    non_negative = serializers.BooleanField(required=False, default=True)
    integer = serializers.BooleanField(required=False, default=False)
    censoring_column = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    save_as_active = serializers.BooleanField(required=False, default=True)
