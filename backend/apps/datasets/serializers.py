from rest_framework import serializers

from .models import Dataset


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = [
            "id", "name", "slug", "source", "path", "original_filename", "size_bytes",
            "n_rows", "n_columns", "mapping", "contract_path", "created_at", "updated_at",
        ]


class JoinSerializer(serializers.Serializer):
    """A side table merged onto the main file before the panel is built.

    The competition ships demand, accommodation/destination and booking data as
    separate files; this is how they are stitched together without leaving the
    Data Lab.
    """

    dataset_id = serializers.IntegerField(required=False, allow_null=True)
    path = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    on = serializers.ListField(child=serializers.CharField(), allow_empty=False)

    def validate(self, attrs):
        if not attrs.get("dataset_id") and not attrs.get("path"):
            raise serializers.ValidationError(
                "A join needs either dataset_id or path."
            )
        return attrs


class ColumnMappingSerializer(serializers.Serializer):
    """The Data Lab form. Everything except timestamp/target is optional."""

    dataset_id = serializers.IntegerField(required=False, allow_null=True)
    timestamp = serializers.CharField()
    # With aggregation="count" the demand IS the row count, so a transactional
    # booking table needs no target column at all.
    target = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    entity_id = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    destination = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    category = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    frequency = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    aggregation = serializers.ChoiceField(
        choices=["sum", "mean", "max", "min", "first", "count"], required=False, default="sum"
    )
    calendar = serializers.ChoiceField(
        choices=["auto", "gregorian", "jalali"], required=False, default="auto"
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
    joins = JoinSerializer(many=True, required=False, default=list)

    def validate(self, attrs):
        if attrs.get("aggregation") != "count" and not attrs.get("target"):
            raise serializers.ValidationError(
                {
                    "target": (
                        "A target column is required unless aggregation is "
                        '"count", where demand is the number of rows.'
                    )
                }
            )
        return attrs
