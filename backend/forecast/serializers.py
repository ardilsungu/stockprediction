from rest_framework import serializers

from .models import ForecastJob, ForecastResult


class ForecastResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastResult
        fields = ('predictions', 'mae', 'rmse', 'created_at')


class ForecastJobCreateSerializer(serializers.ModelSerializer):
    asset_symbol = serializers.CharField(source='asset.symbol', read_only=True)
    # Model fonksiyonlarindaki 1-365 siniriyla tutarli: gecersiz horizon
    # failed job yerine 400 ile reddedilir.
    horizon_days = serializers.IntegerField(min_value=1, max_value=365, default=30)

    class Meta:
        model = ForecastJob
        fields = (
            'id', 'asset', 'asset_symbol', 'model_type', 'horizon_days',
            'status', 'celery_task_id', 'created_at',
        )
        read_only_fields = ('id', 'status', 'celery_task_id', 'created_at')


class ForecastJobDetailSerializer(serializers.ModelSerializer):
    asset_symbol = serializers.CharField(source='asset.symbol', read_only=True)
    result = ForecastResultSerializer(source='forecastresult', read_only=True)

    class Meta:
        model = ForecastJob
        fields = (
            'id', 'asset', 'asset_symbol', 'model_type', 'horizon_days',
            'status', 'error_message', 'celery_task_id', 'created_at', 'result',
        )
        read_only_fields = (
            'id', 'asset', 'model_type', 'horizon_days',
            'status', 'error_message', 'celery_task_id', 'created_at',
        )
