from rest_framework import serializers
from .models import PortfolioJob, PortfolioResult


class PortfolioResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioResult
        fields = ('id', 'surviving_assets', 'pareto_solutions', 'strategies', 'weights', 'created_at')


PARAM_RANGES = {
    'lookback_days': (250, 750),
    'n_coins': (20, 100),
    'n_gen': (20, 500),
    'pop_size': (20, 500),
    'max_weight': (0.01, 1.0),
    'min_assets': (2, 20),
    'max_assets': (2, 20),
}


class PortfolioJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioJob
        fields = ('id', 'status', 'params', 'celery_task_id', 'created_at', 'updated_at')
        read_only_fields = ('id', 'status', 'celery_task_id', 'created_at', 'updated_at')

    def validate_params(self, params):
        if not isinstance(params, dict):
            raise serializers.ValidationError('params bir nesne olmalıdır.')

        errors = {}
        for key, (lo, hi) in PARAM_RANGES.items():
            if key not in params:
                continue
            value = params[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors[key] = f'{key} sayısal olmalıdır.'
                continue
            if value < lo or value > hi:
                errors[key] = f'{key} {lo} ile {hi} arasında olmalıdır.'

        min_assets = params.get('min_assets')
        max_assets = params.get('max_assets')
        if (
            isinstance(min_assets, (int, float)) and not isinstance(min_assets, bool)
            and isinstance(max_assets, (int, float)) and not isinstance(max_assets, bool)
            and min_assets > max_assets
        ):
            errors['min_assets'] = 'min_assets max_assets değerinden büyük olamaz.'

        if errors:
            raise serializers.ValidationError(errors)
        return params


class PortfolioJobDetailSerializer(serializers.ModelSerializer):
    result = PortfolioResultSerializer(read_only=True)

    class Meta:
        model = PortfolioJob
        fields = ('id', 'status', 'params', 'celery_task_id', 'created_at', 'updated_at', 'result')
        read_only_fields = ('id', 'status', 'celery_task_id', 'created_at', 'updated_at')
