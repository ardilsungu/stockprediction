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

        # n_obj: PARAM_RANGES'ta yer almıyordu → tasks.py'de doğrudan kullanılıp
        # CryptoPortfolioProblem'e n_obj olarak geçiyordu. Kullanıcı n_obj:10
        # gönderip pymoo'yu (ve worker'ı) kilitleyebiliyordu. Kod _evaluate'te
        # yalnızca 2 (getiri+CVaR) veya 3 (+volatilite) hedef dolduruyor; bu
        # nedenle izin verilen tek değerler {2, 3}. Diğer her şey 400.
        if 'n_obj' in params:
            n_obj = params['n_obj']
            if isinstance(n_obj, bool) or not isinstance(n_obj, int) or n_obj not in (2, 3):
                errors['n_obj'] = 'n_obj yalnızca 2 veya 3 olabilir.'

        if errors:
            raise serializers.ValidationError(errors)
        return params


class PortfolioJobDetailSerializer(serializers.ModelSerializer):
    result = PortfolioResultSerializer(read_only=True)

    class Meta:
        model = PortfolioJob
        fields = ('id', 'status', 'params', 'celery_task_id', 'created_at', 'updated_at', 'result')
        read_only_fields = ('id', 'status', 'celery_task_id', 'created_at', 'updated_at')
