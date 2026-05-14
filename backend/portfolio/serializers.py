from rest_framework import serializers
from .models import PortfolioJob, PortfolioResult


class PortfolioResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioResult
        fields = ('id', 'surviving_assets', 'pareto_solutions', 'strategies', 'weights', 'created_at')


class PortfolioJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioJob
        fields = ('id', 'status', 'params', 'celery_task_id', 'created_at', 'updated_at')
        read_only_fields = ('id', 'status', 'celery_task_id', 'created_at', 'updated_at')


class PortfolioJobDetailSerializer(serializers.ModelSerializer):
    result = PortfolioResultSerializer(read_only=True)

    class Meta:
        model = PortfolioJob
        fields = ('id', 'status', 'params', 'celery_task_id', 'created_at', 'updated_at', 'result')
        read_only_fields = ('id', 'status', 'celery_task_id', 'created_at', 'updated_at')
