import sys
import uuid
import pytest
from unittest.mock import MagicMock, patch
from django.db import IntegrityError
from django.urls import reverse
from rest_framework.test import APIClient
from users.models import User
from assets.models import Asset
from forecast.models import ForecastJob, ForecastResult
from forecast.serializers import ForecastJobCreateSerializer, ForecastJobDetailSerializer
from forecast.tasks import run_forecast_task

FAKE_FORECAST = {
    'predictions': [
        {'date': '2026-06-13', 'predicted': 100.0, 'lower_ci': 92.0, 'upper_ci': 108.0},
        {'date': '2026-06-14', 'predicted': 101.0, 'lower_ci': 93.0, 'upper_ci': 109.0},
    ],
    'mae': 1.5,
    'rmse': 2.5,
    'baseline_mae': 3.5,
    'baseline_rmse': 4.5,
}

# tasks.py'nin ForecastResult.predictions JSONField'ına yazdığı zarf yapısı
FAKE_SAVED_PREDICTIONS = {
    'points': FAKE_FORECAST['predictions'],
    'baseline': {'mae': 3.5, 'rmse': 4.5},
}


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def user(db):
    return User.objects.create_user(
        email='forecast@test.com',
        username='forecastuser',
        password='Test1234!',
    )


@pytest.fixture
def auth_client(client, user):
    response = client.post(reverse('login'), {
        'email': 'forecast@test.com',
        'password': 'Test1234!',
    }, format='json')
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return client


@pytest.fixture
def asset(db):
    return Asset.objects.create(
        symbol='BTC',
        name='Bitcoin',
        market_cap=1_000_000_000,
        category='crypto',
    )


@pytest.fixture
def job(asset):
    return ForecastJob.objects.create(asset=asset, model_type='prophet')


def _mock_model_module(func_name, **func_config):
    """Ağır ML modülünü (prophet/tensorflow) hiç import etmeden mock'lar."""
    module = MagicMock()
    getattr(module, func_name).configure_mock(**func_config)
    return module


@pytest.mark.django_db
class TestForecastJobModel:
    def test_job_defaults(self, asset):
        job = ForecastJob.objects.create(asset=asset, model_type='prophet')
        assert job.horizon_days == 30
        assert job.status == 'pending'
        assert job.celery_task_id == ''
        assert job.error_message == ''

    def test_job_str(self, job):
        assert str(job) == 'BTC - prophet - pending'

    def test_job_meta_ordering_newest_first(self):
        assert ForecastJob._meta.ordering == ['-created_at']


@pytest.mark.django_db
class TestForecastResultModel:
    def test_result_one_to_one_relation(self, job):
        result = ForecastResult.objects.create(
            job=job, predictions=FAKE_FORECAST['predictions'], mae=1.5, rmse=2.5,
        )
        assert job.forecastresult == result

    def test_result_duplicate_for_same_job_rejected(self, job):
        ForecastResult.objects.create(job=job, predictions=[])
        with pytest.raises(IntegrityError):
            ForecastResult.objects.create(job=job, predictions=[])

    def test_result_str(self, job):
        result = ForecastResult.objects.create(job=job, predictions=[])
        assert str(result) == 'BTC - prophet'


@pytest.mark.django_db
class TestForecastSerializers:
    def test_create_serializer_ignores_read_only_fields(self, asset):
        serializer = ForecastJobCreateSerializer(data={
            'asset': str(asset.id),
            'model_type': 'prophet',
            'horizon_days': 14,
            'status': 'completed',
            'celery_task_id': 'kullanicidan-gelen-id',
        })
        assert serializer.is_valid(), serializer.errors
        job = serializer.save()
        assert job.status == 'pending'
        assert job.celery_task_id == ''

    def test_create_serializer_exposes_asset_symbol(self, job):
        data = ForecastJobCreateSerializer(job).data
        assert data['asset_symbol'] == 'BTC'

    def test_detail_serializer_result_null_when_no_result(self, job):
        data = ForecastJobDetailSerializer(job).data
        assert data['result'] is None

    def test_detail_serializer_nested_result_fields(self, job):
        ForecastResult.objects.create(
            job=job, predictions=FAKE_SAVED_PREDICTIONS, mae=1.5, rmse=2.5,
        )
        data = ForecastJobDetailSerializer(job).data
        assert data['result']['predictions']['points'] == FAKE_FORECAST['predictions']
        assert data['result']['predictions']['baseline'] == {'mae': 3.5, 'rmse': 4.5}
        assert data['result']['mae'] == 1.5
        assert data['result']['rmse'] == 2.5


@pytest.mark.django_db
class TestForecastAPI:
    def test_job_create_unauthenticated(self, client, asset):
        response = client.post(reverse('forecast:job_create'), {
            'asset': str(asset.id),
            'model_type': 'prophet',
        }, format='json')
        assert response.status_code == 401

    def test_job_detail_unauthenticated(self, client, job):
        response = client.get(reverse('forecast:job_detail', kwargs={'pk': job.id}))
        assert response.status_code == 401

    def test_job_create_returns_201_and_triggers_task(self, auth_client, asset):
        mock_async = MagicMock()
        mock_async.id = 'celery-task-id-123'
        with patch('forecast.views.run_forecast_task.delay', return_value=mock_async) as mock_delay:
            response = auth_client.post(reverse('forecast:job_create'), {
                'asset': str(asset.id),
                'model_type': 'lstm',
                'horizon_days': 7,
            }, format='json')
        assert response.status_code == 201
        job = ForecastJob.objects.get(id=response.data['id'])
        mock_delay.assert_called_once_with(str(job.id))
        assert job.celery_task_id == 'celery-task-id-123'
        assert job.model_type == 'lstm'
        assert job.horizon_days == 7

    def test_job_create_ignores_status_and_task_id_in_body(self, auth_client, asset):
        mock_async = MagicMock()
        mock_async.id = None
        with patch('forecast.views.run_forecast_task.delay', return_value=mock_async):
            response = auth_client.post(reverse('forecast:job_create'), {
                'asset': str(asset.id),
                'model_type': 'prophet',
                'status': 'completed',
                'celery_task_id': 'sahte-id',
            }, format='json')
        assert response.status_code == 201
        assert response.data['status'] == 'pending'
        assert response.data['celery_task_id'] == ''

    def test_job_detail_authenticated(self, auth_client, job):
        response = auth_client.get(reverse('forecast:job_detail', kwargs={'pk': job.id}))
        assert response.status_code == 200
        assert response.data['id'] == str(job.id)
        assert response.data['status'] == 'pending'
        assert response.data['asset_symbol'] == 'BTC'
        assert response.data['result'] is None

    def test_job_detail_completed_returns_points_and_baseline(self, auth_client, job):
        job.status = 'completed'
        job.save()
        ForecastResult.objects.create(
            job=job, predictions=FAKE_SAVED_PREDICTIONS, mae=1.5, rmse=2.5,
        )
        response = auth_client.get(reverse('forecast:job_detail', kwargs={'pk': job.id}))
        assert response.status_code == 200
        predictions = response.data['result']['predictions']
        assert predictions['points'] == FAKE_FORECAST['predictions']
        assert predictions['baseline'] == {'mae': 3.5, 'rmse': 4.5}


@pytest.mark.django_db
class TestRunForecastTask:
    def test_success_transitions_pending_running_completed(self, job):
        seen = {}

        def fake_forecast(asset, horizon_days):
            seen['status_during_run'] = ForecastJob.objects.get(id=job.id).status
            return FAKE_FORECAST

        module = _mock_model_module('run_prophet_forecast', side_effect=fake_forecast)
        assert job.status == 'pending'
        with patch.dict(sys.modules, {'forecast.models_prophet': module}):
            run_forecast_task.apply(args=[str(job.id)])

        job.refresh_from_db()
        assert seen['status_during_run'] == 'running'
        assert job.status == 'completed'

    def test_success_saves_forecast_result(self, job):
        module = _mock_model_module('run_prophet_forecast', return_value=FAKE_FORECAST)
        with patch.dict(sys.modules, {'forecast.models_prophet': module}):
            run_forecast_task.apply(args=[str(job.id)])

        result = ForecastResult.objects.get(job=job)
        assert result.predictions == FAKE_SAVED_PREDICTIONS
        assert result.mae == 1.5
        assert result.rmse == 2.5

    def test_lstm_branch_calls_lstm_model(self, asset):
        job = ForecastJob.objects.create(asset=asset, model_type='lstm')
        module = _mock_model_module('run_lstm_forecast', return_value=FAKE_FORECAST)
        with patch.dict(sys.modules, {'forecast.models_lstm': module}):
            run_forecast_task.apply(args=[str(job.id)])

        job.refresh_from_db()
        assert job.status == 'completed'
        module.run_lstm_forecast.assert_called_once_with(job.asset, job.horizon_days)

    def test_model_error_sets_failed_and_error_message(self, job):
        module = _mock_model_module(
            'run_prophet_forecast',
            side_effect=ValueError('BTC için yeterli fiyat verisi yok.'),
        )
        with patch.dict(sys.modules, {'forecast.models_prophet': module}):
            run_forecast_task.apply(args=[str(job.id)])

        job.refresh_from_db()
        assert job.status == 'failed'
        assert 'yeterli fiyat verisi yok' in job.error_message
        assert not ForecastResult.objects.filter(job=job).exists()

    def test_retry_updates_existing_result_without_duplicate(self, job):
        ForecastResult.objects.create(
            job=job,
            predictions=[{'date': '2026-01-01', 'predicted': 1.0}],
            mae=9.9, rmse=9.9,
        )
        module = _mock_model_module('run_prophet_forecast', return_value=FAKE_FORECAST)
        with patch.dict(sys.modules, {'forecast.models_prophet': module}):
            run_forecast_task.apply(args=[str(job.id)])

        job.refresh_from_db()
        assert job.status == 'completed'
        assert ForecastResult.objects.filter(job=job).count() == 1
        result = ForecastResult.objects.get(job=job)
        assert result.predictions == FAKE_SAVED_PREDICTIONS
        assert result.mae == 1.5

    def test_unknown_job_id_returns_silently(self, db):
        run_forecast_task.apply(args=[str(uuid.uuid4())])
        assert ForecastJob.objects.count() == 0
        assert ForecastResult.objects.count() == 0

    def test_unexpected_error_writes_generic_message(self, job):
        module = _mock_model_module(
            'run_prophet_forecast',
            side_effect=RuntimeError('connection to db at 10.0.0.5:5432 refused'),
        )
        with patch.dict(sys.modules, {'forecast.models_prophet': module}):
            run_forecast_task.apply(args=[str(job.id)])

        job.refresh_from_db()
        assert job.status == 'failed'
        assert job.error_message == 'Tahmin işlenirken beklenmeyen bir hata oluştu.'
        assert '10.0.0.5' not in job.error_message


def _post_job(auth_client, asset, **overrides):
    """horizon/throttle testleri icin Celery'yi mock'layarak job POST'lar."""
    payload = {'asset': str(asset.id), 'model_type': 'prophet', **overrides}
    mock_async = MagicMock()
    mock_async.id = None
    with patch('forecast.views.run_forecast_task.delay', return_value=mock_async):
        return auth_client.post(reverse('forecast:job_create'), payload, format='json')


@pytest.mark.django_db
class TestHorizonDaysValidation:
    def test_negative_horizon_returns_400(self, auth_client, asset):
        response = _post_job(auth_client, asset, horizon_days=-5)
        assert response.status_code == 400
        assert 'horizon_days' in response.data
        assert ForecastJob.objects.count() == 0

    def test_huge_horizon_returns_400(self, auth_client, asset):
        response = _post_job(auth_client, asset, horizon_days=999999)
        assert response.status_code == 400
        assert 'horizon_days' in response.data
        assert ForecastJob.objects.count() == 0

    def test_omitted_horizon_defaults_to_30(self, auth_client, asset):
        response = _post_job(auth_client, asset)
        assert response.status_code == 201
        assert response.data['horizon_days'] == 30

    def test_boundary_values_accepted(self, auth_client, asset):
        for horizon in (1, 365):
            response = _post_job(auth_client, asset, horizon_days=horizon)
            assert response.status_code == 201
            assert response.data['horizon_days'] == horizon

    def test_serializer_rejects_out_of_range(self, asset):
        for horizon in (0, 366):
            serializer = ForecastJobCreateSerializer(data={
                'asset': str(asset.id),
                'model_type': 'prophet',
                'horizon_days': horizon,
            })
            assert not serializer.is_valid()
            assert 'horizon_days' in serializer.errors


@pytest.mark.django_db
class TestForecastCreateThrottle:
    def test_create_returns_429_after_limit(self, auth_client, asset):
        # ForecastCreateThrottle: 10/min — 11. istek throttle'a takilmali.
        statuses = [_post_job(auth_client, asset).status_code for _ in range(11)]
        assert statuses[:10] == [201] * 10
        assert statuses[10] == 429

    def test_detail_get_is_not_throttled(self, auth_client, job):
        url = reverse('forecast:job_detail', kwargs={'pk': job.id})
        for _ in range(15):
            assert auth_client.get(url).status_code == 200
