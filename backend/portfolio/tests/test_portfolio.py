import pytest
import pandas as pd
import requests
from unittest.mock import patch
from django.urls import reverse
from rest_framework.test import APIClient
from users.models import User
from portfolio.models import PortfolioJob
from portfolio.tasks import run_portfolio_optimization
from core.optimizer.nsga3 import fetch_top_coins, _get_fallback_coin_list


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def user(db):
    return User.objects.create_user(
        email='portfolio@test.com',
        username='portfoliouser',
        password='Test1234!',
    )


@pytest.fixture
def auth_client(client, user):
    response = client.post(reverse('login'), {
        'email': 'portfolio@test.com',
        'password': 'Test1234!',
    }, format='json')
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return client


@pytest.fixture
def existing_job(user):
    return PortfolioJob.objects.create(
        user=user,
        params={'assets': ['BTC', 'ETH'], 'risk_tolerance': 'low'},
    )


@pytest.mark.django_db
class TestPortfolioJob:
    def test_create_job_authenticated(self, auth_client):
        with patch('portfolio.tasks.run_portfolio_optimization.delay'):
            response = auth_client.post(reverse('job_list'), {
                'params': {'assets': ['BTC', 'ETH', 'SOL'], 'risk_tolerance': 'medium'},
            }, format='json')
        assert response.status_code == 201
        assert response.data['status'] == 'pending'
        assert 'id' in response.data

    def test_create_job_unauthenticated(self, client):
        response = client.post(reverse('job_list'), {
            'params': {'assets': ['BTC'], 'risk_tolerance': 'low'},
        }, format='json')
        assert response.status_code == 401

    def test_list_jobs(self, auth_client, existing_job):
        response = auth_client.get(reverse('job_list'))
        assert response.status_code == 200
        assert len(response.data) >= 1

    def test_job_detail(self, auth_client, existing_job):
        response = auth_client.get(reverse('job_detail', kwargs={'pk': existing_job.id}))
        assert response.status_code == 200
        assert str(response.data['id']) == str(existing_job.id)
        assert response.data['status'] == 'pending'

    def test_delete_job(self, auth_client, existing_job):
        response = auth_client.delete(reverse('job_delete', kwargs={'pk': existing_job.id}))
        assert response.status_code == 200
        assert not PortfolioJob.objects.filter(id=existing_job.id).exists()

    def test_job_belongs_to_user(self, db, existing_job):
        other = User.objects.create_user(
            email='other@test.com',
            username='otherportfolio',
            password='Test1234!',
        )
        other_client = APIClient()
        login = other_client.post(reverse('login'), {
            'email': 'other@test.com',
            'password': 'Test1234!',
        }, format='json')
        other_client.credentials(HTTP_AUTHORIZATION=f"Bearer {login.data['access']}")
        response = other_client.get(reverse('job_detail', kwargs={'pk': existing_job.id}))
        assert response.status_code == 404

    def test_job_default_params(self, auth_client):
        with patch('portfolio.tasks.run_portfolio_optimization.delay'):
            response = auth_client.post(reverse('job_list'), {}, format='json')
        assert response.status_code == 201
        job = PortfolioJob.objects.get(id=response.data['id'])
        assert isinstance(job.params, dict)

    def test_celery_task_mock(self, auth_client):
        with patch('portfolio.tasks.run_portfolio_optimization.delay') as mock_delay:
            response = auth_client.post(reverse('job_list'), {
                'params': {'assets': ['BTC', 'ETH'], 'risk_tolerance': 'low'},
            }, format='json')
        assert response.status_code == 201
        mock_delay.assert_called_once()


_MINIMAL_COINS_DF = pd.DataFrame({
    'symbol':       ['BTC', 'ETH'],
    'coingecko_id': ['bitcoin', 'ethereum'],
    'name':         ['Bitcoin', 'Ethereum'],
    'volume_24h':   [1e9, 5e8],
    'market_cap':   [1e12, 5e11],
})


class TestFetchTopCoinsFallback:
    def test_ssl_error_triggers_fallback(self):
        with patch('requests.get',
                   side_effect=requests.exceptions.SSLError("SSL cert verify failed")):
            result = fetch_top_coins(n_target=10)
        fallback_symbols = {c['symbol'].upper() for c in _get_fallback_coin_list()}
        assert len(result) > 0
        assert set(result['symbol'].tolist()).issubset(fallback_symbols)

    def test_connection_error_triggers_fallback(self):
        with patch('requests.get',
                   side_effect=requests.exceptions.ConnectionError("Connection refused")):
            result = fetch_top_coins(n_target=10)
        assert len(result) > 0

    def test_timeout_triggers_fallback(self):
        with patch('requests.get',
                   side_effect=requests.exceptions.Timeout("Request timed out")):
            result = fetch_top_coins(n_target=10)
        assert len(result) > 0


@pytest.mark.django_db
class TestPortfolioJobDataSource:
    def test_data_source_field_exists_with_default(self, user):
        from django.db import models as django_models
        field = PortfolioJob._meta.get_field('data_source')
        assert isinstance(field, django_models.CharField)
        assert field.default == 'coingecko'
        choices_keys = [k for k, _ in field.choices]
        assert set(choices_keys) == {'coingecko', 'fallback', 'cache'}


@pytest.mark.django_db
class TestRetryDoesNotMarkFailed:
    def test_ssl_error_does_not_mark_job_failed(self, user):
        job = PortfolioJob.objects.create(user=user, params={})

        class FakeRetry(Exception):
            pass

        with patch('portfolio.tasks._mark_failed') as mock_mark_failed, \
             patch.object(run_portfolio_optimization, 'retry',
                          side_effect=FakeRetry()), \
             patch('core.optimizer.nsga3.fetch_top_coins',
                   return_value=_MINIMAL_COINS_DF), \
             patch('core.optimizer.nsga3.load_prices_from_yfinance',
                   side_effect=requests.exceptions.SSLError("SSL cert verify failed")):
            run_portfolio_optimization.apply(args=[str(job.id), {}])

        mock_mark_failed.assert_not_called()
