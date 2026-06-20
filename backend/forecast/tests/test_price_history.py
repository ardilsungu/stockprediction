import datetime
import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from users.models import User
from assets.models import Asset, Price


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def user(db):
    return User.objects.create_user(
        email='prices@test.com',
        username='pricesuser',
        password='Test1234!',
    )


@pytest.fixture
def auth_client(client, user):
    response = client.post(reverse('login'), {
        'email': 'prices@test.com',
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


def _create_prices(asset, count, start=datetime.date(2026, 1, 1)):
    Price.objects.bulk_create([
        Price(
            asset=asset,
            date=start + datetime.timedelta(days=i),
            open=100 + i,
            high=110 + i,
            low=90 + i,
            close=105 + i,
            volume=1000,
        )
        for i in range(count)
    ])


def _url(symbol='BTC'):
    return reverse('price_history', kwargs={'symbol': symbol})


@pytest.mark.django_db
class TestPriceHistory:
    def test_unauthenticated(self, client, asset):
        response = client.get(_url())
        assert response.status_code == 401

    def test_unknown_symbol_returns_404(self, auth_client):
        response = auth_client.get(_url('YOKBOYLE'))
        assert response.status_code == 404

    def test_inactive_symbol_returns_404(self, auth_client):
        Asset.objects.create(symbol='OLD', name='Old Coin', is_active=False)
        response = auth_client.get(_url('OLD'))
        assert response.status_code == 404

    def test_non_numeric_days_returns_400(self, auth_client, asset):
        response = auth_client.get(_url(), {'days': 'abc'})
        assert response.status_code == 400

    def test_days_clamped_to_max_750(self, auth_client, asset):
        _create_prices(asset, 755, start=datetime.date(2024, 1, 1))
        response = auth_client.get(_url(), {'days': 99999})
        assert response.status_code == 200
        assert len(response.data) == 750

    def test_days_clamped_to_min_1(self, auth_client, asset):
        _create_prices(asset, 5)
        response = auth_client.get(_url(), {'days': -10})
        assert response.status_code == 200
        assert len(response.data) == 1

    def test_response_format_and_ascending_dates(self, auth_client, asset):
        _create_prices(asset, 3)
        response = auth_client.get(_url(), {'days': 3})
        assert response.status_code == 200
        assert response.data == [
            {'date': '2026-01-01', 'close': 105.0},
            {'date': '2026-01-02', 'close': 106.0},
            {'date': '2026-01-03', 'close': 107.0},
        ]
