import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from users.models import User
from assets.models import Asset, Watchlist


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def user(db):
    return User.objects.create_user(
        email='assets@test.com',
        username='assetsuser',
        password='Test1234!',
    )


@pytest.fixture
def other_user(db):
    return User.objects.create_user(
        email='other@test.com',
        username='otheruser',
        password='Test1234!',
    )


@pytest.fixture
def auth_client(client, user):
    response = client.post(reverse('login'), {
        'email': 'assets@test.com',
        'password': 'Test1234!',
    }, format='json')
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return client


@pytest.fixture
def other_auth_client(other_user):
    c = APIClient()
    response = c.post(reverse('login'), {
        'email': 'other@test.com',
        'password': 'Test1234!',
    }, format='json')
    c.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return c


@pytest.fixture
def asset(db):
    return Asset.objects.create(
        symbol='BTC',
        name='Bitcoin',
        market_cap=1_000_000_000,
        category='crypto',
    )


@pytest.mark.django_db
class TestAssetList:
    def test_asset_list_authenticated(self, auth_client, asset):
        response = auth_client.get(reverse('asset_list'))
        assert response.status_code == 200
        assert len(response.data) >= 1
        symbols = [a['symbol'] for a in response.data]
        assert 'BTC' in symbols

    def test_asset_list_unauthenticated(self, client):
        response = client.get(reverse('asset_list'))
        assert response.status_code == 401


@pytest.mark.django_db
class TestWatchlist:
    def test_watchlist_empty(self, auth_client):
        response = auth_client.get(reverse('watchlist'))
        assert response.status_code == 200
        assert response.data == []

    def test_watchlist_add(self, auth_client, asset):
        response = auth_client.post(reverse('watchlist'), {
            'asset_id': str(asset.id),
        }, format='json')
        assert response.status_code == 201
        assert str(response.data['asset']['id']) == str(asset.id)

    def test_watchlist_add_duplicate(self, auth_client, asset, user):
        Watchlist.objects.create(user=user, asset=asset)
        response = auth_client.post(reverse('watchlist'), {
            'asset_id': str(asset.id),
        }, format='json')
        assert response.status_code == 400

    def test_watchlist_delete(self, auth_client, asset, user):
        entry = Watchlist.objects.create(user=user, asset=asset)
        response = auth_client.delete(reverse('watchlist_delete', kwargs={'pk': entry.id}))
        assert response.status_code == 200
        assert not Watchlist.objects.filter(id=entry.id).exists()

    def test_watchlist_delete_other_user(self, auth_client, asset, other_user):
        entry = Watchlist.objects.create(user=other_user, asset=asset)
        response = auth_client.delete(reverse('watchlist_delete', kwargs={'pk': entry.id}))
        assert response.status_code == 404
