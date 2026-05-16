import pytest
from unittest.mock import patch
from django.urls import reverse
from rest_framework.test import APIClient
from users.models import User
from portfolio.models import PortfolioJob


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
