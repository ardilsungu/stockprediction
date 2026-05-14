import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from users.models import User


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def registered_user(db):
    return User.objects.create_user(
        email='existing@test.com',
        username='existinguser',
        password='Test1234!',
    )


@pytest.fixture
def auth_client(client, registered_user):
    response = client.post(reverse('login'), {
        'email': 'existing@test.com',
        'password': 'Test1234!',
    }, format='json')
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return client


@pytest.mark.django_db
class TestRegister:
    def test_register_success(self, client):
        response = client.post(reverse('register'), {
            'email': 'new@test.com',
            'username': 'newuser',
            'password': 'Test1234!',
            'password_confirm': 'Test1234!',
        }, format='json')
        assert response.status_code == 201
        assert 'access' in response.data
        assert response.data['message'] == 'Kayıt başarılı.'

    def test_register_duplicate_email(self, client, registered_user):
        response = client.post(reverse('register'), {
            'email': 'existing@test.com',
            'username': 'anotheruser',
            'password': 'Test1234!',
            'password_confirm': 'Test1234!',
        }, format='json')
        assert response.status_code == 400

    def test_register_password_mismatch(self, client):
        response = client.post(reverse('register'), {
            'email': 'mismatch@test.com',
            'username': 'mismatchuser',
            'password': 'Test1234!',
            'password_confirm': 'WrongPass!',
        }, format='json')
        assert response.status_code == 400
        assert 'password' in response.data


@pytest.mark.django_db
class TestLogin:
    def test_login_success(self, client, registered_user):
        response = client.post(reverse('login'), {
            'email': 'existing@test.com',
            'password': 'Test1234!',
        }, format='json')
        assert response.status_code == 200
        assert 'access' in response.data
        assert 'refresh' in response.data

    def test_login_wrong_password(self, client, registered_user):
        response = client.post(reverse('login'), {
            'email': 'existing@test.com',
            'password': 'WrongPass!',
        }, format='json')
        assert response.status_code == 401
        assert 'error' in response.data


@pytest.mark.django_db
class TestProfile:
    def test_profile_authenticated(self, auth_client, registered_user):
        response = auth_client.get(reverse('profile'))
        assert response.status_code == 200
        assert response.data['email'] == 'existing@test.com'
        assert response.data['username'] == 'existinguser'

    def test_profile_unauthenticated(self, client):
        response = client.get(reverse('profile'))
        assert response.status_code == 401
