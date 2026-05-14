import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from users.models import User, AuditLog


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def registered_user(db):
    return User.objects.create_user(
        email='audit@test.com',
        username='audituser',
        password='Test1234!',
    )


@pytest.mark.django_db
class TestAuditLog:
    def test_register_creates_auditlog(self, client):
        client.post(reverse('register'), {
            'email': 'newaudit@test.com',
            'username': 'newaudituser',
            'password': 'Test1234!',
            'password_confirm': 'Test1234!',
        }, format='json')
        assert AuditLog.objects.filter(action='register').exists()
        log = AuditLog.objects.get(action='register')
        assert log.detail.get('email') == 'newaudit@test.com'

    def test_login_creates_auditlog(self, client, registered_user):
        client.post(reverse('login'), {
            'email': 'audit@test.com',
            'password': 'Test1234!',
        }, format='json')
        assert AuditLog.objects.filter(action='login').exists()
        log = AuditLog.objects.get(action='login')
        assert log.user == registered_user

    def test_logout_creates_auditlog(self, client, registered_user):
        login_resp = client.post(reverse('login'), {
            'email': 'audit@test.com',
            'password': 'Test1234!',
        }, format='json')
        client.credentials(HTTP_AUTHORIZATION=f"Bearer {login_resp.data['access']}")
        client.post(reverse('logout'), {
            'refresh': login_resp.data['refresh'],
        }, format='json')
        assert AuditLog.objects.filter(action='logout').exists()
        log = AuditLog.objects.get(action='logout')
        assert log.user == registered_user
