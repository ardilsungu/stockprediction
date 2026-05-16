import logging

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.exceptions import TokenError
from django.contrib.auth import authenticate
from .models import AuditLog
from .serializers import RegisterSerializer, LoginSerializer, UserProfileSerializer

logger = logging.getLogger(__name__)


def get_client_ip(request):
    forwarded = request.META.get('HTTP_X_FORWARDED_FOR')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.META.get('REMOTE_ADDR')


def _log(request, user, action):
    AuditLog.objects.create(
        user=user,
        action=action,
        detail={'email': user.email},
        ip_address=get_client_ip(request),
    )


@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        _log(request, user, 'register')
        refresh = RefreshToken.for_user(user)
        return Response({
            'message': 'Kayıt başarılı.',
            'access': str(refresh.access_token),
            'refresh': str(refresh),
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    serializer = LoginSerializer(data=request.data)
    if serializer.is_valid():
        user = authenticate(
            email=serializer.validated_data['email'],
            password=serializer.validated_data['password']
        )
        if user:
            _log(request, user, 'login')
            refresh = RefreshToken.for_user(user)
            return Response({
                'access': str(refresh.access_token),
                'refresh': str(refresh),
            })
        return Response({'error': 'Email veya şifre hatalı.'}, status=status.HTTP_401_UNAUTHORIZED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout(request):
    try:
        refresh_token = request.data['refresh']
    except KeyError:
        return Response(
            {'error': "'refresh' alanı zorunludur."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        token = RefreshToken(refresh_token)
        token.blacklist()
    except TokenError:
        return Response(
            {'error': 'Refresh token geçersiz veya süresi dolmuş.'},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as exc:
        logger.exception('Logout sırasında beklenmedik hata: %s', exc)
        return Response(
            {'error': 'Çıkış işlemi tamamlanamadı.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    _log(request, request.user, 'logout')
    return Response({'message': 'Çıkış başarılı.'})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile(request):
    serializer = UserProfileSerializer(request.user)
    return Response(serializer.data)
