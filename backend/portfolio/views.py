from rest_framework import status
from rest_framework.decorators import (
    api_view, permission_classes, throttle_classes,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import UserRateThrottle
from config.celery import app as celery_app
from .models import PortfolioJob
from .serializers import PortfolioJobSerializer, PortfolioJobDetailSerializer
from .tasks import run_portfolio_optimization


class PortfolioCreateThrottle(UserRateThrottle):
    """Job oluşturma throttle'ı: her job NSGA-III ile dakikalarca CPU harcayan
    bir optimizasyon kuyruklar; sınırsız POST kuyruğu worker'ı kilitleyebilir.

    Forecast app'indeki ForecastCreateThrottle deseniyle tutarlı (10/dakika).
    Tek kullanıcının meşru denemeleri (farklı params) için fazlasıyla geniş,
    ancak kuyruğun tek kullanıcı tarafından doldurulmasını engeller. Yalnızca
    CREATE (POST) sayılır; GET/list serbesttir (job_list hem GET hem POST
    sunduğu için güvenli metotlar throttle dışı bırakılır).
    """

    scope = 'portfolio_create'
    rate = '10/min'

    def allow_request(self, request, view):
        if request.method != 'POST':
            return True
        return super().allow_request(request, view)


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
@throttle_classes([PortfolioCreateThrottle])
def job_list(request):
    if request.method == 'GET':
        jobs = PortfolioJob.objects.filter(user=request.user)
        serializer = PortfolioJobSerializer(jobs, many=True)
        return Response(serializer.data)

    serializer = PortfolioJobSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    job = PortfolioJob.objects.create(
        user=request.user,
        params=serializer.validated_data.get('params', {}),
    )
    run_portfolio_optimization.delay(str(job.id), job.params)
    return Response(PortfolioJobSerializer(job).data, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def job_detail(request, pk):
    try:
        job = PortfolioJob.objects.get(id=pk, user=request.user)
        serializer = PortfolioJobDetailSerializer(job)
        return Response(serializer.data)
    except PortfolioJob.DoesNotExist:
        return Response({'error': 'Bulunamadı.'}, status=status.HTTP_404_NOT_FOUND)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def job_delete(request, pk):
    try:
        job = PortfolioJob.objects.get(id=pk, user=request.user)
        if job.celery_task_id and job.status in ('pending', 'running'):
            try:
                celery_app.control.revoke(job.celery_task_id, terminate=True)
            except Exception:
                pass
        job.delete()
        return Response({'message': 'Job silindi.'})
    except PortfolioJob.DoesNotExist:
        return Response({'error': 'Bulunamadı.'}, status=status.HTTP_404_NOT_FOUND)
