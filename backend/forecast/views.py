from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle

from .models import ForecastJob
from .serializers import ForecastJobCreateSerializer, ForecastJobDetailSerializer
from .tasks import run_forecast_task


class ForecastCreateThrottle(UserRateThrottle):
    """Job olusturma throttle'i: her job dakikalarca CPU harcayan bir model
    egitimi kuyrukladigi icin sinirsiz POST kuyrugu tiklayabilir.

    10/dakika, tek kullanicinin mesru denemeleri (farkli varlik/model/horizon)
    icin fazlasiyla genis, ancak kuyrugun tek kullanici tarafindan doldurulmasini
    engeller. Yalnizca create endpoint'ine uygulanir; detail/GET serbesttir.
    """

    scope = 'forecast_create'
    rate = '10/min'


class ForecastJobCreateView(generics.CreateAPIView):
    """POST /api/forecast/jobs/ — yeni bir tahmin isi olusturur ve Celery task'ini baslatir."""

    queryset = ForecastJob.objects.all()
    serializer_class = ForecastJobCreateSerializer
    permission_classes = [IsAuthenticated]
    throttle_classes = [ForecastCreateThrottle]

    def perform_create(self, serializer):
        job = serializer.save()
        task = run_forecast_task.delay(str(job.id))
        if task and task.id:
            job.celery_task_id = task.id
            job.save(update_fields=['celery_task_id'])


class ForecastJobDetailView(generics.RetrieveAPIView):
    """GET /api/forecast/jobs/<uuid>/ — isi ve (varsa) tahmin sonucunu dondurur."""

    # Bilincli karar: job'larda sahiplik (user FK) modellenmiyor; erisim icin
    # UUID'nin tahmin-edilemezligine guveniliyor. Spec coklu-kullanici izolasyonu
    # gerektirmedigi icin bu kabul edildi.
    queryset = ForecastJob.objects.select_related('asset', 'forecastresult')
    serializer_class = ForecastJobDetailSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'pk'
