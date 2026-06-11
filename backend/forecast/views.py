from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from .models import ForecastJob
from .serializers import ForecastJobCreateSerializer, ForecastJobDetailSerializer
from .tasks import run_forecast_task


class ForecastJobCreateView(generics.CreateAPIView):
    """POST /api/forecast/jobs/ — yeni bir tahmin isi olusturur ve Celery task'ini baslatir."""

    queryset = ForecastJob.objects.all()
    serializer_class = ForecastJobCreateSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        job = serializer.save()
        task = run_forecast_task.delay(str(job.id))
        if task and task.id:
            job.celery_task_id = task.id
            job.save(update_fields=['celery_task_id'])


class ForecastJobDetailView(generics.RetrieveAPIView):
    """GET /api/forecast/jobs/<uuid>/ — isi ve (varsa) tahmin sonucunu dondurur."""

    queryset = ForecastJob.objects.select_related('asset', 'forecastresult')
    serializer_class = ForecastJobDetailSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'pk'
