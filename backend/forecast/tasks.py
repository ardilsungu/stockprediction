from celery import shared_task
from .models import ForecastJob, ForecastResult
from .models_prophet import run_prophet_forecast
from .models_lstm import run_lstm_forecast
import logging

logger = logging.getLogger(__name__)

@shared_task(bind=True, soft_time_limit=600, time_limit=660)
def run_forecast_task(self, job_id: str):
    """
    ForecastJob'u alır, model_type'a göre Prophet veya LSTM çalıştırır,
    sonucu ForecastResult'a kaydeder.
    """
    try:
        job = ForecastJob.objects.get(id=job_id)
    except ForecastJob.DoesNotExist:
        logger.error(f"ForecastJob bulunamadi: {job_id}")
        return

    job.status = 'running'
    job.celery_task_id = self.request.id or ''
    job.save()

    try:
        if job.model_type == 'prophet':
            result = run_prophet_forecast(job.asset, job.horizon_days)
        elif job.model_type == 'lstm':
            result = run_lstm_forecast(job.asset, job.horizon_days)
        else:
            raise ValueError(f"Bilinmeyen model_type: {job.model_type}")

        ForecastResult.objects.update_or_create(
            job=job,
            defaults={
                'predictions': result['predictions'],
                'mae': result['mae'],
                'rmse': result['rmse'],
            }
        )
        job.status = 'completed'
        logger.info(f"ForecastJob tamamlandi: {job_id}, model={job.model_type}")

    except Exception as e:
        job.status = 'failed'
        job.error_message = str(e)
        logger.error(f"ForecastJob basarisiz: {job_id}, hata: {e}")

    finally:
        job.save()
