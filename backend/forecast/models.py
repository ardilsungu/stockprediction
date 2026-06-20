import uuid
from django.db import models
from assets.models import Asset

class ForecastJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'), ('running', 'Running'),
        ('completed', 'Completed'), ('failed', 'Failed'),
    ]
    MODEL_CHOICES = [('lstm', 'LSTM'), ('prophet', 'Prophet')]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    model_type = models.CharField(max_length=10, choices=MODEL_CHOICES)
    horizon_days = models.IntegerField(default=30)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True)
    celery_task_id = models.CharField(max_length=64, blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['status', 'created_at'])]

    def __str__(self):
        return f"{self.asset.symbol} - {self.model_type} - {self.status}"

class ForecastResult(models.Model):
    job = models.OneToOneField(ForecastJob, on_delete=models.CASCADE)
    predictions = models.JSONField()
    mae = models.FloatField(null=True)
    rmse = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.job.asset.symbol} - {self.job.model_type}"
