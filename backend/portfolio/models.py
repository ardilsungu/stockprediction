import uuid
from django.db import models
from users.models import User


class PortfolioJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='portfolio_jobs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    params = models.JSONField(default=dict)
    data_source = models.CharField(
        max_length=20,
        default='coingecko',
        choices=[
            ('coingecko', 'CoinGecko'),
            ('fallback', 'Fallback'),
            ('cache', 'Cache'),
        ],
    )
    celery_task_id = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'portfolio_jobs'
        ordering = ['-created_at']

    def __str__(self):
        return f"Job {self.id} - {self.status}"


class PortfolioResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.OneToOneField(PortfolioJob, on_delete=models.CASCADE, related_name='result')
    surviving_assets = models.JSONField(default=list)
    pareto_solutions = models.JSONField(default=list)
    strategies = models.JSONField(default=dict)
    weights = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'portfolio_results'

    def __str__(self):
        return f"Result for Job {self.job.id}"
