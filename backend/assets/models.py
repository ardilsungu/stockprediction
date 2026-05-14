import uuid
from django.db import models
from users.models import User


class Asset(models.Model):
    CATEGORY_CHOICES = [('crypto', 'Crypto'), ('stock', 'Stock')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    symbol = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100)
    market_cap = models.BigIntegerField(null=True, blank=True)
    category = models.CharField(max_length=10, choices=CATEGORY_CHOICES, default='crypto')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'assets'
        ordering = ['-market_cap']

    def __str__(self):
        return f"{self.symbol} - {self.name}"


class Price(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name='prices')
    date = models.DateField()
    open = models.DecimalField(max_digits=20, decimal_places=8)
    high = models.DecimalField(max_digits=20, decimal_places=8)
    low = models.DecimalField(max_digits=20, decimal_places=8)
    close = models.DecimalField(max_digits=20, decimal_places=8)
    volume = models.BigIntegerField(null=True, blank=True)

    class Meta:
        db_table = 'prices'
        unique_together = ('asset', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.asset.symbol} - {self.date}"


class Watchlist(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watchlist')
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name='watchlisted_by')
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'watchlist'
        unique_together = ('user', 'asset')

    def __str__(self):
        return f"{self.user.email} - {self.asset.symbol}"
