from django.contrib import admin
from .models import Asset, Price, Watchlist


@admin.register(Asset)
class AssetAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'name', 'category', 'market_cap', 'is_active', 'updated_at')
    list_filter = ('category', 'is_active')
    search_fields = ('symbol', 'name')
    ordering = ('-market_cap',)
    readonly_fields = ('id', 'created_at', 'updated_at')


@admin.register(Price)
class PriceAdmin(admin.ModelAdmin):
    list_display = ('asset', 'date', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('date', 'asset__category')
    search_fields = ('asset__symbol',)
    ordering = ('-date',)
    autocomplete_fields = ('asset',)


@admin.register(Watchlist)
class WatchlistAdmin(admin.ModelAdmin):
    list_display = ('user', 'asset', 'added_at')
    list_filter = ('added_at',)
    search_fields = ('user__email', 'asset__symbol')
    ordering = ('-added_at',)
    autocomplete_fields = ('user', 'asset')
    readonly_fields = ('id', 'added_at')
