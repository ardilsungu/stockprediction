from django.contrib import admin
from .models import PortfolioJob, PortfolioResult


@admin.register(PortfolioJob)
class PortfolioJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'status', 'celery_task_id', 'created_at', 'updated_at')
    list_filter = ('status', 'created_at')
    search_fields = ('id', 'user__email', 'celery_task_id')
    ordering = ('-created_at',)
    readonly_fields = ('id', 'celery_task_id', 'created_at', 'updated_at')
    autocomplete_fields = ('user',)


@admin.register(PortfolioResult)
class PortfolioResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'job', 'created_at')
    search_fields = ('id', 'job__id', 'job__user__email')
    ordering = ('-created_at',)
    readonly_fields = ('id', 'job', 'surviving_assets', 'pareto_solutions', 'strategies', 'weights', 'created_at')
