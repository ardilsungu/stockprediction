from django.contrib import admin
from .models import User, AuditLog


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ('email', 'username', 'is_active', 'is_staff', 'created_at')
    list_filter = ('is_active', 'is_staff', 'created_at')
    search_fields = ('email', 'username')
    ordering = ('-created_at',)
    readonly_fields = ('id', 'created_at', 'updated_at', 'last_login', 'password')


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ('action', 'user', 'ip_address', 'timestamp')
    list_filter = ('action', 'timestamp')
    search_fields = ('user__email', 'action', 'ip_address')
    ordering = ('-timestamp',)
    readonly_fields = ('user', 'action', 'detail', 'ip_address', 'timestamp')
