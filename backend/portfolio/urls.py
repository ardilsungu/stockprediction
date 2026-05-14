from django.urls import path
from . import views

urlpatterns = [
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<uuid:pk>/', views.job_detail, name='job_detail'),
    path('jobs/<uuid:pk>/delete/', views.job_delete, name='job_delete'),
]
