from django.urls import path

from . import views

app_name = 'forecast'

urlpatterns = [
    path('jobs/', views.ForecastJobCreateView.as_view(), name='job_create'),
    path('jobs/<uuid:pk>/', views.ForecastJobDetailView.as_view(), name='job_detail'),
]
