from django.urls import path
from . import views

urlpatterns = [
    path('', views.asset_list, name='asset_list'),
    path('watchlist/', views.watchlist, name='watchlist'),
    path('watchlist/<uuid:pk>/', views.watchlist_delete, name='watchlist_delete'),
    path('<str:symbol>/prices/', views.price_history, name='price_history'),
]
