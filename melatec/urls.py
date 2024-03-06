from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_view, name='predict_view'),
    path('convert/', views.youtube_to_melody, name='youtube_to_melody'),
]