from django.urls import path
from . import views

urlpatterns = [
    path('', views.inicio, name='index'),
    path('chat', views.chat, name='chat'),
    path('nosotros', views.nosotros, name='nosotros'),
    path('proyectos', views.proyectos, name='proyectos'),
    path('vozToTexto', views.vozToTexto, name='vozToTexto'),
    path('nuevo', views.nuevo, name='nuevo'),
    path('meForms', views.meForms, name='meForms'),
]