from django.contrib import admin
from django.urls import path

from .views import index, time_domain, wavelet

urlpatterns = [
    path('', index, name='index'),
    path('time_domain/', time_domain, name='time_domain'),
    path('wavelet/', wavelet, name='wavelet'),
    path('admin/', admin.site.urls),
]