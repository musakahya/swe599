from django.contrib import admin
from django.urls import path

from .views import index, time_domain, wavelet, cosine

urlpatterns = [
    path('', index, name='index'),
    path('time_domain/', time_domain, name='time_domain'),
    path('wavelet/', wavelet, name='wavelet'),
    path('cosine/', cosine, name='cosine'),
    path('admin/', admin.site.urls),
]