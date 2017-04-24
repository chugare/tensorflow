"""CNNWeb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url,include
from django.conf.urls.static import static
from . import settings
from django.contrib import admin
from .views import views
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url('^$', views.home, name='home'),
    url('^state', views.home, name='state'),
    url('^upload/', views.upload, name='upload'),
    url('^dataset/', views.dataset, name='dataset'),
    url('^train/(\d)*', views.train, name='train'),
    url('^run_train/', views.run_train, name='run_train'),

    url('^trainset/(\S)*', views.trainset, name='train_set'),
    url('^eval_single/', views.eval_single, name='eval_single'),
    url('^eval_batch/', views.eval_batch, name='eval_batch'),

]
urlpatterns+=static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
