from django.urls import re_path 
from . import views

urlpatterns = [
    re_path(r'^apiTest/$',views.apiTest),
    re_path(r'^$',views.apiTest),
    re_path(r'^getImage/$', views.getImage),
    re_path(r'^getPath/$', views.getPath),
    # re_path(r'^cleanData/$', views.clean_files),
]