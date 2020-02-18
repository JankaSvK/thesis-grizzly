import position_similarity.views
import face_features.views

from django.http import HttpResponseRedirect

"""diplomova_praca URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

urlpatterns = [
    # path('admin/', admin.site.urls), # TODO

    # Position Similarity
    path('', position_similarity.views.index),
    path('position_similarity/', position_similarity.views.position_similarity),
    path('position_similarity/post', position_similarity.views.position_similarity_post,
         name="position_similarity_post"),

    # Face Features
    path('face_features/', face_features.views.face_features),
]
