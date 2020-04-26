import face_features.views
import position_similarity.views
import shared.views

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
from django.urls import path

urlpatterns = [
    # path('admin/', admin.site.urls), # TODO

    path('', position_similarity.views.index),
    path('position_similarity/', position_similarity.views.position_similarity),
    path('face_features/', face_features.views.index),
    # path('face_features/post', face_features.views.select_face_post, name="select_face_post"),
    path('position_similarity/post', position_similarity.views.position_similarity_post,
         name="position_similarity_post"),
    path('face_features/post', face_features.views.repr_tree_post, name="repr_tree_post"),
    path('video_images/post', shared.views.video_images, name="video_images_post"),

]
