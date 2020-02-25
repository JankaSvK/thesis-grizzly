import collections

from django.shortcuts import render
from django.templatetags.static import static

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.face_features.face_features_request import face_features_request


@csrf_exempt
def index(request):
    Image = collections.namedtuple("Image", ['src', 'link'])

    url = static('images/lookup/00001040.jpg')

    context = {"images": [Image(url, "00001040")]}
    return render(request, 'face_features/index.html', context)


def selectImage(request, image_id):
    response = face_features_request(request)
    context = {}
    return render(request, 'face_features/index.html', context)