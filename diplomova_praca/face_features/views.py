import collections

from django.shortcuts import render
from django.templatetags.static import static

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.face_features.face_features_request import face_features_request
from diplomova_praca_lib.position_similarity.models import Crop

THUMBNAILS_PATH = "images/lookup/thumbnails/"


@csrf_exempt
def index(request):
    Image = collections.namedtuple("Image", ['src', 'link'])

    url = static('images/lookup/00001040.jpg')

    context = {"images": [Image(url, "00001040")]}
    return render(request, 'face_features/index.html', context)


def selectImage(request, image_id):
    gallery_ids = face_features_request(request)
    print(gallery_ids)
    context = {
        "images": [
            {"img_src": static("%s/%s.jpg" % (THUMBNAILS_PATH, id)), "link": "/face_features/" + id, "crop": crop_as_css_inset(crop)}
            for id, crop in gallery_ids],
    }
    return render(request, 'face_features/index.html', context)


def crop_as_css_inset(crop: Crop):
    # from top, from right, from bottom, from left
    return "{}% {}% {}% {}%".format(
        *map(lambda x: round(x * 100), [crop.top, 1 - crop.right, 1 - crop.bottom, crop.left]))
