import logging
import os
from random import randrange

from django.conf import settings
from django.contrib.staticfiles.storage import StaticFilesStorage
from django.contrib.staticfiles.utils import get_files
from django.core.files.storage import default_storage
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render
import json
from django.templatetags.static import static

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.position_similarity.models import UrlImage, PositionSimilarityRequest, Crop
from diplomova_praca_lib.position_similarity.position_similarity_request import position_similarity_request, \
    spatial_similarity_request

from .models import PositionRequest

THUMBNAILS_PATH = "images/lookup/thumbnails/"

@csrf_exempt
def index(request):
    return HttpResponseRedirect("position_similarity/")


@csrf_exempt
def position_similarity(request):
    context = {"search_image": get_random_image()}
    return render(request, 'position_similarity/index.html', context)


@csrf_exempt
def position_similarity_post(request):
    save_request = PositionRequest()
    logging.info("Position similarity request.")

    json_request = json.loads(request.POST['json_data'])
    save_request.json_request = json_request
    images, method = json_request['images'], json_request['method']

    if method == 'regions':
        gallery_ids = position_similarity_request(json_to_position_similarity_request(images))
    elif method == 'spatially':
        gallery_ids = spatial_similarity_request(json_to_position_similarity_request(images))
    else:
        raise ValueError("Unknown method.")

    save_request.response = ",".join(gallery_ids)
    gallery_ids = gallery_ids[:100]

    context = {
        "ranking_results": [{"img_src": static("%s/%s.jpg" % (THUMBNAILS_PATH, id))} for id in gallery_ids],
    }

    save_request.save()
    return JsonResponse(context, status=200)


def json_to_position_similarity_request(json_data):
    images = []
    for image in json_data:
        url_image = UrlImage(image["url"], Crop(*[image[attr] for attr in ["top", "left", "width", "height"]]))
        images.append(url_image)
    return PositionSimilarityRequest(images)



def random_image_id():
    import rstr
    return rstr.xeger(r'0[0-9]{7}')

def get_random_image():
    return static(os.path.join(THUMBNAILS_PATH, f"{randrange(101745):08d}.jpg")) # TODO
