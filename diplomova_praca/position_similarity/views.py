import json
import logging
import os
import random
from pathlib import Path

from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.position_similarity.models import UrlImage, PositionSimilarityRequest, Crop
from diplomova_praca_lib.position_similarity.position_similarity_request import position_similarity_request, \
    spatial_similarity_request
from diplomova_praca_lib.utils import Memoize
from .models import PositionRequest

# STATIC_DIR = settings.STATICFILES_DIRS[0]
THUMBNAILS_PATH = os.path.join("static", "images", "lookup", "thumbnails")

@csrf_exempt
def index(request):
    return HttpResponseRedirect("position_similarity/")


@csrf_exempt
def position_similarity(request):
    context = {"search_image": random_image_path().as_posix()}
    return render(request, 'position_similarity/index.html', context)


@csrf_exempt
def position_similarity_post(request):
    save_request = PositionRequest()
    logging.info("Position similarity request.")

    json_request = json.loads(request.POST['json_data'])
    save_request.json_request = json_request
    images, method = json_request['images'], json_request['method']

    if method == 'regions':
        closest_images = position_similarity_request(json_to_position_similarity_request(images))
    elif method == 'spatially':
        closest_images = spatial_similarity_request(json_to_position_similarity_request(images))
    else:
        raise ValueError("Unknown method.")

    save_request.response = ",".join(closest_images)

    images_to_render = closest_images[:100]
    context = {
        "ranking_results": [{"img_src": str(Path("/") / Path(THUMBNAILS_PATH, path).as_posix())} for path in
                            images_to_render],
    }

    save_request.save()
    return JsonResponse(context, status=200)


def json_to_position_similarity_request(json_data):
    images = []
    for image in json_data:
        url_image = UrlImage(image["url"], Crop(*[image[attr] for attr in ["top", "left", "width", "height"]]))
        images.append(url_image)
    return PositionSimilarityRequest(images)


@Memoize
def available_images():
    """
    Traverse and searches for all images under `THUMBNAILS_PATH`
    :return: List of images as relative path to `THUMBNAILS_PATH`
    """
    return [path for path in Path(THUMBNAILS_PATH).rglob('*.jpg')]


def random_image_path():
    return Path("/", random.choice(available_images()))
