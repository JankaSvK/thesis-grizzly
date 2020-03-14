import json
import logging
import os
from pathlib import Path
from random import randrange

from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.templatetags.static import static
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.position_similarity.models import UrlImage, PositionSimilarityRequest, Crop
from diplomova_praca_lib.position_similarity.position_similarity_request import position_similarity_request, \
    spatial_similarity_request
from .models import PositionRequest

# STATIC_DIR = settings.STATICFILES_DIRS[0]
THUMBNAILS_PATH = os.path.join("static", "images", "lookup", "thumbnails")

@csrf_exempt
def index(request):
    return HttpResponseRedirect("position_similarity/")


@csrf_exempt
def position_similarity(request):
    random_image_path = random_image_path()
    context = {"search_image": random_image_path.as_posix()}
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


class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        # Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]


@Memoize
def available_images():
    available_images = [Path(THUMBNAILS_PATH, str(path)[str(path).index(THUMBNAILS_PATH) + len(THUMBNAILS_PATH) + 1:])
                        for path in Path(THUMBNAILS_PATH).rglob('*.jpg')]

    return available_images


def random_image_path():
    images = available_images()
    query_id = randrange(len(images))
    return Path("/", images[query_id])

