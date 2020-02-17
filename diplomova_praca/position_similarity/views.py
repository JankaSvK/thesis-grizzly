import logging

from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render
import json

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.position_similarity.models import UrlImage, PositionSimilarityRequest, Crop
from diplomova_praca_lib.position_similarity.position_similarity_request import position_similarity_request, \
    spatial_similarity_request


def index(request):
    return HttpResponseRedirect("position_similarity/")


@csrf_exempt
def position_similarity(request):
    return render(request, 'position_similarity/index.html', {})


@csrf_exempt
def position_similarity_post(request):
    logging.info("Position similarity request.")

    THUMBNAILS_PATH = "/static/images/lookup/thumbnails/"

    json_request = json.loads(request.POST['json_data'])
    images = json_request['images']
    method = json_request['method']


    if method == 'regions':
        gallery_ids = position_similarity_request(json_to_position_similarity_request(images))
    elif method == 'spatially':
        gallery_ids = spatial_similarity_request(json_to_position_similarity_request(images))
    else:
        raise ValueError("Unknown method.")

    gallery_ids = gallery_ids[:100]

    context = {
        "ranking_results": [{"img_src": "%s/%s.jpg" % (THUMBNAILS_PATH, id)} for id in gallery_ids],
    }

    return JsonResponse(context, status=200)


def json_to_position_similarity_request(json_data):
    images = []
    for image in json_data:
        url_image = UrlImage(image["url"], Crop(*[image[attr] for attr in ["top", "left", "width", "height"]]))
        images.append(url_image)
    return PositionSimilarityRequest(images)
