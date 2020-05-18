import json
import logging

from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.position_similarity.models import UrlImage, PositionSimilarityRequest, Crop, \
    PositionSimilarityResponse
from diplomova_praca_lib.position_similarity.position_similarity_request import position_similarity_request, \
    spatial_similarity_request
from diplomova_praca_lib.utils import images_with_position_from_json, path_from_css_background
from shared.utils import random_image_path, thumbnail_path
from .models import PositionRequest, Collage


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
    images, method, overlay_image = json_request['images'], json_request['method'], json_request['overlay_image']

    if method == 'regions':
        request = PositionSimilarityRequest(images=images_with_position_from_json(images),
                                            query_image=path_from_css_background(overlay_image))
        response = position_similarity_request(request)
        closest_images = response.ranked_paths
    elif method == 'spatially':
        request = PositionSimilarityRequest()
        response = PositionSimilarityResponse()

        closest_images = spatial_similarity_request(images_with_position_from_json(images))
    else:
        raise ValueError("Unknown method.")

    save_request.response = ",".join(closest_images)

    images_to_render = closest_images[:100]
    context = {
        "ranking_results": [{"img_src": thumbnail_path(path)} for path in images_to_render],
        "search_image_rank": response.searched_image_rank
    }

    save_request.save()
    return JsonResponse(context, status=200)



@csrf_exempt
def position_similarity_submit_collage(request):
    json_data = json.loads(request.POST['json_data'])

    collage = Collage()
    collage.overlay_image = json_data['overlay_image']
    collage.images = json_data['images']
    collage.save()

    return JsonResponse({}, status=200)

