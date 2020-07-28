import json
import logging

from django.http import HttpResponseRedirect, JsonResponse, HttpResponseNotFound
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest, PositionMethod
from diplomova_praca_lib.position_similarity.position_similarity_request import positional_request, available_images, \
    initialize_env
from diplomova_praca_lib.utils import images_with_position_from_json, path_from_css_background
from shared.utils import random_image_path, thumbnail_path, random_subset_image_path, THUMBNAILS_PATH
from .models import PositionRequest, Collage


@csrf_exempt
def index(request):
    return HttpResponseRedirect("position_similarity/")


@csrf_exempt
def position_similarity(request):
    default_method = PositionMethod.REGIONS
    initialize_env('regions')

    subset_images_available = available_images(default_method)

    if not subset_images_available:
        query = random_image_path()
    else:
        query = random_subset_image_path(subset_images_available)

    if query is None:
        return HttpResponseNotFound("Could not load any images.")

    context = {"search_image": query.as_posix()}
    return render(request, 'position_similarity/index.html', context)


@csrf_exempt
def position_similarity_post(request):
    save_request = PositionRequest()
    logging.info("Position similarity request.")

    json_request = json.loads(request.POST['json_data'])
    save_request.json_request = json_request
    images, method, overlay_image = json_request['images'], json_request['method'], json_request['overlay_image']

    request = PositionSimilarityRequest(images=images_with_position_from_json(images),
                                        query_image=path_from_css_background(overlay_image, THUMBNAILS_PATH),
                                        method=PositionMethod.parse(method))
    response = positional_request(request)
    save_request.response = ",".join(response.ranked_paths)

    images_to_render = response.ranked_paths[:100]
    context = {
        "ranking_results": [{"img_src": thumbnail_path(path)} for path in images_to_render],
        "search_image_rank": response.searched_image_rank + 1,
    }

    if response.matched_regions:
        context['matched_regions'] =  transform_crops_to_rectangles(response.matched_regions, images_to_render)

    save_request.save()
    return JsonResponse(context, status=200)


def transform_crops_to_rectangles(matched_regions, images_to_render):
    return {thumbnail_path(image): list(map(lambda x: x.as_quadruple(), regions)) for image, regions in
            matched_regions.items() if image in images_to_render}


@csrf_exempt
def position_similarity_submit_collage(request):
    json_data = json.loads(request.POST['json_data'])

    collage = Collage()
    collage.overlay_image = json_data['overlay_image']
    collage.images = json_data['images']
    collage.save()

    return JsonResponse({}, status=200)

