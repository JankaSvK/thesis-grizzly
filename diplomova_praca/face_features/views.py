from django.http import JsonResponse
from django.shortcuts import render
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.face_features.face_features_request import fake_request
from diplomova_praca_lib.position_similarity.models import Crop
from shared.utils import thumbnail_path

THUMBNAILS_PATH = "images/lookup/thumbnails/"


@csrf_exempt
def index(request):
    context = {}
    return render(request, 'face_features/index.html', context)


@csrf_exempt
def select_face_post(request):
    if not request:
        pass

    return JsonResponse(as_context(fake_request(None)), status=200)


def as_context(results):
    context = {'table': []}
    for row in results:
        row_items = []
        for item in row:
            row_items.append({
                "img_src": thumbnail_path(item.src),
                # "crop": crop_as_css_inset(item.crop)
                "crop": {
                    "x": item.crop.left,
                    "y": item.crop.top,
                    "width": item.crop.width,
                    "height": item.crop.height,
                    "inset": crop_as_css_inset(item.crop)
                }
            })
        context['table'].append(row_items)
    return context


# def crop_as_css_inset(crop: Crop):
#     # from top, from right, from bottom, from left
#     return "{}% {}% {}% {}%".format(
#         *map(lambda x: round(x * 100), [crop.top, 1 - crop.right, 1 - crop.bottom, crop.left]))


def crop_as_css_inset(crop: Crop):
    # from top, from right, from bottom, from left
    return "{}% {}% {}% {}%".format(
        *map(lambda x: round(x * 100), [crop.top, 1 - crop.right, 1 - crop.bottom, crop.left]))
