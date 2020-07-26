from django.http import JsonResponse, HttpResponseNotFound
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.face_features.face_features_request import TreeView, images_with_closest_faces, Environment
from diplomova_praca_lib.face_features.models import ClosestFacesRequest
from diplomova_praca_lib.position_similarity.models import Crop
from face_features.models import FaceFeaturesSubmission
from shared.utils import thumbnail_path, random_image_path, random_subset_image_path

@csrf_exempt
def index(request):
    subset_images_available = Environment.available_images()

    if subset_images_available is None:
        query = random_image_path()
    else:
        query = random_subset_image_path(subset_images_available)

    if query is None:
        return HttpResponseNotFound("Could not load any images.")

    context = {"search_image": query.as_posix()}
    return render(request, 'face_features/index.html', context)


@csrf_exempt
def submit(request):
    record = FaceFeaturesSubmission()
    record.request = request.POST.get('request', '')
    record.selected = request.POST.get('selected', '')
    record.num_hints = request.POST.get('num_hints', '')
    record.save()
    return JsonResponse({}, status=200)


@csrf_exempt
def images_with_closest_faces_post(request):
    face_id = int(request.POST.get('id', None))
    request = ClosestFacesRequest(face_id=face_id)
    response = images_with_closest_faces(request)

    context = {
        "images": [{"img_src": thumbnail_path(face_crop.src)} for face_crop in response.closest_faces],
    }

    return JsonResponse(context, status=200)

@csrf_exempt
def repr_tree_post(request):
    json_request = request.POST.dict()
    print(json_request)

    tree_view = json_request.pop('tree_view', None)
    if tree_view is None:
        tree_view = TreeView(left=0, top=0, level=0)
    else:
        tree_view = TreeView.deserialize(tree_view)
        tree_view.act(**json_request)

    frontend_response = {
        "images_grid": preprocess_image_grid(tree_view.image_grid()),
        "tree_view": tree_view.serialize()
    }

    return JsonResponse(frontend_response, status=200)


def preprocess_layer_info(layer_info):
    return {
        "top_left": (layer_info.top_left.y, layer_info.top_left.x),
        "shape": layer_info.shape,
        "layer_index": layer_info.layer_index
    }


def preprocess_image_grid(images_grid):
    def crop_preprocess(crop: Crop):
        return {"x": crop.left,
                "y": crop.top,
                "width": crop.width,
                "height": crop.height, }

    def path_preprocess(path):
        return thumbnail_path(path)

    images_grid_preprocessed = []
    for i_row, row in enumerate(images_grid):
        row_items = []
        for i_item, item in enumerate(row):
            row_items.append({
                "img_src": path_preprocess(item.src),
                "crop": crop_preprocess(size_up(item.crop)),
                "x": i_item,
                "y": i_row,
                "face_index": item.idx
            })
        images_grid_preprocessed.append(row_items)
    return images_grid_preprocessed


def size_up(crop: Crop):
    size_up_c = 0.01
    return Crop(left=max(0, crop.left - size_up_c), top=max(0, crop.top - size_up_c),
                right=min(1, crop.right + size_up_c),
                bottom=min(1, crop.bottom + size_up_c))