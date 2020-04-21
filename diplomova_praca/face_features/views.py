import json

from django.http import JsonResponse
from django.shortcuts import render
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.face_features.face_features_request import face_features_request, Action, TreeView
from diplomova_praca_lib.face_features.models import FaceView, Coords
from diplomova_praca_lib.position_similarity.models import Crop
from shared.utils import thumbnail_path

THUMBNAILS_PATH = "images/lookup/thumbnails/"


@csrf_exempt
def index(request):
    context = {}
    return render(request, 'face_features/index.html', context)


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
                "crop": crop_preprocess(item.crop),
                "x": i_item,
                "y": i_row
            })
        images_grid_preprocessed.append(row_items)
    return images_grid_preprocessed

#
# @csrf_exempt
# def select_face_post(request):
#     json_request = json.loads(request.POST['json_data'])
#     print(json_request)
#
#     action = actions[json_request['action']]
#     if json_request['selected']:
#         # Image is selected, find its coordinates
#         print(json_request['selected'])
#         i_row, i_column = map(int, json_request['selected'].split(','))
#         selected_coords = Coords(i_column / json_request['display_size']['width'], i_row / json_request['display_size']['height'])
#     else:
#         selected_coords = None
#
#     view = None
#     if json_request['view']:
#         view = FaceView(**json_request['view'])
#
#     grid, new_view = face_features_request(action, view, selected_coords)
#     return JsonResponse(as_context(grid, new_view), status=200)
#
#
# def as_context(grid, view: FaceView, center_position = (0,0)):
#     context = {}
#     context['view'] = {"top_left_x": view.top_left.x, "top_left_y": view.top_left.y,
#                        "bottom_right_x": view.bottom_right.x, "bottom_right_y": view.bottom_right.y,
#                        "max_width": view.max_width, "max_height": view.max_height}
#
#     context['table'] = []
#     num_items = 0
#     for i_row, row in enumerate(grid):
#         row_items = []
#         num_items += len(row)
#         for i_column, item in enumerate(row):
#
#             row_items.append({
#                 "img_src": thumbnail_path(item.src),
#                 "crop": {
#                     "x": item.crop.left,
#                     "y": item.crop.top,
#                     "width": item.crop.width,
#                     "height": item.crop.height,
#                     "inset": crop_as_css_inset(item.crop)
#                 },
#                 "feature_id": item.index,
#                 "link": item.src,
#                 "i_row": i_row,
#                 "i_column": i_column,
#                 "position": {
#                     "x": None,
#                     "y": None
#                 }
#             })
#         context['table'].append(row_items)
#
#     if grid:
#         context['view_width'] = len(grid[0])
#     context['view_height'] = len(grid)
#     context['view_count'] = num_items
#
#     return context
#
#
# # def crop_as_css_inset(crop: Crop):
# #     # from top, from right, from bottom, from left
# #     return "{}% {}% {}% {}%".format(
# #         *map(lambda x: round(x * 100), [crop.top, 1 - crop.right, 1 - crop.bottom, crop.left]))
#
#
# def crop_as_css_inset(crop: Crop):
#     # from top, from right, from bottom, from left
#     return "{}% {}% {}% {}%".format(
#         *map(lambda x: round(x * 100), [crop.top, 1 - crop.right, 1 - crop.bottom, crop.left]))
