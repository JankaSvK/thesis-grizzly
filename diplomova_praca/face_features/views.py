import json

from django.http import JsonResponse
from django.shortcuts import render
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from diplomova_praca_lib.face_features.face_features_request import face_features_request, Action
from diplomova_praca_lib.face_features.models import FaceView, Coords
from diplomova_praca_lib.position_similarity.models import Crop
from shared.utils import thumbnail_path

THUMBNAILS_PATH = "images/lookup/thumbnails/"


@csrf_exempt
def index(request):
    context = {}
    return render(request, 'face_features/index.html', context)


actions = {None: Action.NONE, 'up': Action.UP, 'down': Action.DOWN, 'left': Action.LEFT, 'right': Action.RIGHT,
           'out': Action.OUT, 'in': Action.IN}

@csrf_exempt
def select_face_post(request):
    json_request = json.loads(request.POST['json_data'])
    print(json_request)

    action = actions[json_request['action']]
    if json_request['selected']:
        # Image is selected, find its coordinates
        print(json_request['selected'])
        i_row, i_column = map(int, json_request['selected'].split(','))
        selected_coords = Coords(i_column / json_request['display_size']['width'], i_row / json_request['display_size']['height'])
    else:
        selected_coords = None

    view = None
    if json_request['view']:
        view = FaceView(**json_request['view'])

    grid, new_view = face_features_request(action, view, selected_coords)
    return JsonResponse(as_context(grid, new_view), status=200)


def as_context(grid, view: FaceView):
    context = {}
    context['view'] = {"top_left_x": view.top_left.x, "top_left_y": view.top_left.y,
                       "bottom_right_x": view.bottom_right.x, "bottom_right_y": view.bottom_right.y,
                       "max_width": view.max_width, "max_height": view.max_height}

    context['table'] = []
    num_items = 0
    for i_row, row in enumerate(grid):
        row_items = []
        num_items += len(row)
        for i_column, item in enumerate(row):

            row_items.append({
                "img_src": thumbnail_path(item.src),
                "crop": {
                    "x": item.crop.left,
                    "y": item.crop.top,
                    "width": item.crop.width,
                    "height": item.crop.height,
                    "inset": crop_as_css_inset(item.crop)
                },
                "link": item.src,
                "i_row": i_row,
                "i_column": i_column
            })
        context['table'].append(row_items)

    if grid:
        context['view_width'] = len(grid[0])
    context['view_height'] = len(grid)
    context['view_count'] = num_items

    return context


# def crop_as_css_inset(crop: Crop):
#     # from top, from right, from bottom, from left
#     return "{}% {}% {}% {}%".format(
#         *map(lambda x: round(x * 100), [crop.top, 1 - crop.right, 1 - crop.bottom, crop.left]))


def crop_as_css_inset(crop: Crop):
    # from top, from right, from bottom, from left
    return "{}% {}% {}% {}%".format(
        *map(lambda x: round(x * 100), [crop.top, 1 - crop.right, 1 - crop.bottom, crop.left]))
