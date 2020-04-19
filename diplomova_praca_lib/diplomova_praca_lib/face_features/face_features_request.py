import collections
from enum import Enum

import numpy as np

from diplomova_praca_lib.face_features import clustering
from diplomova_praca_lib.face_features.map_features import SOM, RepresentativesTree
from diplomova_praca_lib.face_features.models import FaceCrop, FaceView, NoMoveError, Coords
from diplomova_praca_lib.storage import FileStorage, Database
from diplomova_praca_lib.utils import filename_without_extensions

database = Database(FileStorage.load_datafiles(r"C:\Users\janul\Desktop\saved_annotations\750_faces_2ndtry"))
# database = Database(FileStorage.load_datafiles(r"/mnt/c/Users/janul/Desktop/saved_annotations/750_faces_2ndtry"))

class Environment:
    features_info = []
    features = []
    som = None

    def __init__(self):
        i_feature = 0
        for path, all_faces_features in database.records:
            for crop, face_features in all_faces_features:
                Environment.features_info.append(FaceCrop(path, crop, i_feature))
                Environment.features.append(face_features)
                i_feature += 1


        Environment.som = SOM((100, 200), 128)
        Environment.som.train_som(Environment.features, epochs=150)


env = Environment()


class Action(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    IN = 5
    OUT = 6


curr_faceview = FaceView(*env.som.som_shape)


def map_movement(action: Action, faceview: FaceView, chosen_coords=None):
    if action == Action.NONE:
        return

    try:
        if action == Action.LEFT:
            faceview.move_left(2)
        if action == Action.RIGHT:
            faceview.move_right(2)
        if action == Action.DOWN:
            faceview.move_down(2)
        if action == Action.UP:
            faceview.move_up(2)
        if action == Action.IN:
            faceview.move_in(0.5, chosen_coords)
        if action == Action.OUT:
            faceview.move_out(0.5)
    except NoMoveError:
        return


FaceFeaturesResponse = collections.namedtuple("FaceFeaturesResponse", ["grid", "view"])

repr_tree = RepresentativesTree(env.som.representatives)

Response = collections.namedtuple("Response", ["images_grid", "layer"])
LayerInfo = collections.namedtuple("LayerInfo", ["layer_index", "top_left", "shape"])


def face_tree_request(curr_layer, chosen, action):
    if curr_layer == 0 and action is None:
        layer_info = LayerInfo(layer_index=0, top_left=Coords(0, 0), shape=repr_tree.layers[0].shape)
        return Response(images_grid=map_grid_to_infos(repr_tree.neighbourhood(0, repr_tree.center)), layer=layer_info)
    if action == Action.IN:
        new_center = repr_tree.element_position(curr_layer + 1, chosen)
        chosen_neighbourhood = repr_tree.neighbourhood(curr_layer + 1, new_center
        return Response(grid=map_grid_to_infos(chosen_neighbourhood), layer_ind=curr_layer + 1)
    raise ValueError


def map_grid_to_infos(grid: np.ndarray):
    infos = []
    for row in grid:
        row_items = []
        for item in row:
            row_items.append(Environment.features_info[item])
        infos.append(row_items)
    return infos


def face_features_request(action, view, selected_coords=None):
    # Check the original position
    if view == None:
        view = FaceView(*env.som.som_shape)

    true_selected_coords = None
    if selected_coords:
        new_x = view.width() * selected_coords.x
        new_y = view.height() * selected_coords.y

        true_selected_coords = Coords(x = round(new_x + view.top_left.x), y = round(new_y + view.top_left.y))

    map_movement(action, view, true_selected_coords)
    # som_shape = view.shape()
    repr = env.som.view_representatives(view)
    representatives_info = [Environment.features_info[int(i_feature)] for i_feature in repr.flatten()]
    result = [representatives_info[i_slice * repr.shape[1]:(i_slice + 1) * repr.shape[1]]
              for i_slice in range(repr.shape[0])]

    return FaceFeaturesResponse(result, view)

# chyba je ze request ti dava absolutnu poziciu vramci usera, nie ako to je vo view. takze treba prepocitaat



def face_features_request_old(request):
    clusters = clustering.group_clusters_to_lists(Environment.features,
                                                  clustering.split_to_n_clusters(Environment.features, 16))
    images_path = []
    for cluster in clusters:
        representative = clustering.find_representative(cluster)
        path, crop = find_source(representative)
        images_path.append((path, crop))
    return [(filename_without_extensions(path), crop) for path, crop in images_path]


def find_source(query_feature):
    for path, features in database.records:
        for crop, f in features:
            if np.array_equal(f, query_feature):
                return (path, crop)
