import collections
import pickle
from bisect import bisect_left
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import euclidean_distances

from diplomova_praca_lib.face_features.map_features import SOM, RepresentativesTree
from diplomova_praca_lib.face_features.models import FaceView, NoMoveError, FaceCrop, ClosestFacesRequest, \
    ClosestFacesResponse
from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import load_from_file, closest_match, Serializable


class Environment:
    features_info = []
    features = []
    som = None
    use_random_grid = False
    initialized = False

    def __init__(self, data_pathm, som_path):
        data = FileStorage.load_multiple_files_multiple_keys(path=data_path, retrieve_merged=['features', 'crops', 'paths'])

        if not data:
            print("Data for faces could not be obtined.")
            return

        Environment.features = data['features']
        Environment.paths = data['paths']
        Environment.crops = data['crops']

        Environment.features_info = []
        for i_crop, (path, crop) in enumerate(zip(Environment.paths, Environment.crops)):
            Environment.features_info.append(FaceCrop(src=path, crop=crop, idx=i_crop))

        self.som = SOM((50, 50), 128)

        if not Path(som_path).exists():
            print("Underlying SOM data not found.")
            return

        self.som.som = load_from_file(som_path)
        self.som.set_representatives(Environment.features)

        if self.use_random_grid:
            # self.som.representatives = []
            max_display_width = 20
            random_grid = np.arange(len(self.features))
            if len(random_grid) % max_display_width:
                suffix = np.ones(max_display_width - len(random_grid) % max_display_width, dtype=np.int32) * random_grid[-1]
                random_grid = np.concatenate([random_grid, suffix])
            self.som.representatives = random_grid.reshape(-1, max_display_width)

        self.initialized = True

        # self.som = load_from_file(r"C:\Users\janul\Desktop\thesis_tmp_files\som\2020-05-25_12-41-30_PM\som.pickle")

    def train_som(self, shape, epochs):
        Environment.som = SOM(shape, 128)
        Environment.som.train_som(Environment.features, epochs=epochs)

    def load_som(self, path):
        with open(path, 'rb') as handle:
            Environment.som = pickle.load(handle)


# database = Database(FileStorage.load_datafiles(r"C:\Users\janul\Desktop\saved_annotations\750_faces"))
# env = Environment(r"C:\Users\janul\Desktop\thesis_tmp_files\transformed_face_features")
# env = Environment(r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_only_bigger_10percent_316videos")
# som_path = r"C:\Users\janul\Desktop\thesis_tmp_files\cosine_som\euclidean\200k-original\som-euclidean,200000-200000.pickle"

env = None
class Action(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    IN = 5
    OUT = 6


actions = {None: Action.NONE, 'up': Action.UP, 'down': Action.DOWN, 'left': Action.LEFT, 'right': Action.RIGHT,
           'out': Action.OUT, 'in': Action.IN}


# curr_faceview = FaceView(*env.som.som_shape)


def images_with_closest_faces(request: ClosestFacesRequest) -> ClosestFacesResponse:
    query_features = Environment.features[request.face_id]
    matches_sorted, distances = closest_match(query_features, Environment.features, distance=euclidean_distances)
    last_matched_image_idx = bisect_left(distances, 0.6)

    response = ClosestFacesResponse()
    response.closest_faces = [Environment.features_info[i] for i in matches_sorted[:last_matched_image_idx]]
    response.distances = distances[:last_matched_image_idx]

    return response


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
Response = collections.namedtuple("Response", ["images_grid", "layer"])
LayerInfo = collections.namedtuple("LayerInfo", ["layer_index", "top_left", "shape"])


class TreeView(Serializable):
    raw_init_params = ['level', 'left', 'top']
    serializable_slots = {}
    repr_tree = None

    def __init__(self, **kwargs):
        self._level = 0
        self._top = 0
        self._left = 0

        if self.__class__.repr_tree is None:
            self.__class__.repr_tree = RepresentativesTree(env.som.representatives)

        super().__init__(**kwargs)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = min(max(0, level), len(self.repr_tree.layers))

    def maximal_position(self, axis, level):
        """Prevents using only a part of the display by scrolling too far."""
        if self.repr_tree.layers[level].shape[axis] < self.display_size[axis]:
            return 0  # Smaller view is returned

        return self.repr_tree.layers[level].shape[axis] - self.display_size[axis]

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, top):
        max_position = self.maximal_position(axis=0, level=self.level)
        self._top = min(max(0, top), max_position)

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, left):
        max_position = self.maximal_position(axis=1, level=self.level)
        self._left = min(max(0, left), max_position)

    @property
    def display_size(self):
        return self.repr_tree.display_size

    def image_grid(self):
        chosen_view = self.repr_tree.topleft_view(self.level, (self.top, self.left))
        return map_grid_to_infos(chosen_view)

    def act(self, action, **kwargs):
        action = actions[action]
        if action == Action.IN:
            self.zoom_in(**kwargs)
        elif action == Action.OUT:
            self.zoom_out()
        else:
            self.move_direction(action)

    def move_direction(self, action):
        if not self.can_go(action):
            return

        if action == Action.LEFT:
            self.left -= 1
        elif action == Action.RIGHT:
            self.left += 1
        elif action == Action.DOWN:
            self.top += 1
        elif action == Action.UP:
            self.top -= 1
        else:
            raise ValueError("Not supported direction for move.")

    def can_go(self, action):
        if action == Action.LEFT:
            return self.left > 0
        elif action == Action.RIGHT:
            return self.left < self.maximal_position(axis=1, level=self.level)
        elif action == Action.DOWN:
            return self.top < self.maximal_position(axis=0, level=self.level)
        elif action == Action.UP:
            return self.top > 0
        elif action == Action.OUT:
            return self.level > 0
        elif action == Action.IN:
            return self.level < len(self.repr_tree.layers) - 1
        else:
            raise ValueError("Not supported action for checking.")

    def zoom_in(self, x, y):
        print(x, y)
        x, y = int(x), int(y)
        if not self.can_go(Action.IN):
            return

        self.level += 1
        self.top = (self.top + y) * self.repr_tree.FACTOR - self.display_size[0] // 2
        self.left = (self.left + x) * self.repr_tree.FACTOR - self.display_size[1] // 2


    def zoom_out(self):
        if not self.can_go(Action.OUT):
            return

        self.level -= 1
        self.top = round(
            self.top / self.repr_tree.FACTOR
            - self.display_size[0] / 2
            + self.display_size[0] / self.repr_tree.FACTOR / 2
        )
        self.left = round(
            self.left / self.repr_tree.FACTOR
            - self.display_size[1] / 2
            + self.display_size[1] / self.repr_tree.FACTOR / 2
        )



def map_grid_to_infos(grid: np.ndarray) -> List[List[FaceCrop]]:
    infos = []
    for row in grid:
        row_items = []
        for item in row:
            row_items.append(Environment.features_info[item])
        infos.append(row_items)
    return infos

