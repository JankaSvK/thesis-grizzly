import collections
from enum import Enum

import numpy as np

from diplomova_praca_lib.face_features.map_features import SOM, RepresentativesTree
from diplomova_praca_lib.face_features.models import FaceCrop, FaceView, NoMoveError
from diplomova_praca_lib.models import Serializable
from diplomova_praca_lib.storage import FileStorage, Database

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
                Environment.features_info.append(FaceCrop(path, crop))
                Environment.features.append(face_features)
                i_feature += 1


        Environment.som = SOM((100, 200), 128)
        Environment.som.train_som(Environment.features, epochs=22)


env = Environment()


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


Response = collections.namedtuple("Response", ["images_grid", "layer"])

LayerInfo = collections.namedtuple("LayerInfo", ["layer_index", "top_left", "shape"])


class TreeView(Serializable):
    raw_init_params = ['level', 'left', 'top']
    serializable_slots = {}
    repr_tree = RepresentativesTree(env.som.representatives)

    def __init__(self, **kwargs):
        self._level = 0
        self._top = 0
        self._left = 0
        super().__init__(**kwargs)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, level):
        self._level = min(max(0, level), len(self.repr_tree.layers))

    @property
    def top(self):
        return self._top

    @top.setter
    def top(self, top):
        max_position_down = max(0, self.repr_tree.layers[self.level].shape[0] - self.display_size[0] - 1)
        self._top = min(max(0, top), max_position_down)

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, left):
        max_position_left = max(0, self.repr_tree.layers[self.level].shape[1] - self.display_size[1] - 1)
        self._left = min(max(0, left), max_position_left)

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
            max_position_right = self.repr_tree.layers[self.level].shape[1] - self.display_size[1] - 1
            return self.left < max_position_right
        elif action == Action.DOWN:
            max_position_down = self.repr_tree.layers[self.level].shape[0] - self.display_size[0] - 1
            return self.top < max_position_down
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



def map_grid_to_infos(grid: np.ndarray):
    infos = []
    for row in grid:
        row_items = []
        for item in row:
            row_items.append(Environment.features_info[item])
        infos.append(row_items)
    return infos

