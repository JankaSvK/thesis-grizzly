import typing
from pathlib import Path

import numpy as np
from sklearn import metrics
from minisom import MiniSom

from diplomova_praca_lib.face_features.models import FaceView
from diplomova_praca_lib.utils import timestamp_directory, dump_to_file


class RepresentativesTree:
    FACTOR = 2

    def __init__(self, representatives: np.ndarray, display_size=(11, 21)):
        self.representatives = representatives
        self.layers = [representatives] # Smallest has index 0
        self.display_size = display_size
        self.center = self.display_size[0] // 2, self.display_size[1] // 2
        self.build_tree(self.FACTOR)

    def build_tree(self, factor):
        assert factor >= 1
        assert factor == self.FACTOR
        previous_layer = self.representatives
        while any((a > b for a, b in zip(previous_layer.shape, self.display_size))):
            layer = previous_layer[::factor, ::factor]
            self.layers.append(layer)
            previous_layer = layer

        self.layers.reverse()


    def element(self, layer, element_position):
        return self.layers[layer][element_position]

    def element_position(self, layer_ind, center):
        return np.argwhere(self.layers[layer_ind] == center)

    def topleft_view(self, level, top_left):
        return self.layers[level][top_left[0]: top_left[0] + self.display_size[0],
               top_left[1]: top_left[1] + self.display_size[1]]

    def neighbourhood(self, layer_ind: int, center_position: np.ndarray):
        return self.layers[layer_ind][
               center_position[0] - self.display_size[0] // 2:center_position[0] + self.display_size[0] // 2,
               center_position[1] - self.display_size[1] // 2:center_position[1] + self.display_size[1] // 2]

    def top_left(self, center_position: np.ndarray):
        return max(0, center_position[0] - self.display_size[0] // 2), max(0, center_position[1] - self.display_size[
            1] // 2)


class SOM:
    log_dir = Path(r"C:\Users\janul\Desktop\thesis_tmp_files\som")

    def __init__(self, som_shape=(6, 6), num_features=128):
        self.som_shape = som_shape
        self.num_features = num_features
        self.som = MiniSom(*som_shape, num_features, sigma=0.3, learning_rate=0.5)
        self.som_size = self.som_shape[0] * self.som_shape[1]

    def train_som(self, features, epochs=None, save_som=True):
        if epochs is None:
            epochs = len(features) * 30 # Each sample is on average provided 30 times

        self.som.train_random(features, epochs, verbose=True)

        if save_som and self.log_dir:
            som_log_file = Path(timestamp_directory(self.log_dir), "som.pickle")
            print("SOM saved in", som_log_file)
            dump_to_file(som_log_file, self.som)

    def set_representatives(self, features):
        self.representatives = self.closest_representatives(self.som.get_weights().reshape(-1, self.num_features),
                                                            features)
        self.representatives = self.representatives.reshape(self.som_shape)

    @staticmethod
    def closest_representatives(weights, features):
        """
        For each weight in weights find closest feature out of features (returns its index).
        """
        return np.argmin(metrics.pairwise.euclidean_distances(weights, features), axis=1)


    @staticmethod
    def sample_grid(som_shape: typing.Tuple[int, int], factor:int=2):
        """
        Returns an IDs of elements chosen to be displayed
        :param som_shape: Original SOM shape, i.e. from which it is sampled
        :param factor: Keep every n-th element (applied to both axis)
        :return: Indexes of chosed elements out of original (starting from 0 top left, 1,2,3,... in row)
        """
        assert factor >= 1
        factor = int(factor)
        step_row, step_col = factor, factor
        max_row, max_col = map(int, som_shape)
        selected_idxs = [row * max_col + col for row in range(0, max_row, step_row) for col in range(0, max_col, step_col)]
        return ((max_row - 1) // step_row + 1, (max_col - 1) // step_col + 1), selected_idxs


    def view_representatives(self, view: FaceView):
        # Fits into a display
        if view.width() * view.height() <= self.display_count:
            # All weights, not sampled
            return self.representatives[view.top_left.y:view.bottom_right.y, view.top_left.x:view.bottom_right.x]

        # Sampling is needed
        factor_to_fit = max(view.width() // self.display_size[1], view.height() // self.display_size[0])

        sampled_grid_shape, sampled_grid_idx = self.sample_grid(view.shape(), factor=factor_to_fit)
        repr = np.zeros(sampled_grid_shape)
        i_row, i_col = 0,0
        for chosen in sampled_grid_idx:
            chosen_y, chosen_x = np.unravel_index(chosen, view.shape())
            repr[i_row, i_col] = self.representatives[chosen_y + view.top_left.y, chosen_x + view.top_left.x]
            i_col += 1
            if i_col >= sampled_grid_shape[1]:
                i_col = 0
                i_row += 1


        return repr