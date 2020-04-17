import numpy as np
import sklearn
import typing
from minisom import MiniSom

from diplomova_praca_lib.face_features.models import FaceView


class SOM:
    def __init__(self, som_shape=(6, 6), num_features=128):
        self.som_shape = som_shape
        self.num_features = num_features
        self.som = MiniSom(*som_shape, num_features, sigma=0.3, learning_rate=0.5)
        self.som_size = self.som_shape[0] * self.som_shape[1]
        self.display_size = (10, 20)  # rows, columns
        self.display_count = self.display_size[0] * self.display_size[1]


    def train_som(self, features, epochs=10000):
        self.som.train_random(features, epochs, verbose=True)
        self.representatives = self.closest_representatives(self.som.get_weights().reshape(-1, self.num_features),
                                                            features)
        self.representatives = self.representatives.reshape(self.som_shape)

    @staticmethod
    def closest_representatives(weights, features):
        """
        For each weight in weights find closest feature out of features (returns its index).
        """
        return np.argmin(sklearn.metrics.pairwise.cosine_distances(weights, features), axis=1)


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