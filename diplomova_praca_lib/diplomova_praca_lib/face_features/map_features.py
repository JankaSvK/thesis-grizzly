import sklearn
from minisom import MiniSom
import numpy as np
from sklearn.neighbors.kd_tree import KDTree


class SOM:
    def __init__(self, som_shape=(6, 6), num_features=128):
        self.som_shape = som_shape
        self.num_features = num_features
        self.som = MiniSom(*som_shape, num_features, sigma=0.3, learning_rate=0.5)
        self.som_size = self.som_shape[0] * self.som_shape[1]
        self.som_weights = None

    def train_som(self, features, epochs=10000):
        self.som.train_random(features, epochs)

    def closest_representatives(self, dataset_features):
        """
        Returns a dictionary with every neuron position and corresponding index of closest feature vector from dataset.
        """
        representatives_features = {}
        representatives_idxs = {}
        som_weights = self.som.get_weights()

        for neuron_id in range(self.som_size):
            position = np.unravel_index(neuron_id, self.som_shape)
            i, j = position
            weights = som_weights[i][j]
            for i_features, features in enumerate(dataset_features):
                if i_features == 0:
                    representatives_features[position] = features
                    representatives_idxs[position] = i_features
                    continue

                current_representative_distance = \
                    sklearn.metrics.pairwise.cosine_distances([weights],
                                                              [representatives_features[position]])[0]
                new_adept_distance = \
                    sklearn.metrics.pairwise.cosine_distances([weights], [features])[0]

                if new_adept_distance < current_representative_distance:
                    representatives_features[position] = features
                    representatives_idxs[position] = i_features

        return representatives_idxs

    @staticmethod
    def closest_representatives2(weights, features):
        """
        For each weight in weights find closest feature out of features (returns its index).
        """

        return np.argmin(sklearn.metrics.pairwise.cosine_distances(weights, features), axis=1)


    @staticmethod
    def sample_som(som_shape, factor = 2):
        step_row, step_col = factor, factor
        max_row, max_col = som_shape
        selected_idxs = [row * max_col + col for row in range(0, max_row, step_row) for col in range(0, max_col, step_col)]
        return (max_row // step_row, max_col // step_col), selected_idxs

        # return [np.unravel_index(neuron_id, som_shape) for neuron_id in selected_idxs]

    # def dic_to_lists(self, data):
    #
    #
    #     result = np.empty(shape=self.som_shape + (0,)).tolist()
    #     for cluster_id, face_crop in data.items():
    #         i, j = np.unravel_index(cluster_id, som_shape)
    #         result[i][j] = face_crop
    #     return result
    # features = []
    # hash_features = []
    # for path, all_faces_features in database.records:
    #     for crop, face_features in all_faces_features:
    #         hash_features.append(FaceCrop(path, crop))
    #         features.append(face_features)

    # som_shape = (6,6)
    # som = MiniSom(*som_shape, 128, sigma=0.3, learning_rate=0.5)
    # som.train_random(features, 100000)
    # winner_coordinates = np.array([som.winner(x) for x in features]).T
    # cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

    # Get random image from that particular cluster
    # ma 36 neuronov, kazdy obrazok je priradeny k danemu neuronu.

    # choices = {}
    # for cluster_id in set(cluster_index):
    #     features_idxs = [idx for idx, value in enumerate(cluster_index) if value == cluster_id]
    #     choice = random.choice(features_idxs)
    #     choices[cluster_id] = hash_features[choice]

    # print(choices)
    # def transform_dic_to_lists(shape, data):

    # som = SOM(features)
    # a = (transform_dic_to_lists(som_shape, choices))
