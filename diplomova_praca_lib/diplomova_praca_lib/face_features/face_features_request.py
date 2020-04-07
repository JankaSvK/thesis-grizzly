import collections

import random
from minisom import MiniSom
import numpy as np

from diplomova_praca_lib.face_features import clustering
from diplomova_praca_lib.position_similarity.models import Crop
from diplomova_praca_lib.storage import FileStorage, Database
from diplomova_praca_lib.utils import filename_without_extensions
FaceCrop = collections.namedtuple("FaceCrop", ['src', 'crop'])

database = Database(FileStorage.load_datafiles(r"C:\Users\janul\Desktop\saved_annotations\750_faces_2ndtry"))

features = []
hash_features = []
for path, all_faces_features in database.records:
    for crop, face_features in all_faces_features:
        hash_features.append(FaceCrop(path, crop))
        features.append(face_features)

som_shape = (6,6)
som = MiniSom(*som_shape, 128, sigma=0.3, learning_rate=0.5)
som.train_random(features, 100000)
winner_coordinates = np.array([som.winner(x) for x in features]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

# Get random image from that particular cluster
# ma 36 neuronov, kazdy obrazok je priradeny k danemu neuronu.


choices = {}
for cluster_id in set(cluster_index):
    features_idxs = [idx for idx, value in enumerate(cluster_index) if value == cluster_id]
    choice = random.choice(features_idxs)
    choices[cluster_id] = hash_features[choice]

# print(choices)
def transform_dic_to_lists(shape, data):
    result = np.empty(shape=shape + (0,)).tolist()
    for cluster_id, face_crop in data.items():
        i,j = np.unravel_index(cluster_id, som_shape)
        result[i][j] = face_crop
    return result
# som = SOM(features)
a = (transform_dic_to_lists(som_shape, choices))


def fake_request(request):
    return transform_dic_to_lists(som_shape, choices)

def face_features_request(request):
    clusters = clustering.group_clusters_to_lists(features, clustering.split_to_n_clusters(features, 16))
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
