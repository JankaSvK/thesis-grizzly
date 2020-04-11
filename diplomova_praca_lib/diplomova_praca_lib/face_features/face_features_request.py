import numpy as np

from diplomova_praca_lib.face_features import clustering
from diplomova_praca_lib.face_features.map_features import SOM
from diplomova_praca_lib.face_features.models import FaceCrop
from diplomova_praca_lib.storage import FileStorage, Database
from diplomova_praca_lib.utils import filename_without_extensions

database = Database(FileStorage.load_datafiles(r"C:\Users\janul\Desktop\saved_annotations\750_faces_2ndtry"))


# database = Database(FileStorage.load_datafiles(r"/mnt/c/Users/janul/Desktop/saved_annotations/750_faces_2ndtry"))
class Environment:
    features_info = []
    features = []
    som = None

    def __init__(self):
        for path, all_faces_features in database.records:
            for crop, face_features in all_faces_features:
                Environment.features_info.append(FaceCrop(path, crop))
                Environment.features.append(face_features)

        Environment.som = SOM((12, 26), 128)
        Environment.som.train_som(Environment.features)


env = Environment()


def face_features_request(request):
    som_weights = env.som.som.get_weights()
    representatives = SOM.closest_representatives2(np.reshape(som_weights, (-1, som_weights.shape[2])), env.features)
    sampled_grid_shape, sample_grid_idxs = SOM.sample_som(env.som.som_shape, 2)


    representatives_info = [Environment.features_info[i_feature] for i_feature in representatives]
    result = [representatives_info[i_slice * env.som.som_shape[1]:(i_slice + 1) * env.som.som_shape[1]]
               for i_slice in range(env.som.som_shape[0])]


    # result = np.empty(shape=Environment.som.som_shape + (0,)).tolist()
    # for (i, j), i_feature in Environment.som.closest_representatives(env.features).items():
    #     result[i][j] = Environment.features_info[i_feature]
    #
    sampled_som_idxs = SOM.sample_som(env.som.som_shape, 2)


    return result


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
