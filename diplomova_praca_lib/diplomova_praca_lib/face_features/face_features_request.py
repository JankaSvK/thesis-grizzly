import numpy as np

from diplomova_praca_lib.face_features import clustering
from diplomova_praca_lib.face_features.som import SOM
from diplomova_praca_lib.storage import FileStorage, Database
from diplomova_praca_lib.utils import filename_without_extensions

database = Database(FileStorage.load_data_from_file(r"C:\Users\janul\Desktop\saved_annotations\300-faces.npy"))
features = [f for path, features in database.records for crop, f in features]
# som = SOM(features)
#
# def face_features_request(request):
#     print(som.som_representants(list(range(0,40)), (4,4)))
#     return None


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
