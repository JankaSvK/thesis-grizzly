from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from diplomova_praca_lib.storage import FileStorage

data_path = r"C:\Users\janul\Desktop\thesis_tmp_files\transformed_face_features\faces.npz"
data = np.load(data_path, allow_pickle=True)
features = data['features']
print("Videos with faces", len(set([prefix.split("/")[0] for prefix in data['paths']])))

y_pred = fclusterdata(features, t=0.6, criterion='distance', method='complete')
print(len(set(y_pred)))

representatives = []
for cluster_id in set(y_pred):
    features_ids = np.argwhere(y_pred == cluster_id)
    cluster_items = features[features_ids]

    centroid = np.mean(cluster_items, axis=0)
    closest_to_centroid_idx = np.argmin([np.linalg.norm(x - centroid) for x in cluster_items])

    closest = features_ids[closest_to_centroid_idx]
    assert y_pred[closest] == cluster_id

    representatives.append(closest)

new_data_path = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_representatives\faces.npz"
new_data = {}
new_data['crops'] = data['crops'][representatives]
new_data['paths'] = data['paths'][representatives]
new_data['features'] = data['features'][representatives]

FileStorage.save_data(Path(new_data_path), **new_data)