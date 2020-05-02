import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from diplomova_praca_lib import storage

# DATA_PATH = r"C:\Users\janul\Desktop\saved_annotations\750_resnet50_new"
DATA_PATH = r"C:\Users\janul\Desktop\saved_annotations\50-regions-mobilenetv2"
OUTPUT_DIR = r"C:\Users\janul\Desktop\saved_annotations\experiments\compressed_features_mobilenetv2"


def main():
    loaded_data = storage.FileStorage.load_datafiles(DATA_PATH)

    paths, crops, dataset_features = [], [], []
    for src_path, all_features in loaded_data:
        paths += [src_path] * len(all_features)
        crops += [x.crop for x in all_features]
        dataset_features += [x.features for x in all_features]

    dataset_features = np.asarray(dataset_features)

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(dataset_features)

    pca = PCA(n_components=0.8)
    features_transformed = pca.fit_transform(features_normalized)

    print("Components = ", pca.n_components_, ";\nTotal explained variance = ",
          round(pca.explained_variance_ratio_.sum(), 5))

    storage.FileStorage.save_data(Path(OUTPUT_DIR, "data"), features=features_transformed, crops=crops, paths=paths,
                                  pca=pickle.dumps(pca), scaler=pickle.dumps(scaler))


if __name__ == '__main__':
    main()