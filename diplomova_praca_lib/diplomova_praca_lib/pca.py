import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from diplomova_praca_lib import storage

# DATA_PATH = r"C:\Users\janul\Desktop\saved_annotations\750_mobilenetv2"
DATA_PATH = r"C:\Users\janul\Desktop\saved_annotations\750_mobilenetv2-12regions"
OUTPUT_DIR = r"C:\Users\janul\Desktop\saved_annotations\experiments\750_mobbilenetv2-12regions"


def get_data(datafile):
    paths, crops, dataset_features = [], [], []
    for src_path, all_features in datafile:
        paths += [src_path] * len(all_features)
        crops += [x.crop for x in all_features]
        dataset_features += [x.features for x in all_features]

    return (paths, crops, dataset_features)


def get_features(data):
    dataset_features = []
    for src_path, all_features in data:
        dataset_features += [x.features for x in all_features]
    return np.vstack(dataset_features)


def main():
    first_file = storage.FileStorage.load_data_from_file(next(Path(DATA_PATH).rglob('*.npz')))

    scaler = StandardScaler()
    ipca = IncrementalPCA(n_components=400)

    load_scaler = False
    if load_scaler:
        with open(OUTPUT_DIR + 'scaler.pickle', 'rb') as f:
            scaler = pickle.load(f)
    else:
        for filename in Path(DATA_PATH).rglob("*.npz"):
            print("Scaler", filename)
            data = storage.FileStorage.load_data_from_file(filename)['data']
            features = get_features(data)
            scaler.partial_fit(features)

        with open(OUTPUT_DIR + 'scaler.pickle', 'wb') as f:
            pickle.dump(scaler, f)

    for filename in Path(DATA_PATH).rglob("*.npz"):
        print("IPCA", filename)
        data = storage.FileStorage.load_data_from_file(filename)['data']
        features = get_features(data)
        transformed_f = scaler.transform(features)
        try:
            ipca.partial_fit(transformed_f)
        except ValueError:
            print("ERROR:", filename)

    with open(OUTPUT_DIR + 'ipca.pickle', 'wb') as f:
        pickle.dump(ipca, f)

    transformed = []
    paths = []
    crops = []

    for filename in Path(DATA_PATH).rglob("*.npz"):
        print("Transforming", filename)
        data = storage.FileStorage.load_data_from_file(filename)['data']
        dpaths, dcrops, features = get_data(data)
        paths.extend(dpaths)
        crops.extend(dcrops)
        transformed.extend(ipca.transform(scaler.transform(features)))

    print("Components = ", ipca.n_components_, ";\nTotal explained variance = ",
          round(ipca.explained_variance_ratio_.sum(), 5))

    storage.FileStorage.save_data(Path(OUTPUT_DIR, "data"), features=transformed, crops=crops, paths=paths,
                                  pca=pickle.dumps(ipca), scaler=pickle.dumps(scaler), model=first_file['model'])


if __name__ == '__main__':
    main()
