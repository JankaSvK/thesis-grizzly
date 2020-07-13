from pathlib import Path
from typing import List, Tuple

import numpy as np

import matplotlib.pyplot as plt
from diplomova_praca_lib import storage
import seaborn as sns

from diplomova_praca_lib.position_similarity.models import Crop
from diplomova_praca_lib.storage import Database, FileStorage


def convert_individual_records_to_groups(database: Database, crop_min_size = 0.0) -> Tuple[List[str], List[Crop], np.ndarray]:
    features = []
    paths = []
    crops = []

    i_feature = 0
    for path, all_faces_features in database.records:
        for crop, face_features in all_faces_features:
            if crop.width * crop.height < crop_min_size:
                continue

            paths.append(path)
            crops.append(crop)
            features.append(face_features)
            i_feature += 1

    features = np.vstack(features)
    return paths, crops, features


def crop_sizes_investigation(crops):
    crops_sizes = [c.width * c.height for c in crops]

    sns.set()
    sns.set(font_scale=1.2)
    sns.distplot(crops_sizes)
    plt.ylabel('Number of queries')
    plt.xlabel('Ratio of area covered')
    plt.title("Relative size of the queries")
    plt.show()

    crops_sizes = np.array(crops_sizes)
    cum = [(i, np.count_nonzero(crops_sizes >= i)) for i in np.linspace(0,0.2,25)]
    print(cum)

def main():
    database = Database(FileStorage.load_datafiles(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_faces_july"))
    paths, crops, features = convert_individual_records_to_groups(database, crop_min_size=0.1)

    output_path = Path(r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_only_bigger_10percent_316videos", "faces.npz")
    storage.FileStorage.save_data(path=output_path, features=features, crops=crops, paths=paths)


if __name__ == '__main__':
    main()
