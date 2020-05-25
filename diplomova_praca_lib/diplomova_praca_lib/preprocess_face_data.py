from pathlib import Path
from typing import List, Tuple

import numpy as np

from diplomova_praca_lib import storage
from diplomova_praca_lib.position_similarity.models import Crop
from diplomova_praca_lib.storage import Database, FileStorage


def convert_individual_records_to_groups(database: Database) -> Tuple[List[str], List[Crop], np.ndarray]:
    features = []
    paths = []
    crops = []

    i_feature = 0
    for path, all_faces_features in database.records:
        for crop, face_features in all_faces_features:
            paths.append(path)
            crops.append(crop)
            features.append(face_features)
            i_feature += 1

    features = np.vstack(features)
    return paths, crops, features


def main():
    database = Database(FileStorage.load_datafiles(r"C:\Users\janul\Desktop\saved_annotations\750_faces"))
    paths, crops, features = convert_individual_records_to_groups(database)
    output_path = Path(r"C:\Users\janul\Desktop\thesis_tmp_files\transformed_face_features", "faces.npz")
    storage.FileStorage.save_data(path=output_path, features=features, crops=crops, paths=paths)


if __name__ == '__main__':
    main()
