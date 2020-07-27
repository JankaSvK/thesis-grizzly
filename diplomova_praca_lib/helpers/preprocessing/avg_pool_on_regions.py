import argparse
from pathlib import Path

import numpy as np

from diplomova_praca_lib.models import DatabaseRecord
from diplomova_praca_lib.position_similarity.models import RegionFeatures
from diplomova_praca_lib.storage import FileStorage

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=None, type=str)
parser.add_argument('--output', default=None, type=str)
args = parser.parse_args()

for file in Path(args.input).rglob("*.npz"):
    save_location = Path(args.output, file.name)
    if save_location.exists():
        print("Skipping {}. Already present.".format(save_location))
        continue

    data = np.load(str(file), allow_pickle=True)

    new_db_records = []
    for filepath, features in data['data']:
        image_features = []
        for regions_features in features:
            avg_pool_features = np.mean(regions_features.features, axis=(0, 1))  # There is no batch
            image_features.append(RegionFeatures(crop=regions_features.crop, features=avg_pool_features))
        new_db_records.append(DatabaseRecord(filename=filepath, features=image_features))

    FileStorage.save_data(Path(args.output, file.name), data=new_db_records, src_dir=data['src_dir'], model=data[
        'model'])
