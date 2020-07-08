import argparse
from pathlib import Path

import numpy as np

from diplomova_praca_lib.storage import FileStorage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str)
    parser.add_argument('--to_fix', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    data_original = FileStorage.load_multiple_files_multiple_keys(args.original, retrieve_merged=['features'], num_files_limit=2)['features']
    features_mean = np.mean(np.stack(data_original), axis=0)

    data_preprocessed = FileStorage.load_multiple_files_multiple_keys(args.to_fix, retrieve_merged=['features'], num_files_limit=5)
    preprocessed_features = np.stack(data_preprocessed['features'])
    preprocessed_fixed = preprocessed_features + features_mean

    new_data = {}
    for key in data_preprocessed.keys():
        if key == 'features':
            new_data['features'] = preprocessed_fixed
        else:
            new_data[key] = data_preprocessed[key]

    FileStorage.save_data(Path(args.output, 'fixed_data'), **new_data)

if __name__ == '__main__':
    main()
