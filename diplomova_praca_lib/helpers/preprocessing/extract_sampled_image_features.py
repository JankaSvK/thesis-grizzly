import argparse
import random
from pathlib import Path

import numpy as np

from helpers.experiments_plots.get_responses import get_queries
from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import sample_image_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input' , type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--sample_size', type=int, default=100)
    args = parser.parse_args()

    random.seed(42)

    requests = get_queries()
    queries_paths = [r.query_image for r in requests]
    selected_paths = sample_image_paths(args.input, args.sample_size)
    selected_paths += queries_paths

    sample_args = ['paths', 'features', 'crops']

    for file in Path(args.input).rglob("*.npz"):
        if Path(args.output, file.name).exists():
            print("skipping", file.name, "already exists")
            continue

        data = np.load(str(file), allow_pickle=True)
        idxs = np.array([i_path for i_path, path in enumerate(data['paths']) if path in selected_paths])

        if len(idxs) == 0:
            continue

        new_data = {}
        for key in data.keys():
            if key in sample_args:
                new_data[key] = data[key][idxs]
            else:
                new_data[key] = data[key]

        FileStorage.save_data(Path(args.output, file.name), **new_data)


if __name__ == '__main__':
    main()
