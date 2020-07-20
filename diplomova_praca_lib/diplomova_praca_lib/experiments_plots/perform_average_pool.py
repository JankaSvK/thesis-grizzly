import argparse
from pathlib import Path

import numpy as np
from tensorflow.python.keras.layers import GlobalAveragePooling2D

from diplomova_praca_lib.storage import FileStorage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    for file in Path(args.input).rglob("*.npz"):
        print(file.name)
        data = np.load(str(file), allow_pickle=True)

        new_data = {}
        for key in data.keys():
            if key == 'features':
                new_data['features'] = GlobalAveragePooling2D()(data['features']).numpy()
            else:
                new_data[key] = data[key]

        FileStorage.save_data(Path(args.output, file.name), **new_data)

if __name__ == '__main__':
    main()
