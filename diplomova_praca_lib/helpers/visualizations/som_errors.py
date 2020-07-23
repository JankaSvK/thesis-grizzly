import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import load_from_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str)
    parser.add_argument("--som", default=None, type=str)
    args = parser.parse_args()

    data = FileStorage.load_multiple_files_multiple_keys(path=args.input,
                                                         retrieve_merged=['features', 'crops', 'paths'])
    features = data['features']
    data = np.vstack(features)

    q_error = []
    t_error = []
    files = list(Path(args.som).rglob("*.pickle"))

    for file in sorted(files, key=lambda f: f.stat().st_mtime):
        som = load_from_file(file)

        q_error.append(som.quantization_error(data))
        t_error.append(som.topographic_error(data))

    step = 1000
    plt.plot(np.arange(len(files) * step, step=step) / 1000, q_error, label='Quantization error')
    plt.plot(np.arange(len(files) * step, step=step) / 1000, t_error, label='Topographic error')
    plt.ylabel('Error')
    plt.xlabel('Iteration ($\\times 10^3$)')
    plt.legend()

    plt.savefig("som_errors.pdf", bbox_inches='tight')
    plt.show()

    print(q_error)
    print(t_error)

if __name__ == '__main__':
    main()
