import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom

from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import load_from_file, dump_to_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_som", default=None, type=str)
    parser.add_argument("--input_data", default=None, type=str)
    parser.add_argument('--output', default=None, type=str)
    args = parser.parse_args()

    data = FileStorage.load_multiple_files_multiple_keys(path=args.input, retrieve_merged=['features', 'crops', 'paths'])
    features = data['features']
    data = np.vstack(features)

    seed_sets = [(10, 100), (42, 4242), (24, 2424), (4242, 24), (1, 1), (4242, 42), (71, 37), (678, 123), (321, 87),
                 (3, 980)]
    np_seed, som_seed = seed_sets[8]

    np.random.seed(np_seed)
    som = MiniSom(50, 50, data.shape[1], random_seed=som_seed, activation_distance=args.distance,
                  learning_rate=args.learning_rate, sigma=args.sigma)

    if args.pretrained:
        som = load_from_file(args.pretrained)
        som._learning_rate = args.learning_rate
        som._sigma = args.sigma

    max_iter = args.iterations

    q_error = []
    t_error = []
    errors_step = 1000

    for i in range(max_iter + 1):
        if (i + 1) % errors_step == 0:
            print("Iteration", i + 1, "/", max_iter)

        if i % errors_step == 0:
            q_error.append(som.quantization_error(data))
            t_error.append(som.topographic_error(data))

            print("Quantization error:", q_error[-1])
            print("Topographic error:", t_error[-1])

            if args.output:
                som_log_file = Path(args.output, "som-{},{}-{}.pickle".format(args.distance, i, args.iterations))
                dump_to_file(som_log_file, som)

        rand_i = np.random.randint(len(data))
        som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)

    experiment_description = ";".join([args.distance, str(args.iterations), str(np_seed), str(som_seed)])
    plt.plot(np.arange(max_iter + 1, step=errors_step), q_error, label='quantization error')
    plt.plot(np.arange(max_iter + 1, step=errors_step), t_error, label='topographic error')
    plt.ylabel('quantization error')
    plt.xlabel('iteration index')
    plt.title(experiment_description)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
