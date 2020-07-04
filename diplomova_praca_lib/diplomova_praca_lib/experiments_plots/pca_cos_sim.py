import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from diplomova_praca_lib.storage import FileStorage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_input", default=None, type=str)
    parser.add_argument('--x_input', default=None, type=str)
    parser.add_argument('--sample_size', default=500, type=int)
    args = parser.parse_args()

    y_dataset = FileStorage.load_multiple_files_multiple_keys(args.y_input, retrieve_merged=['features', 'paths'], num_files_limit=10)
    x_dataset = FileStorage.load_multiple_files_multiple_keys(args.x_input, retrieve_merged=['features', 'paths'], num_files_limit=10)

    y_features = np.array(y_dataset['features'])
    x_features = np.array(x_dataset['features'])

    y_paths = y_dataset['paths']
    x_paths = x_dataset['paths']

    assert y_paths == x_paths

    sampled_idxs = np.random.choice(np.arange(len(y_features)), args.sample_size, replace=False)

    y_sampled = y_features[sampled_idxs]
    x_sampled = x_features[sampled_idxs]

    y_similarities = cosine_similarity(y_sampled)
    x_similarities = cosine_similarity(x_sampled)

    y_similarities = y_similarities.reshape(-1)
    x_similarities = x_similarities.reshape(-1)

    arg_sorted = np.argsort(x_similarities)

    fig, ax = plt.subplots()
    ax.plot(x_similarities[arg_sorted], y_similarities[arg_sorted], 'x', markersize=0.02)
    ax.plot((0,1))
    ax.set_xlim((-1,1))
    ax.set_xlabel(Path(args.x_input).name)
    ax.set_ylabel(Path(args.y_input).name)
    ax.set_ylim((-1,1))
    plt.show()

    mse = ((x_similarities - y_similarities)**2).mean()
    print("mse:", mse)


if __name__ == '__main__':
    main()
