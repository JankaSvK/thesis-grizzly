import argparse
import logging
from _sha256 import sha256
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np

from diplomova_praca_lib.position_similarity.models import PositionSimilarityResponse
from diplomova_praca_lib.storage import FileStorage

logging.basicConfig(level=logging.INFO)


def load_data(path):
    return FileStorage.load_multiple_files_multiple_keys(path=path, retrieve_merged=['features', 'paths'],
                                                         retrieve_once=['pipeline', 'model'])


class DatasetInfo:
    def __init__(self, num_images):
        self.num_images = num_images


def graph_of_search_rank(results: Dict[str, List[int]], input_paths: List[str], save_plot=None):
    fig, ax = plt.subplots()
    for (func_name, ranks), input_src in zip(results.items(), input_paths):
        # ranks_space = np.arange(max(ranks) + 1)
        num_images = len(set(load_data(input_src)['paths']))
        print(num_images)

        ranks_space = np.arange(num_images + 1)
        x = ranks_space / num_images
        y = np.array([np.count_nonzero(ranks <= r) for r in ranks_space]) / len(ranks)

        ax.plot(x, y, label=func_name)

    ax.hlines(y=0.9, xmin=0, xmax=max(x), color = '0.75')

    ax.set_title("Discovery Rate")
    ax.set_xlabel("Rank of searched image [%]")
    ax.set_ylabel("Requests [%]")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc='best')

    if save_plot:
        graph_dir = r"C:\Users\janul\Desktop\thesis_tmp_files\graphs"
        filename_hash = sha256(save_plot.encode('utf-8')).hexdigest()
        plt.savefig(Path(graph_dir, filename_hash + ".pdf"), bbox_inches='tight')

    plt.show()


def searched_rank_only(responses: List[PositionSimilarityResponse]) -> List[int]:
    return [response.searched_image_rank for response in responses]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', action='store', type=str, nargs='+')
    parser.add_argument('--names', action='store', type=str, nargs='+')
    parser.add_argument('--assert_dataset', action='store_true')
    args = parser.parse_args()

    plot_info = {}
    dataset = None
    input_data = []
    for path, name in zip(args.paths, args.names):
        responses_input = Path(path)
        responses_data = FileStorage.load_data_from_file(responses_input)
        responses = responses_data['responses']
        if responses.any():
            plot_info[name] = searched_rank_only(responses)
        else:
            continue

        # Assert that the original data correspondence
        features_src = responses_data['experiment'].item().get('input_data')
        input_data.append(features_src)

        if args.assert_dataset:
            processed_data = set(load_data(features_src)['paths'])

            if not dataset:
                dataset = processed_data

            assert dataset == processed_data

    # dataset_info = DatasetInfo(num_images=len(dataset))
    graph_of_search_rank(plot_info, input_paths=input_data, save_plot=",".join(args.paths))





if __name__ == '__main__':
    main()
