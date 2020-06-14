import argparse
import logging
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


def graph_of_search_rank(results: Dict[str, List[int]], dataset_info: DatasetInfo) -> None:
    fig, ax = plt.subplots()
    for func_name, ranks in results.items():
        ranks_space = np.arange(max(ranks))
        x = ranks_space / dataset_info.num_images
        y = np.array([np.count_nonzero(ranks < r) for r in ranks_space]) / len(ranks)

        ax.plot(x, y, label=func_name)

    ax.set_title("Discovery Rate")
    ax.set_xlabel("Rank of searched image [%]")
    ax.set_ylabel("Requests [%]")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc='best')

    plt.show()


def searched_rank_only(responses: List[PositionSimilarityResponse]) -> List[int]:
    return [response.searched_image_rank for response in responses]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', action='store', type=str, nargs='+')
    parser.add_argument('--names', action='store', type=str, nargs='+')
    args = parser.parse_args()

    plot_info = {}
    dataset = None
    for path, name in zip(args.paths, args.names):
        responses_input = Path(path)
        responses_data = FileStorage.load_data_from_file(responses_input)
        responses = responses_data['responses']
        plot_info[name] = searched_rank_only(responses)

        input_data = responses_data['experiment'].item().get('input_data')
        assert not dataset or input_data == dataset
        dataset = input_data

    data = load_data(dataset)['paths']
    dataset_info = DatasetInfo(num_images=len(set(data)))
    graph_of_search_rank(plot_info, dataset_info=dataset_info)


if __name__ == '__main__':
    main()
