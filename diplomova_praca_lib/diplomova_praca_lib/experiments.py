import collections
import logging
import sqlite3
from typing import *

import matplotlib.pyplot as plt
import numpy as np

import diplomova_praca_lib
from diplomova_praca_lib.position_similarity.models import UrlImage, PositionSimilarityRequest, \
    PositionSimilarityResponse
from diplomova_praca_lib.position_similarity.position_similarity_request import position_similarity_request, \
    RegionsEnvironment
from diplomova_praca_lib.utils import images_with_position_from_json, path_from_css_background

logging.basicConfig(level=logging.INFO)

mobilenet_full_data_with_pca = r"C:\Users\janul\Desktop\output\2020-05-11_05-43-12_PM"


def plot(x, y):

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Rank', ylabel='Num Queries',
           title='Recall based on ranking')
    plt.show()



def graph_of_search_rank(results: Dict[str, List[int]]) -> None:
    dataset_size = len(diplomova_praca_lib.position_similarity.position_similarity_request.regions_env.regions_data.unique_src_paths)
    model_title = diplomova_praca_lib.position_similarity.position_similarity_request.regions_env.model_title()

    fig, ax = plt.subplots()
    for func_name, ranks in results.items():
        ranks_space = np.arange(max(ranks))
        x = ranks_space / dataset_size
        y = np.array([np.count_nonzero(ranks < r) for r in ranks_space]) / len(ranks)

        ax.plot(x, y, label=func_name)

    ax.set_title("Recall - %s" % model_title)
    ax.set_xlabel("Rank [%]")
    ax.set_ylabel("Requests [%]")
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc='best')

    plt.show()

def test_ranking_funcs(requests):
    results = {}
    for func in [np.min, np.max, np.average]:
        print("Processing %s" % func.__name__)
        diplomova_praca_lib.position_similarity.position_similarity_request.regions_env.ranking_func = func
        responses = send_requests(requests)
        results[func.__name__] = searched_rank_only(responses)

    graph_of_search_rank(results)

def searched_rank_only(responses: List[PositionSimilarityResponse]) -> List[int]:
    return [response.searched_image_rank for response in responses]


def send_requests(requests: List[PositionSimilarityRequest]) -> List[PositionSimilarityResponse]:
    return [position_similarity_request(request) for request in requests]

def collage_as_request(collage):
    query_image = path_from_css_background(collage.query)
    images = eval(collage.images)
    return PositionSimilarityRequest(images=images_with_position_from_json(images), query_image=query_image)


def load_environment_again(dataset):
    diplomova_praca_lib.position_similarity.position_similarity_request.regions_env = RegionsEnvironment(
        data_path=dataset, ranking_func=np.min)

Collage = collections.namedtuple('Collage', "id timestamp query images")
def retrieve_collages() -> List[Collage]:
    conn = sqlite3.connect(r'C:\Users\janul\Desktop\thesis\code\diplomova_praca\db.sqlite3')
    conn.row_factory = (lambda cursor, row: Collage(*row))
    c = conn.cursor()
    c.execute('SELECT * FROM position_similarity_collage')
    fetched_queries = c.fetchall()
    return fetched_queries



def recall_graph(x, y):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Rank', ylabel='Num Queries',
           title='Recall based on ranking')
    plt.show()


def main():
    # load_environment_again(r"C:\Users\janul\Desktop\output\2020-05-24_01-02-43_PM")
    fetched_collages = retrieve_collages()
    fetched_collages = fetched_collages
    print("Fetched")
    requests = [collage_as_request(collage) for collage in fetched_collages]
    print("Prepared")
    test_ranking_funcs(requests=requests)


if __name__ == '__main__':
    main()
