import sqlite3

import numpy as np
import collections

import logging

from diplomova_praca_lib.position_similarity.models import UrlImage, Crop, PositionSimilarityRequest
from diplomova_praca_lib.position_similarity.position_similarity_request import position_similarity_request, Environment


logging.basicConfig(level=logging.INFO)

Collage = collections.namedtuple('Collage', "id timestamp query images")

def retrieve_collages():
    conn = sqlite3.connect(r'C:\Users\janul\Desktop\thesis\code\diplomova_praca\db.sqlite3')
    conn.row_factory = (lambda cursor, row: Collage(*row))
    c = conn.cursor()
    c.execute('SELECT * FROM position_similarity_collage')
    fetched_queries = c.fetchall()
    return fetched_queries


def json_to_position_similarity_request(json_data):
    images = []
    for image in json_data:
        url_image = UrlImage(image["url"], Crop(*[image[attr] for attr in ["top", "left", "width", "height"]]))
        images.append(url_image)
    return PositionSimilarityRequest(images)


def recall_graph(x, y):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Rank', ylabel='Num Queries',
           title='Recall based on ranking')
    plt.show()

fetched_collages = retrieve_collages()

Environment.results_limit = 30000

query_images = []
ranks = []
for collage in fetched_collages:
    query_image = collage.query[collage.query.index('thumbnails/') + len('thumbnails/'):-2]
    logging.info('Processing query image %s' % query_image)
    query_images.append(query_image)
    images = eval(collage.images)
    closest_images = position_similarity_request(json_to_position_similarity_request(images))

    try:
        rank = closest_images.index(query_image)
    except ValueError:
        rank = Environment.results_limit + 1
    ranks.append(rank)

ranks = np.array(ranks)
print("\n".join(map(str, zip(query_images, ranks))))

x = np.arange(150)
y = [np.count_nonzero(ranks < r) for r in x]
recall_graph(x, y)
