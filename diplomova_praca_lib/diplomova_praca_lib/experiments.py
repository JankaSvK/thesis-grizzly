import sqlite3
from decimal import Decimal

import numpy as np
import collections

import logging
from typing import *

from diplomova_praca_lib.position_similarity.models import UrlImage, Crop, PositionSimilarityRequest
from diplomova_praca_lib.position_similarity.position_similarity_request import position_similarity_request, Environment
from diplomova_praca_lib.utils import images_with_position_from_json, path_from_css_background

logging.basicConfig(level=logging.INFO)

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

fetched_collages = retrieve_collages()


query_images = []
ranks = []
for collage in fetched_collages:

    query_image = path_from_css_background(collage.query)
    logging.info('Processing query image %s' % query_image)
    query_images.append(query_image)
    images = eval(collage.images)

    request = PositionSimilarityRequest(images=images_with_position_from_json(images),
                                        query_image=query_image)
    response = position_similarity_request(request)
    closest_images = response.ranked_paths

    if query_image in closest_images:
        rank = closest_images.index(query_image)
    else:
        rank = 20000
    ranks.append(rank)

ranks = np.array(ranks)
print("Average rank", np.average(ranks))
print("\n".join(map(str, zip(query_images, ranks))))

x = np.arange(max(ranks))
y = [np.count_nonzero(ranks < r) for r in x]
recall_graph(x, y)
