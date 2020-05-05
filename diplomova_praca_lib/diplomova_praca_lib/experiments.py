import sqlite3

import numpy as np

from diplomova_praca_lib.position_similarity.models import UrlImage, Crop, PositionSimilarityRequest
from diplomova_praca_lib.position_similarity.position_similarity_request import position_similarity_request, Environment


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


conn = sqlite3.connect(r'C:\Users\janul\Desktop\thesis\code\diplomova_praca\db.sqlite3')
c = conn.cursor()
c.execute('SELECT * FROM position_similarity_collage')
fetched_queries = c.fetchall()
Environment.results_limit = 100000
overlay_image = fetched_queries[0][2]
overlay_image = overlay_image[overlay_image.index('thumbnails/') + len('thumbnails/'):-2]

ranks = []
for query in fetched_queries:
    id, created, overlay_image, request_images = query

    overlay_image = overlay_image[overlay_image.index('thumbnails/') + len('thumbnails/'):-2]
    print(overlay_image)
    images = eval(fetched_queries[0][3])
    closest_images = position_similarity_request(json_to_position_similarity_request(images))

    try:
        rank = closest_images.index(overlay_image)
    except ValueError:
        rank = Environment.results_limit + 1
    ranks.append(rank)

ranks = np.array(ranks)
print(ranks)

x = np.arange(150)
y = [np.count_nonzero(ranks < r) for r in x]
recall_graph(x, y)
