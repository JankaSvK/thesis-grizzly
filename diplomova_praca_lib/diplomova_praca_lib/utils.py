from pathlib import Path

import numpy as np

from diplomova_praca_lib.position_similarity.models import UrlImage, Crop


def cap_value(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def filename_without_extensions(path):
    return Path(path).stem

def batches(iterator, batch_size):
    if batch_size < 1:
        raise ValueError("`batch_size` must be greater than or equal to 1.")

    batch = []
    for el in iterator:
        batch.append(el)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def k_smallest_sorted(a, k):
    """Uses hybrid sorting. Finds firstly k-smallest elements (with no ordering) and then orders this smaller set."""
    k_smallest_idxs = np.argpartition(a, k)[:k]
    return k_smallest_idxs[np.argsort(a[k_smallest_idxs])]


def sorted_indexes(a, reverse=True):
    return list(sorted(range(len(a)), key=lambda k: a[k], reverse=reverse))

def concatenate_lists(list_of_lists):
    import itertools
    return list(itertools.chain.from_iterable(list_of_lists))

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        # Warning: You may wish to do a deepcopy here if returning objects
        return self.memo[args]


def images_with_position_from_json(json_data):
    images = []
    for image in json_data:
        url_image = UrlImage(image["url"],
                             Crop(top=image['top'], left=image['left'], width=image['width'], height=image['height']))
        images.append(url_image)
    return images


def path_from_css_background(long_path):
    return long_path[long_path.index('thumbnails/') + len('thumbnails/'):-2]