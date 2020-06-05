import json
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

def timestamp_directory(path_prefix):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    new_dir_path = Path(path_prefix, timestamp)
    new_dir_path.mkdir(parents=True, exist_ok=True)
    return new_dir_path

def dump_to_file(path, object):
    import pickle
    with open(path, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_file(path):
    import pickle
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def reduce_dims(data):
    return data.reshape(-1, data.shape[-1])


def enhance_dims(data, shape):
    return data.reshape(-1, *shape, data.shape[1])


def closest_match(query, features, num_results = None, distance = None):
    distances = distance([query], features)[0]
    if num_results == None:
        sorted_idxs = np.argsort(distances)
        return sorted_idxs, distances[sorted_idxs]

    sorted_idxs = k_smallest_sorted(distances, num_results)
    return sorted_idxs, distances[sorted_idxs]


class Serializable:
    raw_init_params = []
    serializable_init_params = {}

    def __init__(self, **kwargs):
        assert set(self.raw_init_params) | set(self.serializable_init_params.keys()) >= set(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)

    def serialize(self):
        to_serialize = {}
        for key in self.raw_init_params:
            to_serialize[key] = getattr(self, key)

        for key in self.serializable_init_params:
            to_serialize[key] = getattr(self, key).serialize()

        return json.dumps(to_serialize)

    @classmethod
    def deserialize(cls, serialized):
        deserialized = json.loads(serialized)
        for key in cls.serializable_init_params:
            deserialized[key] = cls.serializable_init_params[key].deserialize(deserialized[key])

        return cls(**deserialized)
