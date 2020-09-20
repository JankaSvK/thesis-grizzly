import base64
import json
import random
import re
from io import BytesIO
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import PIL
import numpy as np
import requests
from PIL import Image

from diplomova_praca_lib.position_similarity.models import UrlImage, Crop
from diplomova_praca_lib.storage import FileStorage


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

def images_with_position_from_json_somhunter(json_data):
    images = []
    for image in json_data:
        url_image = UrlImage(image["src"],
                             Crop(top=image['top'], left=image['left'], width=image['width'], height=image['height']))
        images.append(url_image)
    return images


def path_from_css_background(long_path, thumbnails_prefix = None):
    import os
    url_string = '/'.join(thumbnails_prefix.split(os.sep))
    return long_path[long_path.index(url_string) + len(url_string) + 1:-2]

def timestamp_directory(path_prefix):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    new_dir_path = Path(path_prefix, timestamp)
    new_dir_path.mkdir(parents=True, exist_ok=True)
    return new_dir_path

def dump_to_file(path, object):
    Path(path.parents[0]).mkdir(parents=True, exist_ok=True)
    import pickle
    with open(path, 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_file(path):
    import pickle

    if not Path(path).exists():
        return None

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



def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def download_image(image_src:str)-> PIL.Image:
    if is_url(image_src):
        response = requests.get(image_src)
        image = Image.open(BytesIO(response.content))
    else:
        image_data = re.sub('^data:image/.+;base64,', '', image_src)
        image = Image.open(BytesIO(base64.b64decode(image_data)))

    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha = image.convert('RGBA').split()[-1]
        bg_colour = (255, 255, 255)
        image_without_transparency = Image.new("RGBA", image.size, bg_colour + (255,))
        image_without_transparency.paste(image, mask=alpha)

        return image_without_transparency.convert('RGB')
    else:
        return image


def resize_with_padding(img, expected_size, fill='black'):
    from PIL import ImageOps

    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    expanded = ImageOps.expand(img, padding, fill=fill)
    return expanded


def download_and_preprocess(images, shape, padding=None):
    if padding:
        return [resize_with_padding(download_image(request_image.url), expected_size=shape, fill=padding)
                for request_image in images]
    else:
        return download_and_resize(images, shape)

def download_and_resize(images, shape):
    return [download_image(request_image.url).resize(shape[:2]) for request_image in images]


def sample_image_paths(path:str, samples:int) -> List[str]:
    """Reads preprocessed features and extracts only paths to randomly selected images."""
    source_images = FileStorage.load_multiple_files_multiple_keys(path, retrieve_merged=['paths'])['paths']
    unique_source_images = set(source_images)

    sampled_paths = random.sample(unique_source_images, samples)
    return sampled_paths


def sample_features_from_data(path:str, num_samples:int, total_count:int):
    """Reads annotated features and returns randomly selected subset."""
    sampled_idxs = sorted(np.random.choice(np.arange(total_count), num_samples, replace=False))
    retrieved_samples = []
    already_seen_samples = 0
    print("Sampling")
    for file in Path(path).rglob("*.npz"):
        samples_from_file = 0
        loaded_data = np.load(str(file), allow_pickle=True)['data']
        datafile_samples = len(loaded_data)
        i_sample = sampled_idxs[len(retrieved_samples)] - already_seen_samples
        while i_sample < datafile_samples:
            retrieved_samples.append(loaded_data[i_sample])
            samples_from_file += 1

            if len(retrieved_samples) == num_samples:
                break

            i_sample = sampled_idxs[len(retrieved_samples)] - already_seen_samples

        already_seen_samples += datafile_samples
        print("From %s obtained %d samples out of %d samples" % (str(file), samples_from_file, datafile_samples))

    assert len(retrieved_samples) == num_samples
    return retrieved_samples
