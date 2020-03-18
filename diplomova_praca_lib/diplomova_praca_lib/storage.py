import glob
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow

from diplomova_praca_lib.position_similarity.models import ImageData


class Storage:
    def __init__(self):
        pass


class FileStorage(Storage):
    @staticmethod
    def save_data(directory, filename, data, compressed=True):
        Path(directory).mkdir(parents=True, exist_ok=True)
        if compressed:
            FileStorage.save_compressed(Path(directory, filename), data)
        else:
            FileStorage.save_data_to_file(Path(directory, filename), data)

    @staticmethod
    def save_data_to_file(filename, data):
        np.save(filename, data)

    @staticmethod
    def save_compressed(filename, data):
        np.savez_compressed(filename, data)

    @staticmethod
    def load_data_from_file(filename):
        return np.load(filename, allow_pickle=True)

    @staticmethod
    def load_image_from_file(filename):
        return tensorflow.keras.preprocessing.image.load_img(filename)

    @staticmethod
    def directories(path):
        return [i for i in Path(path).glob('*/') if i.is_dir()]

    @staticmethod
    def load_images_continuously(dir_path, recursively=True) -> Iterable[ImageData]:
        if recursively:
            image_files = list(Path(dir_path).rglob('*.jpg'))
        else:
            image_files = glob.glob(os.path.join(dir_path, "*.jpg"))

        print("Found %d images." % len(image_files))
        for i, filename in enumerate(image_files):
            if i % 20 == 0: print("Processing %d out of %d." % (i, len(image_files)))
            logging.debug("Loading image %s" % filename)
            image = FileStorage.load_image_from_file(filename)
            yield ImageData(filename=filename, image=image)


class Database:
    def __init__(self, records):
        self.records = records
