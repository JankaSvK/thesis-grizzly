import collections
import json
from pathlib import Path

import numpy as np
import glob, os
import logging

import tensorflow


class Storage:
    def __init__(self):
        pass


class FileStorage(Storage):
    @staticmethod
    def save_data_to_file(filename, data):
        np.save(filename, data)

    @staticmethod
    def load_data_from_file(filename):
        return np.load(filename, allow_pickle=True)

    @staticmethod
    def load_image_from_file(filename):
        return tensorflow.keras.preprocessing.image.load_img(filename)

    @staticmethod
    def load_images_continuously(dir_path, recursively=True):
        if recursively:
            image_files = Path(dir_path).rglob('*.jpg')
        else:
            image_files = glob.glob(os.path.join(dir_path, "*.jpg"))

        print("Found %d images." % len(image_files))
        Image = collections.namedtuple("Image", ["filename", "image"])
        for i, filename in enumerate(image_files):
            if i % 5 == 0: print("Processing %d out of %d." % (i, len(image_files)))
            logging.debug("Loading image %s" % filename)
            image = FileStorage.load_image_from_file(filename)
            yield Image(filename=filename, image=image)


class Database:
    def __init__(self, records):
        self.records = records
