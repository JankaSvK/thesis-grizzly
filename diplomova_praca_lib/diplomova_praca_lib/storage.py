import glob
import itertools
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
    def save_data(path, compressed=True, **kwargs):
        Path(path.parents[0]).mkdir(parents=True, exist_ok=True)

        if compressed:
            np.savez_compressed(path, **kwargs)
        else:
            np.save(path, **kwargs)

        print("Results saved in {}".format(path))

    @staticmethod
    def load_data_from_file(filename):
        return np.load(filename, allow_pickle=True)

    @staticmethod
    def load_datafiles(dir_path):
        datafiles = Path(dir_path).rglob('*.npz')
        data = [(FileStorage.load_data_from_file(f))['data'] for f in datafiles]
        return np.concatenate(data)
        # return list(itertools.chain(*data))

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
    def __init__(self, records, src_path=None, model=None):
        self.records = records
        self.src_path = src_path
        self.model = model
