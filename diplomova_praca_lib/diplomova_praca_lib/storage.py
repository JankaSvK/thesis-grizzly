import glob
import logging
import os
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

import numpy as np

from diplomova_praca_lib.position_similarity.models import ImageData


class Storage:
    def __init__(self):
        pass


class FileStorage(Storage):
    @staticmethod
    def load_multiple_files_multiple_keys(path, retrieve_merged=None, retrieve_once=None, filename_regex="*.npz",
                                          num_files_limit=None, key_filter: Optional[Tuple[str, List[str]]] = None):
        if not retrieve_once:
            retrieve_once = []
        if not retrieve_merged:
            retrieve_merged = []

        filenames = list(Path(path).rglob(filename_regex))
        files = [FileStorage.load_data_from_file(d) for d in filenames]

        if num_files_limit:
            files = files[:num_files_limit]

        result = {}
        # for key in retrieve_merged + retrieve_once:
        #     for i_f, f in enumerate(files):
        for i_f, f in enumerate(files):
            for key in retrieve_merged + retrieve_once:
                if key not in f:
                    logging.warning("`%s` not available in `%s`" % (key, str(filenames[i_f])))
                    continue

                if key in retrieve_merged:
                    if key not in result:
                        result[key] = []

                    if key_filter:
                        filter_data = f[key_filter[0]]
                        idxs = [i_item for i_item, item in enumerate(filter_data) if item in key_filter[1]]
                        result[key] += list(np.array(f[key])[idxs])
                    else:
                        result[key] += list(f[key])
                if key in retrieve_once and key not in result:
                    logging.info("`%s` retrieved from `%s`" % (key, str(filenames[i_f])))
                    result[key] = f[key]

        return result



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
        import tensorflow
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
