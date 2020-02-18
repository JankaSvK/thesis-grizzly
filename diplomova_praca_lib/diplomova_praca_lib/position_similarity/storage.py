import collections
import json

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
    def load_images_continuously(dir_path):
        Image = collections.namedtuple("Image", ["filename", "image"])
        image_files = glob.glob(os.path.join(dir_path, "*.jpg"))
        print("Found %d images." % len(image_files))
        for i, filename in enumerate(image_files):
            if i % 100 == 0: print("Processing %d out of %d." % (i, len(image_files)))
            logging.debug("Loading image %s" % filename)
            image = FileStorage.load_image_from_file(filename)
            yield Image(filename=filename, image=image)


class Database:
    def __init__(self, records):
        self.records = records


#
#
# class Database:
#     def __init__(self):
#         self.records = {}
#
#     def save_database_to_file(self, path):
#         numpy.save(path, self.records)
#
#     def load_database_from_file(self, path):
#         if os.path.exists(path):
#             self.records = numpy.load(path, allow_pickle=True).item()
#         else:
#             raise FileNotFoundError
#
#     def load_images_from_directory(self, dir_path):
#         import tensorflow as tf
#
#         image_files = glob.glob(os.path.join(dir_path, "*.jpg"))
#         for i_path in image_files:
#             print("Loading image %s" % i_path)
#             image = self.load_image(i_path)
#             self.records[i_path] = Record(image, ImageInformation(i_path))
#
#     @staticmethod
#     def load_image(path):
#         return tensorflow.keras.preprocessing.image.load_img(path)