from typing import List

import PIL
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from diplomova_praca_lib.position_similarity.models import Crop
from diplomova_praca_lib.utils import cap_value


def split_image_to_regions(image, num_horizontal_regions, num_vertical_regions):
    h, w, _ = np.shape(image)
    regions_crops = compute_image_regions(w, h, num_horizontal_regions, num_vertical_regions, 0.5)
    return [(crop, crop_image(image, crop)) for crop in regions_crops]


def crop_image(image: PIL.Image, crop: Crop):
    width, height = image.size
    return image.crop((crop.left * width, crop.top * height, crop.right * width, crop.bottom * height))

def resize_image(image):
    # type: (Image) -> np.ndarray
    image = tf.keras.preprocessing.image.img_to_array(image)
    return cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

def images_as_model_inputs(images):
    # type: (List[Image]) -> np.ndarray
    """Preprocess PIL.Images and returns as a batch"""
    images = [resize_image(x) for x in images]
    normalized_images = tf.keras.applications.imagenet_utils.preprocess_input(np.stack(images))
    return normalized_images

def image_array_as_model_input(image):
    # type: (Image) -> np.ndarray
    image = resize_image(image)
    image = np.expand_dims(image, axis=0)
    return tf.keras.applications.imagenet_utils.preprocess_input(image)


def pil_image_to_np_array(pil_image):
    return np.array(pil_image)


def compute_image_regions(image_width, image_height, num_horizontal_regions, num_vertical_regions,
                          percentage_size_up):
    if percentage_size_up < 0 or percentage_size_up > 1:
        # TODO: makes probably sense to size_up more than 100%
        raise ValueError("`percentage_size_up` has to be a value between 0 and 1")

    region_width = image_width / num_horizontal_regions
    region_height = image_height / num_vertical_regions
    horizontal_splits = [int(i * region_width) for i in range(num_horizontal_regions + 1)]
    vertical_splits = [int(i * region_height) for i in range(num_vertical_regions + 1)]

    regions = []
    for i_vertical in range(num_vertical_regions):
        for i_horizontal in range(num_horizontal_regions):
            xmin = horizontal_splits[i_horizontal] - (percentage_size_up * region_width / 2)
            xmax = horizontal_splits[i_horizontal + 1] + (percentage_size_up * region_width / 2)

            ymin = vertical_splits[i_vertical] - (percentage_size_up * region_height / 2)
            ymax = vertical_splits[i_vertical + 1] + (percentage_size_up * region_height / 2)

            xmin = cap_value(xmin, 0, image_width)
            xmax = cap_value(xmax, 0, image_width)
            ymin = cap_value(ymin, 0, image_height)
            ymax = cap_value(ymax, 0, image_height)

            regions.append(Crop(top=ymin / image_height, left=xmin / image_width, right=xmax / image_width,
                                bottom=ymax / image_height))

    assert len(regions) == num_vertical_regions * num_horizontal_regions
    return regions
