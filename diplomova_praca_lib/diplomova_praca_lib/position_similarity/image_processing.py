import glob, os

import tensorflow

from .utils import cap_value
import cv2
import numpy as np
import PIL


def split_image_to_regions(image, num_horizontal_regions, num_vertical_regions):
    h, w, c = np.shape(image)
    regions = compute_image_regions(w, h, num_horizontal_regions, num_vertical_regions, 0.5)
    return [((top_left, bottom_right), crop_image(image, *top_left, *bottom_right)) for (top_left, bottom_right) in
            regions]


def crop_image(image: PIL.Image, xmin, ymin, xmax, ymax):
    return image.crop((xmin, ymin, xmax, ymax))


def image_array_as_model_input(image):
    image = tensorflow.keras.preprocessing.image.img_to_array(image)
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    image = np.expand_dims(image, axis=0)
    return tensorflow.keras.applications.imagenet_utils.preprocess_input(image)


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

            regions.append(((xmin, ymin), (xmax, ymax)))

    assert len(regions) == num_vertical_regions * num_horizontal_regions
    return regions
