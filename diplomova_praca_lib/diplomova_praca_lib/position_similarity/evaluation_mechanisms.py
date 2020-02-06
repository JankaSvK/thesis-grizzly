import collections

from .image_processing import image_array_as_model_input, split_image_to_regions
import numpy as np

from collections import namedtuple
from  diplomova_praca_lib.position_similarity.models import  RegionFeatures

class EvaluatingBasedOnSpatialDimensions:
    def __init__(self, similarity_measure, model):
        self.similarity_measure = similarity_measure
        self.model = model

    def crop_feature_vector_to_query(self, query_position, feature_vector):
        x, y, width, height = query_position
        batch_size, x_features, y_features, channels = feature_vector.shape
        xmin, ymin, xmax, ymax = x * x_features, y * y_features, (x + width) * x_features, (y + height) * y_features
        xmin, ymin, xmax, ymax = [round(x) for x in [xmin, ymin, xmax, ymax]]

        if xmax - xmin < 1 or ymax - ymin < 1:
            raise ValueError

        subimage_features = feature_vector[:, xmin:xmax, ymin:ymax, :]
        return np.mean(subimage_features, axis=(1, 2))


class EvaluatingRegions:
    def __init__(self, similarity_measure, model):
        self.similarity_measure = similarity_measure
        self.model = model

    def features_on_image_regions(self, image, regions=(4, 3)):
        image_features = []

        # Evaluate whole image
        image_features.append(RegionFeatures(crop=None, features=self.model.predict(image_array_as_model_input(image))))

        # Evaluate each subregion
        for region_coords, region_image in split_image_to_regions(image, *regions):
            image_features.append(RegionFeatures(crop=region_coords,
                                                 features=self.model.predict(image_array_as_model_input(region_image))))

        return image_features

    def best_match(self, query: np.ndarray, regions_features):
        HighestSimilarity = collections.namedtuple("HighestSimilarity", ["similarity", "index"])
        similarities = [self.similarity_measure(query, region_features) for region_features in regions_features]
        return HighestSimilarity(similarity=np.max(similarities), index=np.argmax(similarities))
