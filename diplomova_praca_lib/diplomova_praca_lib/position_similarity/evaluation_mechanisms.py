import collections
from typing import List

from .image_processing import image_array_as_model_input, split_image_to_regions
import numpy as np

from diplomova_praca_lib.position_similarity.models import RegionFeatures, Crop


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
        """
        Computes regions representation over the image
        :param image: PIL.Image
        :param regions: Tuple (width, height) of designed geomery of regions
        :return: List of tuples (crop, feature) for each region.
        """
        image_features = []

        # Evaluate whole image
        image_features.append(
            RegionFeatures(crop=Crop(0, 0, 1, 1), features=self.model.predict(image_array_as_model_input(image))))

        # Evaluate each subregion
        regions = split_image_to_regions(image, *regions)
        for region_crop, region_image in regions:
            image_features.append(RegionFeatures(crop=region_crop,
                                                 features=self.model.predict(image_array_as_model_input(region_image))))

        return image_features

    @staticmethod
    def regions_overlap_ordering(query_crop: Crop, image_crops: List[Crop]):
        # Returns ordering of image  crops (their indexes) based on the overlap
        ious = [query_crop.iou(image_crop) for image_crop in image_crops]
        ious_sorted_indexes = list(sorted(range(len(ious)), key=lambda k: ious[k], reverse=True))

        return ious_sorted_indexes

    def best_matches(self, query_crop, query_image, database_items: List[RegionFeatures]):
        query_image_features = self.model.predict(image_array_as_model_input(query_image))

        scores = []
        for db_item in database_items:
            highest_iou_region_idx = \
                (EvaluatingRegions.regions_overlap_ordering(query_crop, [rf.crop for rf in db_item[1]]))[0]
                    # TODO: why accesssing directly regions features did not work
            # print(db_item[1][highest_iou_region_idx].crop)
            db_feature = db_item[1][highest_iou_region_idx].features

            scores.append(self.similarity_measure(db_feature, query_image_features))

        # Get the items path based on score
        sorted_scores_idx = list(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
        return [database_items[score_idx][0] for score_idx in sorted_scores_idx]

        # Order them and return their index unique ID.

    # def best_match(self, query: np.ndarray, regions_features):
    #     HighestSimilarity = collections.namedtuple("HighestSimilarity", ["similarity", "index"])
    #     similarities = [self.similarity_measure(query, region_features) for region_features in regions_features]
    #     return HighestSimilarity(similarity=np.max(similarities), index=np.argmax(similarities))
