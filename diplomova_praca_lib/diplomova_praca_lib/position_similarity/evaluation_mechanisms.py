from typing import List

import PIL
import numpy as np

from diplomova_praca_lib.image_processing import split_image_to_regions, normalized_images, \
    split_image_to_square_regions, crop_image
from diplomova_praca_lib.models import EvaluationMechanism
from diplomova_praca_lib.position_similarity.models import RegionFeatures, Crop
from diplomova_praca_lib.utils import batches, sorted_indexes, k_smallest_sorted


class EvaluatingSpatially(EvaluationMechanism):
    def __init__(self, similarity_measure, model, database):
        self.similarity_measure = similarity_measure
        self.model = model
        self.database = database


    def features(self, images):
        # type: (List[PIL.Image]) -> np.ndarray
        return self.model.predict(normalized_images(images))

    @staticmethod
    def avg_pool(features):
        # type: (np.ndarray) -> np.ndarray
        """
        Average pool over spatial dimension in (batch, x, y, channels)
        :param features: 4D vector (batch, x, y, channels)
        :return: Average pool --> (batch, channels)
        """
        return np.mean(features, axis=(1,2))

    def crop_features_vectors_to_query(self, query_crop: Crop, features_vectors):
        """
        Crops 3D feature vector to specified crop
        :param query_crop: Region of interest
        :param feature_vector: 4D feature vector (batch, x, y, layers).
        :return: Subset of feature vector from defined crop as result of mean over multiple channels (batch, .
        """
        batch_size, x_features, y_features, channels = features_vectors.shape

        xmin, ymin, xmax, ymax = query_crop.left * x_features, query_crop.top * y_features, query_crop.right * x_features, query_crop.bottom * y_features
        xmin, ymin, xmax, ymax = [round(x) for x in [xmin, ymin, xmax, ymax]]

        if xmax - xmin < 1 or ymax - ymin < 1:
            raise ValueError

        subimage_features = features_vectors[:, xmin:xmax, ymin:ymax, :]
        return EvaluatingSpatially.avg_pool(subimage_features)

    def best_matches(self, query_crop: Crop, query_image: PIL.Image):
        """
        Sorts the database items based on the similarity to the query.
        :param query_crop: Position of queried image
        :param query_image:  PIL.Image of query
        :param database_items: List of features (result of apentultimate layer -- i.e. 3D)
        :return: List of sorted database items based on their similarity to query
        """
        database_items = self.database.records

        query_image_features = self.model.predict(normalized_images([query_image]))[0]
        query_image_features = np.expand_dims(query_image_features, axis=0)

        scores = []
        for batch in batches(database_items, 32):
            batch_features = np.asarray([feature_vector for path, feature_vector in batch])
            cropped_features = self.crop_features_vectors_to_query(query_crop, batch_features)
            batch_scores = self.similarity_measure(cropped_features, EvaluatingSpatially.avg_pool(query_image_features))
            scores.extend(batch_scores.flatten())  # Use queue heap that stores only best 100

        sorted_scores_idx = list(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
        return [database_items[score_idx][0] for score_idx in sorted_scores_idx]

def closest_match(query, features, num_results, distance):
    distances = distance([query], features)[0]
    return k_smallest_sorted(distances, num_results)

class EvaluatingRegions(EvaluationMechanism):
    def __init__(self, model, database, num_regions = None):
        self.model = model
        self.database = database
        self.num_regions = num_regions


    def features(self, images):
        crops = split_image_to_square_regions(region_size=self.model.input_shape, num_regions=self.num_regions)
        images_features = []
        for image in images:
            image_crops = [crop_image(image, crop) for crop in crops]
            crops_features = self.model.predict_on_images(image_crops)

            images_features.append([RegionFeatures(crop=crop, features=features) for crop, features in
                                    zip(crops, crops_features)])
        return images_features


    # def features(self, images):
    #     # type: (List[PIL.Image]) -> List[List[RegionFeatures]]
    #     return [self.features_on_image_regions(image) for image in images]

    #
    # def features_on_image_regions(self, image, regions=(4, 3)):
    #     crops, regions_images = map(list, zip(*split_image_to_regions(image, *regions)))
    #     regions_images.append(image)
    #
    #     regions_features = self.model.predict_on_images(regions_images)
    #
    #     images_features = [RegionFeatures(crop=crop, features=features) for crop, features in
    #                        zip(crops, regions_features)]
    #     images_features.append(RegionFeatures(crop=Crop(0, 0, 1, 1), features=regions_features[-1]))
    #
    #     return images_features

    # @staticmethod
    # def regions_overlap_ordering(query_crop: Crop, image_crops: List[Crop]):
    #     # Returns ordering of image  crops (their indexes) based on the overlap
    #     ious = [query_crop.iou(image_crop) for image_crop in image_crops]
    #     ious_sorted_indexes = list(sorted(range(len(ious)), key=lambda k: ious[k], reverse=True))
    #
    #     return ious_sorted_indexes
