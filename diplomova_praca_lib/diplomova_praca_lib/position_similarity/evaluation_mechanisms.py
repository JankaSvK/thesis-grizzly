from typing import List

import PIL
import numpy as np

from diplomova_praca_lib.image_processing import split_image_to_regions, normalized_images
from diplomova_praca_lib.models import EvaluationMechanism
from diplomova_praca_lib.position_similarity.models import RegionFeatures, Crop
from diplomova_praca_lib.utils import batches, k_largest, sorted_indexes


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

class EvaluatingRegions2:
    def __init__(self, similarity_measure):
        self.similarity_measure = similarity_measure

    def best_match(self, query, features, num_results):
        similarities = self.similarity_measure([query], features)[0]
        largest_idxs = k_largest(similarities, num_results)
        largest_values = similarities[largest_idxs]
        return list(np.array(largest_idxs)[sorted_indexes(largest_values)])


class EvaluatingRegions(EvaluationMechanism):
    def __init__(self, similarity_measure, model, database):
        self.similarity_measure = similarity_measure
        self.model = model
        self.database = database

    def features(self, images):
        # type: (List[PIL.Image]) -> List[List[RegionFeatures]]
        return [self.features_on_image_regions(image) for image in images]

    def features_on_image_regions(self, image, regions=(4, 3)):
        crops, regions_images = map(list, zip(*split_image_to_regions(image, *regions)))
        regions_images.append(image)

        inputs = normalized_images(regions_images)
        regions_features = self.model.predict(inputs)

        images_features = [RegionFeatures(crop=crop, features=features) for crop, features in
                           zip(crops, regions_features)]
        images_features.append(RegionFeatures(crop=Crop(0, 0, 1, 1), features=regions_features[-1]))

        return images_features

    @staticmethod
    def regions_overlap_ordering(query_crop: Crop, image_crops: List[Crop]):
        # Returns ordering of image  crops (their indexes) based on the overlap
        ious = [query_crop.iou(image_crop) for image_crop in image_crops]
        ious_sorted_indexes = list(sorted(range(len(ious)), key=lambda k: ious[k], reverse=True))

        return ious_sorted_indexes

    def best_matches(self, query_crop, query_image):
        database_items = self.database.records
        query_image_features = self.model.predict(normalized_images([query_image]))[0]

        scores = []
        for path, features in database_items:
            highest_iou_region_idx = \
                (EvaluatingRegions.regions_overlap_ordering(query_crop, [rf.crop for rf in features]))[0]
            # TODO: why accesssing directly regions features did not work
            # print(db_item[1][highest_iou_region_idx].crop)
            db_feature = features[highest_iou_region_idx].features
            scores.append(self.similarity_measure([db_feature], [query_image_features])[0])

        sorted_scores_idx = list(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))
        return [database_items[score_idx][0] for score_idx in sorted_scores_idx]
