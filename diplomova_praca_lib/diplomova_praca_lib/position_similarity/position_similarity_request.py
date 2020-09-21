import logging
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from tensorflow.python.keras.layers import GlobalAveragePooling2D

from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingSpatially
from diplomova_praca_lib.position_similarity.feature_vector_models import model_factory
from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest, Crop, PositionSimilarityResponse, \
    PositionMethod
from diplomova_praca_lib.position_similarity.ranking_mechanisms import RankingMechanism
from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import concatenate_lists, closest_match, download_and_preprocess


class RegionsData:
    def __init__(self, data):
        self.data = data
        self.features = data['features']
        self.src_paths = data['paths']
        self.crops = data['crops']

        self.crop_idxs = self._get_crop_idxs()
        self.unique_crops = list(self.crop_idxs.keys())
        self.unique_src_paths = {x: i for i, x in enumerate(set(self.src_paths))}
        max_string_len = max(len(src_path) for src_path in self.unique_src_paths.keys())
        self.unique_src_paths_list = np.zeros(len(self.unique_src_paths), dtype='<U{}'.format(max_string_len))
        for x, i in self.unique_src_paths.items():
            self.unique_src_paths_list[i] = x

    def _get_crop_idxs(self):
        crops_lists = defaultdict(list)
        for i_crop, crop in enumerate(self.crops):
            crops_lists[crop].append(i_crop)
        return crops_lists

    def highest_iou_crop(self, crop: Crop):
        max_iou = 0
        max_crop = None
        for c in self.unique_crops:
            if c.iou(crop) > max_iou:
                max_iou = c.iou(crop)
                max_crop = c

        return max_crop

    def related_crops(self, crop: Crop):
        """Sorted by IOU"""
        overlapping = [c for c in self.unique_crops if crop.iou(c) > 0]
        overlapping.sort(key=crop.iou, reverse=True)
        return overlapping

class Environment:
    def __init__(self):
        self.data = None
        self.padding = None
        self.initialized = False

    def available_images_in_dataset(self):
        if not self.data:
            return None
        return set(self.data['paths'])


class RegionsEnvironment(Environment):
    def __init__(self, data_path, ranking_func=np.mean, distance_func=cosine_distances):
        super().__init__()
        self.data_path = data_path
        self.ranking_func = ranking_func
        self.initialized = False
        self.maximum_related_crops = 1
        self.distance_func = distance_func

    def init(self):
        if self.initialized:
            return
        self.initialized = True

        print("Initializing environment, this may take a while.")
        self.data = FileStorage.load_multiple_files_multiple_keys(path=self.data_path,
                                                                  retrieve_merged=['features', 'crops', 'paths'],
                                                                  retrieve_once=['pipeline', 'model'])

        if not self.data:
            print("Data for Regions do not contain the correct information. Environment not initialized.")
            self.initialized = False
            return

        self.preprocessing = pickle.loads(self.data['pipeline'])
        self.model = model_factory(str(self.data['model']))
        self.data['features'] = np.array(self.data['features'])
        self.regions_data = RegionsData(self.data)

    def model_title(self):
        return str(self.data['model'])

class SpatialEnvironment(Environment):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.initialized = False
        self.files_limit = None
        self.full_image = False

    def init(self, **kwargs):
        if self.initialized:
            return
        self.initialized = True

        print("Initializing environment, this may take a while.")
        self.data = FileStorage.load_multiple_files_multiple_keys(path=self.data_path,
                                                                  retrieve_merged=['features', 'paths'],
                                                                  retrieve_once=['pipeline', 'model'],
                                                                  num_files_limit=self.files_limit, **kwargs)
        if not self.data:
            print("Data for Spatial do not contain the correct information. Environment not initialized.")
            self.initialized = False
            return

        self.preprocessing = pickle.loads(self.data['pipeline'])
        self.model = model_factory(str(self.data['model']))
        self.data['features'] = np.array(self.data['features'])
        self.features = self.data['features']
        print("Environment initialized.")

    def model_title(self):
        return str(self.data['model'])


class WholeImageEnvironment(Environment):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.initialized = False

    def init(self):
        if self.initialized:
            return
        self.initialized = True

        print("Initializing environment, this may take a while.")
        self.data = FileStorage.load_multiple_files_multiple_keys(path=self.data_path,
                                                                  retrieve_merged=['features', 'paths'],
                                                                  retrieve_once=['pipeline', 'model'])
        self.preprocessing = pickle.loads(self.data['pipeline'])
        self.model = model_factory(str(self.data['model']))
        self.data['features'] = np.array(self.data['features'])
        self.features = self.data['features']

    def model_title(self):
        return str(self.data['model'])

regions_env = None
spatial_env = None
whole_image_env = None

def initialize_env(env):
    global regions_env, spatial_env, whole_image_env

    if env is 'regions':
        regions_env.init()
    if env is 'spatial':
        spatial_env.init()
    if env is 'whole':
        whole_image_env.init()


def positional_request(request: PositionSimilarityRequest) -> PositionSimilarityResponse:
    if request.position_method == PositionMethod.REGIONS:
        return position_similarity_request(request)
    elif request.position_method == PositionMethod.SPATIALLY:
        return spatial_similarity_request(request)
    elif request.position_method == PositionMethod.WHOLE_IMAGE:
        return whole_image_similarity_request(request)
    else:
        return position_similarity_request(request)

def available_images(method: PositionMethod) -> Set[str]:
    return environment_select(method).available_images_in_dataset()

def environment_select(method: PositionMethod) -> Environment:
    if method == PositionMethod.REGIONS:
        return regions_env
    elif method == PositionMethod.SPATIALLY:
        return spatial_env
    elif method == PositionMethod.WHOLE_IMAGE:
        return whole_image_env


def position_similarity_request(request: PositionSimilarityRequest) -> PositionSimilarityResponse:
    global regions_env
    env = regions_env
    initialize_env('regions')
    downloaded_images = download_and_preprocess(request.images, regions_env.model.input_shape, padding=env.padding)

    if not downloaded_images:
        return PositionSimilarityResponse(ranked_paths=[])

    images_features = regions_env.model.predict_on_images(downloaded_images)
    images_features_transformed = regions_env.preprocessing.transform(images_features)

    crop_ranking_per_image = []

    for image_info, image_features in zip(request.images, images_features_transformed):
        related_crops = regions_env.regions_data.related_crops(image_info.crop)
        if regions_env.maximum_related_crops:
            related_crops = related_crops[:regions_env.maximum_related_crops]

        logging.info("%d related crops" % len(related_crops))
        related_crops_idxs = concatenate_lists((regions_env.regions_data.crop_idxs[c] for c in related_crops))
        features = regions_env.regions_data.features[related_crops_idxs]

        closest_features_idxs, distances = closest_match(image_features, features, distance=regions_env.distance_func)

        # Indexes with features are only a subset of whole data, we have to transform back
        closest_crops_idxs = np.array(related_crops_idxs)[closest_features_idxs]
        crop_ranking_per_image.append(zip(reversed(closest_crops_idxs), reversed(distances)))

    images_with_best_crops_and_distances = np.ones((len(regions_env.regions_data.unique_src_paths), len(request.images)), dtype=np.float32)

    for i_ranking, ranking in enumerate(crop_ranking_per_image):
        for crop_idx, distance in ranking:
            image = crop_idx_to_src_idx(crop_idx)
            images_with_best_crops_and_distances[image, i_ranking] = distance

    distances = regions_env.ranking_func(images_with_best_crops_and_distances, axis=1)
    ranked_results = np.argsort(distances)
    matched_paths = regions_env.regions_data.unique_src_paths_list[ranked_results]

    return PositionSimilarityResponse(
        ranked_paths=list(matched_paths),
        searched_image_rank=searched_image_rank(request.query_image, matched_paths),
        # matched_regions=image_src_with_best_regions(images_with_best_crops_and_distances),
        dissimilarity_scores=distances,
    )


def searched_image_rank(query_path: str, matched_paths: np.ndarray) -> Optional[int]:
    if not query_path:
        return None
    for i, this_path in enumerate(matched_paths):
        if this_path == query_path:
            return i
    return None

def image_src_with_best_regions(obtained_regions: Dict[int, List[Tuple[int, float]]]) -> Dict[str, List[Crop]]:
    return {
        regions_env.regions_data.unique_src_paths_list[id_]:
            [regions_env.regions_data.crops[crop_id] for crop_id, distance in values]
        for id_, values in obtained_regions.items()
    }


def best_crop_only(ranking):
    images_closest_crop = defaultdict(list)
    for crop_idx, distance in ranking:
        if crop_idx_to_src_idx(crop_idx) not in images_closest_crop:
            images_closest_crop[crop_idx_to_src_idx(crop_idx)] = (crop_idx, distance)
    return images_closest_crop


def crop_idx_to_src_idx(crop_idx):
    crop_src = regions_env.regions_data.src_paths[crop_idx]
    return regions_env.regions_data.unique_src_paths[crop_src]


def whole_image_similarity_request(request: PositionSimilarityRequest) -> PositionSimilarityResponse:
    global whole_image_env
    initialize_env('whole')

    downloaded_images = download_and_preprocess(request.images, whole_image_env.model.input_shape)

    if not downloaded_images:
        return PositionSimilarityResponse(ranked_paths=[])

    images_features = whole_image_env.model.predict_on_images(downloaded_images)
    images_features = whole_image_env.preprocessing.transform(images_features)

    rankings_per_query_image = []  # type: List[Tuple[List[int], List[float]]]
    for image_info, image_features in zip(request.images, images_features):
        features = whole_image_env.features
        closest_images, distances = closest_match(image_features, features, distance=cosine_distances)
        rankings_per_query_image.append((closest_images, distances))

    distances_per_image_per_query = defaultdict(list)
    for close_images, distances in rankings_per_query_image:
        for image, distance in zip(close_images, distances):
            distances_per_image_per_query[image].append(distance)

    ranked_results = [match_id for match_id, _ in
                      RankingMechanism.rank_func(list(distances_per_image_per_query.items()), func=np.mean)]

    matched_paths = [whole_image_env.data['paths'][match] for match in ranked_results]

    return PositionSimilarityResponse(
        ranked_paths=matched_paths,
        searched_image_rank=searched_image_rank(request.query_image, matched_paths),
    )


def spatial_similarity_request(request: PositionSimilarityRequest):
    global spatial_env
    env = spatial_env
    env.padding = None
    initialize_env('spatial')

    downloaded_images = download_and_preprocess(request.images, spatial_env.model.input_shape, padding=env.padding)

    if not downloaded_images:
        return PositionSimilarityResponse(ranked_paths=[])

    images_features = spatial_env.model.predict_on_images(downloaded_images)
    if 'enhance_dims' in spatial_env.preprocessing.named_steps:
        spatial_env.preprocessing.steps[-1][1].set_params(kw_args={"shape": images_features.shape[1:-1]})
    images_features = spatial_env.preprocessing.transform(images_features)
    images_features = GlobalAveragePooling2D()(images_features).numpy()

    rankings_per_query_image = []  # type: List[Tuple[List[int], List[float]]]
    for image_info, image_features in zip(request.images, images_features):
        db_features = spatial_env.features
        if not env.full_image:
            db_features = EvaluatingSpatially.crop_features_vectors_to_query(image_info.crop, db_features)
        db_features = GlobalAveragePooling2D()(db_features).numpy()

        closest_images, distances = closest_match(image_features, db_features, distance=cosine_distances)
        rankings_per_query_image.append((closest_images, distances))

    distances_per_image_per_query = defaultdict(list)
    for close_images, distances in rankings_per_query_image:
        for image, distance in zip(close_images, distances):
            distances_per_image_per_query[image].append(distance)

    ranked_results = [match_id for match_id, _ in
                      RankingMechanism.rank_func(list(distances_per_image_per_query.items()), func=np.mean)]

    matched_paths = [spatial_env.data['paths'][match] for match in ranked_results]

    return PositionSimilarityResponse(
        ranked_paths=matched_paths,
        searched_image_rank=searched_image_rank(request.query_image, matched_paths),
    )

