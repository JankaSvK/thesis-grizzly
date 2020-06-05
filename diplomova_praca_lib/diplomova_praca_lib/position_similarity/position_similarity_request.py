import base64
import logging
import pickle
import re
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from diplomova_praca_lib.image_processing import resize_with_padding
from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions, EvaluatingSpatially
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50, Resnet50Antepenultimate, \
    model_factory
from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest, Crop, PositionSimilarityResponse
from diplomova_praca_lib.position_similarity.ranking_mechanisms import RankingMechanism
from diplomova_praca_lib.storage import FileStorage, Database
from diplomova_praca_lib.utils import concatenate_lists, closest_match

logging.basicConfig(level=logging.INFO)

class Environment:
    database_spatially = None
    database_regions = None
    evaluating_regions = None
    evaluating_spatially = None
    results_limit = 10000

    @staticmethod
    def initialize(regions_path, spatially_path):
        Environment.database_regions = Database(FileStorage.load_datafiles(regions_path))
        Environment.evaluating_regions = EvaluatingRegions(model=Resnet50(), database=Environment.database_regions)

        Environment.database_spatially = Database(FileStorage.load_datafiles(spatially_path))

        Environment.evaluating_spatially = EvaluatingSpatially(similarity_measure=cosine_similarity,
                                                               model=Resnet50Antepenultimate(),
                                                               database=Environment.database_spatially)

class RegionsData:
    def __init__(self, data):
        self.data = data
        self.features = data['features']
        self.src_paths = data['paths']
        self.crops = data['crops']

        self.crop_idxs = self._get_crop_idxs()
        self.unique_crops = list(self.crop_idxs.keys())
        self.unique_src_paths = {x: i for i, x in enumerate(set(self.src_paths))}
        self.unique_src_paths_list = len(self.unique_src_paths.keys()) * [None]
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


class RegionsEnvironment:
    def __init__(self, data_path, ranking_func=np.min):
        self.data = FileStorage.load_multiple_files_multiple_keys(path=data_path,
                                                                  retrieve_merged=['features', 'crops', 'paths'],
                                                                  retrieve_once=['pipeline', 'model'])
        self.preprocessing = pickle.loads(self.data['pipeline'])
        self.model = model_factory(str(self.data['model']))
        self.data['features'] = np.array(self.data['features'])
        self.regions_data = RegionsData(self.data)
        self.ranking_func = ranking_func
        self.maximum_related_crops = None

    def model_title(self):
        return str(self.data['model'])


class SpatialEnvironment:
    def __init__(self, data_path):
        self.data = FileStorage.load_multiple_files_multiple_keys(path=data_path,
                                                                  retrieve_merged=['features', 'paths'],
                                                                  retrieve_once=['pipeline', 'model'],
                                                                  num_files_limit=400)
        self.preprocessing = pickle.loads(self.data['pipeline'])
        self.model = model_factory(str(self.data['model']))
        self.data['features'] = np.array(self.data['features'])
        self.features = self.data['features']


regions_env = None
spatial_env = None


def initialize_env(env):
    global regions_env, spatial_env

    if not regions_env and env is 'regions':
        regions_env = RegionsEnvironment(r"C:\Users\janul\Desktop\output\2020-05-11_05-43-12_PM")
    if not spatial_env and env is 'spatial':
        # spatial_env = SpatialEnvironment(
        #     r"C:\Users\janul\Desktop\thesis_tmp_files\antepenultimate\resnet50-antepenultimate-preprocessed-no_transform\2020-06-01_04-29-23_PM")
        # spatial_env = SpatialEnvironment(
        #     r"C:\Users\janul\Desktop\thesis_tmp_files\antepenultimate\resnet50-antepenultimate-preprocessed-08pca\2020-06-03_04-25-02_PM")
        spatial_env = SpatialEnvironment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\antepenultimate\resnet50-antepenultimate-preprocessed-08pca\2020-06-03_10-36-16_PM")


def download_and_preprocess(images, shape):
    return [resize_with_padding(download_image(request_image.url), expected_size=shape) for request_image in images]

def position_similarity_request(request: PositionSimilarityRequest) -> PositionSimilarityResponse:
    initialize_env('regions')

    downloaded_images = download_and_preprocess(request.images, regions_env.model.input_shape)

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

        closest_features_idxs, distances = closest_match(image_features, features, distance=cosine_distances)

        # Indexes with features are only a subset of whole data, we have to transform back
        closest_crops_idxs = np.array(related_crops_idxs)[closest_features_idxs]
        crop_ranking_per_image.append(zip(closest_crops_idxs, distances))

    images_with_best_crops_and_distances = defaultdict(list)
    for ranking in crop_ranking_per_image:
        best_crop_per_image = best_crop_only(ranking)
        for image, value in best_crop_per_image.items():
            images_with_best_crops_and_distances[image].append(value)

    images_with_crop_distances = {id_: [distance for crop_id, distance in values]
                                  for id_, values in images_with_best_crops_and_distances.items()}

    images_with_crop_distances = list(images_with_crop_distances.items())

    ranked_results = [img_idx for img_idx, _ in
                      RankingMechanism.rank_func(images_with_crop_distances, func=regions_env.ranking_func)]

    matched_paths = [regions_env.regions_data.unique_src_paths_list[match] for match in ranked_results]

    return PositionSimilarityResponse(
        ranked_paths=matched_paths,
        searched_image_rank=searched_image_rank(request.query_image, matched_paths),
        matched_regions=image_src_with_best_regions(images_with_best_crops_and_distances)
    )


def searched_image_rank(query_path: str, matched_paths: List[str]) -> Optional[int]:
    if not query_path:
        return None
    try:
        return matched_paths.index(query_path)
    except ValueError:
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
        if not crop_idx_to_src_idx(crop_idx) in images_closest_crop:
            images_closest_crop[crop_idx_to_src_idx(crop_idx)] = (crop_idx, distance)
    return images_closest_crop



def crop_idx_to_src_idx(crop_idx):
    crop_src = regions_env.regions_data.src_paths[crop_idx]
    return regions_env.regions_data.unique_src_paths[crop_src]



def spatial_similarity_request(request: PositionSimilarityRequest):
    initialize_env('spatial')

    downloaded_images = download_and_preprocess(request.images, spatial_env.model.input_shape)

    if not downloaded_images:
        return PositionSimilarityResponse(ranked_paths=[])

    images_features = spatial_env.model.predict_on_images(downloaded_images)
    if 'enhance_dims' in spatial_env.preprocessing.named_steps:
        spatial_env.preprocessing.steps[3][1].set_params(kw_args={"shape": images_features.shape[1:-1]})
    images_features = spatial_env.preprocessing.transform(images_features)
    images_features = np.mean(images_features, axis=(1, 2))

    rankings_per_query_image = []  # type: List[Tuple[List[int], List[float]]]
    for image_info, image_features in zip(request.images, images_features):
        features = spatial_env.features
        features = EvaluatingSpatially.crop_features_vectors_to_query(image_info.crop, features)

        closest_images, distances = closest_match(image_features, features, distance=cosine_distances)
        rankings_per_query_image.append((closest_images, distances))

    distances_per_image_per_query = defaultdict(list)
    for close_images, distances in rankings_per_query_image:
        for image, distance in zip(close_images, distances):
            distances_per_image_per_query[image].append(distance)

    ranked_results = [match_id for match_id, _ in
                      RankingMechanism.rank_func(list(distances_per_image_per_query.items()), func=np.average)]

    matched_paths = [spatial_env.data['paths'][match] for match in ranked_results]

    return PositionSimilarityResponse(
        ranked_paths=matched_paths,
        searched_image_rank=searched_image_rank(request.query_image, matched_paths),
    )

def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def download_image(image_src):
    if is_url(image_src):
        response = requests.get(image_src)
        return Image.open(BytesIO(response.content))
    else:
        image_data = re.sub('^data:image/.+;base64,', '', image_src)
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            alpha = image.convert('RGBA').split()[-1]
            bg_colour = (255, 255, 255)
            image_without_transparency = Image.new("RGBA", image.size, bg_colour + (255,))
            image_without_transparency.paste(image, mask=alpha)

            return image_without_transparency.convert('RGB')
        else:
            return image
