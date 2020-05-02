import base64
import pickle
import re
from collections import defaultdict
from io import BytesIO
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions, EvaluatingSpatially, \
    EvaluatingRegions2, closest_match
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50, Resnet50Antepenultimate
from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest, Crop
from diplomova_praca_lib.position_similarity.ranking_mechanisms import RankingMechanism
from diplomova_praca_lib.storage import FileStorage, Database
from diplomova_praca_lib.utils import concatenate_lists


class Environment:
    database_spatially = None
    database_regions = None
    evaluating_regions = None
    evaluating_spatially = None
    results_limit = 100

    @staticmethod
    def initialize(regions_path, spatially_path):
        Environment.database_regions = Database(FileStorage.load_datafiles(regions_path))
        Environment.evaluating_regions = EvaluatingRegions(similarity_measure=cosine_similarity, model=Resnet50(),
                                                           database=Environment.database_regions)

        Environment.database_spatially = Database(FileStorage.load_datafiles(spatially_path))

        Environment.evaluating_spatially = EvaluatingSpatially(similarity_measure=cosine_similarity,
                                                               model=Resnet50Antepenultimate(),
                                                               database=Environment.database_spatially)
#
# def position_similarity_request(request: PositionSimilarityRequest):
#     # TODO: only regions so far
#     downloaded_images = [download_image(request_image.url) for request_image in request.images]
#
#     best_matches = []
#     for image_information, image in zip(request.images, downloaded_images):
#         best_matches.append(Environment.evaluating_regions.best_matches(image_information.crop, image))
#
#     ranking = RankingMechanism.summing(best_matches)
#     return [str(path) for path in ranking[:Environment.results_limit]]


class RegionsData:
    def __init__(self, data):
        self.data = data
        self.features = data['features']
        self.src_paths = data['paths']
        self.crops = data['crops']
        self.pca = pickle.loads(data['pca'])
        self.scaler = pickle.loads(data['scaler'])

        self.crop_idxs = self._get_crop_idxs()
        self.unique_crops = self.crop_idxs.keys()

    def _get_crop_idxs(self):
        crops_lists = defaultdict(list)
        for i_crop, crop in enumerate(self.crops):
            crops_lists[crop].append(i_crop)
        return crops_lists

    def related_crops(self, crop: Crop):
        return [c for c in self.unique_crops if crop.iou(c) > 0]


class RegionsEnvironment:
    def __init__(self, data_path):
        data = FileStorage.load_data_from_file(data_path)
        self.regions_data = RegionsData(data)

        self.pca = pickle.loads(data['pca'])
        self.scaler = pickle.loads(data['scaler'])
        self.model = Resnet50()

    def pca_transform(self, features):
        return self.pca.transform(self.scaler.transform(features))

regions_env = RegionsEnvironment(r"C:\Users\janul\Desktop\saved_annotations\experiments\compressed_featueres2\data.npz")

def position_similarity_request(request: PositionSimilarityRequest):
    downloaded_images = [download_image(request_image.url) for request_image in request.images]
    if not downloaded_images:
        return []

    model_features = regions_env.model.predict_on_images(downloaded_images)
    pca_features = regions_env.pca_transform(model_features)

    matches = []
    for image_info, pca_feature in zip(request.images, pca_features):
        related_crops = regions_env.regions_data.related_crops(image_info.crop)
        related_crops_features_idxs = concatenate_lists([regions_env.regions_data.crop_idxs[c] for c in related_crops])
        features = regions_env.regions_data.features[related_crops_features_idxs]

        highest_similarity_features_idxs = closest_match(pca_feature, features, 100, similarity_measure=cosine_similarity)
        matches.append(np.array(related_crops_features_idxs)[highest_similarity_features_idxs])

    ranked_results = RankingMechanism.summing(matches)
    return [regions_env.regions_data.src_paths[match] for match in ranked_results[:Environment.results_limit]]

def spatial_similarity_request(request: PositionSimilarityRequest):
    # Spatial similarity
    downloaded_images = [download_image(request_image.url) for request_image in request.images]

    best_matches = []
    for image_information, image in zip(request.images, downloaded_images):
        best_matches.append(
            Environment.evaluating_spatially.best_matches(image_information.crop, image))

    ranking = RankingMechanism.summing(best_matches)
    return [str(path) for path in ranking[:Environment.results_limit]]


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
