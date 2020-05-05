import base64
import pickle
import re
from collections import defaultdict
from io import BytesIO
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances

from diplomova_praca_lib.image_processing import resize_with_padding
from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions, EvaluatingSpatially, \
    closest_match
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50, Resnet50Antepenultimate, MobileNetV2
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
        Environment.evaluating_regions = EvaluatingRegions(model=Resnet50(), database=Environment.database_regions)

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
        return [c for c in self.unique_crops if crop.iou(c) > 0]


def model_factory(model_repr):
    # Example: 'MobileNetV2(model=mobilenetv2_1.00_224, input_shape=(50, 50, 3))'
    class_name = model_repr[:model_repr.index('(')]
    input_shape = eval(model_repr[model_repr.index('input_shape=') + len('input_shape='):-1])

    class_object = {"MobileNetV2": MobileNetV2,
                    "Resnet50": Resnet50,
                    "Resnet50Antepenultimate": Resnet50Antepenultimate}[class_name]

    return class_object(input_shape=input_shape)

class RegionsEnvironment:
    def __init__(self, data_path):
        data = FileStorage.load_data_from_file(data_path)
        self.regions_data = RegionsData(data)

        self.pca = pickle.loads(data['pca'])
        self.scaler = pickle.loads(data['scaler'])
        if 'model' in data.keys():
            self.model = model_factory(str(data['model']))
        else:
            self.model = Resnet50(input_shape=(224,224,3))

    def pca_transform(self, features):
        return self.pca.transform(self.scaler.transform(features))


# regions_env = RegionsEnvironment(r"C:\Users\janul\Desktop\saved_annotations\experiments\compressed_featueres2\data.npz")
regions_env = RegionsEnvironment(
    r"C:\Users\janul\Desktop\saved_annotations\experiments\750_mobbilenetv2-12regions\data.npz")

def position_similarity_request(request: PositionSimilarityRequest):
    downloaded_images = [
        resize_with_padding(download_image(request_image.url), expected_size=regions_env.model.input_shape)
        for request_image in request.images]

    # downloaded_images = [downloaded_images[0]]
    if not downloaded_images:
        return []

    model_features = regions_env.model.predict_on_images(downloaded_images)
    pca_features = regions_env.pca_transform(model_features)

    matched_crop_idxs = []
    matched_src_idxs = []
    for image_info, pca_feature in zip(request.images, pca_features):
        related_crops = [regions_env.regions_data.highest_iou_crop(image_info.crop)]
        # related_crops = regions_env.regions_data.related_crops(image_info.crop) # TOOD need to prrocess features semarately
        related_crops_features_idxs = concatenate_lists([regions_env.regions_data.crop_idxs[c] for c in related_crops])
        features = regions_env.regions_data.features[related_crops_features_idxs]

        highest_similarity_features_idxs = closest_match(pca_feature, features, Environment.results_limit,
                                                         distance=cosine_distances)
        idxs_crops_sorted_by_distance = np.array(related_crops_features_idxs)[highest_similarity_features_idxs]
        # The address of image is needed, not the crop
        matched_crop_idxs.append(idxs_crops_sorted_by_distance)
        matched_src_idxs.append(
            [regions_env.regions_data.unique_src_paths[regions_env.regions_data.src_paths[i_crop]] for i_crop in
             idxs_crops_sorted_by_distance])

    # ranked_results = RankingMechanism.summing(matches)
    # ranked_results = RankingMechanism.borda_count(matched_crop_idxs)
    ranked_results = RankingMechanism.borda_count(matched_src_idxs)
    return [regions_env.regions_data.unique_src_paths_list[match] for match in ranked_results[:Environment.results_limit]]

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
