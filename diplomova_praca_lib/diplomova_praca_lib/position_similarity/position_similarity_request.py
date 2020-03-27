import base64
import re
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions, EvaluatingSpatially
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50, Resnet50Antepenultimate
from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest
from diplomova_praca_lib.position_similarity.ranking_mechanisms import RankingMechanism
from diplomova_praca_lib.storage import FileStorage, Database


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

def position_similarity_request(request: PositionSimilarityRequest):
    # TODO: only regions so far
    downloaded_images = [download_image(request_image.url) for request_image in request.images]

    best_matches = []
    for image_information, image in zip(request.images, downloaded_images):
        best_matches.append(Environment.evaluating_regions.best_matches(image_information.crop, image))

    ranking = RankingMechanism.summing(best_matches)
    return [str(path) for path in ranking[:Environment.results_limit]]


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
