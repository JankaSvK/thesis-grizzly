import base64
import re
from urllib.parse import urlparse

from PIL import Image
import requests
from io import BytesIO

from sklearn.metrics.pairwise import cosine_similarity

from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions, EvaluatingSpatially
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50, Resnet50Antepenultimate
from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest
from diplomova_praca_lib.position_similarity.ranking_mechanisms import RankingMechanism
from diplomova_praca_lib.storage import FileStorage, Database
from diplomova_praca_lib.utils import filename_without_extensions

database_regions = Database(FileStorage.load_data_from_file(r"C:\Users\janul\Desktop\saved_annotations\1000.npy"))
database_spatially = Database(FileStorage.load_data_from_file(r"C:\Users\janul\Desktop\saved_annotations\1000_spatially.npy"))

evaluating_regions = EvaluatingRegions(similarity_measure=cosine_similarity, model=Resnet50())
evaluating_spatially = EvaluatingSpatially(similarity_measure=cosine_similarity, model=Resnet50Antepenultimate())


def position_similarity_request(request: PositionSimilarityRequest):
    # TODO: only regions so far
    downloaded_images = [download_image(request_image.url) for request_image in request.images]

    best_matches = []
    for image_information, image in zip(request.images, downloaded_images):
        best_matches.append(evaluating_regions.best_matches(image_information.crop, image, database_regions.records))

    ranking = RankingMechanism.summing(best_matches)
    return [filename_without_extensions(path) for path in ranking]


def spatial_similarity_request(request: PositionSimilarityRequest):
    # Spatial similarity
    downloaded_images = [download_image(request_image.url) for request_image in request.images]

    best_matches = []
    for image_information, image in zip(request.images, downloaded_images):
        best_matches.append(evaluating_spatially.best_matches(image_information.crop, image, database_spatially.records))

    ranking = RankingMechanism.summing(best_matches)
    return [filename_without_extensions(path) for path in ranking]



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
