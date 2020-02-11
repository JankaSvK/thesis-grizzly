import os
from pathlib import Path

from PIL import Image
import requests
from io import BytesIO

from sklearn.metrics.pairwise import cosine_similarity

from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50
from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest
from diplomova_praca_lib.position_similarity.storage import FileStorage, Database

database = Database(FileStorage.load_data_from_file(r"C:\Users\janul\Desktop\saved_annotations\1000.npy"))

evaluating_regions = EvaluatingRegions(similarity_measure=cosine_similarity, model=Resnet50())

def position_similarity_request(request: PositionSimilarityRequest):
    # TODO: only regions so far
    downloaded_images = [download_image(request_image.url) for request_image in request.images]
    downloaded_images = [downloaded_images[0]] # TODO: multiple images need ranking mechanism

    best_matches = []
    for image_information, image in zip(request.images, downloaded_images):
        best_matches.append(evaluating_regions.best_matches(image_information.crop, image, database.records))

    # print(best_matches)
    # print([filename_without_extensions(path) for path in best_matches[0]])
    return [filename_without_extensions(path) for path in best_matches[0]]

    # return best_matches[0]


    # return ["00000003", "00000008", "00000004"]

def filename_without_extensions(path):
    return Path(path).stem

def download_image(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))

