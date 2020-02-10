from PIL import Image
import requests
from io import BytesIO

from sklearn.metrics.pairwise import cosine_similarity

from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50
from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest


def position_similarity_request(request: PositionSimilarityRequest):
    # TODO: only regions so far

    downloaded_images = [download_image(request_image.url) for request_image in request.images]
    # downloaded_images = [downloaded_images[0]] # TODO: multiple images need ranking mechanism
    #
    # features_score = []
    # for db_item in db_items:
    #     highest_iou_region_idx = EvaluatingRegions.regions_overlap_ordering(query_crop, db_items.regions)[0]
    #     db_feature = db_item.features[highest_iou_region_idx]
    #
    #     features_score.append(cosine_similarity(db_feature, downloaded_images))

    # best_match_idxs = sorted(lambda k)
    # return best_match_idxs


    return ["00000003", "00000008", "00000004"]

def download_image(image_url):
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))

