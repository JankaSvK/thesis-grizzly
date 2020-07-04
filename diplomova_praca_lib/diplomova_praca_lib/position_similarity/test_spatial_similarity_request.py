from unittest import TestCase

from diplomova_praca_lib.experiments import retrieve_collages, collage_as_request, load_environment_again
from diplomova_praca_lib.position_similarity.models import PositionMethod
from diplomova_praca_lib.position_similarity.position_similarity_request import spatial_similarity_request, \
    whole_image_similarity_request


class TestSpatial_similarity_request(TestCase):
    def test_spatial_similarity_request(self):
        collages= retrieve_collages()
        collages_below_150 = [collages[1], collages[9], collages[6]]
        for collage in collages_below_150:
            request = collage_as_request(collage)
            request.position_method=PositionMethod.SPATIALLY
            response = spatial_similarity_request(request)
            # response = whole_image_similarity_request(request)
            # response = whole_image_similarity_request(request)
            print(response.searched_image_rank)
