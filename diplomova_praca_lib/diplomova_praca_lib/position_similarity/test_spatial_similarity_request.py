from unittest import TestCase

from diplomova_praca_lib.experiments import retrieve_collages, collage_as_request, load_environment_again
from diplomova_praca_lib.position_similarity.position_similarity_request import spatial_similarity_request


class TestSpatial_similarity_request(TestCase):
    def test_spatial_similarity_request(self):
        for collage in retrieve_collages():
            request = collage_as_request(collage)
            response = spatial_similarity_request(request)
            print(response.searched_image_rank)
