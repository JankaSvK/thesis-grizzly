from unittest import TestCase

from diplomova_praca_lib.experiments import retrieve_collages, collage_as_request
from diplomova_praca_lib.position_similarity.models import PositionMethod
from diplomova_praca_lib.position_similarity.position_similarity_request import positional_request


class TestSpatial_similarity_request(TestCase):
    def test_regions_similarity_request(self):
        collages= retrieve_collages()
        searched_ranks  = []
        for collage in collages[:10]:
            request = collage_as_request(collage)

            request.position_method = PositionMethod.REGIONS
            response = positional_request(request)
            searched_ranks.append(response.searched_image_rank)

        print(searched_ranks)
