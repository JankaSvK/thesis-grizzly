import os

from django.apps import AppConfig

import diplomova_praca_lib
from diplomova_praca_lib.position_similarity.position_similarity_request import WholeImageEnvironment, RegionsEnvironment, SpatialEnvironment
from shared.utils import FEATURES_PATH


class PositionSimilarityConfig(AppConfig):
    name = 'position_similarity'
    def ready(self):
        regions_env = RegionsEnvironment(os.path.join(FEATURES_PATH, 'regions'))
        diplomova_praca_lib.position_similarity.position_similarity_request.regions_env = regions_env

        spatial_env = SpatialEnvironment(os.path.join(FEATURES_PATH, 'spatial'))
        diplomova_praca_lib.position_similarity.position_similarity_request.spatial_env = spatial_env
