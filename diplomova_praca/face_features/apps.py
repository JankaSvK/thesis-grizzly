import os

from django.apps import AppConfig

import diplomova_praca_lib
from diplomova_praca_lib.face_features.face_features_request import Environment
from shared.utils import FEATURES_PATH


class FaceFeaturesConfig(AppConfig):
    name = 'face_features'

    def ready(self):
        environment = Environment(os.path.join(FEATURES_PATH, 'faces', 'features'),
                                  os.path.join(FEATURES_PATH, 'faces', 'som'))
        diplomova_praca_lib.face_features.face_features_request.env = environment
