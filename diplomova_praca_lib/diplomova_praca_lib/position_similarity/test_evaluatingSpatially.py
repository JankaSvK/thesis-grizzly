from unittest import TestCase

import numpy as np

from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingSpatially
from diplomova_praca_lib.position_similarity.models import Crop


class TestEvaluatingSpatially(TestCase):
    def test_crop_features_vectors_to_query(self):
        features = EvaluatingSpatially.crop_features_vectors_to_query(Crop(left=0.55, top=0, right=0.58, bottom=1),
                                                           np.ones((3, 7, 7, 1280)))
        self.assertEqual(features.shape, (3,7,1,1280))
