import logging
from unittest import TestCase

import numpy as np

from diplomova_praca_lib.storage import FileStorage

logging.basicConfig(level=logging.INFO)


class TestFileStorage(TestCase):
    def test_load_features_datafiles(self):
        result = FileStorage.load_multiple_files_multiple_keys(r"C:\Users\janul\Desktop\output\test",
                                                               retrieve_merged=['crops', 'paths', 'features'],
                                                               retrieve_once=['pipeline', 'model'])

        self.assertEqual(len(result['paths']), len(result['crops']))
        self.assertEqual(len(result['features']), len(result['crops']))
        self.assertTrue('pipeline' in result)
        self.assertTrue('model' in result)
