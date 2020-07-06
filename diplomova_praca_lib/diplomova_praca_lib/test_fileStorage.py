import logging
from unittest import TestCase

from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import sample_image_paths

logging.basicConfig(level=logging.INFO)


class TestFileStorage(TestCase):
    regions_dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca64"
    antepenultimate_dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_antepenultimate_preprocess"
    antepenultimate_small = r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\test_antepenultimate"

    def test_load_features_datafiles(self):
        result = FileStorage.load_multiple_files_multiple_keys(
            self.regions_dataset,
            retrieve_merged=['crops', 'paths', 'features'],
            retrieve_once=['pipeline', 'model'])

        self.assertGreater(len(result['paths']), 0)
        self.assertEqual(len(result['paths']), len(result['crops']))
        self.assertEqual(len(result['features']), len(result['crops']))
        self.assertTrue('pipeline' in result)
        self.assertTrue('model' in result)

    def test_load_multiple_files_multiple_keys(self):
        paths = sample_image_paths(self.regions_dataset, 100)
        result = FileStorage.load_multiple_files_multiple_keys(
            self.antepenultimate_small,
            retrieve_merged=['paths', 'features'], key_filter=('paths', paths))

        self.assertEqual(len(paths), len(set(result['paths'])))
