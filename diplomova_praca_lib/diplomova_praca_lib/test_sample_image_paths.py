from unittest import TestCase

from diplomova_praca_lib.utils import sample_image_paths


class TestSample_image_paths(TestCase):
    def test_sample_image_paths(self):
        num_samples = 100
        sampled_images = sample_image_paths(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca64", num_samples)
        self.assertEqual(len(set(sampled_images)), num_samples)
