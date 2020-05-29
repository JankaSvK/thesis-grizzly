from unittest import TestCase

from diplomova_praca_lib.face_features.face_features_request import images_with_closest_faces
from diplomova_praca_lib.face_features.models import ClosestFacesRequest


class TestImages_with_closest_faces(TestCase):
    def test_images_with_closest_faces(self):
        images_with_closest_faces(ClosestFacesRequest())
