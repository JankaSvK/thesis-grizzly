import face_recognition
import numpy as np

from diplomova_praca_lib.face_features.models import FaceDetection
from diplomova_praca_lib.image_processing import image_as_array
from diplomova_praca_lib.models import EvaluationMechanism
from diplomova_praca_lib.position_similarity.models import Crop


class EvaluatingFaces(EvaluationMechanism):
    def __init__(self):
        self.model = None

    def features(self, images):
        images_np_array = [image_as_array(image) for image in images]
        image_height, image_width = images_np_array[0].shape[
                                    :2]  # All images need to be same size, otherwise dont use batch
        batch_face_locations = face_recognition.batch_face_locations(images_np_array, batch_size=len(images),
                                                                     number_of_times_to_upsample=1)

        detections = []
        for face_locations, image in zip(batch_face_locations, images_np_array):
            if not face_locations:
                detections.append([])
                continue

            crops = self.create_crops(width=image_width, height=image_height, face_locations=face_locations)
            face_features = self.face_features(image=image, face_locations=face_locations)

            detections.append(
                [FaceDetection(crop=crop, encoding=features) for crop, features in zip(crops, face_features)])

        return detections

    def face_features(self, image: np.ndarray, face_locations):
        return face_recognition.face_encodings(image, known_face_locations=face_locations)

    def create_crops(self, width, height, face_locations):
        return [Crop(top=top / height, left=left / width, bottom=bottom / height,
                     right=right / width) for top, right, bottom, left in face_locations]
