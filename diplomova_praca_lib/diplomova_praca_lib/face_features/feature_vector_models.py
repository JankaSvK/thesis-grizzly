import PIL
import face_recognition

from diplomova_praca_lib import image_processing
from diplomova_praca_lib.face_features.models import FaceDetection
from diplomova_praca_lib.position_similarity.models import Crop


def face_features(image: PIL.Image):
    """
    Returns a position of all detected faces on the image with their feature encoding.
    :param image: PIL.Image
    :return: List of `FaceDetection`
    """
    np_image = image_processing.pil_image_to_np_array(image)
    # (top, right, bottom, left)
    face_locations = face_recognition.face_locations(np_image)  # , model='cnn'
    if not face_locations:
        return []

    face_features = face_recognition.face_encodings(np_image, known_face_locations=face_locations)

    im_width, im_height = image.size
    face_locations_relative = []
    for top, right, bottom, left in face_locations:
        face_locations_relative.append(
            Crop(top=top / im_height, left=left / im_width, bottom=bottom / im_height, right=right / im_width))

    return [FaceDetection(location, encoding) for location, encoding in zip(face_locations_relative, face_features)]
