import face_recognition
import numpy as np

from diplomova_praca_lib.face_features.models import FaceDetection
from diplomova_praca_lib.image_processing import image_as_array
from diplomova_praca_lib.models import EvaluationMechanism
from diplomova_praca_lib.position_similarity.models import Crop


def features(self, images):
    images_np_array = [image_as_array(image) for image in images]
    image_height, image_width = images_np_array[0].shape[
                                :2]  # All images need to be same size, otherwise dont use batch
    batch_face_locations = face_recognition.batch_face_locations(images_np_array, batch_size=len(images),
                                                                 number_of_times_to_upsample=2)

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

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

# Load an image of Jacques Chirac
jc_orig = load_image(metadata[2].image_path())

# Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(jc_orig)

# Transform image using specified face landmark indices and crop image to 96x96
jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Show original image
plt.subplot(131)
plt.imshow(jc_orig)

# Show original image with bounding box
plt.subplot(132)
plt.imshow(jc_orig)
plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

# Show aligned image
plt.subplot(133)
plt.imshow(jc_aligned);

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]