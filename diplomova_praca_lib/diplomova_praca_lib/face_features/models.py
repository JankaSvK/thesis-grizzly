import collections

FaceCrop = collections.namedtuple("FaceCrop", ['src', 'crop'])
FaceDetection = collections.namedtuple('FaceDetection', ['crop', 'encoding'])
FaceDetectionsRecord = collections.namedtuple('Record', ['filename', 'detections'])