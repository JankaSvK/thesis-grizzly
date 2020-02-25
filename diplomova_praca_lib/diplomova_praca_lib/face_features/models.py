import collections

FaceDetection = collections.namedtuple('FaceDetection', ['crop', 'encoding'])
FaceDetectionsRecord = collections.namedtuple('Record', ['filename', 'detections'])