import collections


class ImageRecord:
    def __init__(self):
        self.filename = None
        self.representation = None
        self.regions_features = None
        self.spatial_features = None


RegionFeatures = collections.namedtuple("RegionFeatures", ["crop", "features"])
