import collections


class ImageRecord:
    def __init__(self):
        self.filename = None
        self.representation = None
        self.regions_features = None
        self.spatial_features = None


class Crop:
    # Always stores information between [0,1]
    def __init__(self, top, left, width=None, height=None, bottom=None, right=None):
        self.top = top
        self.left = left
        if width and height:
            self.width = width
            self.height = height
        if bottom and right:
            self.bottom = bottom
            self.right = right

    def __repr__(self):
        return "%s(left=%s, top=%s, right=%s, bottom=%s)" % (
        self.__class__.__name__, self.left, self.top, self.right, self.bottom)

    def normalize(self, image_width, image_height):
        self.top /= image_height
        self.left /= image_width
        self.width /= image_width
        self.height /= image_height

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @right.setter
    def right(self, val):
        self.width = val - self.left

    @bottom.setter
    def bottom(self, val):
        self.height = val - self.top

    def iou(self, crop: "Crop"):

        # Determine coordinates of intersection
        left = max(self.left, crop.left)
        top = max(self.top, crop.top)
        right = min(self.right, crop.right)
        bottom = min(self.bottom, crop.bottom)

        # Compute the area of intersection rectangle
        intersection_area = max(0, right - left) * max(0, bottom - top)

        # Compute the area of both original crops
        a_area = self.width * self.height
        b_area = crop.width * crop.height

        iou = intersection_area / float(a_area + b_area - intersection_area)

        assert iou >= 0

        return iou


RegionFeatures = collections.namedtuple("RegionFeatures", ["crop", "features"])
UrlImage = collections.namedtuple('UrlImage', ['url', 'crop'])
PositionSimilarityRequest = collections.namedtuple('PositionSimilarityRequest', ['images'])
RegionsFeaturesRecord = collections.namedtuple("RegionsFeaturesRecord", ["filename", "regions_features"])
