import collections


class ImageRecord:
    def __init__(self):
        self.filename = None
        self.representation = None
        self.regions_features = None
        self.spatial_features = None


class Crop:
    # Always stores information between [0,1]
    def __init__(self, top, left, width, height):
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def normalize(self, image_width, image_height):
        self.top /=  image_height
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
        intersection_area = max(0, right - left + 1) * max(0, bottom - top + 1)

        # Compute the area of both original crops
        a_area = self.width * self.height
        b_area = crop.width * crop.height

        iou = intersection_area / float(a_area + b_area - intersection_area)

        return iou


RegionFeatures = collections.namedtuple("RegionFeatures", ["crop", "features"])
UrlImage = collections.namedtuple('UrlImage', ['url', 'crop'])
PositionSimilarityRequest = collections.namedtuple('PositionSimilarityRequest', ['images'])
