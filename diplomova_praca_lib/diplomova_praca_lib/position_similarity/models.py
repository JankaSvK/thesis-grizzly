import collections


class ImageRecord:
    def __init__(self):
        self.filename = None
        self.representation = None
        self.regions_features = None
        self.spatial_features = None


class Crop:
    """
    Always stores information between [0,1]
    """
    def __init__(self, left, top,  width=None, height=None, right=None, bottom=None):
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

    def __hash__(self):
        return hash((self.top, self.left, self.width, self.height))

    def __eq__(self, other):
        return (
                isinstance(other, Crop)
                and self.top == other.top
                and self.left == other.left
                and self.width == other.width
                and self.height == other.height
        )

    def normalize(self, image_width, image_height):
        self.top /= image_height
        self.left /= image_width
        self.width /= image_width
        self.height /= image_height

    def size_up(self, top, left, bottom, right):
        """
        Stretches the borders of regions by `value` in each direction and returns the new Crop
        :param value: Size of the stretch
        :returns: New stretched crop
        """
        new_crop = Crop(self.top - top, self.left - left, self.bottom + bottom, self.right + right)
        new_crop.check_01()
        return new_crop

    def size_up_to_square(self):
        """
        Size up current crop to the square (centered).
        :return: New crop
        """
        max_dim = max(self.height, self.width)
        x_missing = max_dim - self.width
        y_missing = max_dim - self.height
        return self.size_up(y_missing / 2, x_missing / 2, y_missing / 2, x_missing / 2)

    def area(self):
        return self.width * self.height

    def check_01(self):
        """
        Sets all the borders to [0,1].
        """
        self.top = max(0, self.top)
        self.left = max(0, self.left)
        self.bottom = min(1, self.bottom)
        self.right = min(1, self.right)

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
ImageData = collections.namedtuple("Image", ["filename", "image"])
