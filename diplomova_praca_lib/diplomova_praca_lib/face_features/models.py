import collections
from typing import List, NamedTuple

from diplomova_praca_lib.position_similarity.models import Crop

FaceDetection = collections.namedtuple('FaceDetection', ['crop', 'encoding'])
FaceDetectionsRecord = collections.namedtuple('Record', ['filename', 'detections'])


Coords = collections.namedtuple('Coords', ['x', 'y'])


class FaceCrop(NamedTuple):
    crop: Crop
    src: str
    idx: int

    def __eq__(self, other):
        return self.src == other.src and self.crop == other.crop



class NoMoveError(Exception):
    pass


class ClosestFacesRequest(NamedTuple):
    face_id: int


class ClosestFacesResponse:
    closest_faces: List[FaceCrop]
    distances: List[float]

class FaceView:
    def __init__(self, max_height, max_width, top_left_x: int = 0, top_left_y: int = 0, bottom_right_x=None,
                 bottom_right_y=None):
        self.max_width = round(max_width)
        self.max_height = round(max_height)
        self.top_left = Coords(round(top_left_x), round(top_left_y))
        if bottom_right_x == None or bottom_right_y == None:
            self.bottom_right = Coords(max_width, max_height)
        else:
            self.bottom_right = Coords(bottom_right_x, bottom_right_y)

    def move_left(self, step):
        if self.top_left.x - step < 0:
            raise NoMoveError

        self.top_left = Coords(self.top_left.x - step, self.top_left.y)
        self.bottom_right = Coords(self.bottom_right.x - step, self.bottom_right.y)

    def move_right(self, step=1):
        if self.bottom_right.x + step >= self.max_width:
            raise NoMoveError

        self.top_left = Coords(self.top_left.x + step, self.top_left.y)
        self.bottom_right = Coords(self.bottom_right.x + step, self.bottom_right.y)

    def move_up(self, step):
        if self.top_left.y - step < 0:
            raise NoMoveError

        self.top_left = Coords(self.top_left.x, self.top_left.y - step)
        self.bottom_right = Coords(self.bottom_right.x, self.bottom_right.y - step)

    def move_down(self, step=1):
        if self.bottom_right.y + step >= self.max_height:
            raise NoMoveError

        self.top_left = Coords(self.top_left.x, self.top_left.y - step)
        self.bottom_right = Coords(self.bottom_right.x, self.bottom_right.y - step)

    def move_out(self, factor=0.5):
        self.top_left = Coords(x=max(0, round(self.top_left.x - 0.5 * (self.width() * factor))),
                               y=max(0, round(self.top_left.y - 0.5 * (self.height() * factor))))

        self.bottom_right = Coords(x=max(0, round(self.bottom_right.x + 0.5 * (self.width() * factor))),
                                   y=max(0, round(self.bottom_right.y + 0.5 * (self.height() * factor))))

    def move_in(self, factor, request: Coords):
        new_width = round(self.width() * factor)
        new_height = round(self.height() * factor)

        self.top_left = Coords(x=round(request.x - new_width / 2), y=round(request.y - new_height / 2))
        self.bottom_right = Coords(x=round(self.top_left.x + new_width / 2), y=round(self.top_left.y + new_height / 2))

    def width(self):
        return self.bottom_right.x - self.top_left.x

    def height(self):
        return self.bottom_right.y - self.top_left.y

    def shape(self):
        return self.bottom_right.y - self.top_left.y, self.bottom_right.x - self.top_left.x

    def __repr__(self):
        return "%s(left=%s, bottom=%s)" % (
            self.__class__.__name__, self.top_left, self.bottom_right)
