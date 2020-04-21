# Create your models here.
class LayerInfo:
    def __init__(self, layer_index, top_left_x, top_left_y, shape):
        self.shape = shape
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.layer_index = layer_index
