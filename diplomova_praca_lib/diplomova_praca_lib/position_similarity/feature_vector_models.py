import logging
import tensorflow
import numpy as np


class FeatureVectorModel:
    def __init__(self):
        self.model = None

    def predict(self, model_input: np.ndarray):
        return self.model.predict(model_input)


class Resnet50(FeatureVectorModel):
    def __init__(self):
        super().__init__()

        # returns (1,2048)
        self.model = tensorflow.keras.applications.resnet50.ResNet50(weights='imagenet',
                                                                     pooling='avg',
                                                                     include_top=False)
        # logging.debug(self.model.summary())


class Resnet50Antepenultimate(FeatureVectorModel):
    def __init__(self):
        super().__init__()
        resnet50antepenultimate = tensorflow.keras.applications.resnet50.ResNet50(weights='imagenet', pooling='avg',
                                                                                  include_top=False)

        # returns shape (1, 7, 7, 2048)
        self.model = tensorflow.keras.models.Model(inputs=resnet50antepenultimate.input,
                                                   outputs=resnet50antepenultimate.get_layer('conv5_block3_out').output)
        # logging.debug(self.model.summary())


