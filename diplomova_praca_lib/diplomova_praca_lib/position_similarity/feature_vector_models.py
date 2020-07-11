import re

import numpy as np
import tensorflow
import tensorflow as tf
from classification_models.tfkeras import Classifiers

from diplomova_praca_lib.image_processing import resize_image


def model_factory(model_repr):
    # Example: 'MobileNetV2(model=mobilenetv2_1.00_224, input_shape=(50, 50, 3))'

    # Obsolete: to support old generated data
    # if model_repr == 'resnet50antepenultimate':
    #     return Resnet50Antepenultimate(input_shape=(224, 224, 3))


    class_name, options = re.search(r'(\S+)\((.*)\)', model_repr).groups()
    input_shape = eval(re.search('input_shape=\(([^)]*)\)', options).groups()[0])

    class_object = {"MobileNetV2": MobileNetV2,
                    "MobileNetV2Antepenultimate": MobileNetV2Antepenultimate,
                    "Resnet50V2": Resnet50V2,
                    "Resnet50V2Antepenultimate": Resnet50V2Antepenultimate,
                    "Resnet50_11k_classes": Resnet50_11k_classes}[class_name]

    return class_object(input_shape=input_shape)

class FeatureVectorModel:
    def __init__(self, input_shape=(50, 50, 3)):
        self.model = None
        self.input_shape = input_shape

    def __repr__(self):
        return "%s(model=%s, input_shape=%s, output_shape=%s)" % (
            self.__class__.__name__, self.model.name, self.input_shape, self.model.output_shape)

    def predict(self, model_input: np.ndarray):
        return self.model.predict(model_input, batch_size=128)

    def resize_and_preprocess(self, images):
        images = [resize_image(tf.keras.preprocessing.image.img_to_array(x), target_shape=self.input_shape[:2])
                  for x in images]
        normalized_images = self.preprocess_input(np.stack(images))
        return normalized_images

    def preprocess_input(self, images):
        raise NotImplementedError()

    def predict_on_images(self, images):
        images = self.resize_and_preprocess(images)
        return self.predict(images)


class Resnet50V2(FeatureVectorModel):
    def __init__(self, input_shape=(224, 224, 3)):
        super().__init__(input_shape=input_shape)
        self.model = tensorflow.keras.applications.ResNet50V2(weights='imagenet', pooling='avg', include_top=False,
                                                              input_shape=input_shape)

    def preprocess_input(self, images):
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        return preprocess_input(images)

class Resnet50_11k_classes(FeatureVectorModel):
    def __init__(self, input_shape=(224, 224, 3)):
        super().__init__(input_shape=input_shape)
        Resnet50, self.preprocess_input_f = Classifiers.get('resnet50')
        self.model = Resnet50(input_shape=input_shape, weights='imagenet11k-places365ch', include_top=False, classes= 11586)

    def preprocess_input(self, images):
        return self.preprocess_input_f(images)

class Resnet50V2Antepenultimate(FeatureVectorModel):
    def __init__(self, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.model = tensorflow.keras.applications.ResNet50V2(weights='imagenet', pooling=None, include_top=False,
                                                              input_shape=input_shape)

    def preprocess_input(self, images):
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        return preprocess_input(images)


class MobileNetV2(FeatureVectorModel):
    def __init__(self, input_shape=(50, 50, 3)):
        super().__init__(input_shape=input_shape)
        self.model = tensorflow.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', pooling='avg',
                                                                            include_top=False, input_shape=input_shape)

    def preprocess_input(self, images):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input(images)


class MobileNetV2Antepenultimate(FeatureVectorModel):
    def __init__(self, input_shape=(50, 50, 3)):
        super().__init__(input_shape=input_shape)
        self.model = tensorflow.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', pooling=None,
                                                                            include_top=False, input_shape=input_shape)

    def preprocess_input(self, images):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input(images)
