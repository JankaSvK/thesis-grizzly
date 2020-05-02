import numpy as np
import tensorflow
import tensorflow as tf

from diplomova_praca_lib.image_processing import resize_image


class FeatureVectorModel:
    def __init__(self, input_shape=(224, 224, 3)):
        self.model = None
        self.input_shape = input_shape

    def __repr__(self):
        return "%s(model=%s, input_shape=%s)" % (
            self.__class__.__name__, self.model.name, self.input_shape)

    def predict(self, model_input: np.ndarray):
        return self.model.predict(model_input)

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


class Resnet50(FeatureVectorModel):
    def __init__(self, input_shape=(224, 224, 3)):
        super().__init__()

        self.input_shape = input_shape

        # returns (1,2048)
        self.model = tensorflow.keras.applications.resnet50.ResNet50(weights='imagenet',
                                                                     pooling='avg',
                                                                     include_top=False, input_shape=input_shape)
        # logging.debug(self.model.summary())

    def preprocess_input(self, images):
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(images)


class MobileNetV2(FeatureVectorModel):
    def __init__(self, input_shape=(50, 50, 3)):
        super().__init__(input_shape)
        self.model = tensorflow.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',
                                                                            pooling='avg',
                                                                            include_top=False, input_shape=input_shape)

    def preprocess_input(self, images):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input(images)



class Resnet50Antepenultimate(FeatureVectorModel):
    def __init__(self):
        super().__init__()
        resnet50antepenultimate = tensorflow.keras.applications.resnet50.ResNet50(weights='imagenet', pooling='avg',
                                                                                  include_top=False)

        # returns shape (1, 7, 7, 2048)
        self.model = tensorflow.keras.models.Model(inputs=resnet50antepenultimate.input,
                                                   outputs=resnet50antepenultimate.get_layer('conv5_block3_out').output)
        # logging.debug(self.model.summary())


