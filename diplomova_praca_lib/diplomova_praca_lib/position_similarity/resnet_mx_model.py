import mxnet as mx
import numpy as np
from collections import namedtuple

class Resnet_MX():
    def __init__(self, input_shape):
        self.name = "resnet_mx"
        self.output_shape = 4096
        self.input_shape = input_shape
        self.resnext = self.get_network_fc("/resnet/resnext-101-1", 40, True)
        self.resnet = self.get_network_fc("/resnet/resnet-152", 0, False)

    def predict(self, images, batch_size):
        out =  np.concatenate([self.resnext(images), self.resnet(images)], 1)
        return out

    def get_network_fc(self, network_path, network_epoch, normalize_inputs):
        batch_def = namedtuple('Batch', ['data'])
        sym, arg_params, aux_params = mx.model.load_checkpoint(network_path, network_epoch)

        network = mx.mod.Module(symbol=sym.get_internals()['flatten0_output'],
                                label_names=None,
                                context=mx.gpu())
        network.bind(for_training=False,
                     data_shapes=[("data", (1, self.input_shape[2], self.input_shape[0], self.input_shape[2]))])
        network.set_params(arg_params, aux_params)

        def fc(image):
            image = image.astype(np.float32)
            if normalize_inputs:  # true for resnext101
                image = image - np.array([[[[123.68, 116.779, 103.939]]]], dtype=np.float32)
            image = np.transpose(image, [0, 3, 1, 2])
            inputs = batch_def([mx.nd.array(image)])

            network.forward(inputs)
            return network.get_outputs()[0].asnumpy()

        return fc
