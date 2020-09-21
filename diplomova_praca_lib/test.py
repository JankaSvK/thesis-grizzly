import numpy as np
import mxnet as mx
from collections import namedtuple

def get_network_fc(network_path, network_epoch, normalize_inputs):
    batch_def = namedtuple('Batch', ['data'])
    sym, arg_params, aux_params = mx.model.load_checkpoint(network_path, network_epoch)

    network = mx.mod.Module(symbol=sym.get_internals()['flatten0_output'],
                            label_names=None,
                            context=mx.gpu())
    network.bind(for_training=False,
                 data_shapes=[("data", (6, 3, 224, 224))])
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

resnext = get_network_fc("//diplomova_praca_lib/resnet/resnext-101-1", 40, True)
resnet = get_network_fc("//diplomova_praca_lib/resnet/resnet-152", 0, False)

image = np.zeros([6, 224, 224, 3], dtype=np.uint8)
features_for_image = np.concatenate([resnext(image), resnet(image)], 1)
print(features_for_image.shape)
