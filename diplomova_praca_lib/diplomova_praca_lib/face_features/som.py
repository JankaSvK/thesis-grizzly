# Author: TODO
# Source:
# Limited to Windows
import itertools
from ctypes import *
import os


class DLLWithSOM:
    class PointWithId(Structure):
        _fields_ = [("first", c_double),
                    ("second", c_double),
                    ("clustId", c_int),
                    ("imageId", c_int),
                    ("sigma", c_double)]

    def __init__(self, dll_path, features_dimension, dataset_size):
        self.lib = WinDLL(dll_path)
        self.features_dimension = features_dimension
        self.dataset_size = dataset_size
        self.features_type = c_double * (dataset_size * features_dimension)
        self.lib.initSOM.restype = c_void_p
        self.features = self.features_type()
        self.som_object = self.lib.initSOM(pointer(self.features), self.dataset_size, self.features_dimension)

    def load_features(self, features):
        for i, f in enumerate(self.features):
            self.features[i] = features[i]

    def test_dll(self):
        number_of_input_ids = 1000
        Probabilities_t = c_double * number_of_input_ids

        test_probabs = Probabilities_t()
        for i in range(len(test_probabs)):
            test_probabs[i] = i / len(test_probabs)

        self.lib.testFunc2.restype = POINTER(Probabilities_t)
        test_new_probabs = self.lib.testFunc2(pointer(test_probabs), len(test_probabs))[0]

        epsilon = 1 / len(test_probabs)
        for i in range(len(test_probabs)):
            if abs(test_new_probabs[i] - test_probabs[i] * 2) > epsilon:
                print("DLL malfunctioned.")
                break
        else:
            self.lib.deleteTest(test_new_probabs)
            print("DLL is working correctly.")

    def som_representants(self, ids, dims=(4, 4)):
        xdim, ydim = dims

        Input_ids_t = c_size_t * len(ids)
        input_ids = Input_ids_t()
        for i, _ in enumerate(input_ids):
            input_ids[i] = ids[i]

        Probabilities_t = c_double * len(ids)
        probabilities = Probabilities_t()
        for i, _ in enumerate(probabilities):
            probabilities[i] = i / len(probabilities)

        # True to take point with highest probability as representative (false for weighted random)
        take_most_probable = False

        self.lib.getSOMRepresentants.restype = POINTER(DLLWithSOM.PointWithId)
        result = self.lib.getSOMRepresentants(pointer(input_ids), pointer(c_long(len(input_ids))),
                                              pointer(probabilities),
                                              c_void_p(self.som_object), take_most_probable, xdim, ydim, 15)

        output_ids = [r.imageId for r in result]
        self.lib.deletePointWithIdArray(result)
        return output_ids

class SOM:
    def __init__(self, features):
        self.features = features
        features_dimension = len(self.features[0])
        dataset_size = len(self.features)

        dll_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'CEmbedSomDLL.dll')
        self.dll_som = DLLWithSOM(dll_path, features_dimension, dataset_size)
        # self.dll_som = DLLWithSOM('./CEmbedSomDLL.dll', features_dimension, dataset_size)
        self.dll_som.test_dll()
        self.dll_som.load_features(list(itertools.chain(*self.features)))

    def som_representants(self, ids, dims):
        return self.dll_som.som_representants(ids, dims)
