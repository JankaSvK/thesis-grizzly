import abc
import collections

import json


class EvaluationMechanism:
    @abc.abstractmethod
    def features(self, images):
        pass

DatabaseRecord = collections.namedtuple("DatabaseRecord", ["filename", "features"])


class Serializable:
    raw_init_params = []
    serializable_init_params = {}

    def __init__(self, **kwargs):
        assert set(self.raw_init_params) | set(self.serializable_init_params.keys()) >= set(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)
    def serialize(self):
        to_serialize = {}
        for key in self.raw_init_params:
            to_serialize[key] = getattr(self, key)

        for key in self.serializable_init_params:
            to_serialize[key] = getattr(self, key).serialize()

        return json.dumps(to_serialize)

    @classmethod
    def deserialize(cls, serialized):
        deserialized = json.loads(serialized)
        for key in cls.serializable_init_params:
            deserialized[key] = cls.serializable_init_params[key].deserialize(deserialized[key])

        return cls(**deserialized)
