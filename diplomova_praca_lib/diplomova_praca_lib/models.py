import abc
import collections


class EvaluationMechanism:
    @abc.abstractmethod
    def features(self, images):
        pass

DatabaseRecord = collections.namedtuple("DatabaseRecord", ["filename", "features"])
