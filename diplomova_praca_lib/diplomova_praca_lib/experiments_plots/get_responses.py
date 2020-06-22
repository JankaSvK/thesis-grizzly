import abc
import collections
import logging
import sqlite3
from pathlib import Path
from typing import List

import numpy as np

import diplomova_praca_lib
from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest, PositionMethod
from diplomova_praca_lib.position_similarity.position_similarity_request import positional_request, RegionsEnvironment, \
    SpatialEnvironment
from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import images_with_position_from_json, path_from_css_background

logging.basicConfig(level=logging.INFO)

Collage = collections.namedtuple('Collage', "id timestamp query images")


def retrieve_collages() -> List[Collage]:
    conn = sqlite3.connect(r'C:\Users\janul\Desktop\thesis\code\diplomova_praca\db.sqlite3')
    conn.row_factory = (lambda cursor, row: Collage(*row))
    c = conn.cursor()
    c.execute('SELECT * FROM position_similarity_collage')
    fetched_queries = c.fetchall()
    return fetched_queries


def collage_as_request(collage: Collage) -> PositionSimilarityRequest:
    query_image = path_from_css_background(collage.query)
    images = eval(collage.images)
    return PositionSimilarityRequest(images=images_with_position_from_json(images), query_image=query_image)


def get_queries() -> List[PositionSimilarityRequest]:
    return [collage_as_request(collage) for collage in retrieve_collages()]


class Experiment:
    def __init__(self):
        self.input_data = None
        self.ranking_func = None
        self.method = None

    def set_method(self, requests):
        for r in requests:
            r.position_method = self.method
        return

    def save_results(self):
        pass

    @abc.abstractmethod
    def update_env(self):
        pass

    def run(self, requests):
        self.update_env()
        self.set_method(requests)

        return [positional_request(r) for r in requests]

    def __repr__(self):
        options = dict(self.__dict__)
        options['input_data'] = Path(options['input_data']).parts[-1]
        for k, v in options.items():
            options[k] = getattr(v, "__name__", v)

        options_formatted = ",".join("{}={}".format(k,v) for k, v in sorted(options.items()))

        return "{}({})".format(self.__class__.__name__, options_formatted)

class RegionsExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.method = PositionMethod.REGIONS
        self.input_data = None
        self.ranking_func = np.min
        self.maximum_related_crops = None

    def update_env(self):
        new_env = RegionsEnvironment(self.input_data)
        new_env.maximum_related_crops = self.maximum_related_crops
        new_env.ranking_func = self.ranking_func

        diplomova_praca_lib.position_similarity.position_similarity_request.regions_env = new_env

    def get_env(self):
        return diplomova_praca_lib.position_similarity.position_similarity_request.regions_env


class SpatialExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.method = PositionMethod.SPATIALLY
        self.input_data = None
        self.ranking_func = np.min

    def update_env(self):
        new_env = SpatialEnvironment(self.input_data)
        new_env.ranking_func = self.ranking_func

        diplomova_praca_lib.position_similarity.position_similarity_request.spatial_env = new_env


def experiments(id):
    exp = None
    if id == 1:
        exp = RegionsExperiment()
        exp.input_data  =r"C:\Users\janul\Desktop\output\2020-05-11_05-43-12_PM"
    elif id == 2:
        exp = SpatialExperiment()
        exp.input_data = r""

    return exp

def main():
    experiment_save_dir = r"C:\Users\janul\Desktop\thesis_tmp_files\responses"

    exp = experiments(1)

    responses_save_path = Path(experiment_save_dir, repr(exp)).with_suffix(".npz")
    if (responses_save_path.exists()):
        print("Results already present.")
        return

    print("Output path:", responses_save_path)
    requests = get_queries()
    responses = exp.run(requests)
    FileStorage.save_data(responses_save_path, responses=responses, experiment=exp.__dict__,
                          model=repr(exp.get_env().model))



if __name__ == '__main__':
    main()


