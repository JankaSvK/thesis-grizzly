import abc
import collections
import logging
import os
import sqlite3
from _sha256 import sha256
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances

import diplomova_praca_lib
from diplomova_praca_lib.position_similarity.models import PositionSimilarityRequest, PositionMethod
from diplomova_praca_lib.position_similarity.position_similarity_request import positional_request, RegionsEnvironment, \
    SpatialEnvironment, WholeImageEnvironment
from diplomova_praca_lib.position_similarity.ranking_mechanisms import RankingMechanism
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
    THUMBNAILS_PATH = os.path.join("static", "images", "lookup", "thumbnails")
    query_image = path_from_css_background(collage.query, thumbnails_prefix=THUMBNAILS_PATH)
    images = eval(collage.images)
    return PositionSimilarityRequest(images=images_with_position_from_json(images), query_image=query_image)


def get_queries() -> List[PositionSimilarityRequest]:
    return [collage_as_request(collage) for collage in retrieve_collages()]


class Experiment:
    def __init__(self):
        self.input_data = None
        self.ranking_func = None
        self.method = None

    def num_images(self):
        return len(set(self.get_env().data['paths']))

    def set_method(self, requests):
        for r in requests:
            r.position_method = self.method
        return

    def save_results(self):
        pass

    @abc.abstractmethod
    def get_env(self):
        pass

    @abc.abstractmethod
    def update_env(self):
        pass

    def run(self, requests):
        print("Updating environment")
        self.update_env()
        self.set_method(requests)

        print("Running experiment")
        responses = []
        for i_request, request in enumerate(requests):
            print("Processing request", i_request + 1, "out of", len(requests))
            responses.append(positional_request(request))
        return responses

    def __repr__(self):
        options = dict(self.__dict__)
        for k, v in options.items():
            options[k] = getattr(v, "__name__", v)

        options_formatted = ",".join("{}={}".format(k,v) for k, v in sorted(options.items()))
        exp_repr = "{}({})".format(self.__class__.__name__, options_formatted)

        return exp_repr

class RegionsExperiment(Experiment):
    def __init__(self, input_data, ranking_func=np.min, maximum_related_crops=None, distance_func = cosine_distances, padding=None):
        super().__init__()
        self.method = PositionMethod.REGIONS
        self.input_data = input_data
        self.ranking_func = ranking_func
        self.maximum_related_crops = maximum_related_crops
        self.distance_metric = distance_func
        self.padding = padding



    def update_env(self):
        new_env = RegionsEnvironment(self.input_data)
        new_env.maximum_related_crops = self.maximum_related_crops
        new_env.ranking_func = self.ranking_func
        new_env.distance_func = self.distance_metric
        new_env.padding = self.padding

        diplomova_praca_lib.position_similarity.position_similarity_request.regions_env = new_env

    def get_env(self):
        return diplomova_praca_lib.position_similarity.position_similarity_request.regions_env


class SpatialExperiment(Experiment):
    def __init__(self, input_data, ranking_func, files_limit=None, paths_source=None, full_image=False):
        super().__init__()
        self.method = PositionMethod.SPATIALLY
        self.input_data = input_data
        self.ranking_func = ranking_func
        self.files_limit = files_limit
        self.selected_paths = paths_source
        self.full_image = full_image

    def update_env(self):
        new_env = SpatialEnvironment(self.input_data)
        new_env.ranking_func = self.ranking_func
        new_env.files_limit = self.files_limit
        new_env.full_image = self.full_image

        if self.selected_paths:
            new_env.init(key_filter=('paths', self.selected_paths))

        diplomova_praca_lib.position_similarity.position_similarity_request.spatial_env = new_env

    def get_env(self):
        return diplomova_praca_lib.position_similarity.position_similarity_request.spatial_env


class FullImageExperiment(Experiment):
    def __init__(self, input_data, ranking_func):
        super().__init__()
        self.method = PositionMethod.WHOLE_IMAGE
        self.input_data = input_data
        self.ranking_func = ranking_func

    def update_env(self):
        new_env = WholeImageEnvironment(self.input_data)
        new_env.ranking_func = self.ranking_func
        diplomova_praca_lib.position_similarity.position_similarity_request.whole_image_env = new_env

    def get_env(self):
        return diplomova_praca_lib.position_similarity.position_similarity_request.whole_image_env

def experiments(id, queries_paths=None):
    maximum_related_crops = [1, 2, 3, None]
    ranking_funcs = [np.min, np.max, np.mean]


    if id == 0:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_96x96_preprocess",
                                 np.min, 1)
    elif id == 1:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_96x96_preprocess",
                                 np.min, 2)
    elif id == 2:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_96x96_preprocess",
                                 np.mean, 3)
    elif id == 3:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_96x96_preprocess",
                                 np.min, None)
    elif id == 4:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_96x96_preprocess",
                                 np.mean, 1)
    elif id == 5:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_96x96_preprocess",
                                 np.max, 1)
    elif id == 6:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess",
                                 np.mean, None)
    elif id == 11:
        return FullImageExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_224x224_preprocess_pca08", np.min)
    elif id == 12:
        return FullImageExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_224x224_preprocess", np.mean)
    elif id == 13:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca64",
            np.mean, 3)
    elif id == 14:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca512",
            np.mean, 3)
    elif id == 15:
        return SpatialExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_antepenultimate_preprocess_sampled_40k", np.mean, files_limit=10)
    elif id == 16:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_5x3_96x96_preprocess_pca64",
            np.mean, 3)
    elif id == 17:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_5x3_96x96_preprocess_pca128",
            np.mean, 3)
    elif id == 18:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_5x3_96x96_preprocess_pca256",
            np.mean, 3)
    elif id == 19:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_5x3_96x96_preprocess",
            np.mean, 3)
    elif id == 20:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_5x3_96x96_preprocess_pca32",
            np.mean, 3)
    elif id == 21:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_5x3_96x96_preprocess_pca32",
            np.mean, 3, distance_func=euclidean_distances)
    elif id == 22:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca64",
            np.mean, 3, distance_func=euclidean_distances)
    elif id == 23:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_Resnet50_11k_classes_5x3_96x96_avg_pool_preprocess",
            np.mean, 3, distance_func=cosine_distances)
    elif id == 24:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_Resnet50_11k_classes_5x3_96x96_avg_pool_preprocess",
            np.mean, 3, distance_func=cosine_distances)
    elif id == 25:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_Resnet50_11k_classes_5x3_96x96_preprocess_pca32",
            np.mean, 3, distance_func=cosine_distances)
    elif id == 26:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_5x3_96x96_preprocess",
            np.mean, 3, distance_func=cosine_distances)
    elif id == 27:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_Resnet50_11k_classes_5x3_96x96_preprocesspreprocess_20k",
            np.mean, 3, distance_func=cosine_distances
        )
    elif id == 28:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_5x3_96x96_preprocess_20k",
            np.mean, 3, distance_func=cosine_distances
        )
    elif id == 29:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_20k",
            np.mean, 3, distance_func=cosine_distances
        )
    elif id == 30:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess",
                                 np.mean, 1)
    elif id == 31:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess",
                                 np.mean, 2)
    elif id == 32:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess",
                                 np.mean, 3)
    elif id == 33:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_3x2_160x160_preprocess",
            np.mean, 3)
    elif id == 34:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_96x96_preprocess",
                                 np.mean, 3)
    elif id == 35:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.mean, 3)
    elif id == 36:
        return SpatialExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_antepenultimate_preprocess_10k", np.mean)
    elif id == 37:
        return SpatialExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_antepenultimate_preprocess_10k", np.mean)
    elif id == 38:
        return SpatialExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_3ndtry_antepenultimate_preprocess_10k",
            np.mean)
    elif id == 39:
        return SpatialExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4thtry_antepenultimate_preprocess_10k",
            np.mean
        )
    elif id == 40:
        return SpatialExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4thtry_antepenultimate_preprocess_10k",
            np.mean, full_image=True
        )

    elif id == 41:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.mean, 3,
            distance_func=cosine_distances)

    elif id == 42:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.mean, 3,
            distance_func=euclidean_distances)
    elif id == 43:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.mean, 3,
            distance_func=manhattan_distances)
    elif id == 44:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess",
            RankingMechanism.mean_with_threshold, 3, distance_func=cosine_distances)
    elif id == 45:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.min, 3,
            distance_func=cosine_distances)
    elif id == 46:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.mean, 3,
            distance_func=cosine_distances)
    elif id == 47:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.max, 3,
            distance_func=cosine_distances)
    elif id == 48:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.mean, 3,
            distance_func=cosine_distances, padding='black')
    elif id == 49:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.mean, 3,
            distance_func=cosine_distances, padding='white')
    elif id == 50:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_4x2_128x128_preprocess", np.mean, 3,
            distance_func=cosine_distances, padding=None)
    elif id == 51:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca64",
                                 np.mean, 1)
    elif id == 52:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca128",
                                 np.mean, 1)
    elif id == 53:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca256",
                                 np.mean, 1)
    elif id == 54:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca512",
                                 np.mean, 1)
    elif id == 55:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca8",
                                 np.mean, 1)
    elif id == 56:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca16",
                                 np.mean, 1)
    elif id == 57:
        return RegionsExperiment(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca32",
                                 np.mean, 1)
    elif id == 58:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_Resnet50_11k_classes_5x3_96x96_preprocess_pca128",
            np.mean, 1)
    elif id == 59:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_resnet50v2_5x3_96x96_preprocess_pca128",
            np.mean, 1)
    elif id == 60:
        return RegionsExperiment(
            r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\750_mobilenetv2_5x3_96x96_preprocess_pca128",
            np.mean, 1)
    else:
        raise ValueError("Unknown experiment ID")


def main():
    experiment_save_dir = r"C:\Users\janul\Desktop\thesis_tmp_files\responses"

    requests = get_queries()

    exps = [experiments(i) for i in [58, 59, 60]]

    for exp in exps:
        try:
            print(exp.__repr__())
            if not exp:
                continue
            filename_hash = sha256(repr(exp).encode('utf-8')).hexdigest()
            responses_save_path = Path(experiment_save_dir, filename_hash).with_suffix(".npz")
            if (responses_save_path.exists()):
                print("Results already present.", responses_save_path)
                continue

            print("Output path:", responses_save_path)

            responses = exp.run(requests)
            FileStorage.save_data(responses_save_path, responses=responses, experiment=exp.__dict__, exp_repr=repr(exp),
                                  model=repr(exp.get_env().model), num_images=exp.num_images())
        except Exception:
            continue


if __name__ == '__main__':
    main()


