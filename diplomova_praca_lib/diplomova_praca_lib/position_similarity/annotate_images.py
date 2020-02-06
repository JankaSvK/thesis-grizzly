import argparse
import collections
import os

from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50
from diplomova_praca_lib.position_similarity.storage import FileStorage
from sklearn.metrics.pairwise import cosine_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="", type=str, help="Path to image directory.")
    parser.add_argument('--feature_model',
                        default='resnet50',
                        const='all',
                        nargs='?',
                        choices=['resnet50', 'resnet50antepenultimate'],
                        help='Feature vector model to compute (default: %(default)s)')
    parser.add_argument("--save_location", default="", type=str,
                        help="Path to directory where precomputed models are saved.")
    args = parser.parse_args()


    RegionsFeaturesRecord = collections.namedtuple("RegionsFeaturesRecord", ["filename", "regions_features"])
    features_model = Resnet50()  # TODO: change based on arguments
    evaluation_mechanism = EvaluatingRegions(similarity_measure=cosine_similarity, model=features_model)

    images_regions_features = []
    for image in FileStorage.load_images_continuously(args.images_dir):
        regions_features = evaluation_mechanism.features_on_image_regions(image.image)
        images_regions_features.append(
            RegionsFeaturesRecord(filename=image.filename, regions_features=regions_features))

    FileStorage.save_data_to_file(os.path.join(args.save_location, "tmp.npy"), images_regions_features)


if __name__ == '__main__':
    main()
