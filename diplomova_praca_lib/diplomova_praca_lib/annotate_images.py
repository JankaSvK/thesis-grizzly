import argparse
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from diplomova_praca_lib.face_features.feature_vector_models import face_features
from diplomova_praca_lib.face_features.models import FaceDetectionsRecord
from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions, EvaluatingSpatially
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50, Resnet50Antepenultimate
from diplomova_praca_lib.position_similarity.models import RegionsFeaturesRecord
from diplomova_praca_lib.storage import FileStorage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="", type=str, help="Path to image directory.")
    parser.add_argument('--feature_model',
                        default='resnet50',
                        const='all',
                        nargs='?',
                        choices=['resnet50', 'resnet50antepenultimate', 'face_features'],
                        help='Feature vector model to compute (default: %(default)s)')
    parser.add_argument("--save_location", default="", type=str,
                        help="Path to directory where precomputed models are saved.")
    args = parser.parse_args()

    if args.feature_model == 'resnet50' or args.feature_model == 'resnet50antepenultimate':
        evaluation_mechanism = None
        if args.feature_model == 'resnet50':
            features_model = Resnet50()
            evaluation_mechanism = EvaluatingRegions(similarity_measure=cosine_similarity, model=features_model)
        elif args.feature_model == 'resnet50antepenultimate':
            features_model = Resnet50Antepenultimate()
            evaluation_mechanism = EvaluatingSpatially(similarity_measure=cosine_similarity, model=features_model)

        images_features = []
        directories = FileStorage.directories(args.images_dir) or [args.images_dir]
        for directory in directories:
            for image in FileStorage.load_images_continuously(directory):
                features = evaluation_mechanism.features(image.image)
                images_features.append(RegionsFeaturesRecord(filename=image.filename, regions_features=features))

            FileStorage.save_data(args.save_location, filename(args.feature_model, Path(directory).name),
                                  images_features)

    elif args.feature_model == 'face_features':
        # TODO fix
        images_features = []
        for image in FileStorage.load_images_continuously(args.images_dir):
            images_features.append(FaceDetectionsRecord(filename=image.filename, detections=face_features(image.image)))
        FileStorage.save_data_to_file(args.save_location, images_features)

    else:
        raise ValueError('Unknown `feature_model`.')


def filename(feature_model, directory):
    return "model-{},dir-{}".format(feature_model, directory)

if __name__ == '__main__':
    main()
