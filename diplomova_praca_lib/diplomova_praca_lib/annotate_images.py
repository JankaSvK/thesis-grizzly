import argparse
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions, EvaluatingSpatially
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50, Resnet50Antepenultimate
from diplomova_praca_lib.position_similarity.models import RegionsFeaturesRecord
from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default=None, type=str, help="Path to image directory.")
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
            evaluation_mechanism = EvaluatingRegions(similarity_measure=cosine_similarity, model=features_model,
                                                     database=None)
        elif args.feature_model == 'resnet50antepenultimate':
            features_model = Resnet50Antepenultimate()
            evaluation_mechanism = EvaluatingSpatially(similarity_measure=cosine_similarity, model=features_model,
                                                       database=None)

        images_features = []
        directories = FileStorage.directories(args.images_dir) or [args.images_dir]
        print()
        print("Found %d directories." % len(directories))
        for directory in directories:
            print("Processing directory {}".format(directory))
            for images_data in batches(FileStorage.load_images_continuously(directory), batch_size=32):
                features = evaluation_mechanism.features([sample.image for sample in images_data])
                for image_features, image_data in zip(features, images_data):
                    images_features.append(
                        RegionsFeaturesRecord(filename=Path(image_data.filename).relative_to(args.images_dir),
                                              regions_features=image_features))

            FileStorage.save_data(args.save_location, filename(args.feature_model, Path(directory).name),
                                  data=images_features, src_dir=args.images_dir, model=args.feature_model)
            images_features = []

    elif args.feature_model == 'face_features':
        # TODO fix
        images_features = []
        # for image in FileStorage.load_images_continuously(args.images_dir):
        #     images_features.append(FaceDetectionsRecord(filename=image.filename, detections=face_features(image.image)))
        # FileStorage.save_data_to_file(args.save_location, images_features)

    else:
        raise ValueError('Unknown `feature_model`.')


def filename(feature_model, directory):
    return "model-{},dir-{}".format(feature_model, directory)

if __name__ == '__main__':
    main()
