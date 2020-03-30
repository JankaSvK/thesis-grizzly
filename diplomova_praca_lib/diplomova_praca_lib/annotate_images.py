import argparse
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

from diplomova_praca_lib.face_features.feature_vector_models import EvaluatingFaces
from diplomova_praca_lib.models import DatabaseRecord
from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions, EvaluatingSpatially
from diplomova_praca_lib.position_similarity.feature_vector_models import Resnet50, Resnet50Antepenultimate
from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default=None, type=str, help="Path to image directory.")
    parser.add_argument('--feature_model',
                        default='resnet50',
                        const='all',
                        nargs='?',
                        choices=['resnet50', 'resnet50antepenultimate', 'faces'],
                        help='Feature vector model to compute (default: %(default)s)')
    parser.add_argument("--save_location", default="", type=str,
                        help="Path to directory where precomputed models are saved.")
    args = parser.parse_args()

    evaluation_mechanism = None
    if args.feature_model == 'resnet50':
        features_model = Resnet50()
        evaluation_mechanism = EvaluatingRegions(similarity_measure=cosine_similarity, model=features_model,
                                                 database=None)
    elif args.feature_model == 'resnet50antepenultimate':
        features_model = Resnet50Antepenultimate()
        evaluation_mechanism = EvaluatingSpatially(similarity_measure=cosine_similarity, model=features_model,
                                                   database=None)
    elif args.feature_model == 'faces':
        evaluation_mechanism = EvaluatingFaces()  # images_features.append(FaceDetectionsRecord(filename=image.filename, detections=face_features(image.image)))
    else:
        raise ValueError('Unknown `feature_model`.')

    directories = FileStorage.directories(args.images_dir) or [args.images_dir]
    print("Found %d directories." % len(directories))

    images_features = []
    for directory in directories:
        save_location = Path(args.save_location, filename(args.feature_model, Path(directory).name, extension='.npz'))
        if save_location.exists():
            print("Skipping directory {}".format(directory))
            continue

        print("Processing directory {}".format(directory))
        for images_data in batches(FileStorage.load_images_continuously(directory), batch_size=32):
            features = evaluation_mechanism.features([sample.image for sample in images_data])
            for image_features, image_data in zip(features, images_data):
                images_features.append(
                    DatabaseRecord(filename=str(Path(image_data.filename).relative_to(args.images_dir).as_posix()),
                                   features=image_features))

        FileStorage.save_data(Path(args.save_location, filename(args.feature_model, Path(directory).name)),
                              data=images_features, src_dir=args.images_dir, model=args.feature_model)
        images_features = []



def filename(feature_model, directory, extension = ''):
    return "model-{},dir-{}{extension}".format(feature_model, directory, extension=extension)

if __name__ == '__main__':
    main()
