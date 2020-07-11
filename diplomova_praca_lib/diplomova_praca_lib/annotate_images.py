import argparse
from pathlib import Path

from diplomova_praca_lib.face_features.feature_vector_models import EvaluatingFaces
from diplomova_praca_lib.models import DatabaseRecord
from diplomova_praca_lib.position_similarity.evaluation_mechanisms import EvaluatingRegions, EvaluatingSpatially, \
    EvaluatingWholeImage
from diplomova_praca_lib.position_similarity.feature_vector_models import MobileNetV2, MobileNetV2Antepenultimate, \
    Resnet50V2Antepenultimate, Resnet50V2, Resnet50_11k_classes
from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default=None, type=str, help="Path to image directory.")
    parser.add_argument("--save_location", default="", type=str,
                        help="Path to directory where precomputed models are saved.")
    parser.add_argument("--input_size", default=96, type=int, help="Input shape for model (square width)")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for processing")
    parser.add_argument("--num_regions", default=None, type=str, help="Number of regions \"vertically,horizzontaly\".")
    parser.add_argument('--feature_model', default='resnet50v2', type=str,
                        help='Feature vector model to compute (default: %(default)s)')
    args = parser.parse_args()

    input_shape = (args.input_size, args.input_size, 3)
    num_regions =  tuple(map(int, args.num_regions.split(","))) if args.num_regions else None

    if args.feature_model == 'resnet50v2' and num_regions:
        features_model = Resnet50V2(input_shape=input_shape)
        evaluation_mechanism = EvaluatingRegions(model=features_model, num_regions=num_regions)
    elif args.feature_model == 'resnet50v2':
        features_model = Resnet50V2(input_shape=input_shape)
        evaluation_mechanism = EvaluatingWholeImage(model=features_model)
    elif args.feature_model == 'resnet50v2antepenultimate':
        features_model = Resnet50V2Antepenultimate(input_shape=input_shape)
        evaluation_mechanism = EvaluatingSpatially(model=features_model)
    elif args.feature_model == 'mobilenetv2' and num_regions:
        features_model = MobileNetV2(input_shape=input_shape)
        evaluation_mechanism = EvaluatingRegions(model=features_model, num_regions=num_regions)
    elif args.feature_model == 'mobilenetv2':
        features_model = MobileNetV2(input_shape=input_shape)
        evaluation_mechanism = EvaluatingWholeImage(model=features_model)
    elif args.feature_model == 'mobilenetv2antepenultimate':
        features_model = MobileNetV2Antepenultimate(input_shape=input_shape)
        evaluation_mechanism = EvaluatingSpatially(model=features_model)
    elif args.feature_model == 'Resnet50_11k_classes':
        features_model = Resnet50_11k_classes()
        evaluation_mechanism = EvaluatingRegions(model=features_model, num_regions=num_regions)
    elif args.feature_model == 'faces':
        evaluation_mechanism = EvaluatingFaces()
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
        for images_data in batches(FileStorage.load_images_continuously(directory), batch_size=args.batch_size):
            features = evaluation_mechanism.features([sample.image for sample in images_data])
            for image_features, image_data in zip(features, images_data):
                images_features.append(
                    DatabaseRecord(filename=str(Path(image_data.filename).relative_to(args.images_dir).as_posix()),
                                   features=image_features))

        FileStorage.save_data(Path(args.save_location, filename(args.feature_model, Path(directory).name)),
                              data=images_features, src_dir=args.images_dir, model=repr(evaluation_mechanism.model))
        images_features = []



def filename(feature_model, directory, extension = ''):
    return "model-{},dir-{}{extension}".format(feature_model, directory, extension=extension)

if __name__ == '__main__':
    main()
