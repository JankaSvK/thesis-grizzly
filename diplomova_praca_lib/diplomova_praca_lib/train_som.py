import argparse
from pathlib import Path

from diplomova_praca_lib.face_features.map_features import SOM
from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import load_from_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str)
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--batch_epochs', default=100000, type=int)
    args = parser.parse_args()


    # input = r"C:\Users\janul\Desktop\thesis_tmp_files\transformed_face_features"
    data = FileStorage.load_multiple_files_multiple_keys(path=args.input, retrieve_merged=['features', 'crops', 'paths'])
    features = data['features']

    som = SOM(som_shape=(50, 50))
    som.log_dir = Path(args.output)

    if args.pretrained:
        som.som = load_from_file(args.pretrained)

    if args.epochs:
        training_epochs = args.epochs
    else:
        training_epochs = len(features) * 1000

    iterations = training_epochs // args.batch_epochs
    for i in range(iterations):
        print("Batch for training SOM", i, "out of", iterations)
        som.train_som(features=features, epochs=args.batch_epochs)



if __name__ == '__main__':
    main()
