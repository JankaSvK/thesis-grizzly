import argparse
from pathlib import Path

from diplomova_praca_lib.face_features.map_features import SOM
from diplomova_praca_lib.storage import FileStorage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str)
    parser.add_argument('--output', default=None, type=str)
    args = parser.parse_args()

    # input = r"C:\Users\janul\Desktop\thesis_tmp_files\transformed_face_features"
    data = FileStorage.load_multiple_files_multiple_keys(path=args.input, retrieve_merged=['features', 'crops', 'paths'])
    features = data['features']
    som = SOM(som_shape=(300, 300))
    som.log_dir = Path(args.output)

    som.train_som(features=features, epochs=len(features) * 100)


if __name__ == '__main__':
    main()
