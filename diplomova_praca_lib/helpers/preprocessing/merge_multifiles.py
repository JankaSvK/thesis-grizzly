import argparse
from pathlib import Path

from diplomova_praca_lib.storage import FileStorage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    keys_merged = {'crops', 'paths', 'features'}
    first_file_name = str(next(Path(args.input).rglob("*.npz")))
    first_file = FileStorage.load_data_from_file(first_file_name)
    keys_available = set(first_file.keys())
    keys_once = keys_available - keys_merged

    data = FileStorage.load_multiple_files_multiple_keys(args.input, retrieve_merged=list(keys_available - keys_once),
                                                         retrieve_once=list(keys_once))

    filename = Path(first_file_name).name.split(',')[0]
    FileStorage.save_data(Path(args.output, filename), **data)

if __name__ == '__main__':
    main()
