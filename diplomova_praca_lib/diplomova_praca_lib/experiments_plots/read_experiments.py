from pathlib import Path

import numpy as np


def main():
    dir = r"C:\Users\janul\Desktop\thesis_tmp_files\responses"
    for file in Path(dir).iterdir():
        if file.is_dir():
            continue
        data = np.load(str(file), allow_pickle=True)
        try:
            exp_repr = data['exp_repr']
        except KeyError:
            exp_repr = data['experiment']
        print("File: {} contains {}".format(file.name, exp_repr))

if __name__ == '__main__':
    main()