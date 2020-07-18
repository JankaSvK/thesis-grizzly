from pathlib import Path

import numpy as np
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt


def gallery(array, ncols=8):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols

    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols, intensity))
    return result

def make_array(idxs, images_path, paths, crops):
    res = []
    for i in idxs:
        path = str(Path(images_path, paths[i]))
        image = Image.open(path).convert('RGB')
        crop = crops[i]

        image = image.crop(
            (crop.left * image.width, crop.top * image.height, crop.right * image.width, crop.bottom * image.height))

        image = image.resize((70, 70))
        res.append(np.asarray(image))

    return np.array(res)

def main():
    images_path = r"C:\Users\janul\Desktop\thesis\images\selected_frames_first750"
    dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_only_bigger_10percent_316videos"
    # dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_representatives"

    data = np.load(Path(dataset, "faces.npz"), allow_pickle=True)
    paths, crops, features = data['paths'], data['crops'], data['features']

    chunk_size = 32
    for i in range(len(paths) // chunk_size):
        # last non complete batch is not displayed
        array = make_array(range(i * chunk_size, (i + 1) * chunk_size), images_path, paths, crops)
        result = gallery(array)
        # plt.imshow(result)
        # plt.show()

    random_idxs = np.random.choice(len(paths), size=32, replace=False)
    array = make_array(random_idxs, images_path, paths, crops)
    result = gallery(array)
    plt.imshow(result)
    plt.show()


if __name__ == '__main__':
    main()

