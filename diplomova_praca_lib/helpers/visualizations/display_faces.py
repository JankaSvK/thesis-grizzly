import random
from pathlib import Path

import numpy as np
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt

from helpers.visualizations.visualize_request_stats import save_plot


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

        crop.left = max(0, crop.left - 0.05)
        crop.top = max(0, crop.top - 0.05)
        crop.bottom = min(1, crop.bottom + 0.05)
        crop.right = min(1, crop.right + 0.05)

        # w = min(crop.width, crop.height)
        # w *= image.width/image.height

        image = image.crop(
            (crop.left * image.width, crop.top * image.height, (crop.left + crop.width) * image.width, (crop.top + crop.height) * image.height))
            # (crop.left * w, crop.top * w, crop.right * w, crop.bottom * w))

        # print(image.size)
        image = image.resize((150, 150))
        res.append(np.asarray(image))

    return np.array(res)

def main():
    images_path = r"C:\Users\janul\Desktop\thesis\images\selected_frames_first750"
    # dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_only_bigger_10percent_316videos"
    dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_only_bigger_10percent_316videos"
    # dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_316videos"
    # dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_representatives"


    data = np.load(Path(dataset, "faces.npz"), allow_pickle=True)
    paths, crops, features = data['paths'], data['crops'], data['features']

    crops_sizes = [c.width * c.height for c in crops]
    import seaborn as sns
    sns.distplot(crops_sizes, kde=False, bins=np.linspace(0, 1, 21))
    plt.xlim(0,0.8)
    plt.xlabel("Area of image covered by a face")
    plt.ylabel("Number of faces")
    save_plot(plt, "faces_size_distribution")
    plt.show()


    # xs,ys = [],[]
    # for i, c_size in enumerate(sorted(crops_sizes)):
    #     xs.append(c_size)
    #     ys.append((i + 1) / len(crops_sizes))
    #
    # plt.plot(xs, ys)
    # plt.show()

    chunk_size = 32
    for i in range(len(paths) // chunk_size):
        # last non complete batch is not displayed
        array = make_array(range(i * chunk_size, (i + 1) * chunk_size), images_path, paths, crops)
        result = gallery(array)
        # plt.imshow(result)
        # plt.show()

    np.random.seed(42)
    random_idxs = np.random.choice(len(paths), size=100, replace=False)
    print(list(random_idxs))
    print(paths[random_idxs])
    array = make_array(random_idxs, images_path, paths, crops)
    result = gallery(array, ncols=10)

    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')

    plt.imshow(result)
    plt.savefig("random_faces.pdf", bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    main()

