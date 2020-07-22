from pathlib import Path

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import euclidean_distances
from matplotlib.gridspec import GridSpec

from diplomova_praca_lib.utils import closest_match
from helpers.visualizations.display_faces import make_array, gallery


def main():
    images_path = r"C:\Users\janul\Desktop\thesis\images\selected_frames_first750"
    dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_only_bigger_10percent_316videos"

    data = np.load(Path(dataset, "faces.npz"), allow_pickle=True)
    paths, crops, features = data['paths'], data['crops'], data['features']

    interesting_face = 150
    selected_features = features[interesting_face]
    closest_features_idxs, distances = closest_match(selected_features, features, distance=euclidean_distances)

    idxs = closest_features_idxs[:100]

    images = []
    for i in idxs:
        path = str(Path(images_path, paths[i]))
        image = Image.open(path).convert('RGB')
        crop = crops[i]

        image = image.crop(
            (crop.left * image.width, crop.top * image.height, crop.right * image.width, crop.bottom * image.height))

        image = image.resize((70, 70))
        images.append(image)
    #
    # faces_to_display = make_array(idxs[:12], images_path, paths, crops)
    # result = gallery(faces_to_display, ncols=6)
    # plt.imshow(result)
    # plt.show()

    matplotlib.rcParams.update({'font.size': 8})

    plt_w, plt_h = 8, 5

    fig, axs = plt.subplots(plt_h, plt_w, gridspec_kw = {'wspace':0, 'hspace':0.5})
    for y in range(plt_h):
        for x in range(plt_w):
            i = y * plt_w + x
            axs[y, x].imshow(images[i])
            axs[y, x].set_title("{:.2f}".format(distances[i]))
            axs[y, x].axis('off')
    plt.show()

if __name__ == '__main__':
    main()
