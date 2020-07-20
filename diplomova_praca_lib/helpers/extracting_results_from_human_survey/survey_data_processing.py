from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from cv2.cv2 import norm

from helpers.extracting_results_from_human_survey.process_csv import heat_map_from_responses

images_path = r"C:\Users\janul\Desktop\thesis\images\selected_frames_first750"
dataset = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_only_bigger_10percent_316videos"
graph_dir = r"C:\Users\janul\Desktop\thesis_tmp_files\graphs"

target_images = ["v00125_s00001(f00000223-f00000261)_f00000246",
                 "v00236_s00001(f00000095-f00000323)_f00000212",
                 "v00005_s00039(f00003156-f00003195)_f00003177",
                 "v00309_s00051(f00004296-f00004753)_f00004338",
                 "v00012_s00008(f00001106-f00001737)_f00001169",
                 "v00016_s00086(f00009708-f00009754)_f00009733",
                 "v00026_s00016(f00001365-f00001433)_f00001403",
                 "v00029_s00059(f00003342-f00003408)_f00003375",
                 "v00030_s00089(f00004206-f00004258)_f00004232",
                 "v00037_s00018(f00003167-f00003255)_f00003212"]

displayed_faces = [1475, 692, 100, 1826, 1761, 1247, 393, 1993, 2006, 1556, 2042, 1057, 1865, 1151, 70, 1728, 361, 1857,
                   857, 855, 1317, 1753, 812, 861, 1319, 56, 1477, 1650, 1271, 420, 233, 1442, 485, 1114, 921, 931, 254,
                   411, 745, 1132, 532, 1265, 1605, 1749, 1040, 29, 1356, 519, 1576, 1831, 1318, 299, 1786, 1571, 576,
                   650, 432, 1913, 1764, 1519, 1245, 1226, 588, 966, 1627, 1553, 730, 65, 1567, 1090, 680, 231, 342,
                   124, 705, 275, 1473, 1417, 922, 1562, 710, 433, 429, 1054, 619, 974, 1192, 1281, 693, 1427, 807,
                   1112, 1986, 1600, 788, 344, 239, 1358, 351, 69]  # first row, second row, etc


def distance(a, b):
    # return cos_dist(a, b)
    return np.linalg.norm(a - b)


def cos_dist(a, b):
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


def distances_from_i(t):
    global features
    return [distance(features[t], features[j_displayed]) for j_displayed in displayed_faces]


def show_image(path):
    im = Image.open(Path(images_path, path))
    im.show()


def compare_random_average_dist_to_user(target_idxs):
    # Average distance from target to images selected by users
    for idx_target, heat_map in zip(target_idxs, heat_map_from_responses()):
        distx = np.mean(distances_from_i(idx_target))
        print("{:.3f}".format(distx), end=' ')

        flatten = lambda l: [item for sublist in l for item in sublist]
        heat_map_flatten = flatten(heat_map)

        dist = np.average(distances_from_i(idx_target), weights=np.array(heat_map_flatten))
        print("{:.3f}".format(dist))


def sort_map_based_on_distance(t):
    return np.argsort(distances_from_i(t))


def merge_heatmap_with_ranking(i_target, heatmap):
    global idxs_target
    assignments = []
    for i_face_map in sort_map_based_on_distance(idxs_target[i_target]):
        face_row, face_column = i_face_map // 10, i_face_map % 10
        assignments.append(heatmap[face_row][face_column])
    return assignments


def main():
    data = np.load(Path(dataset, "faces.npz"), allow_pickle=True)
    global paths, crops, features
    paths, crops, features = data['paths'], data['crops'], data['features']

    global idxs_target
    idxs_target = [list(paths).index(t[1:6] + "/" + t + ".jpg") for t in target_images]
    # compare_random_average_dist_to_user(idxs_target)

    print(distances_from_i(0)[70:80])
    print(np.argmin(distances_from_i(0)))
    # show_image(paths[displayed_faces[np.argmin(distance_from_target(5))]])
    # for i in np.argsort(distance_from_target(0))[:5]:
    #     show_image(paths[displayed_faces[i]])
    print(distance(features[idxs_target[2]], features[displayed_faces[45]]))

    plt.rcParams.update({'font.size': 14})

    all_assignments = []
    for i_target in range(10):
        assignments = merge_heatmap_with_ranking(i_target=i_target, heatmap=heat_map_from_responses()[i_target])
        all_assignments.append(assignments)

    xall_assignments = np.array(all_assignments)[(1, 3, 4, 6, 7, 8, 9), :]
    summed = np.sum(xall_assignments, axis=0)
    summed = summed * 3 / np.sum(summed)

    plt.bar(np.arange(len(summed)), summed, 1)
    plt.xlabel("k")
    plt.ylabel("p(k)")
    plt.ylim(0, 0.3)
    plt.savefig(Path(graph_dir, "survey_distribution_without_the_easy.pdf"))
    plt.show()

    plt.plot(np.cumsum(summed), label="Proposed features")
    print(np.cumsum(summed))
    plt.plot([0, 100], [0, 3], "--", label="Random selection")
    plt.xlabel("Rank")
    plt.ylabel("E[number of selected images]")
    plt.legend(loc='lower right')
    plt.savefig(Path(graph_dir, "survey_cumsum_without_the_easy.pdf"))
    plt.show()

    xall_assignments = np.array(all_assignments)[(1, 3, 6, 7, 8, 9), :]
    summed = np.sum(xall_assignments, axis=0)
    summed = summed * 3 / np.sum(summed)

    plt.bar(np.arange(len(summed)), summed, 1)
    plt.xlabel("k")
    plt.ylabel("p(k)")
    plt.ylim(0, 0.3)
    plt.savefig(Path(graph_dir, "survey_distribution_childless.pdf"))
    # plt.show()

    xall_assignments = np.array(all_assignments)[:, :]
    summed = np.sum(xall_assignments, axis=0)
    summed = summed * 3 / np.sum(summed)

    plt.bar(np.arange(len(summed)), summed, 1)
    plt.xlabel("k")
    plt.ylabel("p(k)")
    plt.savefig(Path(graph_dir, "survey_distribution_all.pdf"), bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    main()
