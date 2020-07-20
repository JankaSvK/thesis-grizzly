from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from diplomova_praca_lib.experiments_plots.get_responses import get_queries


def save_plot(plt, filename):
    graph_dir = r"C:\Users\janul\Desktop\thesis_tmp_files\graphs"
    plt.savefig(Path(graph_dir, filename + ".pdf"), bbox_inches='tight')


def num_queries_in_request_plot(requests):
    num_queries = [len(r.images) for r in requests]
    x_ticks = range(min(num_queries), max(num_queries) + 1)
    y_ticks = [num_queries.count(x) for x in x_ticks]

    print("images placed", sum(num_queries))

    plt.rcParams.update({'font.size': 14})
    plt.bar(x_ticks, y_ticks)
    plt.xlabel('Number of images in collage')
    plt.ylabel('Number of requests')
    plt.title('Number of images in collages')
    save_plot(plt, "num_queries_in_request")
    plt.show()



def queries_size(requests):
    def crop_size(crop):
        return crop.width * crop.height

    all_images = []
    for r in requests:
        all_images += r.images

    all_images_crops = [i.crop for i in all_images]
    crops_sizes = list(map(crop_size, all_images_crops))

    sns.set()
    sns.set(font_scale=1.2)
    sns.distplot(crops_sizes, kde=False)
    plt.ylabel('Number of queries')
    plt.xlabel('Ratio of area covered')
    plt.title("Relative size of the queries")
    save_plot(plt, "queries_size")
    plt.show()


    print("All crops", len(crops_sizes))
    print("Average crop size", np.mean(crops_sizes))
    print("Sizes bigger than 0.8", np.sum(np.array(crops_sizes) > 0.8))


def main():
    requests = get_queries()

    num_queries_in_request_plot(requests)

    queries_size(requests)


if __name__ == '__main__':
    main()
