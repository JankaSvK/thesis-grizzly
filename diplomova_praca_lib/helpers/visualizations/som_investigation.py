import numpy as np

from diplomova_praca_lib.face_features.map_features import SOM
from diplomova_praca_lib.storage import FileStorage
from diplomova_praca_lib.utils import load_from_file
from helpers.extracting_results_from_human_survey.survey_data_processing import show_image


def main():
    data_path = r"C:\Users\janul\Desktop\thesis_tmp_files\face_features_only_bigger_10percent_316videos"
    data = FileStorage.load_multiple_files_multiple_keys(path=data_path, retrieve_merged=['features', 'crops', 'paths'])
    features, paths, crops = data['features'], data['paths'], data['crops']
    som = SOM((50, 50), 128)
    # som.som = load_from_file(r"C:\Users\janul\Desktop\thesis_tmp_files\gpulab\som_45x45_01bigger_316videos.pickle")
    # som.som = load_from_file(r"C:\Users\janul\Desktop\thesis_tmp_files\somky\somcosine;61410.pickle")
    # som.som = load_from_file(r"C:\Users\janul\Desktop\thesis_tmp_files\somky\somcosine;200000.pickle")
    # som.som = load_from_file(r"C:\Users\janul\Desktop\thesis_tmp_files\cosine_som\cosine_2M\som-cosine,1990000-2000000.pickle")
    # som.som = load_from_file(r"C:\Users\janul\Desktop\thesis_tmp_files\cosine_som\cosine_50+50+50\som-cosine,50000-50000.pickle")

    som.som = load_from_file(
        r"C:\Users\janul\Desktop\thesis_tmp_files\cosine_som\euclidean\200k-original\som-euclidean,200000-200000.pickle")
    # som.som = load_from_file(r"C:\Users\janul\Desktop\thesis_tmp_files\cosine_som\cosine_50+50+50+50+50\som-cosine,50000-50000.pickle")

    som.set_representatives(features)

    present_frames = np.unique(som.representatives.flatten())
    print("Unique images included", len(present_frames))

    present_videos_set = {paths[i_present][:6] for i_present in present_frames}
    all_videos_set = {path[:6] for path in paths}
    print(all_videos_set - present_videos_set) # No missing video

    np.random.seed(42)
    selected_images_for_experiment = np.random.choice(paths, 10, replace=False)
    print(selected_images_for_experiment)

    for selected in selected_images_for_experiment:
        # show_image(selected)
        pass

    missing_ids = set(range(0, len(paths))) - set(som.representatives.flatten())
    print(len(missing_ids))

    min_distance = []
    for missing_id in missing_ids:
        distances =[]
        for face_id in set(som.representatives.flatten()):
            distances.append(np.linalg.norm(features[face_id] - features[missing_id]))
        min_distance.append(np.min(distances))

    filt = [i for i in min_distance if i > 0.45]

    print(len(filt))
    print(max(min_distance))

if __name__ == '__main__':
    main()
