import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, FunctionTransformer

from diplomova_praca_lib import storage


def count_total(path):
    total_count = 0
    print("Obtaining total count of samples")
    for file in Path(path).rglob("*.npz"):
        print("Processing", str(file))
        total_count += len(np.load(str(file), allow_pickle=True)['data'])
    print("Total count is ", total_count)
    return total_count


def sample_images(path, num_samples, total_count):
    sampled_idxs = sorted(np.random.choice(np.arange(total_count), num_samples, replace=False))
    retrieved_samples = []
    already_seen_samples = 0
    print("Sampling")
    for file in Path(path).rglob("*.npz"):
        samples_from_file = 0
        loaded_data = np.load(str(file), allow_pickle=True)['data']
        datafile_samples = len(loaded_data)
        i_sample = sampled_idxs[len(retrieved_samples)] - already_seen_samples
        while i_sample < datafile_samples:
            retrieved_samples.append(loaded_data[i_sample])
            samples_from_file += 1

            if len(retrieved_samples) == num_samples:
                break

            i_sample = sampled_idxs[len(retrieved_samples)] - already_seen_samples

        already_seen_samples += datafile_samples
        print("From %s obtained %d samples out of %d samples" % (str(file), samples_from_file, datafile_samples))

    assert len(retrieved_samples) == num_samples
    return retrieved_samples

def report_pca(pca):
    print("Components = ", pca.n_components_, ";\nTotal explained variance = ",
          round(pca.explained_variance_ratio_.sum(), 5))


def regions_features_only(record):
    return np.vstack([x.features for x in record[1]])


def region_records_to_cols(source_data):
    data = []
    for src_path, crops_data in source_data:
        for crop, crop_features in crops_data:
            data.append((crop, src_path, crop_features))

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str)
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument("--fit", default=False, type=bool)
    parser.add_argument("--transform", default=False, type=bool)
    parser.add_argument("--learned_model", default=None, type=str)
    parser.add_argument("--empty_pipeline", default=False, type=bool)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")

    if args.learned_model:
        pipeline = np.load(args.learned_model, allow_pickle=True)
    elif args.empty_pipeline:
        pipeline = make_pipeline(FunctionTransformer(func=None, validate=False))
    else:
        pipeline = make_pipeline(Normalizer(), PCA(n_components=0.8))


    if args.fit or (args.transform and not args.learned_model):
        total_count = count_total(args.input)
        sampled_records = sample_images(args.input, 10000, total_count)
        sampled_features = np.vstack([regions_features_only(x) for x in sampled_records])
        print("Obtained features with following shape", sampled_features.shape)

        pipeline.fit(sampled_features)
        if "pca" in pipeline: report_pca(pipeline['pca'])
        output_path = Path(args.output, timestamp, "pipeline.npz")
        storage.FileStorage.save_data(path=output_path, pipeline=pickle.dumps(pipeline))

    if args.transform:
        for file_path in Path(args.input).rglob('*.npz'):
            loaded_file = np.load(str(file_path), allow_pickle=True)
            data = loaded_file['data']
            crops, paths, crop_features = zip(*region_records_to_cols(data))
            transformed_features = pipeline.transform(crop_features)
            output_path = Path(args.output, timestamp, file_path.name)
            storage.FileStorage.save_data(path=output_path, pipeline=pickle.dumps(pipeline), model=loaded_file['model'],
                                          features=transformed_features, crops=crops, paths=paths)

if __name__ == '__main__':
    main()
