import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, FunctionTransformer

from diplomova_praca_lib import storage
from diplomova_praca_lib.utils import enhance_dims, reduce_dims


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


def spatial_features_only(record):
    return record[1]

def region_records_to_cols(source_data):
    data = []
    for src_path, crops_data in source_data:
        for crop, crop_features in crops_data:
            data.append((crop, src_path, crop_features))

    return data


def sample_from_4dimensional_features(data):
    batch, rows, cols, num_features = data.shape
    sampled_y_idxs = np.random.randint(0, rows, size=batch)
    sampled_x_idxs = np.random.randint(0, cols, size=batch)
    return data[np.arange(batch), sampled_y_idxs, sampled_x_idxs, :]


def pipeline_with_dim_reduction(pipeline_steps, shape):
    decrease_dims = ('reduce_dims', FunctionTransformer(reduce_dims, validate=False))
    increase_dims = ('enhance_dims', FunctionTransformer(enhance_dims, kw_args={"shape": shape}, validate=False))
    return [decrease_dims, *pipeline_steps, increase_dims]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str)
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument("--fit", default=False, type=bool)
    parser.add_argument("--transform", default=False, type=bool)
    parser.add_argument("--learned_model", default=None, type=str)
    parser.add_argument("--empty_pipeline", default=False, type=bool)
    parser.add_argument("--regions", action='store_true')
    parser.add_argument("--count", default=None, type=int)
    parser.add_argument("--samples", default=10000, type=int)
    parser.add_argument("--explained_ratio", default=0.8, type=float)
    args = parser.parse_args()

    if args.learned_model:
        pipeline = np.load(args.learned_model, allow_pickle=True)
    elif args.empty_pipeline:
        pipeline = make_pipeline(FunctionTransformer(func=None, validate=False))
    else:
        pipeline = make_pipeline(Normalizer(), PCA(n_components=args.explained_ratio))

    if args.fit or (args.transform and not args.learned_model):
        if args.count:
            total_count = args.count
        else:
            total_count = count_total(args.input)
        sampled_records = sample_images(args.input, min(total_count, args.samples), total_count)

        if args.regions:
            sampled_features = np.vstack([regions_features_only(x) for x in sampled_records])
        else:
            sampled_features = np.array([spatial_features_only(x) for x in sampled_records])

        print("Obtained features with following shape", sampled_features.shape)

        if sampled_features.ndim > 2:
            pipeline.steps = pipeline_with_dim_reduction(pipeline.steps, sampled_features.shape[1:-1])

        pipeline.fit(sampled_features)
        if "pca" in pipeline.named_steps: report_pca(pipeline['pca'])

    if args.transform:
        for file_path in Path(args.input).rglob('*.npz'):
            output_path = Path(args.output, file_path.name)
            if output_path.exists():
                print("Skipping directory {}".format(file_path))
                continue

            loaded_file = np.load(str(file_path), allow_pickle=True)
            data = loaded_file['data']

            if args.regions:
                crops, paths, features = zip(*region_records_to_cols(data))
                to_save = {"crops": crops, "paths": paths}
            else:
                paths, features = zip(*data)
                features = np.array(features)
                to_save = {"paths": paths}

            if features.ndim > 2 and 'enhance_dims' not in pipeline.named_steps:
                pipeline.steps = pipeline_with_dim_reduction(pipeline.steps, features.shape[1:-1])

            to_save['features'] = pipeline.transform(features)

            storage.FileStorage.save_data(path=output_path, pipeline=pickle.dumps(pipeline), model=loaded_file['model'],
                                          **to_save)

if __name__ == '__main__':
    main()
