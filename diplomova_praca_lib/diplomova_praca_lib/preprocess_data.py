import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, FunctionTransformer

from diplomova_praca_lib import storage
from diplomova_praca_lib.utils import enhance_dims, reduce_dims, sample_features_from_data


def count_total(path):
    total_count = 0
    print("Obtaining total count of samples")
    for file in Path(path).rglob("*.npz"):
        print("Processing", str(file))
        total_count += len(np.load(str(file), allow_pickle=True)['data'])
    print("Total count is ", total_count)
    return total_count


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
    parser.add_argument("--fit", action='store_true')
    parser.add_argument("--no_transform", action='store_false')
    parser.add_argument("--empty_pipeline", action='store_true')
    parser.add_argument("--regions", action='store_true')
    parser.add_argument("--count", default=None, type=int)
    parser.add_argument("--samples", default=10000, type=int)
    parser.add_argument("--explained_ratio", default=0.8, type=float)
    args = parser.parse_args()

    transform = not args.no_transform

    if args.empty_pipeline:
        pipeline = make_pipeline(FunctionTransformer(func=None, validate=False))
    else:
        if args.explained_ratio > 1:
            args.explained_ratio = int(args.explained_ratio)

        dimensionality_reduction = PCA(n_components=args.explained_ratio)
        pipeline = make_pipeline(Normalizer(), dimensionality_reduction)

    if args.fit or (transform and not args.empty_pipeline):
        if args.count:
            total_count = args.count
        else:
            total_count = count_total(args.input)
        sampled_records = sample_features_from_data(args.input, min(total_count, args.samples), total_count)

        if args.regions:
            sampled_features = np.vstack([regions_features_only(x) for x in sampled_records])
        else:
            sampled_features = np.array([spatial_features_only(x) for x in sampled_records])

        print("Obtained features with following shape", sampled_features.shape)

        if sampled_features.ndim > 2:
            pipeline.steps = pipeline_with_dim_reduction(pipeline.steps, sampled_features.shape[1:-1])

        pipeline.fit(sampled_features)
        if "pca" in pipeline.named_steps: report_pca(pipeline['pca'])

    if transform:
        for file_path in Path(args.input).rglob('*.npz'):
            output_path = Path(args.output, file_path.name)
            if output_path.exists():
                print("Skipping directory {}".format(file_path))
                continue

            loaded_file = np.load(str(file_path), allow_pickle=True)
            data = loaded_file['data']

            if args.regions:
                crops, paths, features = zip(*region_records_to_cols(data))
                features = np.array(features)
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
