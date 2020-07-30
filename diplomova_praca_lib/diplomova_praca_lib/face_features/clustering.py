import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans


def closest_node(query, features):
    features = np.asarray(features)
    deltas = features - query
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return features[np.argmin(dist_2)]

def find_representative(np_features):
    # Finds one representative for the cluster
    kmeans = KMeans(n_clusters=1).fit(np_features)
    return closest_node(kmeans.cluster_centers_, np_features)


def split_to_n_clusters(np_features, n):
    # Splits dataset to n clusters
    agglomerative = AgglomerativeClustering(n_clusters=n).fit(np_features)
    return agglomerative.labels_


def group_clusters_to_lists(np_features, labels):
    clusters = []
    for i_cluster in range(max(labels + 1)):
        clusters.append([features for i, features in enumerate(np_features) if labels[i] == i_cluster])
    return clusters