"""Evaluation of cluster by elbow method/cross validation/silhouette analysis"""

from .cluster_eval_fcts import *

if __name__ == '__main':

    algorithm = 'agglomerative'
    feature_lists = ['planar_1_median', 'planar_2_median', 'planar_3_median']
    args = [(features, dim, algorithm)
            for features in feature_lists
            for dim in range(1, 3)]

    elbow = False
    cross_valid = False
    average_cross_valid = False
    silhouette = True

    n_runs = 100
    left_outs = range(3, 40, 5)
    clusters = [0] if algorithm in ("meanshift", "agglomerative") else (2, 3, 4)

    for arg in args:
        if elbow:
            do_elbow(*arg)
        if cross_valid:
            do_cross_validation(None, clusters, *arg)
        if average_cross_valid:
            do_average_cross_validation(n_runs, left_outs, clusters, *arg)
        if silhouette:
            do_silhouette_analysis(clusters, *arg)

        print(f"Finished with arguments {arg}")
