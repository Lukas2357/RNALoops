"""Plot the results of clustering with heatmap/histogram/scatterplot"""

from .cluster_plot_fcts import *


def do_plot(
        data,
        result,
        alg='k_means',
        dim=2,
        path="",
        abline=False,
        dpi=100,
        n_bins=30,
        user_ids=None,
        show_cluster_of=None,
        plot_center=False,
        save=True,
        plot_labels=False,
        s=60,
        scale=1,
        fontsize=10):
    """Do plot function to be called from clustering

    Args:
        see plot_kmeans_centers, cluster_single_plot and cluster_pairs_plot
        
    """
    if alg == 'k_means' and plot_center:
        ax = plot_kmeans_centers(result['center'], result['center_inv'],
                                 path=path, dpi=dpi, save=save)

    if dim == 1:
        ax = cluster_single_plot(data, result['labels'], n_bins=n_bins,
                                 path=path, dpi=dpi, save=save)

    else:
        ax = cluster_pairs_plot(data, result['labels'], abline=abline,
                                path=path, dpi=dpi, user_ids=user_ids,
                                show_cluster_of=show_cluster_of, save=save,
                                plot_labels=plot_labels, s=s, scale=scale,
                                fontsize=fontsize)

    return ax
