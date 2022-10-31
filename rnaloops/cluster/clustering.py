"""Cluster features using various algorithms and parameters"""

from ..evaluater.cluster_eval_help_fcts import elbow_plot, silhouette_analysis
from .cluster_plot import do_plot
from .cluster_fcts import *


def do_cluster(data: pd.DataFrame, features: list, n_clusters=(2, ), dim=1, 
               alg='k_means', scaler='Standard', plot=True, pca=False,
               lineplots=False, abline=False, c_thresh=2, learntypes=None, 
               n_bins=20, dpi=100, show_cluster_of=None, save=True,
               elbow=False, silhouette=False, plot_center=False, 
               verbose=False):
    """Do cluster function to be called from the Analyst

    Args:
        data (pd.DataFrame): The df to cluster
        features (list): The features to use
        n_clusters (tuple, optional): Number of clusters. Defaults to (2, ).
        dim (int, optional): Dimension (-> features to compare). Defaults to 1.
        alg (str, optional): The algorithm used. Defaults to 'k_means'.
        scaler (str, optional): The scaler used. Defaults to 'Standard'.
        plot (bool, optional): Whether to plot results. Defaults to True.
        pca (bool, optional): Whether to do PCA afterwards. Defaults to True.
        lineplots (bool, optional): Whether to show lineplots. Defaults to True.
        abline (bool, optional): Whether to show abline. Defaults to False.
        c_thresh (int, optional): Color threshold of dendrogram. Defaults to 2.
        learntypes (any, optional): Learntypes for lineplots. Defaults to None.
        n_bins (int, optional): Number of bins in histograms. Defaults to 20.
        dpi (int, optional): Dots per inch for figures. Defaults to 100.
        show_cluster_of (list, optional): The features whose clusters are shown 
                                          in the plots. Defaults to None.
        save (bool): Whether to save the resulting plots
        elbow (bool): Whether to show elbow plots (only for kmeans)
        silhouette (bool): Whether to show silhouette plots
        plot_center (bool): Whether to plot the centers of kMeans clustering
        verbose (bool): Whether to log actions
    
    """
    folder_path = get_folder_path(alg, features)
    user_ids = list(data.User)

    if alg == 'hierarchy':

        file_path = os.path.join(folder_path, f"{dim}D")

        kwargs = {'dim': dim, 'c_threshold': c_thresh,
                  'file_path': file_path, 'dpi': dpi}
        
        if verbose:
            print('-> Performing hierarchy clustering, generates and saves '
                  'cluster map and dendrogram ... ')
        res_data, result = hierarchy_clustering(data, features, **kwargs)
        if verbose:
            log_result(result)
        if pca:
            if dim > 2:
                if verbose:
                    print('-> Generate PCA (no preview, just saved)... to '
                          'avoid set an.create_pca=False ...')
                do_pca(res_data, result['labels'], file_path, None, scaler,
                       user_ids=user_ids, save=save)
            else:
                if verbose:
                    print('-> PCA only supported for dim > 2')
        if lineplots:
            if verbose:
                print('-> Generate Lineplots... to avoid set '
                      'an.create_lineplots=False ...')
            result.index = data.User
            do_lineplots(result, dpi, file_path, learntypes, user_ids)

    else:
        for n in n_clusters:
            file_path = os.path.join(folder_path, f"{n}C-{dim}D")

            cluster_kwargs = {'features': features, 'n_cluster': n,
                              'dim': dim, 'scale': scaler, 'alg': alg,
                              'path': file_path}
            
            if verbose:
                if alg == 'k_means':
                    print(f'-> Performing {alg} clustering ... generating '
                          f'cluster centers heatmap ...')
                else:
                    print(f'-> Performing {alg} clustering ... to see '
                          f'center heatmap use k_means ...')
            res_data, result = generic_clustering(data, **cluster_kwargs)
            if verbose:
                log_result(result)
            labels, center = result['labels'], result['center']

            plot_kwargs = {'alg': alg, 'dim': dim, 'path': file_path,
                           'abline': abline, 'dpi': dpi, 'n_bins': n_bins,
                           'user_ids': list(data['User']), 'save': save,
                           'show_cluster_of': show_cluster_of,
                           'plot_center': plot_center}
            if plot:
                if dim < 3:
                    do_plot(res_data, result, **plot_kwargs)
                    if verbose:
                        print('-> Generate feature Plots... to avoid set '
                              'an.create_plots=False ...')
                else:
                    for entry in res_data:
                        combis = get_feature_combis(entry, entry.columns)
                        do_plot(combis, result, **plot_kwargs)
                        if verbose:
                            print('-> Generate scatterplots for possible '
                                  'feature combination ...')

            if pca:
                if dim > 2:
                    if verbose:
                        print('-> Generate PCA (no preview)... to '
                              'avoid set create_pca=False ...')
                    do_pca(res_data, labels, file_path, center, scaler,
                           user_ids=user_ids)
                else:
                    if verbose:
                        print('-> PCA only supported for dim > 2')

            if lineplots:
                if verbose:
                    print('-> Generate Lineplots... to avoid set '
                          'an.user_lineplots=False ...')
                do_lineplots(result, dpi, file_path, learntypes, user_ids)
                if verbose:
                    print('-> Saved all plots to corresponding folders and '
                          'results_new folder.')

        if elbow:
            if alg == "k_means":
                file_path = os.path.join(folder_path, f"{dim}D")
                if verbose:
                    print('-> Generate elbow plots... to avoid set '
                          'an.elbow=False ...')
                elbow_plot(data, features, dim, max_cluster=5, scale=scaler,
                           save=save, path=file_path, dpi=dpi)
            else:
                if verbose:
                    print("-> Elbow plot only supported for kmeans.")

        if silhouette:
            if dim < 3:
                file_path = os.path.join(folder_path, f"{dim}D")
                if verbose:
                    print('-> Generate silhouette plots... to avoid set '
                          'an.silhouette=False ...')
                silhouette_analysis(data, features, dim, n_clusters=[2, 3, 4],
                                    alg=alg, scale=scaler, save=save,
                                    path=file_path, dpi=dpi)
            else:
                if verbose:
                    print("-> Silhouette plots only supported for dim < 3.")
