"""Functions for the clustering.py module"""
import os.path
from random import sample

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from plotly import express as px


from .cluster_help_fcts import *


def generic_clustering(df: pd.DataFrame, features: list, n_cluster=3, dim=2,
                       scale='MinMax', left_out=0, alg='k_means',
                       path="", save=False) -> tuple:
    """Perform clustering on different types of feature pairs

    Args:
        df (pd.DataFrame): The dataframe with all features as columns
        features (list): The features to use
        n_cluster (int): The number of clusters to find
        dim (int): The number of dimensions (features) to be included
        scale (str|None): 'MinMax' for MinMax scaler, 'Standard' for standard
        left_out (int): Number of rows left out for cross validation
        alg (str): The clustering algorithm to use
        path (str): Path to where result raw_data will be saved to
        save (bool): Whether to save the result raw_data

    Returns:
        tuple: A list of dfs for each pair of features, a dict of labels for
               each pair from the clustering and the clustering object for
               each feature

    """
    combis = get_feature_combis(df, features, dim)
    result = initialize_result_dict(combis)

    for combi in combis:

        combi.drop(sample(list(combi.index), left_out), inplace=True)
        tag = tuple((combi.columns[i] for i in range(dim)))

        if scale is not None:
            scaler = StandardScaler() if scale == 'Standard' else MinMaxScaler()
            data = scaler.fit_transform(combi)
        else:
            scaler = None
            data = combi

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if n_cluster > len(data):
            print('More clusters than samples, reduce cluster number...')
            n_cluster = len(data)

        if len(data) < 3:
            print('To few samples, skip...')
            continue

        model = generate_model(alg, n_cluster, data)
        model.fit(data)

        result['labels'][tag] = model.labels_
        result['labels'] = result['labels'].copy()
        result['labels'].index = data.index

        if alg == 'k_means':
            center = model.cluster_centers_
            if scale is not None:
                center_inv = scaler.inverse_transform(center)
            else:
                center_inv = center
            for idx, feature in enumerate(combi.columns):
                f_center = pd.Series([c[idx] for c in center], name=feature)
                f_center_inv = pd.Series([c[idx] for c in center_inv],
                                         name=feature)
                result['center'][tag, feature] = f_center
                result['center_inv'][tag, feature] = f_center_inv
            result['model'][tag] = model

    if save:
        save_cluster_result(combis, result, path)

    return combis, result


def hierarchy_clustering(df: pd.DataFrame, features: list,
                         explanations=tuple(), save=True, dpi=300,
                         c_threshold=2, dim=2, file_path='') -> dendrogram:
    """Get the dendrogram from the seaborn cluster map and plot it

    Args:
        df (pd.DataFrame): The prepared df
        features (list): The list of features to use
        explanations (Iterable, optional): Explanations for the clusters.
        save (bool, optional): Whether to save the plot. Defaults to True.
        dpi (int): Dots per inch for saving the figure
        c_threshold (float): The color threshold for the dendrogram
        dim (int): The number of dimensions to cluster
        file_path (str) = The path where to save the plots

    Returns:
        dendrogram: The dendrogram object handle

    """

    if dim < 2:
        if len(features) > 1:
            print("Need at least 2 dimensions for hierarchy clustering. "
                  "Set dim=2.")
            dim = 2
        else:
            raise ValueError("Need at least 2 features for hierarchy "
                             "clustering, choose dim>1 and more feature.")

    combis = get_feature_combis(df, features, dim)
    result = initialize_result_dict(combis)

    for combi in combis:

        tag = tuple((combi.columns[i] for i in range(dim)))
        title = '+'.join(combi.columns) if len(combi.columns) < 7 else \
            '+'.join(combi.columns[:7] + f"and {len(combi.columns) - 7} more")
        clustermap_path = os.path.join(file_path, "clustermap_" + title)

        c_map = cluster_map(pd.concat([combi, df.index], axis=1), save,
                            clustermap_path, dpi)

        fig = plt.figure(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        plt.xlabel('Index', fontsize=13)
        plt.ylabel('Abweichung', fontsize=13)
        plt.title(title)

        handle = dendrogram(c_map.dendrogram_col.linkage, leaf_rotation=90.,
                            leaf_font_size=12, color_threshold=c_threshold)

        labels = pd.Series(handle['leaves_color_list']).unique()
        if len(explanations) > 0:
            exps = [e for e in explanations[1:]] + [explanations[0]]
            labs = [i for i in labels[1:]] + [labels[0]]
            legend_entries = [l + ' - ' + exps[idx]
                              for idx, l in enumerate(labs)]
            plt.legend(legend_entries, fontsize=13)

        save_figure("dendrogram_" + title, folder=file_path, save=save, dpi=dpi)
        plt.show()

        result['labels'][tag] = handle['leaves_color_list']

    return combis, result


def do_pca(data: list, clusters: pd.DataFrame, file_path: str,
           scaler='Standard', dpi=300, save=True, scale=1, s=2):
    """Perform PCA on given raw_data and plot the results

    Args:
        data (list): A list of df for which PCA should be performed
        clusters (pd.DataFrame): The clusters for coloring markers in PC plot
        file_path (str): File path where to save the plot
        scaler (str, optional): The scaler to use. Defaults to 'Standard'.
        dpi (int, optional): Dots per inch for the figure. Defaults to 300.
        save (bool): Whether to save the figures
        scale (int): the scale of the plot.
        s (float): marker size
    
    """

    for idx, df in enumerate(data):

        comps, pca, cols = scaled_pca(df, scaler)
        title = '+'.join(cols) if len(cols) < 7 else \
                '+'.join(cols[:7] + f"and {len(cols) - 7} more")

        pc_combis = [2 * i for i in range(min(5, len(cols) // 2))]
        rows = (len(pc_combis) + 1)//2

        fig, ax = plt.subplots(rows, 2, figsize=(scale * 13, rows * scale * 6),
                               dpi=dpi)
        for c, a in zip(pc_combis, ax.ravel()):
            plot_pca(pca, comps, clusters, cols, ca=c, cb=c+1,
                     dpi=dpi, scale=scale, s=s, ax=a)

        save_figure('PCA'+title, folder=file_path, save=save,
                    create_if_missing=True)


def scaled_pca(df: pd.DataFrame, scaler="MinMax") -> tuple:
    """Apply scaling and PCA to dataframe

    Args:
        df (pd.DataFrame): The input dataframe
        scaler (string): The type of scaler to use

    Returns:
        tuple: The PCs and the PCA handle as well as headers of used columns

    """
    n_components = len(df.columns)

    if scaler is not None:
        scaler = MinMaxScaler() if scaler == 'MinMax' else StandardScaler()
        data = scaler.fit_transform(df)
    else:
        data = df

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(data)

    var_percent = pca.explained_variance_ratio_

    pcs = pd.DataFrame({f"PC{i + 1} - {var:.1f}%": components[:, i]
                        for i, var in enumerate(var_percent * 100)})

    return pcs, pca, df.columns


def plot_pca(pca: PCA, components: pd.DataFrame, clusters: pd.DataFrame,
             columns: list, ca=0, cb=1, dpi=300, scale=1, s=2,
             ax=None) -> px.scatter:
    """Plot the result of a PCA analysis

    Args:
        pca (PCA): The PCA object from sklearn decomposition
        components (pd.DataFrame): List of PCA components
        clusters (pd.DataFrame): The clusters of the raw_data points
        columns (list): The columns used as features
        ca (int, optional): The first PC to use. Defaults to 0.
        cb (int, optional): The second PC to use. Defaults to 1.
        dpi (int, optional): Dots per inch.
        scale (float, optional): Scaling factor for plot.
        s (float): marker size
        ax (plt.Axes): axes to plot in

    Returns:
        px.scatter: The plotly scatter plot handle

    """
    pca_variance = pca.explained_variance_ratio_
    loadings = pca.components_.T * np.sqrt(pca_variance)

    pcs = components.columns
    clusters = clusters[tuple(columns)]

    max_load = max(max(loadings[:, ca]), max(loadings[:, cb]))
    val_range = min(max(components[pcs[ca]]) - min(components[pcs[ca]]),
                    max(components[pcs[cb]]) - min(components[pcs[cb]]))

    loadings = loadings / max_load * val_range / 4

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(scale*16, scale*12), dpi=dpi)

    sns.scatterplot(x=components[pcs[ca]],
                    y=components[pcs[cb]],
                    hue=clusters,
                    palette='Greys',
                    ax=ax,
                    s=s,
                    legend=False)

    for i, feature in enumerate(columns):
        color = list(plt.cm.tab20.colors)[i] if i < 20 else 'k'
        ax.plot([0, loadings[i, ca]],
                [0, loadings[i, cb]],
                alpha=0.75,
                color=color,
                label=feature)
    
    if len(columns) < 20:
        ax.legend()
