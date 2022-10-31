"""Functions for the clustering.py module"""
import warnings
from random import sample

import numpy as np
import pandas as pd
from plotly.io import write_image
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from plotly import express as px

from .cluster_help_fcts import *
from .cluster_plot_help_fcts import get_centers_list


def generic_clustering(df: pd.DataFrame, features: list, n_cluster=3, dim=2,
                       scale='MinMax', left_out=0, alg='k_means',
                       path="", save=True) -> tuple:
    """Perform clustering on different types of feature pairs

    Args:
        df (pd.DataFrame): The dataframe with all features as columns
        features (list): The features to use
        n_cluster (int): The number of clusters to find
        dim (int): The number of dimensions (features) to be included
        scale (str): 'MinMax' for MinMax scaler, 'Standard' for standard scaler
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

        model = generate_model(alg, n_cluster, data)
        model.fit(data)

        result['labels'][tag] = model.labels_
        result['labels'] = result['labels'].copy()

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
        save_cluster_result(combis, result, path, df[["User"]])

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
        dendrogram_path = os.path.join(file_path, "dendrogram_" + title)
        clustermap_path = os.path.join(file_path, "clustermap_" + title)

        c_map = cluster_map(pd.concat([combi, df['User']], axis=1), save,
                            clustermap_path, dpi)

        fig = plt.figure(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        plt.xlabel('User', fontsize=13)
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

        save_figure(save, dpi, dendrogram_path, fig)
        plt.show()

        result['labels'][tag] = handle['leaves_color_list']

    return combis, result


def do_pca(data: list, clusters: pd.DataFrame, file_path: str, centers=None,
           scaler='Standard', dpi=300, user_ids=None, save=True):
    """Perform PCA on given raw_data and plot the results

    Args:
        data (list): A list of df for which PCA should be performed
        clusters (pd.DataFrame): The clusters for coloring markers in PC plot
        file_path (str): File path where to save the plots
        centers (list, optional): kMeans cluster center. Defaults to None.
        scaler (str, optional): The scaler to use. Defaults to 'Standard'.
        dpi (int, optional): Dots per inch for the figure. Defaults to 300.
        user_ids (list, optional): User ids to annotate. Defaults to None.
        save (bool): Whether to save the figures
    
    """
    centers_dfs = get_centers_list(centers) if centers is not None else []

    for idx, df in enumerate(data):

        center = centers_dfs[idx] if centers is not None else None
        comps, pca, cols, center = scaled_pca(df, center, scaler)

        title = '+'.join(cols) if len(cols) < 7 else \
            '+'.join(cols[:7] + f"and {len(cols) - 7} more")

        for c in [2 * i for i in range(min(5, len(cols) // 2))]:

            fig = plot_pca(pca, comps, clusters, center, cols, ca=c, cb=c+1,
                           user_ids=user_ids)

            if save:
                file_name = f'PC{c + 1}{c + 2}_'
                c_file_path = os.path.join(file_path, file_name + title)
                write_image(fig, c_file_path, format='png', scale=2,
                            width=800, height=700)
                scale = 5 if dpi >= 300 else 2
                write_image(fig, mypath('RESULTS_RECENT'),
                            format='png', scale=scale, width=800, height=700)


def scaled_pca(df: pd.DataFrame, center=None, scaler="MinMax") -> tuple:
    """Apply scaling and PCA to dataframe

    Args:
        df (pd.DataFrame): The input dataframe
        center (optional): Kmeans cluster center to transform
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
    if center is not None:
        center.columns = center.columns.droplevel(0)
        center = np.array(center)
        center = pca.transform(center)

    var_percent = pca.explained_variance_ratio_

    pcs = pd.DataFrame({f"PC{i + 1}_{var:.1f}%": components[:, i]
                        for i, var in enumerate(var_percent * 100)})

    return pcs, pca, df.columns, center


def plot_pca(pca: PCA, components: pd.DataFrame, clusters: pd.DataFrame,
             center: pd.DataFrame, columns: list, ca=0, cb=1,
             user_ids=None) -> px.scatter:
    """Plot the result of a PCA analysis

    Args:
        pca (PCA): The PCA object from sklearn decomposition
        components (pd.DataFrame): List of PCA components
        clusters (pd.DataFrame): The clusters of the raw_data points
        center (pd.DataFrame): The center of the cluster (only for kmeans)
        columns (list): The columns used as features
        ca (int, optional): The first PC to use. Defaults to 0.
        cb (int, optional): The second PC to use. Defaults to 1.
        user_ids (any, optional): The user ids to annotate in the plot

    Returns:
        px.scatter: The plotly scatter plot handle

    """
    pca_variance = pca.explained_variance_ratio_
    loadings = pca.components_.T * np.sqrt(pca_variance)
    loadings = loadings * np.sqrt(len(loadings))
    center = pd.DataFrame(center)

    labels = {str(i + ca): f"PC {i + 1 + cb} ({var:.1f}%)"
              for i, var in enumerate(pca_variance[ca:cb + 1] * 100)}

    pcs = components.columns
    components['size'] = 1
    clusters = clusters[tuple(columns)]

    text = clusters.index if user_ids is None else user_ids
    fig = px.scatter(components, x=pcs[ca], y=pcs[cb], labels=labels,
                     color=[str(c) for c in clusters], text=text,
                     size='size', size_max=15)

    if center is not None and len(center.index) != 0:
        fig.add_scatter(x=center[0], y=center[1], line=Line(width=0),
                        marker=Marker(size=15, color='rgb(0,0,0)', symbol=17,
                                      opacity=0.75))

        fig['raw_data'][-1]['name'] = 'Cluster Zentren'

    for i, feature in enumerate(columns):
        fig.add_shape(type='line', x0=0, y0=0, x1=loadings[i, ca],
                      y1=loadings[i, cb])

        fig.add_annotation(x=loadings[i, ca], y=loadings[i, cb], ax=0, ay=0,
                           yanchor="top" if loadings[i, cb] < 0 else 'bottom',
                           xanchor="center", text=feature)

    fig.update_traces(textfont_size=12)
    fig.update_layout(legend=dict(x=0.7, y=1, bgcolor='rgba(0,0,0,0)',
                                  title=''))

    return fig