"""Helper functions for the cluster_fct.py module"""

import os
from itertools import combinations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn import clustermap

from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, \
    SpectralClustering, estimate_bandwidth, DBSCAN, OPTICS

from ..config.helper import save_figure, save_data, mypath


def get_folder_path(alg: str, features: list) -> str:
    """Get folder path to store results of raw_data and plots

    Args:
        alg (str): The algorithm used for clustering
        features (list): The features used for clustering

    Returns:
        str: The folder path to store results
        
    """
    folder_path = mypath('cluster', alg, create_if_missing=True)

    tag = '-'.join(features) if len(features) < 7 \
        else '-'.join(features[:7]) + f'-and_{len(features)-7}_more'

    folder_path = os.path.join(folder_path, tag)

    return folder_path


def save_cluster_result(combis: list, result: dict):
    """Save the result raw_data of a clustering process

    Args:
        combis (list): The feature combis used
        result (dict): The result dict with labels, centers, ...
        
    """

    for idx, df in enumerate(combis):
        save_data(df, filename=f'combi_{idx}', folder='results/csv')
    for param, df in result.items():
        if param != "model":
            save_data(df, filename=param, folder='results/csv')


def get_feature_combis(df: pd.DataFrame, features: list, dim=2) -> list:
    """Get combinations of columns for the given feature type or selection

    Args:
        df (pd.DataFrame): The input dataframe
        features (str): The list of features to use
        dim (int): The dimension to use, i.e. number of columns to gather

    Returns:
        list: A list of combinations of features

    """
    
    selected_df = df[features]

    combis = [pd.concat([selected_df[combi[i]] for i in range(dim)], axis=1)
              for combi in combinations(selected_df.columns, dim)]

    return combis


def cluster_map(df: pd.DataFrame, save=True, path='', dpi=300) -> clustermap:
    """Generate seaborns cluster map for specific features and return it

    Args:
        df (pd:DataFrame): The df including features to show in cluster map
        save (bool, optional): Whether to save th plot. Defaults to True.
        path (str): The path where to save the figure
        dpi (int): Dots per inch for saving the figure

    Returns:
        sns.clustermap: The generated seaborn cluster map object

    """
    df = df.set_index('User')

    c_map = sns.clustermap(df.transpose(), xticklabels=True, z_score=0,
                           annot=df.transpose(), fmt='.0f', vmin=-2, vmax=2)

    c_map.ax_heatmap.figure.set_size_inches(len(df.index)/2,
                                            len(df.columns)/2+1)

    plt.setp(c_map.ax_heatmap.xaxis.get_majorticklabels(), rotation=0)
    plt.setp(c_map.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    c_map.cax.set_visible(False)
    plt.tight_layout()
    save_figure(path.split('/')[-1], save=save,
                folder='/'.join(path.split('/')[:-1]),
                dpi=dpi, fformat='jpg')
    plt.show()

    return c_map


def generate_model(alg: str, n_cluster: int, data: pd.DataFrame) -> any:
    """Generate a cluster model of the selected algorithm

    Args:
        alg (str): The algorithm used
        n_cluster (int): The number of clusters to generate
        data (pd.DataFrame): The raw_data to use for fitting afterwards

    Raises:
        ValueError: If wrong algorithm is selected

    Returns:
        any: The cluster model constructed for the given algorithm
        
    """
    if alg == 'k_means':
        model = KMeans(n_clusters=n_cluster)
        
    elif alg == 'meanshift':
        bandwidth = estimate_bandwidth(data)
        bandwidth = None if bandwidth < 10 ** -6 else bandwidth
        print(f'MeanShift bandwidth: {bandwidth}')
        model = MeanShift(bandwidth=bandwidth)
        
    elif alg == 'spectral':
        model = SpectralClustering(n_clusters=n_cluster)
        
    elif alg == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_cluster, linkage='ward',
                                        compute_full_tree=False,
                                        distance_threshold=None)
    elif alg == 'dbscan':
        model = DBSCAN()
    elif alg == 'optics':
        model = OPTICS()
    else:
        raise ValueError('The chosen algorithm does not exist. Use '
                         'k_means, meanshift, spectral or agglomerative')

    return model


def initialize_result_dict(combis: list) -> dict:
    """Initialize an empty dict to use to store cluster modelling results
    
    Args:
        combis (list): The feature combis to use

    Returns:
        dict: The empty dict with the proper structure
        
    """
    combi_tags = [(tuple(combi.columns), feature)
                  for combi in combis for feature in combi.columns]

    multi_idx = pd.MultiIndex.from_tuples(combi_tags)

    result = {'labels': pd.DataFrame(),
              'model': {},
              'center': pd.DataFrame(columns=multi_idx),
              'center_inv': pd.DataFrame(columns=multi_idx)}

    return result


def log_result(result: dict):
    """Log the result of clustering to the console

    Args:
        result (dict): Dict of labels, center etc. from clustering

    """
    print('-> Finished clustering, print results for clusters ...')
    
    for col in result['labels'].columns:
        lab = result['labels'][col]
        
        print(f'Cluster indices found for {col}:')
        
        ids = [' ' + str(i) for i in range(10)] + \
              [str(i) for i in range(10, len(lab))]
        
        print('ids:', *ids)
        print('Cluster:', *[' ' + str(i) for i in lab])
