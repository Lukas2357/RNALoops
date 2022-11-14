"""Helper functions for the cluster_eval_fcts.py module"""

import os
from itertools import permutations
from typing import Iterable
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_samples

from ..config.helper import save_figure
from ..cluster.cluster_fcts import generic_clustering
from ..cluster.cluster_plot_help_fcts import init_subplots_plot, \
    get_current_axis


def _get_labels(df: pd.DataFrame, features: list, n_cluster: int, left_out: int,
                dim=2, alg='k_means', scale='MinMax') -> dict[dict]:
    """Perform clustering and get labels with index of raw_data point

    Args:
        df (pd.DataFrame): The raw input dataframe
        features (list): The features to use
        n_cluster (int): The number of clusters
        left_out (int): The number of points left out for cross validation
        dim (int): The dimension (number of features to include)
        alg (str): The clustering algorithm to use
        scale (str): The scaler to use

    Returns:
        dict[dict]: For each feature combination a dict of indices of raw_data
                    points as keys and the corresponding labels as values
                    
    """
    data, result = generic_clustering(df, features, n_cluster, dim, scale,
                                      alg=alg, save=False, left_out=left_out)

    result = {}
    for idx, (key, value) in enumerate(result['labels'].items()):
        result[key] = dict(zip(data[idx].index, value))

    return result


def get_match_labels(df: pd.DataFrame, features: list, left_out: int,
                     n_cluster: int, dim=2, alg='k_means',
                     scale='MinMax') -> dict:
    """Get fraction of matching labels between reference and sample labels

    Args:
        df (pd.DataFrame): The raw dataframe to cluster
        features (list): The features to use
        left_out (int): The number of points left out for cross validation
        n_cluster (int): The number of clusters
        dim (int): The dimension (number of features to include)
        alg (str): The clustering algorithm to use
        scale (str): The scaler to use

    Returns:
        dict: A dictionary with feature combinations as keys and the fraction
              of correctly classified points under the given left_out value
              
    """
    reference = _get_labels(df, features, n_cluster, 0, dim, alg, scale)
    sample = _get_labels(df, features, n_cluster, left_out, dim, alg, scale)
    
    matches = {}
    for feature, value in sample.items():
        max_matches = max((sum(mapping[label] == reference[feature][idx] 
                               for idx, label in value.items()) 
                           for mapping in permutations(range(n_cluster))))
        matches[feature] = max_matches/len(value.keys())
    
    return matches


def cross_validation(df: pd.DataFrame, features: list, left_outs=None,
                     clusters=(2, 3, 4), dim=2, alg='k_means', plot=False,
                     save=False, path='', scale='MinMax') -> dict:
    """Perform cross validation on k-means clustering

    Args:
        df (pd.DataFrame): The raw dataframe to be cross-validated
        features (list): The features to use
        left_outs (range): The range of numbers of raw_data points to left out
        clusters (tuple): The range of cluster numbers to check
        dim (int): The dimension (number of features to include)
        alg (str): The clustering algorithm to use
        plot (boolean, optional): Whether to plot the result
        save (boolean, optional): Whether to save the result figure
        path (str, optional): The path to save the result figure
        scale (str): The scaler to use

    Returns:
        dict: Feature combinations as keys and a dictionary as value, containing
              the number of clusters as key and the fraction of correctly 
              classified samples as value.
              
    """
    left_outs = left_outs if left_outs else range(1, len(df.index)-5)
    cross_valid = {}

    selected = get_match_labels(df, features, 0, 1, dim, alg, scale).keys()
    
    for select in selected:

        results = dict(zip(clusters, [[] for _ in clusters]))

        for n_cluster in clusters:
            
            matches = []

            for left_out in left_outs:
                match = list(get_match_labels(df, features, left_out, n_cluster,
                                              dim, alg, scale).values())[0]
                matches.append(match)
                    
            results[n_cluster] = matches

        cross_valid[select] = results
        
    if plot:
        _, ax = plt.subplots(len(selected), 1, figsize=(16, 3*len(selected)))
        
        for idx, (feature, values) in enumerate(cross_valid.items()):
            
            c_ax = ax[idx] if isinstance(ax, Iterable) else ax
            
            for _, value in values.items():
                c_ax.plot(left_outs, value, "x--")
                
            c_ax.set_title(f"{feature}")
            c_ax.set_ylabel("classified similar")
            c_ax.set_xlabel("Number of left out datapoints")
            c_ax.set_ylim([0.5, 1])
            
        _ = plt.legend([f"cluster: {i}" for i in clusters])
        plt.tight_layout()
        
        if save:
            plt.savefig(path)
        
        plt.show()
    
    return cross_valid


def average_cross_validation(runs: list, clusters=range(2, 5), save=False,
                             path='') -> tuple:
    """Get average result_plots of cross validation runs and plot as heatmap

    Args:
        runs (list): A list of dfs one for each run of cross validation.
        clusters (range, optional): Range of clusters. Defaults to range(2, 5).
        save (bool, optional): Whether to save the figure. Defaults to False.
        path (str, optional): The path to save the figure. Defaults to ''.

    Returns:
        tuple: A tuple of axis objects of the figure
        
    """
    data_list = []
    
    for run in runs:      
          
        data = {}
        for feature, values in run.items():    
            data[feature] = [sum(value)/len(value) for value in values.values()]
            
        data = pd.DataFrame({"n_cluster": list(clusters)} | data)
        data = data.set_index("n_cluster")
        data_list.append(data)

    data_mean = data_list[0]
    for data in data_list[1:]:
        data_mean = data_mean.add(data)
    data_mean = data_mean/len(runs)

    data_std = data_list[0].add(-data_mean)**2
    for data in data_list[1:]:
        data_std = data_std.add(-data)**2
    data_std = data_std**0.5/len(runs)

    data_min = pd.concat([df for df in data_list]).groupby(level=0).min()

    _, ax = plt.subplots(3, 1, figsize=(6, 4+1.5*len(data_min.columns)))
    ax[0].set_title('Mean correctly classified')
    sns.heatmap(data_mean.transpose(), ax=ax[0], annot=True, fmt='.3f')
    ax[1].set_title('Mean-Std correctly classified')
    sns.heatmap(data_mean.transpose()-data_std.transpose(), ax=ax[1],
                annot=True, fmt='.3f')
    ax[2].set_title('Min correctly classified')
    sns.heatmap(data_min.transpose(), ax=ax[2], annot=True, fmt='.3f')
    plt.tight_layout()

    if save:
        plt.savefig(path)

    plt.show()
    
    return ax


def elbow_plot(df: pd.DataFrame, features: list, dim=2, max_cluster=5,
               scale='MinMax', alg='k_means', save=False, path='',
               dpi=300) -> list:
    """Elbow plots for clustering result_plots to get best number of clusters

    Args:
        df (pd.DataFrame): The input dataframe
        features (list): List of features to use
        dim (int): The dimension (number of features) to cluster
        max_cluster (int): The maximum number of clusters to check
        scale (str): The scaler to use for clustering
        alg (str): The algorithm to use
        save (bool): Whether to save the figure
        path (string): Path to save the figure
        dpi (int): Dots per inch for saving figure. Defaults to 300.
        
    Returns: 
        List: A tuple of axis objects of the figure 

    """
    sse_dict = {}
    for n_cluster in range(1, max_cluster+1):
        _, result = generic_clustering(df, features, n_cluster, dim, scale,
                                       alg=alg, save=False)
        for combi in result['labels'].columns:
            if combi in sse_dict.keys():
                sse_dict[combi].append(result['model'][combi].inertia_)
            else:
                sse_dict[combi] = [result['model'][combi].inertia_]

    n_plots = len(list(sse_dict.keys()))
    ax = init_subplots_plot(n_plots)

    for idx, (key, value) in enumerate(sse_dict.items()):
        
        c_ax = get_current_axis(list(sse_dict.values()), ax, idx)

        c_ax.plot(range(1, max_cluster+1), value, 'o:b')

        tag = f"{key[0]}" if dim == 1 else f"{key[0]} + {key[1]}"
        c_ax.set_title(f"Elbow method - " + tag)
        c_ax.set_xlabel("Anzahl der cluster")
        c_ax.set_ylabel("SSE innerhalb der cluster")

    plt.tight_layout()

    save_figure(os.path.join(path, "elbow"), save=save, dpi=dpi)

    plt.show()
        
    return ax


def silhouette_analysis(df: pd.DataFrame, features: list, dim=2,
                        n_clusters=(2, 3, 4), alg='k_means', scale='MinMax',
                        save=False, path='', dpi=300,
                        show_labels=False, plot=True) -> pd.DataFrame:
    """Perform silhouette analysis to check separation of clusters
    
    For details see: 
    https://towardsdatascience.com/k-means-clustering-algorithm-applications-
    evaluation-methods-and-drawbacks-aa03e644b48a

    Args:
        df (pd.DataFrame): The standard input dataframe for
        features (list): The features to use.
        dim (int, optional): Number of dimensions (features). Defaults to 2.
        n_clusters (int, optional): Number of clusters to check.
        alg (str): The algorithm to use. Defaults to k_means.
        scale (str): The scaler to use.
        save (bool, optional): Whether to save figures. Defaults to False.
        path (str, optional): Where to save figures. Defaults to ''.
        dpi (int): Dots per inch for saving figure. Defaults to 300.
        show_labels (bool, optional): Whether to show labels
        plot (bool, optional): Whether to plot figure

    Returns:
        pd.DataFrame: Summary df of silhouette scores
        
    """    
    values = {}
    labels = {}
    n_combis = 0
    n_cl = len(n_clusters)
    combis = []
    
    for n_cluster in n_clusters:
        data, result = generic_clustering(df, features, n_cluster, dim, scale,
                                          alg=alg, save=False)
        values[n_cluster] = data
        labels[n_cluster] = result['labels']
        n_combis = len(data)
        combis = result['labels'].columns

    key = list(labels.keys())[0]

    if plot:
        _, ax = plt.subplots(n_combis, 2*n_cl, figsize=(n_cl*8, 3.5*n_combis),
                             dpi=dpi)
    else:
        ax = [[plt.Axes]]

    result = np.zeros([n_combis, n_cl])
    for idx, feature in enumerate(combis):

        for i, k in enumerate(n_clusters):

            c_labels = [label+1 for label in labels[k][feature]]
            c_values = values[k][idx]
            if plot:
                c_ax1 = (ax[idx][2*i] if isinstance(ax[0], Iterable)
                         else ax[2*i])
                c_ax2 = (ax[idx][2*i+1] if isinstance(ax[0], Iterable)
                         else ax[2*i+1])
            else:
                c_ax1, c_ax2 = plt.Axes, plt.Axes

            silhouette_vals = silhouette_samples(c_values, c_labels)
            avg_score = np.mean(silhouette_vals)
            result[idx][i] = avg_score

            y_lower, y_upper = 0, 0
            for s, n_cluster in enumerate(np.unique(c_labels)):
                cluster_silhouette_vals = silhouette_vals[c_labels == n_cluster]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)
                if plot:
                    c_ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals,
                               edgecolor='none', height=1)
                if show_labels and plot:
                    c_ax1.text(-0.1, (y_lower + y_upper) / 2, str(s + 1))
                y_lower += len(cluster_silhouette_vals)

            if plot:
                c_ax1.axvline(avg_score, linestyle='--', linewidth=2,
                              color='green')
                c_ax1.set_yticks([])
                c_ax1.set_xlim([-1, 1])
                c_ax1.set_xlabel('Silhouette coefficient values')
                c_ax1.set_ylabel('Cluster labels')

                if dim == 2:
                    sns.scatterplot(x=c_values[c_values.columns[0]],
                                    y=c_values[c_values.columns[1]], ax=c_ax2,
                                    hue=c_labels, palette="tab10",
                                    legend=False, s=5)
                    c_ax2.set_ylabel(feature[1])
                else:
                    data = pd.concat([c_values[c_values.columns[0]],
                                      pd.Series(c_labels)], axis=1)
                    data.columns = [feature, 'label']
                    sns.histplot(ax=c_ax2, data=data, x=feature, hue='label',
                                 multiple='stack', bins=30, palette="tab10")
                    c_ax2.set_ylabel("occurrences")

                c_ax2.set_xlabel(feature[0])

                plt.tight_layout()

    result = pd.DataFrame(result, columns=n_clusters, index=labels[key].keys())

    if plot:
        save_figure(os.path.join(path, "silhouette"), save=save, dpi=dpi)
        plt.show()
        plt.figure(figsize=(8, 2+0.8*len(result.index)), dpi=dpi)
        sns.heatmap(result, annot=True, fmt='.3f')
        plt.ylabel('')
        plt.tight_layout()
        save_figure(os.path.join(path, "silhouette_summary"),
                    save=save, dpi=dpi)
        plt.show()
    
    return result
