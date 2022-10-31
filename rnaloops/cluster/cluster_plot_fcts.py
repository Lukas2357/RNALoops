"""Functions for the cluster_plot.py module"""
import warnings

import numpy as np
import seaborn as sns
import colorcet as cc

from .cluster_plot_help_fcts import *
from ..config.helper import save_figure, mypath

pd.options.mode.chained_assignment = None  # default='warn'


def heatmap(title, data, save, path=None, filename='map', dpi=120, **kwargs):
    """Generic heatmap function

    Args:
        title (string): Title of the heatmap
        data (pd.DataFrame): The df to plot
        save (bool): Whether to save the heatmap
        dpi (int): Dots per inch to save the plot
        filename (str): The filename to save the plot
        path (str): Path to save the plots
        kwargs (dict): Additional keyword arguments for sns.heatmap

    Returns:
        plt.figure: The figure handle

    """
    rows, cols = len(data.index), len(data.columns)
    fig = plt.figure(figsize=(rows, cols / 4 + 2), dpi=dpi)
    sns.heatmap(data, annot=True, **kwargs)
    plt.title(title, fontweight='bold')
    plt.tight_layout()

    if path is None:
        spath = mypath('RESULTS', filename, subfolder='heatmaps')
    else:
        spath = os.path.join(path, filename)

    save_figure(save, dpi, spath, fig)

    return fig


def correlation_heatmap(df, features=None, save=False, dpi=150,
                        filename='cor_heatmap', cbar=False, path=None,
                        all_with_these_features=False, method='pearson',
                        plot_corr=True, pvalues=True, significant=True,
                        plot_significant=True, threshold=10, min_cor=0):
    """Heatmap of all correlations of the prepared dataframe

    Args:
        features (list): The features for which all correlations are shown
        df (any): The data to plot, if None it will be loaded from data prep
        save (bool): Whether to save the plot
        dpi (int): Dots per inch to save the plot
        filename (str): The filename to save the plot
        cbar (bool): Whether to show the colorbar
        path (str): Path to save the plots
        all_with_these_features (bool): Show all cors of given features
        method (str): Either 'pearson' or 'spearman' for the correlation type
        plot_corr (bool): Whether to plot the correlation heatmap
        pvalues (bool): Whether to show the p-values as separate heatmap
        significant (bool): Whether to show the significant correlation values
        plot_significant (bool): Whether to plot the significant correlations'
        threshold (float): The threshold value for the correlations
        min_cor (float): The minimum correlation

    """

    df = df.drop(['User'], axis=1)

    if features is None:
        features = df.columns

    if isinstance(features, str):
        features = [features]

    if not all_with_these_features:
        df = df[features]

    pearson = df.corr(method='pearson')[features]
    spearman = df.corr(method='spearman')[features]
    correlation = pearson if method == 'pearson' else spearman
    figures = []

    if plot_corr:
        kwargs = dict(fmt='.3f', vmin=-1, vmax=1, cbar=cbar, filename=filename)
        fig = heatmap('Feature correlations', correlation, save, **kwargs)
        figures.append(fig)

    if pvalues or significant:

        p_pearson = 100 * df.corr(method=cor_test('pearson'))[features]
        p_spearman = 100 * df.corr(method=cor_test('spearman'))[features]
        p = p_pearson if method == 'pearson' else p_spearman

        cmap = sns.diverging_palette(133, 10, as_cmap=True)

        if pvalues:
            kwargs = dict(fmt='.2f', vmin=0, vmax=100, cbar=cbar, cmap=cmap,
                          filename=filename + '_pvalues')
            title = 'Two sided corr p-values / %'
            fig = heatmap(title, p, save, **kwargs)
            figures.append(fig)

        if significant:

            args = (method, pearson, p_pearson, spearman, p_spearman,
                    threshold, min_cor)
            df, numeric = get_significant_correlations(*args)

            numeric = numeric.drop(['Prio 1', 'Prio 2'], axis=1)
            df = df.drop(['Prio 1', 'Prio 2'], axis=1)

            if len(df.index) > 0 and plot_significant:
                fig = plot_significant_corrs(numeric, df, dpi, save, path,
                                             filename + '_sgnf')
                figures.append(fig)

    plt.show()
    return figures, df


def cluster_pairs_plot(pairs: list[pd.DataFrame], labels: pd.DataFrame,
                       abline=False, save=True, path="", dpi=300,
                       user_ids=None, show_cluster_of=None,
                       plot_labels=False, s=60, legend=False,
                       scale=1, fontsize=10) -> list:
    """Plot the kmeans clustering result for a pair of features

    Args:
        pairs (list[pd.DataFrame]): The input dataframes to be plotted
        labels (pd.DataFrame): Df with labels for each kmeans result
        abline (bool): Whether to show bisector line in the plot
        save (bool): Whether to save the figure
        path (string): Path to save the figure
        dpi (int): Dots per inch for saving the figure
        user_ids (any): User ids to annotate in the plot, uses index if None
        show_cluster_of (list): Feature combi whose clusters shown in all plots
        s
        fontsize
        scale
        legend
        plot_labels

    """
    ax, label = init_subplots_plot(len(pairs), scale=scale, dpi=dpi), ''

    if show_cluster_of is not None:
        label = get_color_label(show_cluster_of, labels)

    for idx, data in enumerate(pairs):

        c_ax = get_current_axis(pairs, ax, idx)

        x_data, y_data = data.iloc[:, 0], data.iloc[:, 1]
        x_tag, y_tag = data.columns[0], data.columns[1]

        if show_cluster_of is None or len(label) == 0:
            if (x_tag, y_tag) in labels.columns:
                c_label = [i + 1 for i in labels[(x_tag, y_tag)]]
            else:
                c_label = [i + 1 for i in labels.iloc[:, 0]]
        else:
            c_label = label

        legend = True if (idx == 0 and legend) else False

        nc = len(labels[labels.columns[0]].unique())
        palette = sns.color_palette(cc.glasbey, n_colors=nc)

        sns.scatterplot(x=x_data, y=y_data, ax=c_ax, hue=c_label,
                        palette=palette, legend=legend, s=s)

        user_ids = data.index if user_ids is None else user_ids

        if plot_labels:
            for i, j in enumerate(user_ids):
                c_ax.annotate(str(j), [
                    list(x_data)[i],
                    list(y_data)[i]
                ], horizontalalignment='center',
                              verticalalignment='center', size=5)

            c_ax.set_title(f"{x_tag} vs {y_tag}", fontweight='semibold')
            set_labels(pairs, x_tag, c_ax, 2, y_tag)
        if abline:
            end = min(max(y_data), max(x_data))
            c_ax.plot([0, end], [0, end], 'r-')

        for item in ([c_ax.title, c_ax.xaxis.label, c_ax.yaxis.label] +
                     c_ax.get_xticklabels() + c_ax.get_yticklabels()):
            item.set_fontsize(fontsize * scale)

        x_range, y_range = max(x_data) - min(x_data), max(y_data) - min(y_data)
        c_ax.set_xlim(min(x_data) - 0.005 * x_range,
                      max(x_data) + 0.005 * x_range)
        c_ax.set_ylim(min(y_data) - 0.005 * y_range,
                      max(y_data) + 0.005 * y_range)

    plt.tight_layout()
    save_figure(save=save, dpi=dpi, name="scatterplot", folder=path)

    return ax


def cluster_single_plot(columns: list[pd.DataFrame], labels: pd.DataFrame,
                        save=True, path="", dpi=300, n_bins=20) -> list:
    """Plot the kmeans clustering result for single features

    Args:
        columns (list[pd.DataFrame]): The input dataframes to be plotted
        labels (list): List of pandas series with labels for each kmeans result
        save (bool): Whether to save the figure
        path (string): Path to save the figure
        dpi (int): Dots per inch for saving the figure
        n_bins (int): Number of bins to show in histogram
        
    Returns:
        list: List of axes of the plots

    """
    ax = init_subplots_plot(len(columns))

    for idx, data in enumerate(columns):
        c_ax = get_current_axis(list(labels.columns), ax, idx)
        data = data.reset_index().drop('index', axis=1)
        tag = data.columns[0]
        label = [i + 1 for i in labels[(tag,)]]
        data = pd.concat([data.iloc[:, 0], pd.Series(label)], axis=1)
        data.columns = [tag, 'label']
        legend = True if idx == 0 else False
        if len(data[tag].unique()) < 10:
            data['count'] = 1
            data = data.groupby([tag, 'label']).count().reset_index()
            sns.barplot(ax=c_ax, data=data, x=tag, y='count', hue='label',
                        palette="tab10")
        else:
            sns.histplot(ax=c_ax, data=data, x=tag, hue='label',
                         multiple='stack', bins=n_bins, palette="tab10",
                         legend=legend)
        c_ax.set_title(f"{tag}", fontweight='semibold')
        c_ax.set_ylabel("occurrences")
        set_labels(columns, tag, c_ax)

    plt.tight_layout()
    save_figure(save, dpi, os.path.join(path, "histogram"))
    plt.show()

    return ax


def plot_kmeans_centers(centers: pd.DataFrame, centers_inv: pd.DataFrame,
                        save=True, path="", dpi=300):
    """Plot the centers of the kmeans clustering in terms of a heatmap

    Args:
        centers (pd.DataFrame): The centers determined
        centers_inv (pd.DataFrame): The centers inverse transformed
        save (bool, optional): Whether to save the plot. Defaults to True.
        path (str, optional): The path where to save. Defaults to "".
        dpi (int, optional): The dpi to use for the figure. Defaults to 300.
    
    """
    centers_dfs = get_centers_list(centers)
    centers_inv_dfs = get_centers_list(centers_inv)

    first_df = centers_dfs[0]
    scales = len(first_df.columns) * 1.5, len(first_df.index) * 0.5 + 1

    ax = init_subplots_plot(len(centers_dfs), scales=scales)

    for idx, df in enumerate(centers_dfs):
        c_ax_0 = get_current_axis(centers_dfs, ax, idx)
        df.columns = [c[1] for c in df.columns]
        sns.heatmap(df, ax=c_ax_0, annot=centers_inv_dfs[idx], fmt='.2f',
                    cbar=False)

    plt.suptitle('Cluster centers of kMeans', fontweight='bold')
    plt.tight_layout()
    save_figure(save, dpi, os.path.join(path, "center"))
    plt.show()


def get_color_label(show_cluster_of: list, labels: pd.DataFrame) -> list:
    """Get the proper color labels for the clusters using show_cluster_of

        Args:
            show_cluster_of (list): List of features to show the clusters of
            labels (pd.DataFrame): The user labels of the corresponding clusters

        """
    label = []

    all_cols = ' or '.join([str(list(lab)) for lab in labels.columns])
    warn_m = 'The chosen feature combi in show_cluster_of does not exist, ' \
             'showing individual clusters instead. ' \
             f'\nSelect {all_cols} to show this cluster in each plot.'
    if tuple(show_cluster_of) in list(labels.columns):
        label = list(labels[tuple(show_cluster_of)])
    else:
        warnings.warn(warn_m)

    return label


def string_ratio_grid(ratios_df, clustermap=True, pad='max', exclude='G',
                      replace=None, save=False, dpi=300) -> np.array:
    """Get a grid for string ratios and possibly plot it as clustermap

    Args:
        ratios_df (pd.DataFrame): Df as returned by get_learntype_string_ratios
        clustermap (bool): Whether to plot the clustermap
        pad (str) pad parameter passed to get_learntype_string_ratios
        exclude (str): exclude parameter passed to get_learntype_string_ratios
        replace (dict): replace parameter passed to get_learntype_string_ratios
        save (bool): Whether to save the clustermap
        dpi (int): Dpi of the clustermap

    Returns:
        np.array: The grid of user vs user similarity ratios

    """

    if ratios_df is None:
        ratios_df = get_learntype_string_ratios(pad=pad, exclude=exclude,
                                                replace=replace)

    max_user = max(ratios_df.user1)
    grid = np.zeros([max_user + 1, max_user + 1])

    for idx in ratios_df.index:
        row = ratios_df.loc[idx]
        grid[row.user1, row.user2] = row.sim_ratio

    if clustermap:
        sns.clustermap(
            pd.DataFrame(grid), xticklabels=True, yticklabels=True
        ).ax_col_dendrogram.set_title("Similarity of User LearnType Strings")

        path = mypath('RESULTS', 'string_similarities', subfolder='heatmaps')
        save_figure(save=save, path=path, dpi=dpi, fig_format='jpg')

    return grid


def plot_significant_corrs(numeric, df, dpi=300, save=False,
                           path=None, filename='cor_heatmap',
                           title='Significant correlations'):
    """Plot the significant correlations as a heatmap

    Args:
        numeric (pd.Dataframe): numeric df to show
        df (pd.DataFrame): corresponding df potentially with strings
        dpi (int): dots per inch for the figure
        save (bool): Whether to save the plot
        path (str): Path to save the plots
        filename (str): Name of the file to save
        title (str): The title of the plot

    """
    cmap1 = sns.diverging_palette(133, 10, as_cmap=True)
    cmap2 = sns.diverging_palette(220, 50, as_cmap=True)

    fig = plt.figure(figsize=(12, len(numeric) // 4 + 2), dpi=dpi)

    for col in numeric:
        mask = numeric.copy()
        for col_m in mask:
            mask[col_m] = col != col_m
        if col in ['pearson', 'spearman', 'cor', 'correlation',
                   'spear', 'pears']:
            sns.heatmap(numeric, annot=True, fmt='.3f', vmin=-1,
                        vmax=1, cbar=False, mask=mask)
        elif 'F' in col:
            sns.heatmap(numeric, annot=df, fmt='', cbar=False,
                        mask=mask, cmap=cmap2, vmin=1, vmax=8)
        elif 'p-value' in col or col in ['spear_p', 'pears_p']:
            sns.heatmap(numeric, annot=True, fmt='.4f', mask=mask,
                        vmin=0, vmax=5, cbar=False, cmap=cmap1)
        elif 'mean' in col:
            sns.heatmap(numeric, annot=True, fmt='.2f', mask=mask,
                        vmin=-3, vmax=3, cbar=False)

    plt.title(label=title, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelbottom=False,
                    bottom=False, top=False, labeltop=True)

    if path is None:
        spath = mypath('RESULTS', filename, subfolder='heatmaps')
    else:
        spath = os.path.join(path, filename)

    save_figure(save, dpi, spath, fig)

    return fig


def significant_corrs_heatmap(features=None, method='pearson', p_threshold=5,
                              min_cor=0, cor_threshold=0.8, save=True, path=''):
    """Plot an overview heatmap for significant correlations

    Args:
        features: list of features to use
        method (str): The type of correlation to plot (spearman, pearson)
        p_threshold (float): The threshold for the p-value in %
        min_cor (float): The minimum correlation to consider
        cor_threshold (float): The threshold for abs(correlation)
        save (bool): Whether to save the plot
        path (str): Path to save the plots

    Returns:
        plt.figure: The figure handle

    """
    path = os.path.join(path, "Heatmaps")

    kwargs = dict(plot_corr=False, pvalues=False, significant=True,
                  method=method, plot_significant=False, threshold=p_threshold,
                  min_cor=min_cor)

    _, df = correlation_heatmap(features=features, save=False, **kwargs)

    for idx in df.index:

        prio1, prio2 = df.loc[idx, "Prio 1"], df.loc[idx, "Prio 2"]
        feature1, feature2 = df.loc[idx, "Feature 1"], df.loc[idx, "Feature 2"]
        corr = df.loc[idx, method]

        if abs(corr) > cor_threshold:
            if prio2 >= prio1:
                if feature1 in features:
                    features.remove(feature1)
            elif prio1 > prio2:
                if feature2 in features:
                    features.remove(feature2)

    kwargs['plot_significant'] = True

    fig, _ = correlation_heatmap(features=features, save=save, path=path,
                                 **kwargs)

    return fig, features
