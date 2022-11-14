import random
from typing import Iterable

import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt
import seaborn as sns
import colorcet as cc
from sklearn.neighbors import LocalOutlierFactor

from .help_fcts import get_frequent_sequences, get_sequence_mean_angles, \
    confidence_ellipse
from ..cluster.cluster_fcts import generic_clustering
from ..cluster.cluster_plot import do_plot
from ..config.helper import save_figure
from ..prepare.data_loader import load_data


def plot_angles(
        feature,
        df=None,
        sequence='',
        show_other=True,
        ms=20,
        hue_col="parts_seq",
        marker_col=None,
        legend=True,
        save=True,
        ax=None,
        cat="whole_sequence",
        outlier_std=None,
        color=None,
        w_size=5,
        title=True,
        cut='min_max',
        plot_mean=False
):
    if df is None:
        df = load_data("_cleaned_L2")

    df["id"] = df.index
    df = df.set_index(cat)
    if isinstance(sequence, int):
        sequence = (
            df.groupby(by=cat)
            .count()
            .sort_values("id", ascending=False)
            .index[sequence]
        )

    data = df.loc[sequence]
    if isinstance(data, pd.Series):
        data = pd.DataFrame([data])
    other = df[df.index != sequence]
    hue = data[hue_col]
    markers = data[marker_col] if marker_col is not None else None

    plot_data, plot_other = [], []
    for f in feature:
        temp = data[f]
        if outlier_std is not None:
            inlier = abs(temp - np.mean(temp)) < outlier_std * np.std(temp)
            hue = hue[inlier]
            if marker_col is not None:
                markers = markers[inlier]
            temp = temp[inlier]
        plot_data.append(temp)
        plot_other.append(other)

    if ax is None:
        if len(feature) == 3:
            plt.figure(figsize=(6, 4), dpi=100)
            ax = plt.axes(projection='3d')
        else:
            _, ax = plt.subplots(figsize=(8, 6), dpi=300)

    if show_other:

        if len(feature) == 2:
            sns.scatterplot(
                x=plot_other[0],
                y=plot_other[1],
                s=ms / 10,
                legend=False,
                ax=ax,
                color="k",
            )

        if len(feature) == 3:
            ax.scatter3D(
                xs=plot_other[0],
                ys=plot_other[1],
                zs=plot_other[2],
                s=ms / 10,
                color="k",
            )

    n_colors = len(hue.unique())
    palette = sns.color_palette(cc.glasbey, n_colors=n_colors)
    legend = False if hue_col == "home_structure" else legend

    xd = plot_data[0]

    if len(feature) == 1:
        sns.histplot(
            x=xd.values,
            legend=legend,
            ax=ax,
            hue=hue,
            palette=palette
        )

    if len(feature) == 2:
        yd = plot_data[1]
        sns_legend = False if isinstance(legend, Iterable) else legend
        kwargs = dict(x=xd,
                      y=yd,
                      s=ms,
                      legend=sns_legend,
                      ax=ax,
                      style=markers)
        if color is None:
            a = sns.scatterplot(palette=palette, hue=hue, **kwargs)
        else:
            a = sns.scatterplot(color=color, **kwargs)

        if isinstance(legend, Iterable):
            ax.legend(legend)

        elif sns_legend:

            handles, labels = a.get_legend_handles_labels()
            hue_label = labels.index(hue_col)
            style_label = labels.index(marker_col)

            max_n = 8

            if style_label <= max_n:
                hue_handles = handles[hue_label + 1:style_label]
                hue_labels = labels[hue_label + 1:style_label]
            else:
                hue_handles = handles[hue_label + 1:max_n+1]
                hue_labels = labels[hue_label + 1:max_n] + ['...']
            if style_label + 5 <= len(labels):
                style_handles = handles[style_label + 1:style_label + max_n + 1]
                style_labels = (labels[style_label + 1:style_label + max_n] +
                                ['...'])
            else:
                style_handles = handles[style_label + 1:]
                style_labels = labels[style_label + 1:]

            a.legend(hue_handles + style_handles, hue_labels + style_labels,
                     title='color = ' + labels[hue_label] +
                           ' and ' + 'marker = ' + labels[style_label])

        mean_x, mean_y = np.mean(xd), np.mean(yd)

        if plot_mean:
            confidence_ellipse(xd, yd, ax, n_std=1.0, facecolor=color, ls='-',
                               alpha=0.3, label='_nolegend_', edgecolor='k',
                               zorder=10)
            confidence_ellipse(xd, yd, ax, n_std=1.0 / np.sqrt(len(xd)),
                               facecolor=color, alpha=0.6, edgecolor='k',
                               lw=1, ls='-', label='_nolegend_', zorder=20)

        if cut == 'std':
            x_range = np.std(xd)
            y_range = np.std(yd)
            x_range = np.std(xd[abs(xd - mean_x) < 3 * x_range])
            y_range = np.std(yd[abs(yd - mean_y) < 3 * y_range])
            if not np.isnan(x_range) and x_range > 0 and y_range > 0:
                ax.set_xlim(mean_x - w_size * x_range,
                            mean_x + w_size * x_range)
                ax.set_ylim(mean_y - w_size * y_range,
                            mean_y + w_size * y_range)

        elif cut == 'min_max':
            x_range = max(xd) - min(xd)
            y_range = max(yd) - min(yd)
            if x_range > 0 and y_range > 0:
                ax.set_xlim(min(xd) - 0.1 * x_range,
                            max(xd) + 0.1 * x_range)
                ax.set_ylim(min(yd) - 0.1 * y_range,
                            max(yd) + 0.1 * y_range)

    if len(feature) == 3:

        yd, zd = plot_data[1], plot_data[2]
        ax.scatter3D(
            xs=xd,
            ys=yd,
            zs=zd,
            s=ms,
            color='r'
        )

        if cut:
            x_range = max(xd) - min(xd)
            y_range = max(yd) - min(yd)
            z_range = max(zd) - min(zd)
            ax.set_xlim(min(xd) - 0.1 * x_range,
                        max(xd) + 0.1 * x_range)
            ax.set_ylim(min(yd) - 0.1 * y_range,
                        max(yd) + 0.1 * y_range)
            ax.set_zlim(min(zd) - 0.1 * z_range,
                        max(zd) + 0.1 * z_range)

    if isinstance(title, str):
        ax.set_title(title)
    elif title:
        ax.set_title(sequence + f' ({data.loop_type.iloc[0]})')

    save_figure(sequence, folder='angles/' + '-'.join(feature),
                create_if_missing=True, save=save, recent=False)

    return ax


def cluster_angles(
        df,
        feature,
        n_cluster=25,
        n_neighbors=1,
        contam=1,
        save=False,
        ms_data=5,
        ms_other=1,
        plot_labels=False,
        dpi=600,
        annot_perc=101,
        alg="agglomerative",
        title=None,
        fs=4,
        do_cluster=True,
        hue_col="whole_sequence",
        size_col=None,
        extension='',
        ls=1.5,
        labels=None
):
    data = df[feature]

    if contam == 1:
        inlier, outlier = data, data
    else:
        clf = LocalOutlierFactor(n_neighbors=n_neighbors,
                                 contamination=contam,
                                 p=2)
        outlier = clf.fit_predict(data)
        df = df[outlier > 0]
        inlier = data[outlier > 0]
        outlier = data[outlier < 0]

    kwargs = dict(dim=len(data.columns), alg=alg, path="", save=False)

    c_data = inlier if do_cluster else inlier.iloc[:n_cluster]

    data, result = generic_clustering(
        c_data,
        features=data.columns,
        n_cluster=n_cluster,
        scale=None,
        left_out=0,
        **kwargs,
    )

    if not do_cluster:
        labels = pd.Categorical(df[hue_col]).codes
        labels_df = pd.DataFrame()
        labels_df[result["labels"].columns[0]] = labels
        result["labels"] = labels_df

    ax = do_plot(
        [inlier], result, s=ms_data, scale=1, dpi=dpi, fontsize=fs, **kwargs
    )

    texts = []
    if plot_labels:
        labels = inlier.index if labels is None else labels
        for i, j in enumerate(labels):
            if random.randint(0, 100) < annot_perc:
                x = list(inlier[feature[0]])[i]
                y = list(inlier[feature[1]])[i]
                texts.append(plt.text(x, y, str(j), fontsize=ls))

    adjust_text(texts)

    x_other, y_other = outlier[feature[0]], outlier[feature[1]]
    _ = sns.scatterplot(x=x_other, y=y_other, s=ms_other, legend=False, ax=ax,
                        color='k', size=size_col, alpha=0.3)

    if title is None:
        if not do_cluster:
            title = f"Angles colored by {hue_col}"
        else:
            title = f"Predicted {alg} angle clusters"

    ax.set_title(title, fontsize=fs + 2)

    if do_cluster:
        name = f'{feature[0]}-{feature[1]}-{alg}-{extension}-cluster'
    else:
        name = f'{feature[0]}-{feature[1]}-by-{hue_col}'

    save_figure(name=name, folder='cluster', save=save)

    return ax


def planar_angles_diff_clustermap(df, min_n=50, save=True):
    data = get_frequent_sequences(df, min_n=min_n)
    data = get_sequence_mean_angles(data, category="index")

    diff = (
            pd.DataFrame(abs(data.planar_1.values -
                             data.planar_1.values[:, None]))
            + pd.DataFrame(abs(data.planar_2.values -
                               data.planar_2.values[:, None]))
            + pd.DataFrame(abs(data.planar_3.values -
                               data.planar_3.values[:, None]))
    )

    closest = pd.DataFrame(columns=["s2", "diff"])
    for seq, col in zip(data.index, diff.columns):
        closest.loc[seq] = (
            data.index[diff[diff > 0][col].argmin()],
            diff[diff > 0][col].min(),
        )

    diff.index, diff.columns = data.index, data.index
    cmap = sns.clustermap(diff, xticklabels=True, yticklabels=True)
    cmap.ax_heatmap.figure.set_size_inches(
        len(data.index) // 4 + 5, len(data.index) // 4 + 5
    )

    cmap.cax.set_visible(False)
    plt.tight_layout()

    save_figure(name="planar_angles_diff_cluster", folder="cluster", save=save)
