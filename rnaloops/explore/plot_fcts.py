import random

import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt
import seaborn as sns
import colorcet as cc
from sklearn.neighbors import LocalOutlierFactor

from .help_fcts import get_frequent_sequences, get_sequence_mean_angles
from ..cluster.cluster_fcts import generic_clustering
from ..cluster.cluster_plot import do_plot
from ..config.helper import save_figure
from ..prepare.data_loader import load_data


def plot_angles(
        feature,
        df=None,
        sequence=0,
        show_other=True,
        ms=20,
        hue_col="parts_seq",
        legend=True,
        save=True,
        ax=None,
        cat="whole_sequence",
        outlier_std=100
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
    other = df[df.index != sequence]
    hue = data[hue_col]

    plot_data, plot_other = [], []
    for f in feature:
        temp = data[f]
        inlier = abs(temp - np.mean(temp)) < outlier_std * np.std(temp)
        hue = hue[inlier]
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
    legend = False if hue_col != "parts_seq" else legend

    if len(feature) == 1:
        sns.histplot(
            x=plot_data[0].values,
            legend=legend,
            ax=ax,
            hue=hue,
            palette=palette
        )

    if len(feature) == 2:
        sns.scatterplot(
            x=plot_data[0],
            y=plot_data[1],
            hue=hue,
            palette=palette,
            s=ms,
            legend=legend,
            ax=ax,
        )

        x_range = max(plot_data[0]) - min(plot_data[0])
        y_range = max(plot_data[1]) - min(plot_data[1])
        ax.set_xlim(min(plot_data[0]) - 0.01 * x_range,
                    max(plot_data[0]) + 0.01 * x_range)
        ax.set_ylim(min(plot_data[1]) - 0.01 * y_range,
                    max(plot_data[1]) + 0.01 * y_range)

    if len(feature) == 3:
        ax.scatter3D(
            xs=plot_data[0],
            ys=plot_data[1],
            zs=plot_data[2],
            s=ms,
            color='r'
        )

        x_range = max(plot_data[0]) - min(plot_data[0])
        y_range = max(plot_data[1]) - min(plot_data[1])
        z_range = max(plot_data[2]) - min(plot_data[2])
        ax.set_xlim(min(plot_data[0]) - 0.1 * x_range,
                    max(plot_data[0]) + 0.1 * x_range)
        ax.set_ylim(min(plot_data[1]) - 0.1 * y_range,
                    max(plot_data[1]) + 0.1 * y_range)
        ax.set_zlim(min(plot_data[2]) - 0.1 * z_range,
                    max(plot_data[2]) + 0.1 * z_range)

    ax.set_title(sequence + f' ({data.loop_type.iloc[0]})')

    save_figure(sequence, folder='angles/'+'-'.join(feature),
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

    kwargs = dict(dim=2, alg=alg, path="", save=False)

    c_data = inlier if do_cluster else inlier.iloc[:n_cluster]

    _, result = generic_clustering(
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
