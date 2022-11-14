"""Module to analyse mean values of angles for distinct sequences"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from rnaloops.explore.help_fcts import get_frequent_sequences, \
    get_sequence_mean_angles
from rnaloops.explore.plot_fcts import cluster_angles


def cluster_planar_angles(df, min_n=100, label_kind='sequence', **kwargs):
    """Cluster planar angles and label the label_kind attribute"""

    frequent = get_frequent_sequences(df, min_n=min_n)
    df = get_sequence_mean_angles(frequent, "index")
    labels = get_labels(df, label_kind)

    combis = [["planar_1", "planar_2"], ["planar_1", "planar_3"]]

    for feature in combis[:1]:
        std_cols = [c + "_std" for c in feature]
        size_col = np.sqrt(df[std_cols[0]] ** 2 / 2 + df[std_cols[1]] ** 2 / 2)

        cluster_angles(
            df,
            feature=feature,
            size_col=size_col,
            plot_labels=True,
            title=f"Angles by sequence with {label_kind} annotates",
            labels=labels,
            **kwargs,
        )


def get_labels(df, kind):
    """Get labels for given label kind"""

    if kind == 'sequence':
        return df.index
    if kind == 'sequence length':
        return [len(x) for x in df.index]
    if kind == 'parts length':
        return ['|'.join([str(len(x.split('-')[0]))
                for x in y.split('|')])
                for y in df.index]
    if kind == 'strand length':
        return ['|'.join([str(len(x))
                for x in y.split('|') if '-' not in x])
                for y in df.index]
    if kind == 'helix length':
        return ['|'.join([str(len(x.split('-')[0]))
                for x in y.split('|') if '-' in x])
                for y in df.index]


def feature_barplot(agg, cat, feature='planar_1_mean'):
    numeric = agg[cat].dtype in (np.int64, np.float32, np.float64)
    agg[cat] = agg[cat] if numeric else agg[cat].str.upper()
    x = agg[cat]

    if not numeric:
        groups = x.groupby(x).count().sort_values(ascending=False)
    else:
        groups = x.groupby(x).count().sort_index()

    groups = groups[groups.values > 4]

    agg = agg[agg[cat].isin(groups.index)]
    x = agg[cat]

    fig, ax = plt.subplots(figsize=(12, 3), dpi=600)
    sns.barplot(data=agg, x=x, y=agg[feature], order=groups.index,
                errorbar=('ci', 95))

    ax.bar_label(ax.containers[0], labels=groups, label_type='center')
    ax.bar_label(ax.containers[0], fmt='%.1f', padding=10)

    ax.set_ylim([0, max(ax.containers[0].datavalues * 1.3)])

    ax.set_title('Mean angles by categorical feature with category counts and' +
                 ' 95% confidence intervalls')
