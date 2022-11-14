import os
import random
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from rnaloops.cluster.clustering import do_cluster
from rnaloops.config.helper import mypath, save_figure
from rnaloops.engineer.add_features import load_agg_df
from rnaloops.prepare.data_loader import load_data
from rnaloops.prepare.explore_fcts import open_svg_image


def main_clustering(agg_df, way, n_cluster, alg, chunk_size):
    files, missing = [], False
    for file_idx in range(int(np.ceil(n_cluster / chunk_size))):
        min_idx, max_idx = file_idx * chunk_size, (file_idx + 1) * chunk_size
        f = mypath(folder=f'cluster_ids/n_{n_cluster}/way{way}/csv',
                   file=f'{alg}_{min_idx}-{max_idx}.csv',
                   create_if_missing=True)
        if os.path.isfile(f):
            files.append(f)
        else:
            missing = True

    if missing:

        feature = [f'planar_{i}_median' for i in range(1, way + 1)]
        cluster = do_cluster(agg_df, feature, (n_cluster,), dim=way, save=False,
                             alg=alg, plot=False, scaler=None)[0]

        for chunk in range(int(np.ceil(n_cluster / chunk_size))):
            min_idx, max_idx = chunk * chunk_size, (chunk + 1) * chunk_size
            label_col = cluster.columns[-1]
            c_chunk = cluster[cluster[label_col].isin(range(min_idx, max_idx))]
            f = mypath(folder=f'results/cluster_ids/n_{n_cluster}/way{way}/csv',
                       file=f'{alg}_{min_idx}-{max_idx}.csv',
                       create_if_missing=True)
            if f not in files:
                c_chunk.to_csv(f)
                files.append(f)

    return files


def main_cluster_analysis(
        files=None,
        way=3,
        agg_df=None,
        raw_df=None,
        agg_all=None,
        save=True,
        max_devs=(5,),
        min_density=0,
        min_n=10,
        dpi=300,
        verbose=False
):
    for file in files:

        cluster = pd.read_csv(file, index_col='parts_seq')
        file_split = file.split('/')
        n_cluster = int(file_split[2][2:])
        indices = file_split[-1].split('_')[1].split('.')[0].split('-')
        min_idx, max_idx = int(indices[0]), int(indices[1])

        for idx in range(min_idx, max_idx):
            density, *out = analyse_cluster(
                way, idx, raw_df, cluster, agg_df, agg_all
            )

            show_cluster(density,
                         way,
                         f'{n_cluster}-{idx}',
                         *out,
                         save=save,
                         max_devs=max_devs,
                         min_density=min_density,
                         n_clusters=n_cluster,
                         min_n=min_n,
                         dpi=dpi)

            if verbose:
                print(f'Checked n={n_cluster}, min_density={min_density}, '
                      f'max_angle_deviation={max_devs}, way='
                      f'{way}, cluster {idx}')


def analyse_cluster(way=3, idx=0, raw_df=None, cluster=None, agg_way=None,
                    agg_all=None, all_seq=False):
    if raw_df is None:
        raw_df = load_data('_cleaned_L2_with_chains')
    if agg_way is None:
        agg_way = load_agg_df(way=way)
    if cluster is None:
        if all_seq:
            cluster = agg_way
        else:
            cluster = pd.read_csv(mypath('CLUSTER', f'{way}.csv'))
    if agg_all is None:
        agg_all = load_agg_df()

    if all_seq:
        cluster_seq = agg_way.index
    else:
        cluster_seq = cluster[cluster[cluster.columns[-1]] == idx].index

    cluster_df = agg_way.loc[cluster_seq]
    cluster_raw_df = raw_df[raw_df.parts_seq.isin(cluster_seq)]

    diff_inner = pd.DataFrame(index=cluster_seq)
    diff_all = pd.DataFrame(index=agg_all.index)

    for i in range(1, way + 1):
        median = cluster_df[f'planar_{i}_mean'].median()
        diff_inner[f'diff_{i}'] = abs(median - cluster_df[f'planar_{i}_mean'])
        diff_all[f'diff_{i}'] = abs(median - agg_all[f'planar_{i}_mean'])

    diff_inner = diff_inner.mean(axis=1).sort_values()
    diff_all = diff_all.mean(axis=1).sort_values().iloc[:len(diff_inner)]

    cluster_df = cluster_df.loc[diff_inner.index]
    n = len(cluster_seq)

    std = get_rolling_std(cluster_df, way, n)

    std = np.array([1, 1] + [x for x in std if x != 0])
    density = np.array([0, 0] + list(range(1, len(std) - 1))) / std ** way

    return density, diff_inner, diff_all, agg_all, cluster_df, cluster_raw_df


def get_rolling_std(cluster_df, way, n):
    std = [np.mean([cluster_df[f'planar_{i}_mean'][:end].std()
                    for i in range(1, way + 1)])
           for end in range(2, n)]
    return std


def get_cut(min_density, max_dev, density, diff_inner):
    cut = 0
    for i, (den, div) in enumerate(zip(density, diff_inner)):
        if den > min_density and div < max_dev:
            cut = i

    return cut


def plot_cluster_analysis(density, diff_inner, diff_all, ax=None, cut=0,
                          min_density=15, max_dev=5, return_ax=False):
    min_x = 5

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    ax.plot(diff_all.values, 'k--', label='all sequences')
    ax.plot(diff_inner.values, 'k', label='cluster sequences')
    ax2 = ax.twinx()
    ax2.plot(density, 'r', label='sequence density')

    ax.set_xlabel('number of sequences')
    ax.set_ylabel('Mean distance from cluster median / °', color='k')
    ax2.set_ylabel('Mean density in sequences / std', color='r')

    y_max = 3 * max_dev
    plt.xlim([min_x, len(diff_inner[min_x:]) * 1.1])
    ax.set_ylim([0, 1.1 * y_max])
    ax2.set_ylim([0, 1.1 * 3 * min_density])
    ax.vlines(cut, 0, y_max * 1.1, 'g', label=f'density > {min_density}, '
                                              f'deviation < {max_dev}°')
    ax.legend(framealpha=1)
    ax2.legend(loc='lower right', framealpha=1)

    if return_ax:
        return ax, ax2

    return cut


def show_cluster(density, way, c_id, diff_inner, diff_all, agg, cluster_df,
                 cluster_raw_df, s=25, save=False, min_density=15,
                 max_devs=(5,), n_clusters=75, min_n=10, dpi=300):
    m = {i: f"planar_{i}_median" for i in range(1, way + 1)}
    rows, cols = way + 1, 8
    widths = [1, 0.8, 0.8, 0.8, 0.8, 0.8, 1, 1.3]

    cuts = [get_cut(min_density, max_dev, density, diff_inner)
            for max_dev in max_devs]

    if all(cut < min_n for cut in cuts):
        return

    for max_dev, cut in zip(max_devs, cuts):

        if cut < min_n:
            return

        min_density = int(max(density[cut] // 4 * 4, min_density, 1))

        fig, ax = plt.subplots(rows, cols,
                               figsize=(4.4 * cols, 4 * rows),
                               dpi=dpi,
                               gridspec_kw={'width_ratios': widths})

        j1 = joined_axis(fig, ax, 0, 1, m=2)
        j1.axis('off')

        j2 = joined_axis(fig, ax, 0, 3, m=2)
        j2.set_title('Cluster sequence density and mixing characteristic',
                     weight='semibold')

        plot_cluster_analysis(density, diff_inner, diff_all, ax=j2, cut=cut,
                              min_density=min_density, max_dev=max_dev)

        cut_df, n = cluster_df[:cut], cut

        ax[1, 0].set_title('Location of cluster', weight='semibold')
        ax[1, 1].set_title('Sequences angle distribution', weight='semibold')
        ax[1, 2].set_title('Multiloops angle distribution', weight='semibold')
        ax[0, 6].set_title('Organisms of cluster sequences', weight='semibold')
        ax[1, 6].set_title('Chains of cluster sequences', weight='semibold')
        ax[1, 3].set_title('Length of strands', weight='semibold')
        ax[1, 4].set_title('Length of helices', weight='semibold')
        ax[2, 6].set_title('Length of sequences', weight='semibold')
        ax[3, 6].set_title('Length of inner loop', weight='semibold')
        ax[0, 5].set_title('Fraction of bases', weight='semibold')

        cluster_raw_df_cut = cluster_raw_df[
            cluster_raw_df.parts_seq.isin(cut_df.index)]

        j3 = joined_axis(fig, ax, 0, 7, n=rows)
        j3.axis('off')
        sequences = []
        for j, seq in enumerate(cluster_df.index[:50]):
            if j == n:
                sequences.append('\n-------- end of sequences '
                                 'in threshold cut --------\n')
            if len(seq) < 60:
                sequences.append(seq)
            else:
                sequences.append(seq[:60] + '...')
        indices = []
        for k, idx in enumerate(cluster_raw_df_cut.index[:300]):
            indices.append(str(idx))
            if (k + 1) % 8 == 0:
                indices.append('\n')
        j3.text(s=f'Parts sequences in Cluster (max 50 shown):\n\n'
                  + '\n'.join(sequences), x=-0.1, y=1, fontsize=8,
                verticalalignment='top')
        j3.text(s=f'Multiloop indices in Cluster (max 300 shown):\n\n | '
                  + ' | '.join(indices), x=-0.1, y=0.48 + 0.07 * (way - 3),
                fontsize=8,
                verticalalignment='top')

        plot_clusters(ax[1, 0], agg, cluster_df, cut_df, m[1], m[2], s, True)

        x, y = (m[2], m[3]) if way == 3 else (m[3], m[4])
        plot_clusters(ax[2, 0], agg, cluster_df, cut_df, x, y, s, False)

        row = 2
        for idx, w in enumerate([5, 7, 9, 11, 13]):
            if way >= w:
                x, y = (m[w - 1], m[w]) if way == w else (m[w], m[w + 1])
                row = idx + 3
                plot_clusters(ax[row, 0], agg, cluster_df, cut_df, x, y, s,
                              False)

        for i in range(3, way + 3, 3):
            plot_3d(fig, ax, rows, cols, cluster_df, cut_df, m, i, s, row)

        plot_hists(ax, cluster_df, cut_df, m, cluster_raw_df)
        most_common = plot_bars(cluster_raw_df_cut, ax)
        plot_length(cluster_df, cut, ax, way)
        plot_fractions(cluster_df, cut_df, ax)

        pseudo = any('[' in x for x in cluster_raw_df.db_notation)
        pseudo = 'yes' if pseudo else 'no'

        j1.annotate(f'{way}-way cluster (ID {c_id})',
                    xy=(0, 0.85),
                    weight='bold',
                    fontsize=26)

        p = [density[n], diff_inner[n], most_common[0][:30],
             most_common[1][:30], pseudo]
        text = '\n'.join([
            f'Characteristics at threshold cut:',
            f'- number of distinct sequences (in secondary structure): {n}',
            f'- mean density of sequences / planar angle stddev: {p[0]:.2f}',
            f'- mean planar angle deviation from cluster median: {p[1]:.2f}°',
            f'- most common organism: {p[2]}',
            f'- most common chain label: {p[3]}',
            f'- contains pseudoknots (based on [] in dotbracket): {p[4]}'
        ])
        j1.annotate(text, xy=(0, 0.05), fontsize=16)

        for col in [5, 6]:
            for row in range(4, way + 1):
                ax[row, col].remove()

        if way == 6 or way == 8:
            ax[rows - 1, 0].remove()

        example_image = get_seq_png(indices, allow_download=True,
                                    allow_load=False)
        ax[0, 0].imshow(example_image)
        ax[0, 0].set_title('Example multiloop of cluster', weight='semibold')
        ax[0, 0].axis('off')

        plt.tight_layout()

        folder = f'cluster_ids/n_{n_clusters}/way{way}/density_{min_density}'
        save_figure(f'{c_id}_dev{max_dev}', dpi=dpi, save=save,
                    folder=folder, create_if_missing=True, recent=False,
                    tight=True)

        out_df = pd.DataFrame(index=cluster_df.index)
        out_df[cut] = 0
        out_df.iloc[:n, 0] = 1
        out_df.to_csv(mypath('RESULTS', subfolder=folder + '/sequences',
                             file=f'cluster_{c_id}', create_if_missing=True))

        plt.close()

    return


def get_seq_png(indices=None, n=1, verbose=False, allow_load=True,
                label='', shuffle=False, allow_download=True):
    success, i, image, count = False, 0, None, 0

    if indices is None:
        indices = load_data('_cleaned_L2_final').index

    if shuffle:
        indices = random.choices(indices, k=n)

    while not success or count < n:
        idx = indices[i]
        x = 1000
        subfolder = f'{int((int(idx) // x) * x)}-{int((int(idx) // x + 1) * x)}'
        path = mypath('SEQ_PNG', subfolder=subfolder, file=f'{idx}.png',
                      create_if_missing=True)
        try:
            if allow_load:
                image = plt.imread(path)
                success = True
                count += 1
                i += 1
            else:
                raise OSError
        except OSError:
            if allow_download:
                try:
                    open_svg_image(indices[i], display_svg=False, get_png=True)
                    image = plt.imread(path)
                    success = True
                    count += 1
                    i += 1
                    if verbose:
                        print(f'{label} downloaded {idx}, in total {count}')
                except (ValueError, FileExistsError):
                    if verbose:
                        print(f"Could not open svg of {idx}, try next...")
                    i += 1
            else:
                i += 1

    return image


def plot_clusters(ax, agg, cluster_df, cut_df, x, y, s, legend):
    labels = ["all sequences", "agglom. cluster", "threshold cut"]

    sns.scatterplot(
        data=agg,
        x=x,
        y=y,
        ax=ax,
        s=s / 2,
        alpha=0.2,
        color="k",
        label=labels[0],
        zorder=-10 ** 6,
        legend=legend
    )

    for df, c, tag, a in zip([cluster_df, cut_df], ["r", "g"], labels[1:],
                             [0.5, 1]):
        sns.scatterplot(data=df,
                        x=x,
                        y=y,
                        ax=ax,
                        s=s,
                        color=c,
                        label=tag,
                        zorder=-len(df),
                        legend=legend,
                        alpha=a)


def plot_hists(ax, cluster_df, cut_df, m, raw):
    labels = ["agglom. cluster", 'threshold cut']
    for i, x in enumerate(m.values()):
        for df, c, label, a in zip([cluster_df, cut_df], ["r", "g"], labels,
                                   [0.5, 1]):
            legend = True if i == 0 else False
            sns.histplot(
                data=df,
                x=x,
                ax=ax[i + 1, 1],
                color=c,
                label=label,
                zorder=-len(df),
                binwidth=2,
                legend=legend,
                alpha=a,
                edgecolor=None
            )
            all_x = '_'.join(x.split('_')[:-1])
            all_df = raw[raw.parts_seq.isin(df.index)]
            sns.histplot(
                data=all_df,
                x=all_x,
                ax=ax[i + 1, 2],
                color=c,
                label=label,
                zorder=-len(df),
                binwidth=1,
                legend=legend,
                alpha=a,
                edgecolor=None
            )
        ax[1, 1].legend()


def plot_3d(fig, ax, rows, cols, cluster_df, cut_df, m, i, s, row):
    pos = (row + (i - 3) // 3 + 1) * cols + 1
    a = fig.add_subplot(rows, cols, pos, projection='3d')

    ax.ravel()[pos - 1].set_yticklabels([])
    ax.ravel()[pos - 1].set_xticklabels([])
    ax.ravel()[pos - 1].set_yticks([])
    ax.ravel()[pos - 1].set_xticks([])

    way = max(m.keys())
    for df, c, al in zip([cluster_df, cut_df], ["r", "g"], [0.5, 1]):
        if i == 3:
            x, y, z = m[1], m[2], m[3]
            kwargs = dict(xs=df[x], ys=df[y], zs=df[z])
        else:
            w = i if i < way else way
            x, y, z = m[-2 + w], m[-1 + w], m[w]
            kwargs = dict(xs=df[x], ys=df[y], zs=df[z])
        a.scatter3D(
            **kwargs,
            color=c,
            s=s,
            zorder=-len(df),
            alpha=al
        )
        a.set_xlabel(x, fontsize=8)
        a.set_ylabel(y, fontsize=8)
        a.set_zlabel(z, fontsize=8)
        a.tick_params(axis='both', which='major', labelsize=6)
        warnings.filterwarnings("ignore")
        a.dist = 14
        warnings.filterwarnings("default")


def plot_length(cluster_df, cut, ax, way):
    c1 = cluster_df[cut:]
    c2 = cluster_df[:cut]
    c1['c'] = "ab"
    c2['c'] = "bg"
    c = pd.concat([c1, c2])
    idx = [i for i in range(1, way + 1)]

    labels = [f'strand_{i}_nts' for i in idx]
    labels += [f'helix_{i}_bps' for i in idx]
    labels += ['seq_length', 'loop_length']
    rows = idx + idx + [2, 3]
    cols = [3] * way + [4] * way + [6] * 2

    for row, col, label in zip(rows, cols, labels):
        grouped = (
            c.groupby([label, 'c'])
            .size().reset_index()
            .pivot(columns='c', index=label, values=0)
        )
        grouped.plot(
            kind='bar',
            ax=ax[row, col],
            color=['r', 'g'],
            alpha=0.7,
            stacked=True,
            legend=False
        )


def plot_fractions(cluster_df, cut_df, ax):
    col = 5
    for idx, kind in enumerate(['frac__A', 'frac__C', 'frac__G', 'frac__U']):
        for df, c, a in zip([cluster_df, cut_df], ["r", "g"], [0.5, 1]):
            sns.histplot(
                df[kind],
                ax=ax[idx, col],
                alpha=a,
                legend=False,
                color=c,
                zorder=-len(df),
                edgecolor=None
            )


def plot_bars(cluster_raw_df, ax, align=False):
    col = 6

    most_common = []
    for idx, kind in enumerate(['main_organism', 'chain_label']):
        category = get_sequences_per_category(cluster_raw_df, kind)
        try:
            g = sns.barplot(y=category.index,
                            x=category,
                            orientation='horizontal',
                            ax=ax[idx, col])
            cut_off = 0.5 * max(category)
            for p, label in zip(ax[idx, col].patches, category.index):
                perc = 100 * p.get_width() / sum(category)
                annot = f"{label[:30]}: {perc:.2f}%"
                x = p.get_width() if align else max(category) / 100
                y = p.get_y() + p.get_height() / 1.2
                if x >= cut_off:
                    ax[idx, col].annotate(annot, (x * 0.1, y), color="black")
                else:
                    ax[idx, col].annotate(annot, (x * 1.1, y), color="black")

            ax[idx, col].set_ylabel(None)
            g.set(yticklabels=[], yticks=[], xlabel=kind)

            most_common.append(category.index[0])
        except ValueError:
            most_common.append('')

    return most_common


def joined_axis(fig, ax, x, y, n=1, m=1):
    gs = ax[x, y].get_gridspec()
    for a in ax[x:x + n, y:y + m].ravel():
        a.remove()
    return fig.add_subplot(gs[x:x + n, y:y + m])


def get_sequences_per_category(df, kind, max_n=15):
    category = (
                   df
                   .groupby(['parts_seq', kind])
                   .head(1)
                   .groupby(kind)
                   .count()
                   .sort_values('loop_type', ascending=False)
               )['loop_type'][:max_n]

    return category


def get_overall_densities():
    ylim = [0.5, 0.0025, 0.003, 0.0025, 0.0025, 0.0025]

    for idx, way in enumerate(range(3, 9)):

        density, diff_inner, diff_all, agg_all, cluster_df, cluster_raw_df = (
            analyse_cluster(way=way, all_seq=True)
        )

        ax, ax2 = plot_cluster_analysis(
            density,
            diff_inner,
            diff_all,
            ax=None,
            min_density=1,
            max_dev=10,
            return_ax=True
        )
        ax2.set_ylim([10 ** -10, ylim[idx]])
        if way > 4:
            ax2.set_yscale('log')
        ax.set_xlim([10, len(density)])
        ax.legend(['all sequences', 'loop_type sequences'], loc='upper right')
        ax2.legend(['sequence density'], loc='lower right')

        plt.savefig(f'results/densities/way_{way}_seq_density.png')
