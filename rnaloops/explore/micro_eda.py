"""A module to explore the distribution of angles within similar sequences"""
import random
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from rnaloops.config.helper import save_figure
from rnaloops.engineer.add_features import p_norm
from rnaloops.explore.help_fcts import get_sequences_ordered_by_nstructures
from rnaloops.explore.plot_fcts import plot_angles
from rnaloops.prepare.data_loader import load_data


def analyse_distributions(
        save=True,
        cats=("chain_config", "parts_seq"),
        ns=(40, 100, 200),
        n_samples=(120, 75, 50),
        features=("planar_1", "planar_2", "planar_3"),
        plot='auto'
):
    """Analyse the distribution of angles within unique sequences

    The parameters might be set below. This function can be called immediately
    to get the analysis, but is limited in flexibility, so changing it might be
    difficult.

    Parameters
    -------
    save : bool
        Whether to save the resulting plots
    cats : tuple
        The categories to split the data by
    ns : tuple
        The number of unique category entries to analyse. Will start with one
        that contains the most multiloops and goes up to ns[i]
    n_samples : tuple
        The size of samples to take from distribution for calculating p-values
        of normal distribution tests. Make sure to choose a value as large as 
        possible, but less than the minimum number of entries found in 
        categories. This makes p-value calculation independent of sample size
    features : tuple
        The features to analyse (usually planar angles)
    plot : bool or str, default 'auto'
        Whether to show histograms for each unique sequence, by default
        histograms are shown if less than 40 sequences are analysed.

    Returns
    -------
    dict
        key=tuple of param configuration, value=dict with keys of mean, median,
        std and p-value containing each a list of the corresponding values for
        each unique sequence.

    """

    # We need the chain data to compare it with parts seq:
    df = load_data("_cleaned_L2_with_chains")

    n_samples = dict(zip(ns, n_samples))

    # The minimum standard deviation for outlier removal. By default, >=6 is
    # interpreted as keeping all data:
    min_stds = (3, 6)

    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    norm_result = {
        # normale_test does all the job for us:
        f"{cat}-{n}-{min_std}-{feature}": normale_test(
            df, cat, n, min_std, n_samples[n], feature, plot=plot
        )
        for cat in cats
        for n in ns
        for min_std in min_stds
        for feature in features
    }

    # Return the results as dict with tuple of params as key. For each param
    # config we store list of mean values, median values, stds and p-values:
    configs = {}
    for key in norm_result.keys():
        configs[tuple(key.split('-'))] = norm_result[key]

    plot_results(cats, ns, features, configs, save=save)

    return configs


def normale_test(df, cat, n, min_std, n_sample, feature,
                 qqplot=True, plot='auto', verbose=False):
    """Performs a test on normal distribution of values and adds some plots

    Parameters
    ----------
    df : pd.DataFrame
        The data from RNALoops to use
    cat : str
        The column to split data by
    n : int
        The number of unique entries from that column to keep
    min_std : int
        The minimum standard deviation to detect outliers
    n_sample : int
        Sample size for p-value calculation
    feature : str
        The column to use as feature
    qqplot : bool, default True
        Whether to show qqplot
    plot : bool or str, default 'auto'
        Whether to show histograms for each unique sequence, by default
        histograms are shown if less than 40 sequences are analysed.
    verbose : bool
        Whether to log progress

    Returns
    -------
    dict
        A dict like {'mean': [...], 'med': [...], 'std': [...], 'pnorm': [...]}
        with the lists being filled by values for each unique sequence analysed.

    """
    print(cat, n, min_std, n_sample, feature, sep=' | ')

    if plot == 'auto':
        plot = True if n < 40 else False

    sequences, _ = get_sequences_ordered_by_nstructures(df, n=n, cat=cat)

    k = 2 if qqplot else 1

    if plot:
        fig, ax = plt.subplots(
            (len(sequences) - 1) // 2 + 1,
            2 * k,
            figsize=(16 * k, 1 + k * len(sequences)),
            dpi=200,
        )

        ax = ax.ravel()
    else:
        ax = None

    params = {'mean': [], 'med': [], 'std': [], 'pnorm': []}

    for i in range(len(sequences)):

        sequence = sequences[i]

        data, mean, std, n_std, outlier = find_outlier(
            df, feature, sequence, cat, min_std=min_std
        )

        if verbose:
            print(f"Outlier ids of {sequence}:\n{list(outlier)}\n")
            if i % 100 == 0:
                print(i)

        med = np.median(data)
        skew = stats.skew(data)
        pnorm = p_norm(data)

        if plot:
            a = ax[2 * i] if qqplot else ax[i]

            plot_angles(
                [feature],
                save=False,
                legend=False,
                df=df.copy(),
                sequence=sequence,
                ax=a,
                cat=cat,
                hue_col="helix_1_bps",
                outlier_std=n_std,
            )

            n_outlier = 0 if outlier is None else len(outlier)

            a.legend(
                [
                    "\n".join(
                        [
                            f"Arithm. Mittel: {mean:.3f} +/- {std:.3f}",
                            f"Median : {med:.3f}",
                            f"Schiefe: {skew:.3f}",
                            f"Outlier bound: {float(n_std):.2}*std",
                            f"n: {len(data)}",
                            f"n_outlier: {n_outlier}",
                        ]
                    )
                ]
            )

            a.set_xlabel(feature)

        # Use to get equal size samples:
        data = random.sample(list(data), min(n_sample, len(data)))

        if plot and qqplot:
            stats.probplot(data, plot=ax[2 * i + 1])
            ax[2 * i + 1].legend([f"p-value norm-test: {pnorm:.3g}"])

        params['std'].append(std)
        params['med'].append(med)
        params['pnorm'].append(pnorm)
        params['mean'].append(mean)

    if plot:
        plt.tight_layout()
        post = '-outlier_removed' if min_std < 6 else ''
        save_figure(
            f"{cat}-{feature}" + post,
            folder="micro",
            create_if_missing=True,
            save=True,
            recent=False,
        )

    return params


def find_outlier(df, feature, sequence, cat, min_std=6):
    """Find outliers of feature in df for given sequence

    The function maximizes the p-value of normal distribution test for 100
    values between min_std and 6 (if min_std=6, no outliers are detected).
    The best fitting data to normal distribution is returned as data and the
    removed outliers as outlier. Also mean, std of the new data and the found
    optimal n_std is returned.

    Parameters
    ----------
    df : pd.DataFrame
        The data.
    feature : str
        The feature column.
    sequence : str
        The sequence to select.
    cat : str
        The category of the sequence (parts_seq, whole_sequence, chain_config)
    min_std : int, default 6
        The minimum standard deviation for outliers.

    Returns
    -------
    tuple
        (data, mean, std, n_std, outlier) found by analysis

    """
    data = (
        df.copy()
        .reset_index()
        .set_index(cat)
        .loc[sequence, ["index", feature]]
        .set_index("index")
    )[feature]

    if len(data) > 0:
        std = np.std(data)
        mean = np.mean(data)
    else:
        return data[False], np.nan, np.nan, min_std, data

    best, n_std = 0, 0

    if min_std < 6:

        for t in np.linspace(min_std, 6, 100):
            temp = data[abs(data - mean) < t * std]
            pnorm = p_norm(temp)
            if pnorm > best:
                best = pnorm
                n_std = t

        outlier = data[abs(data - mean) >= n_std * std].index
        outlier = sorted(outlier, key=lambda x: -abs(x - mean))
        data = data[abs(data - mean) < n_std * std]

        std = np.std(data)
        mean = np.mean(data)

    else:
        outlier, n_std = None, 6

    return data, mean, std, n_std, outlier


def plot_results(cats, ns, features, configs, save=True):
    """Plots results of micro analysis

    Parameters
    ----------
    cats : list
        The categories analysed (chain_config, parts_seq)
    ns : list
        The number of unique sequences to include
    features : tuple
        The features analysed (planar_1, planar_2, planar_3)
    configs : dict
        The configs result as returned by analyse_distributions
    save : bool
        Whether to save the resulting plots

    """
    for value in ["pnorm", "mean", "std"]:
        for n in ns:
            n_row = len(features)
            fig, ax = plt.subplots(n_row, 2, figsize=(16, n_row * 4), dpi=300)
            for i, cat in enumerate(cats):
                for j, feature in enumerate(features):

                    ax[j, i].set_xlabel("# sequence")
                    if j == 0:
                        ax[j, i].set_title(cat)

                    if value == "pnorm":
                        plot_values(configs, ax[j, i], cat, n, feature,
                                    'pnorm', 'median', ['r+', 'gx'],
                                    feature + ' --- norm-test p-value')

                    elif value == 'mean':
                        plot_values(configs, ax[j, i], cat, n, feature,
                                    'mean', 'mean', ['r+', 'gx'])
                        plot_values(configs, ax[j, i], cat, n, feature,
                                    'med', 'mean', ['b.', 'k.'],
                                    feature + ' --- mean and median in °')
                    else:
                        a = plot_values(configs, ax[j, i], cat, n, feature,
                                        'std', 'median+std',
                                        ['r+', 'gx'],
                                        feature + ' --- standard dev. in °')
                        a.set_ylim([0, 2])

                    ax[0, 0].legend()

            plt.tight_layout()

            save_figure(name=f'micro_analysis_{value}',
                        folder=f'{cats[0]}_{cats[1]}/micro',
                        create_if_missing=True, save=save)


def plot_values(configs, ax, cat, n, feature, value, metric, c, ylabel=''):
    """A helper function to plot results in plot_results"""

    y1 = np.array(configs[(cat, str(n), "6", feature)][value])
    y2 = np.array(configs[(cat, str(n), "3", feature)][value])

    y1, y2 = (y1[~np.isnan(y1) & ~np.isnan(y2)],
              y2[~np.isnan(y2) & ~np.isnan(y1)])

    metric_fct = np.mean if metric == 'mean' else np.median
    med1 = metric_fct(y1)
    med2 = metric_fct(y2)

    ax.plot(y1, f"{c[0]}", label=value + " with outliers")
    ax.plot(y2, f"{c[1]}", label=value + " without outliers")
    ax.plot([0, n], [med1, med1], f"{c[0][0]}.-", label=metric)
    ax.plot([0, n], [med2, med2], f"{c[1][0]}.--", label=metric)

    if metric == 'median+std':
        ns = np.sqrt(n)
        std1, std2 = np.std(y1) / ns, np.std(y2) / ns
        ax.plot([0, n], [med1 - std1, med1 - std1], f"{c[0][0]}:")
        ax.plot([0, n], [med2 - std2, med2 - std2], f"{c[1][0]}:")
        ax.plot([0, n], [med1 + std1, med1 + std1], f"{c[0][0]}:")
        ax.plot([0, n], [med2 + std2, med2 + std2], f"{c[1][0]}:")

    ax.set_ylabel(ylabel)

    return ax
