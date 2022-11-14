import numpy as np
import pandas as pd

from rnaloops.engineer.add_features import load_agg_df


def get_similar_seq(sequence, df=None, z=3, verbose=1):

    if df is None:
        df = load_agg_df()

    s_df = df.loc[sequence]
    n = s_df.loop_type
    similar = []

    m1s, s1s = [], []
    for i in range(1, n + 1):
        m1s.append(s_df[f'planar_{i}_mean'])
        s1s.append(s_df[f'planar_{i}_std'])

    for other in df.index:
        o_df = df.loc[other]
        deg_diff, std_diff, std_diff_left, m2s = [], [], [], []
        for i in range(1, n + 1):
            m1 = m1s[i - 1]
            s1 = s1s[i - 1]
            m2 = o_df[f'planar_{i}_mean']
            s2 = o_df[f'planar_{i}_std']
            deg_diff.append(abs(m1 - m2))
            if s1 + s2 != 0:
                std_diff.append(abs(m1 - m2) / (s1 + s2))
            else:
                std_diff.append(0)
            if s1 != 0:
                std_diff_left.append(abs(m1 - m2) / s1)
            else:
                std_diff_left.append(0)
            if verbose > 2:
                m2s.append(m2)
        std_diff_mean = np.mean(std_diff)
        deg_diff_mean = np.mean(deg_diff)
        std_diff_left_mean = np.mean(std_diff_left)
        if std_diff_mean <= z and deg_diff_mean <= z:
            entry = (other, std_diff_mean, std_diff_left_mean, deg_diff_mean)
            if verbose == 2:
                entry = *entry, *std_diff
            elif verbose == 3:
                entry = *entry, *std_diff, *m2s
            similar.append(entry)

    if verbose == 0:
        return [x[0] for x in sorted(similar, key=lambda x: x[1])]

    c1 = ['similar_sequence', 'z-diff', 'z-diff-left', 'deg-diff']

    if verbose > 1:
        c1 += [f'std_diff_{i}' for i in range(1, n + 1)]
    if verbose == 3:
        c1 += [f'planar_{i}_mean' for i in range(1, n + 1)]

    sim_df = pd.DataFrame(similar, columns=c1)
    return sim_df.sort_values('z-diff')


def get_similar_df(agg, sequences, n_max=10 ** 6):

    results = []
    sep = pd.DataFrame([['-' * 50, np.nan, np.nan, np.nan]])

    if n_max < len(sequences):
        sequences = sequences[:n_max]

    for counter, sequence in enumerate(sequences):
        print(f'Sequence {counter + 1}/{min(len(sequences), n_max)}')

        result = get_similar_seq(sequence, df=agg, z=2, verbose=1)
        result = result.reset_index().drop('index', axis=1)
        results.append(result)
        sep.columns = results[-1].columns
        results.append(sep)

    results = pd.concat(results)
    results.to_csv('results/similar_sequences.csv')

    return results


def get_mean_diffs(sequences, counts):

    sims = pd.read_csv('results/similar_sequences.csv')

    sequence_counts = {key: value for key, value in zip(sequences, counts)}

    std1 = sims['deg-diff'] / sims['z-diff-left']
    std2 = sims['deg-diff'] / sims['z-diff'] - std1

    sim_seq = list(sims['similar_sequence'])

    seq1, seq1s = sim_seq[0], []
    for idx, seq in enumerate(sim_seq):
        seq1s.append(seq1)
        if '---' in seq and idx + 1 < len(sim_seq):
            seq1 = sim_seq[idx + 1]

    seq1s = np.array(seq1s)

    counts1 = []
    for key in seq1s:
        if '---' in key:
            counts1.append(1)
        else:
            counts1.append(sequence_counts[key])

    counts2 = []
    for key in sims['similar_sequence']:
        if '---' in key:
            counts2.append(1)
        else:
            counts2.append(sequence_counts[key])

    mean_std1 = std1 / np.sqrt(counts1)
    mean_std2 = std2 / np.sqrt(counts2)

    sims['means-diff'] = abs(sims['deg-diff'] / (mean_std1 + mean_std2))

    sims[1:].to_csv('results/similar_sequences_1.csv')

    return sims
