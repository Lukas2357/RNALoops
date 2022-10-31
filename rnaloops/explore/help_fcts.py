import pandas as pd


def get_frequent_sequences(df, min_n=100, category="parts_seq"):

    df_new = df.copy()
    df_new['id'] = df_new.index
    grouped = df_new.groupby(by=category).count().id
    df_new = df_new.set_index(category)
    mask = grouped.index[grouped >= min_n]
    df_new = df_new.loc[mask]

    return df_new


def get_sequences_ordered_by_nstructures(df, min_n=1, n=None,
                                         cat='whole_sequence'):

    frequent = get_frequent_sequences(df, min_n=min_n)
    grouped = frequent.groupby(by=cat).count()
    sort = grouped.sort_values('loop_type', ascending=False)

    sequences = sort.index.values
    counts = sort.loop_type.values

    if n is not None:
        sequences = sequences[:n]
        counts = counts[:n]

    return sequences, counts


def get_sequence_mean_angles(df, category="parts_seq"):
    columns = [x for x in df.columns if 'planar_' in x]
    df_columns = columns if category == 'index' else columns + [category]

    df = df[df_columns]

    if category == 'index':
        seq_mean = df.groupby(level=0).mean()
    else:
        seq_mean = df.groupby(by=category).mean()

    seq_mean.columns = columns

    if category == 'index':
        seq_std = df.groupby(level=0).std()
    else:
        seq_std = df.groupby(by=category).std()

    seq_std.columns = [c + '_std' for c in columns]

    result = pd.concat([seq_mean, seq_std], axis=1)

    return result
