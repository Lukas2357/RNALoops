"""Prepare RNALoops df for the use of DataScience tools"""
import os

import joblib
import pandas as pd

from ..config.helper import mypath
from ..prepare.data_loader import load_data
from ..verify.verify_fcts import load_qualities


def prepare_df(convert=True, save=True) -> pd.DataFrame:
    """Load initial df, prepare it, and save it as rnaloops_data_prepared.pkl

    Parameters
    ----------
    convert=True
        Whether to convert string in categorical columns. Saves memory, but
        prevents string operations on entries and thereby reduces flexibility.
    save=True
        Whether to save to rnaloops_data_prepared.pkl

    Returns
    -------
    pd.DataFrame
        The prepared df

    """
    # obtain.getter_fcts.get_df will load initial df from
    # data/rnaloops_data.pkl:
    df = load_data(kind='')

    # Shorten column names for simplicity:
    short = [column
             .replace(' Angle', '')
             .replace('Length ', '')
             .replace('position', '')
             .replace('(', '')
             .replace(')', '')
             .replace('PDB id', '')
             for column in df.columns]
    # And remove spaces to allow . access:
    col_mapper = {col: f"{'_'.join(new_col.lower().split())}"
                  for col, new_col in zip(df.columns, short)}
    # Rename df now:
    df = df.rename(columns=col_mapper)

    # We tweak the df to reduce memory size:
    for column in df.columns:
        if 'nts' in column:
            # A length of -1 will indicate, that the strand does not exist:
            # ATTENTION: There are strands with length 0 (two helices next to
            # each other, so we need -1 and not 0 as null indicator here:
            df.loc[df[column].isnull(), column] = -1
            # Length in nts will never exceed 32767:
            df[column] = df[column].astype('int16')
            # These columns show strand length. Rename accordingly:
            mapper = {column: f"strand_{column.split('_')[-1]}_nts"}
            df = df.rename(columns=mapper)
        if 'bps' in column:
            # A length of -1 will indicate, that the helix does not exist:
            df.loc[df[column].isnull(), column] = -1
            # Length in bps will never exceed 32767:
            df[column] = df[column].astype('int16')
            # These columns show connection helix length. Rename accordingly:
            mapper = {column: f"helix_{column.split('_')[-1]}_bps"}
            df = df.rename(columns=mapper)
        if convert:
            if any(x in column for x in ['start', 'end', 'notation', 'whole']):
                # Use string[pyarrow] datatype to save memory (new in 1.3.0):
                df[column] = df[column].astype('string[pyarrow]')
            if 'sequence' in column or 'home' in column:
                # We have ~1300 different sequences, here category is worth it:
                df[column] = df[column].astype('category')
            if 'euler' in column or 'planar' in column:
                # Angles are <=180° and measured with 3 decimals, float32 is ok
                df[column] = df[column].astype('float32')

    # ATTENTION: Structures with >9 stems have Loop Type = 1
    # This is a bug from pdf parsing. We will get the correct values
    # from checking the number of helix bps columns > 0:
    helix_cols = [col for col in df.columns if 'helix' in col]
    helix_df = df[helix_cols].copy()
    helix_df[helix_df < 0] = 0
    helix_df = helix_df.astype(bool).sum(axis=1)
    df.loop_type = pd.Series([f'{x:02}-way' for x in helix_df]).values
    # Es gibt ein oder zwei <3-way loops, das ist Quatsch, entferne:
    df = df[df.loop_type != '00-way']
    df = df[df.loop_type != '01-way']
    df = df[df.loop_type != '02-way']
    df.loop_type = df.loop_type.astype('category')

    # ATTENTION: Some structures have 0 helices, this is not true, instead
    # the information are just missing. We remove those structures (~2300):
    df = df[df.helix_1_bps > 0]

    # ATTENTION There are about 3000 duplicated rows. We need to remove them.
    df = df.drop_duplicates()

    if save:
        joblib.dump(df, mypath('DATA_PREP', 'rnaloops_data_prepared.pkl'))

    return df


def add_seq_cols(df=None) -> pd.DataFrame:
    """Add additional sequence columns to RNALoops df

    The new columns are:
    parts_seq: Contains the whole sequence, but with | as separator of single
                sequences representing strands and helices
    ordered_seq: As above, but single sequences ordered alphabetically.

    Note: whole_sequence might match, but parts_seq not due to more or less
          bonds within the multiloop.
          ordered_seq might match, but parts_seq not due to wrong ordering
          of strands/helices in RNALoops database.

    Parameters
    ----------
    df : pd.DataFrame, default=None -> loads from 'rnaloops_data_ordered.pkl'
        The ordered RNALoops df.

    Returns
    -------
    pd.DataFrame
        The same df with the two new columns

    """
    if df is None:
        df = load_data("_ordered")

    cols = [f"sequence_{idx}" for idx in range(1, 29)]
    for row in df.iterrows():
        seq = []
        for col in cols:
            part = row[1][col]
            if pd.isnull(part):
                break
            seq.append(part)
        ordered_seq = '|'.join(sorted(seq))
        parts_seq = '|'.join(seq)
        df.at[row[0], 'parts_seq'] = parts_seq
        df.at[row[0], 'ordered_seq'] = ordered_seq

    return df


def remove_unordered(df) -> pd.DataFrame:
    """Remove wrongly ordered structures

    If structures have the same whole sequence and the same ordered parts
    sequence, but different unordered part sequences, it means that the ordering
    of strands and helices in the RNALoops db is wrong. Note that this is not
    due to different starting positions in the multiloops, since the columns
    where ordered before, to always start with the same helix. Instead, it is an
    error in the db, that happens rarely, so we can simply remove these entries.

    Parameters
    ----------
    df : pd.DataFrame
        The RNALoops df after add_seq_cols was applied

    Returns
    -------
    pd.DataFrame
        The same df but with unordered structures removed

    """
    new_df = df.copy()
    n = len(df.ordered_seq.unique())

    for c, o_seq in enumerate(df.ordered_seq.unique()):
        if c % 100 == 0:
            print(f"Finished {c} of {n} structures")
        c_df = df[df.ordered_seq == o_seq]
        counts = c_df.groupby('parts_seq').count()
        seq = counts.sort_values('loop_type').iloc[-1].name
        d_idx = c_df[c_df.parts_seq != seq].index
        new_df = new_df.drop(d_idx)

    return new_df


def save_clean_dfs():
    """Save dfs at different levels of cleaning. See data_loader for details."""

    l0 = add_seq_cols()
    joblib.dump(l0, mypath('DATA_PREP', 'rnaloops_data_cleaned_L0.pkl'))

    l1 = remove_unordered(l0)
    joblib.dump(l1, mypath('DATA_PREP', 'rnaloops_data_cleaned_L0.pkl'))

    qualities = load_qualities()
    good_mls = qualities[0]
    good_mls.update(qualities[1])
    good_mls.update(qualities[2])
    good_mls.update(qualities[-1])
    l3 = l1.loc[good_mls]
    joblib.dump(l3, mypath('DATA_PREP', 'rnaloops_data_cleaned_L0.pkl'))

    good_mls.update(qualities[3])
    l2 = l1.loc[good_mls]
    joblib.dump(l2, mypath('DATA_PREP', 'rnaloops_data_cleaned_L0.pkl'))


def get_cols(idx: int) -> list[str]:
    """Get all columns of df with given idx"""
    if isinstance(idx, int):
        idx = [idx]
    cols = ([f"helix_{i}_bps" for i in idx] +
            [f"strand_{i}_nts" for i in idx] +
            [f"start_{i}" for i in idx] +
            [f"end_{i}" for i in idx] +
            [f"planar_{i}" for i in idx] +
            [f"euler_{a}_{i}" for a in ["x", "y", "z"] for i in idx])
    return cols


def order_df(save=False, load=False):
    """Order RNALoops df columns to have common sequence order

    ATTENTION:
    Function sets each entry via .at, which is extremely inefficient.
    Rewriting the function to improve performance would be very useful.

    The order_df function takes the dataframe of RNA loop sequences and reorders
    the columns so that the same sequences are in the same order.
    It does this by first finding all possible start helix/strand sequences and
    then selecting the longest of them. If there are multiple, the first one
    in alphabetical order is chosen. This is taken as the first sequence.
    If it is a strand, it is chosen second, since we always start with a helix.

    Parameters
    ----------
    save=False
        Whether to save output df to DATA_PREP/rnaloops_data_ordered.pkl
    load=False
        Whether to load input df from DATA_PREP/rnaloops_data_ordered.pkl
        This can be used if data was ordered before, but some entries might be
        reordered due to changes in the ordering process.

    Returns
    -------
        A dataframe with the same number of rows as the original,
        but with columns in new order.

    """
    file_path = mypath('DATA_PREP', 'rnaloops_data_ordered.pkl')
    if os.path.isfile(file_path) and load:
        df = joblib.load(file_path)
    else:
        df = prepare_df(convert=False, save=False)

    new_df = df.copy()
    s_df = df[[f"sequence_{j}" for j in range(1, 29)]]
    h_df = df[[f"helix_{j}_bps" for j in range(1, 15)]]

    for count, idx in enumerate(df.index):

        if count % 100 == 0:
            print(f"Ordered {count}/{len(df.index)} entries")

        start = s_df.loc[idx].dropna().drop_duplicates(keep=False)
        longest = start[start.str.len() == start.str.len().max()]

        try:
            start = int(longest[longest ==
                                longest.max()].index[0].split("_")[-1])
            start = start if start % 2 else start + 1

            s = start // 2 + 1
            n = len(h_df.loc[idx][h_df.loc[idx] > 0])

            if start != 1 and start != 2 * n + 1:
                for i in range(1, n + 1):
                    j = i - s + 1
                    if j < 1:
                        j = n + j
                    for c1, c2 in zip(get_cols(i), get_cols(j)):
                        new_df.at[idx, c2] = df.at[idx, c1]

                for i in range(1, 2 * n + 1):
                    j = i - start + 1
                    if j < 1:
                        j = 2 * n + j
                    sj = f"sequence_{j}"
                    new_entry = df.at[idx, f"sequence_{i}"]
                    new_df.at[idx, sj] = new_entry

        except IndexError:
            pass

    if save:
        joblib.dump(new_df, file_path)

    return new_df
