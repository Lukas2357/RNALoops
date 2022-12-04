"""Module to prepare RNALoops data for use in neural network"""

import random

import pandas as pd


def prepare_data(agg_df, metric="median", way=3):
    """Prepare the aggregated RNALoops data for neural network input

    Parameters
    ----------
    agg_df : pd.DataFrame
        The aggregated RNALoops data. Can be obtained by load_agg_df if save_agg_df was
        called previously, otherwise call save_agg_df to get it (needs a few seconds)
    metric : str, default 'median'
        Whether to use "mean" or "median" angle values as targets
    way : int, default 3
        The loop type to use. Must match the agg_df loop type (call save_agg_df with
        parameter way=X to obtain aggregated data of loop type X).

    Returns
    -------
    pd.DataFrame
        The data with numeric feature and target columns ready for NN input

    """
    return (
        # Expand the sequence string to df columns:
        pd.DataFrame(
            pd.Series(agg_df.index)
            .apply(prepare_sequence)
            .str.split("", expand=True)
            .values
        ).iloc[:, 1:-1]
        # Rename the columns to start with 'b' for base and its index:
        .pipe(
            lambda d: d.rename(
                {c: f"b{idx}" for idx, c in enumerate(d.columns)}, axis=1
            )
        )
        # Using a category with default values ensures that number of columns is always
        # the same when calling get_dummies below. On the other hand, this can lead to
        # all zero columns then. It depends on the neural network architecture if this
        # is desired. Here it is chosen as the more reliable way of feature engineering:
        .apply(
            lambda s: s.astype("category").cat.set_categories(
                ["A", "C", "G", "U", "-", "_"]
            ),
            axis=1,
        )
        .pipe(pd.get_dummies)
        .assign(input_seq=agg_df.index)
        .set_index("input_seq")
        # targets can be any planar angles of given metric up to the given loop type:
        .assign(
            **{
                target: agg_df[target]
                for target in [
                    col for col in agg_df.columns if "planar" in col and metric in col
                ][:way]
            }
        )
    )


def prepare_sequence(sequence, max_bases=20):
    """Prepare a sequence, so it can be transformed to numeric features reliably

    Parameters
    ----------
    sequence : str
        RNA sequence of RNA multiloop
    max_bases : int, default 20
        The maximum number of bases for each part (helix/strand)

    Returns
    -------
    str
        The prepared sequence

    """
    def pad_part(part):
        # If we aggregated by parts_seq, the helices include '-':
        if "-" in part:
            # We pad on both sides with '_' to ensure always same length
            n_pad = (7 - len(part)) // 2
            if n_pad > 0:
                return "_" * n_pad + part + "_" * n_pad
            elif n_pad < 0:
                return part[-n_pad:n_pad]
            else:
                return part
        # Strands:
        else:
            # We pad at the center of the strand to ensure same length always
            n_pad = max_bases - len(part)
            cut1, cut2 = len(part) // 2, max_bases // 2
            if n_pad > 0:
                return part[:cut1] + "_" * n_pad + part[cut1:]
            elif n_pad < 0:
                return part[:cut2] + part[-cut2:]
            else:
                return part

    return "".join(pad_part(part) for part in sequence.upper().split("|"))


def tweak_seq(seq):
    """Tweak a sequence slightly to check how the NN response changes

    Parameters
    ----------
    seq : str
        RNALoops sequence to tweak

    Returns
    -------
    str
        The tweaked sequence

    """
    letters = ["A", "U", "G", "C"]
    split, item, idx = [x for x in seq], '_', 0
    while item in ["_", "-", "|"]:
        idx = random.randint(0, len(split) - 1)
        item = split[idx]

    letters.remove(split[idx])
    split[idx] = random.choice(letters)
    seq = "".join(split)

    parts = seq.split("|")
    flips = 0
    new_seq = []
    for part in parts:
        part = [x for x in part]
        if "-" not in part:
            for i in range(len(part)):
                if part[i] not in ("_", "-", "|"):
                    if random.randint(0, 10) < 1 and flips == -1:
                        part[i] = random.choice("AUGC")
                        flips += 1
        new_seq.append("".join(part))

    return "|".join(new_seq)