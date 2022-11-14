import pandas as pd


def prep_df(agg, max_bases=20, metric='median'):

    if max_bases is None:
        max_bases = agg.index.str.len().max()

    new_index = []

    for entry in agg.index:
        new_entries = []
        for part in entry.split("|"):
            if "-" in part:
                n_pad = (7 - len(part)) // 2
                if n_pad > 0:
                    new_entries.append("_" * n_pad + part + "_" * n_pad)
                elif n_pad < 0:
                    new_entries.append(part[-n_pad:n_pad])
                else:
                    new_entries.append(part)
            else:
                n_pad = max_bases - len(part)
                c = len(part) // 2
                cut = max_bases // 2
                if n_pad > 0:
                    new_entries.append(part[:c] + "_" * n_pad + part[c:])
                elif n_pad < 0:
                    new_entries.append(part[:cut] + part[-cut:])
                else:
                    new_entries.append(part)

        new_index.append("|".join([x.upper() for x in new_entries]))

    new_df = pd.DataFrame()
    for i in range(len(new_index[0])):
        new_df[f"b{i}"] = [s[i] for s in new_index]

    new_df.index = agg.index

    cols = [c for c in agg.columns if "planar" in c and metric in c]
    for col in cols[:3]:
        new_df[col] = agg[col]

    new_df = pd.get_dummies(new_df)

    return new_df
