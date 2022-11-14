from collections import Counter

import joblib
import numpy as np
from scipy import stats
import pandas as pd

from rnaloops.config.constants import PLANAR_STD_COLS
from rnaloops.config.helper import mypath, save_data
from rnaloops.prepare.data_loader import load_data
from rnaloops.prepare.data_preparer import get_cols


def get_ratio(x, letter, idx=None, kind=''):
    
    if idx is not None:
        x_split = x.split('|')
        if len(x_split) <= idx:
            return np.nan
        else:
            x = x_split[idx]
            
    if kind != '':
        x_split = x.split('|')
        if kind == 'helix':
            x = ''.join([i for i in x_split if '-' in i])
        else:
            x = ''.join([i for i in x_split if '-' not in i])
        
    count = Counter(x).get(letter, 0)
    length = len(x.replace("|", "").replace("-", ""))
    if length == 0:
        return 0
    return count / length


def get_bond(x, idx, n):
    x_split = x.split('|')
    if len(x_split) <= idx:
        return np.nan
    else:
        x = x_split[idx]
    strangs = x.split('-')
    if len(strangs) == 1:
        return ''
    bonds = ['-'.join(f'{a}{b}') for a, b in zip(*strangs)]
    if n >= len(bonds):
        return ''
    return bonds[n]


def p_norm(x):
    if len(x) < 21:
        return np.nan
    try:
        return stats.normaltest(x).pvalue
    except ValueError:
        return np.nan


def get_seq_agg_df(level='L2', cat='parts_seq', way=None):

    df = load_data('_cleaned_' + level)

    if way is not None:
        df = df[df.loop_type == f'0{way}-way']

    cols = [cat]
    cols += [x for x in get_cols(range(1, 15)) 
             if "helix" in x or "strand" in x]

    angles = [x for x in get_cols(range(1, 15)) if
              "euler" in x or "planar" in x]

    col_names = [x + "_mean" for x in angles]
    col_names += [x + "_std" for x in angles]
    col_names += [x + "_median" for x in angles]
    col_names += [x + "_p_norm" for x in angles]

    agg_tuples = [(col, fct) 
                  for fct in [np.mean, np.std, np.median, p_norm] 
                  for col in angles]

    agg = {key: value for key, value in zip(col_names, agg_tuples)}
    agg = agg | {'entries': ('euler_x_1', len)}

    groupby = cols if cat == 'parts_seq' else cat
    new_df = df.groupby(groupby).agg(**agg).reset_index().set_index(cat)

    psc = PLANAR_STD_COLS
    new_df[psc] = new_df[psc].fillna(new_df[psc].mean().mean())
    
    return new_df


def add_features(df, cat='parts_seq'):

    df["seq_length"] = [len(x.replace("|", "").replace("-", ""))
                        for x in df.index]

    for letter in ["A", "U", "C", "G", "u", "c", "g", "a"]:
        
        for kind in ['', 'helix', 'strand']:
            df[f"frac_{kind}_{letter}"] = [get_ratio(x, letter, kind=kind) 
                                           for x in df.index]

        for idx in range(1, 15):
            df[f"strand_{idx}_frac_{letter}"] = [
                get_ratio(x, letter, idx=idx * 2 - 1)
                for x in df.index]

            df = df.copy()

    if cat == 'parts_seq':

        df["loop_type"] = [Counter(x)["-"] for x in df.index]

        for idx in range(1, 15):
            for n in range(3):
                df[f"helix_{idx}_bond_{n}"] = [get_bond(x, 2 * (idx - 1), n)
                                               for x in df.index]

        loop_length = pd.Series(name='loop_length', dtype=np.int16)
        for way in range(3, 15):
            way_df = df[df.loop_type == way]
            strand_cols = [f'strand_{idx}_nts' for idx in range(1, way+1)]
            c_loop_length = way_df[strand_cols].sum(axis=1) + 2*way
            loop_length = pd.concat([loop_length, c_loop_length])
        df['loop_length'] = loop_length

    return df


def save_agg_df(level='L2', cat='parts_seq', way=None):

    df = get_seq_agg_df(level=level, cat=cat, way=way)
    df = add_features(df, cat=cat)

    if way is None:
        filename = f'rnaloops_data_agg_by_{cat}.pkl'
    else:
        filename = f'rnaloops_data_agg_by_{cat}_way{way}.pkl'

    joblib.dump(df, mypath('DATA_PREP', filename))

    return df


def load_agg_df(way=None, cat='parts_seq'):

    if way is None:
        filename = f'rnaloops_data_agg_by_{cat}.pkl'
    else:
        filename = f'rnaloops_data_agg_by_{cat}_way{way}.pkl'

    df = joblib.load(mypath('DATA_PREP', filename))

    for col in df.columns:
        if 'euler' in col:
            split = col.split('_')
            new_name = 'euler_' + split[2] + '_' + split[1] + '_'
            new_name += '_'.join(split[3:])
            df = df.rename(columns={col: new_name})

    if way is not None:
        df = df[[col for col in df.columns
                 if not any(char.isdigit() for char in col) or
                 int(col.split('_')[1]) <= way]]

    return df


def map_chains_and_organisms():

    df = load_data('_cleaned_L2_with_chains')

    label_map = pd.read_csv('rnaloops/data/mappings/labels.csv', index_col=0)
    orga_map = pd.read_csv('rnaloops/data/mappings/organisms.csv', index_col=0)

    df['chain_label'] = df['chain_0_label'].map(
        {key: value
         for key, value in zip(label_map.index, label_map.label)})

    df['main_organism'] = df['chain_0_organism'].map(
        {key: value
         for key, value in zip(orga_map.index, orga_map.main_organism)})

    df['organism'] = df['chain_0_organism'].map(
        {key: value
         for key, value in zip(orga_map.index, orga_map.organism)})

    save_data(df, 'rnaloops_data_cleaned_L2_final', formats=('csv', 'pkl'))
