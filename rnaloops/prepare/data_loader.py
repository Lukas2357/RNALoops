"""Load RNALoops data from prepared dfs as .pkl files"""

import warnings

import joblib
import pandas as pd
from Bio import SeqIO
from pandas.core.dtypes.common import is_numeric_dtype

from ..config.helper import mypath

COLUMN_INFO = """
Information on RNALoops DataFrame columns: \n
- loop_type (int8, NOT NULL): Any of 3-14 = number of structure stems \n
- home_structure (category, NOT NULL): Where the structure is found in PDB \n
- db_notation (str_arrow, NOT NULL): bracket notation, explanation see e.g. 
https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/rna_structure_notations.html#dot-bracketnotation \n
- whole_sequence (str_arrow, NOT NULL): The base sequence of the structure \n
- helix_{1...14}_bps (int8): Length of helices 1...14 in base pairs, -1 for non existent, at least 1, 2 and 3 must not be -1, to have a multiloop \n
- strand_{1...14}_nts (int16): Length of strands 1...14 in units, again -1 for non existent, can be 0 if two helices are next to each other \n
- start_{1...14} (str_arrow): The start position of strand 1...14 \n
- end_{1...14} (str_arrow): The end position of strand 1...14 \n
- euler_{x|y|z}_{1...14} (float32, 0<=...<=180): Euler angles in ° of strand 1...14 in x, y and z \n
- planar_{x|y|z}_{1...14} (float32, 0<=...<=180): Planar angle in °, for Details on angles read https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9438955/
"""


def load_data(kind='_cleaned_L3'):
    """Load prepared dataframes from RNALoops data

    By the kind argument, the level of preparation can be chosen (see below).
    The df will be equipped with attributes for easy access of column groups.

    The kind of df to load can be any of:
        ''              -> The df as obtained from all downloaded RNALoops pdfs
                           containing all information.
        '_prepared'     -> Above after removing duplicates, renaming cols, ...
                            See data_preparer.prepare_df for details.
        '_cleaned_L0'   -> Above with ordered_seq, parts_seq columns added
                           See data_preparer.add_seq_cols for details.
        '_cleaned_L1'   -> Above with wrong ordered sequences removed.
                           See data_preparer.remove_unordered for details.
        '_cleaned_L2'   -> Above with quality 4 and 5 structures removed.
                           See verify.mmcif_parser.check_structure for details.
        '_cleaned_L3'   -> Above with quality 3 structures removed additionally.
                           See verify.mmcif_parser.check_structure for details.

    Parameters
    ----------
    kind : str, default '_cleaned_L3'
        kind of data to load

    Returns
    -------
    pd.DataFrame
        the loaded and modified df

    """

    # Ignore Pandas column creation warning, should be checked in the future:
    warnings.filterwarnings('ignore')

    try:
        df = joblib.load(mypath("DATA_PREP", f'rnaloops_data{kind}.pkl'))
    except OSError:
        df = pd.read_csv(mypath("DATA_PREP", f'rnaloops_data{kind}.csv'),
                         index_col=0)

    # Add column information print method:
    df.columns_info = lambda: print(COLUMN_INFO)
    # Add attributes to easily access parts of the df:
    set_attrs(df)
    for way in range(3, 15):
        setattr(df, f'way{way}', get_loop_types(df, way))
        set_attrs(getattr(df, f'way{way}'))
    df.upto8 = get_loop_types(df, max_way=8)
    set_attrs(df.upto8)

    warnings.filterwarnings('default')

    return df


def set_attrs(df):
    """
    The set_attrs function sets the attributes of a dataframe.
    It takes in a dataframe and returns the same dataframe with new attributes:
    angles, euler, euler_x, euler_y, and euler_z, strand, helix.

    Parameters
    ----------
    df : pd.DataFrame
        Pass the dataframe to the function

    """
    warnings.filterwarnings('ignore')

    df.angles = df[[col for col in df.columns
                    if 'euler' in col or 'planar' in col]]
    df.euler = df[[col for col in df.columns if 'euler' in col]]
    df.euler_x = df[[col for col in df.columns if 'euler_x' in col]]
    df.euler_y = df[[col for col in df.columns if 'euler_y' in col]]
    df.euler_z = df[[col for col in df.columns if 'euler_z' in col]]
    df.planar = df[[col for col in df.columns if 'planar' in col]]
    df.planar_mean = df[[col for col in df.columns if 'planar' in col
                         and 'mean' in col]]
    df.planar_std = df[[col for col in df.columns if 'planar' in col
                        and 'std' in col]]
    df.strand = df[[col for col in df.columns if 'strand' in col]]
    df.helix = df[[col for col in df.columns if 'helix' in col]]


def get_loop_types(df=None, way=None, numeric=False,
                   max_way=None) -> pd.DataFrame:
    """Get relevant rows and columns for given loop_types from full df
    
    Parameters
    ----------
    df (pd.DataFrame): Default None -> Loaded from file in that case
        df containing all RNALoops data in prepared manner.
    way (int):
        The loop type to extract, ignored if max_way is set
    numeric (bool):
        Weather to return only numeric columns
    max_way (int):
        This loop type and lower ones are extracted
    
    Parameters
    ----------
    pd.DataFrame:
        df containing only rows and columns of given loop_types
    
    """
    if df is None:
        df = load_data('_prepared')

    if way is None and max_way is None:
        raise AttributeError('Must either specify way or max_way.')
    if max_way is None:
        cols = [f'{way:02}-way']
    else:
        if way is not None:
            print('Both way and max_way set... ignoring the former.')
        cols = [f'{w:02}-way' for w in range(3, max_way + 1)]

    way_df = df[df.loop_type.isin(cols)]

    numerics = ['int8', 'int16', 'int64', 'float32', 'float64']

    drop_cols = [c for c in way_df.columns if is_numeric_dtype(way_df[c])
                 and all(y < 0 for y in way_df[c])]

    way_df = way_df.drop(drop_cols, axis=1)
    way_df = way_df.dropna(axis=1, how='all')

    if numeric:
        way_df = way_df.select_dtypes(include=numerics)

    rm_cols = [f'{w:02}-way' for w in range(3, 15) if f'{w:02}-way' not in cols]
    try:
        way_df.loop_type = way_df.loop_type.cat.remove_categories(rm_cols)
    except AttributeError:
        pass

    return way_df


def load_sequence_map() -> pd.DataFrame:
    """Load chain sequences from fasta file and create dict to map them to ints

    Returns
    -------
    pd.DataFrame
        key=chain sequence, value=id -> a mapper for unique chains in RNALoops

    """
    sequences = {'label': [], 'organism': [], 'seq': []}
    index = []

    for record in SeqIO.parse(mypath('DATA', "sequences.fasta"), "fasta"):
        description = record.description.split("|")
        label = description[2].lower()
        organism = ' '.join(description[3].split(' ')[:-1])
        organism = organism.lower()
        pdb_id = description[0].split("_")[0]
        if ',' in description[1]:
            chain_ids = description[1].split(',')
        else:
            chain_ids = [description[1]]

        for chain_id in chain_ids:
            chain_id = chain_id.split(" ")[-1].replace("]", "")
            seq = str(record.seq)
            sequences['label'].append(label)
            sequences['organism'].append(organism)
            sequences['seq'].append(seq)
            index.append(pdb_id + "-" + chain_id)

    for key, value in sequences.items():
        sequences[key] = pd.Series(value, name=key)

    unique_seq = sequences['seq'].unique()
    mapper = {x: y for x, y in zip(unique_seq, range(len(unique_seq)))}
    seq_mapped = sequences['seq'].map(mapper)
    values = [seq_mapped] + [value for value in sequences.values()]
    cols = ['idx'] + list(sequences.keys())

    seq_map = pd.DataFrame(columns=cols, index=index)
    for idx, value in enumerate(values):
        seq_map[cols[idx]] = value.values

    return seq_map
