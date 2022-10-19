import os.path
import random
import webbrowser
from time import sleep
from typing import Iterable

from IPython.display import SVG, display
from pandas.api.types import is_numeric_dtype

from ..data_getter.getter_fcts import *


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


def get_prepared_df():
    # make sure that 'rnaloops/data/rnaloops_data_prepared.pkl' exist
    # it is created from rnaloops_data.pkl by _prepare_df from above
    path = os.path.join('rnaloops', 'data', 'rnaloops_data_prepared.pkl')
    # Load the data in a DataFrame in here:
    df = joblib.load(path)
    # Add column information print method:
    df.columns_info = lambda: print(COLUMN_INFO)
    set_attrs(df)
    for way in range(3, 15):
        setattr(df, f'way{way}', get_loop_types(df, way))
        set_attrs(getattr(df, f'way{way}'))
    df.upto8 = get_loop_types(df, max_way=8)
    set_attrs(df.upto8)
    
    return df


def set_attrs(df):
    
    df.angles = df[[col for col in df.columns 
                    if 'euler' in col or 'planar' in col]]
    df.euler = df[[col for col in df.columns if 'euler' in col]]
    df.euler_x = df[[col for col in df.columns if 'euler_x' in col]]
    df.euler_y = df[[col for col in df.columns if 'euler_y' in col]]
    df.euler_z = df[[col for col in df.columns if 'euler_z' in col]]
    df.planar = df[[col for col in df.columns if 'planar' in col]]
    df.strand = df[[col for col in df.columns if 'strand' in col]]
    df.helix = df[[col for col in df.columns if 'helix' in col]]
    
    
def get_loop_types(df=None, way=None, numeric=False, 
                   max_way=None) -> pd.DataFrame:
    """Get relevant rows and columns for given loop_types from full df
    
    Args:
        df (pd.DataFrame): df containing all RNALoops data in prepared manner.
                           Default None -> Loaded from file in that case
        way (int): The loop type to extract, ignored if max_way is set
        numeric (bool): Wheather to return only numeric columns
        max_way (int): This loop type and lower ones are extracted
    
    Returns:
        pd.DataFrame: df containing only rows and columns of given loop_types
    
    """
    
    if df is None:
        df = get_prepared_df()
        
    if way is None and max_way is None:
        raise AttributeError('Must either specify way or max_way.')
    if max_way is None:
        cols = [f'{way:02}-way']
    else:
        if way is not None:
            print('Both way and max_way set... ignoring the former.')
        cols = [f'{w:02}-way' for w in range(3, max_way+1)]
    
    way_df = df[df.loop_type.isin(cols)]
    
    numerics = ['int8', 'int16', 'int64', 'float32', 'float64']
    
    drop_cols = [c for c in way_df.columns if is_numeric_dtype(way_df[c])
                 and all(y<0 for y in way_df[c])]
    
    way_df = way_df.drop(drop_cols, axis=1)
    way_df = way_df.dropna(axis=1, how='all')
    
    if numeric:
        way_df = way_df.select_dtypes(include=numerics)
        
    rm_cols = [f'{w:02}-way' for w in range(3, 15) if f'{w:02}-way' not in cols] 
    way_df.loop_type = way_df.loop_type.cat.remove_categories(rm_cols)
    
    return way_df


def show_structure(df=None, indices=None, pdf=False, web=False, svg=False,
                   keep_files=False, n=1):

    df = get_prepared_df() if df is None else df
    if indices is None:
        indices = random.sample(population=list(df.index), k=n)

    if not isinstance(indices, Iterable):
        indices = [indices]

    structures = pd.DataFrame(columns=df.columns)

    for idx in indices:

        structures.loc[idx] = df.loc[idx]

        if pdf:
            path = open_data_pdf(idx, keep_files=keep_files)
            print(f'Structure pdf file: {path}')
        if web:
            url = open_web_page(idx)
            print(f'Structure web url: {url}')
        if svg:
            path = open_svg_image(idx, keep_files=keep_files)
            print(f'Structure svg file: {path}')

    return structures


def open_data_pdf(idx, keep_files=False):
    try:
        path = get_pdf(idx)
        webbrowser.open_new(path)
    except TypeError:
        get_structure([idx])
        path = [f
                for f in os.listdir() if 'pdf' in f and 'way_junction' in f][0]
        webbrowser.open_new(path)
        if not keep_files:
            sleep(3)
            os.remove(path)
            print(f'File removed to keep cwd clean, keep_files=True to keep')
    return path


def open_web_page(idx):
    url = get_url(idx)
    init_driver(url=url, headless=False, detached=True)
    return url


def open_svg_image(idx, keep_files=False):

    get_structure([idx], pdf=False, svg=True)
    path = [file for file in os.listdir() if 'svg' in file][0]

    display(SVG(path))
    if not keep_files:
        os.remove(path)
        print(f'File removed to keep cwd clean, keep_svg=True to keep')

    if os.path.isfile(path[:-3] + 'cif'):
        os.remove(path[:-3] + 'cif')

    return path


def compare_df_memory(df1, df2):
    # Index dtype is not returned from df.dtypes, so set it here:
    index_dtype = pd.DataFrame([['Index', 'Index', 'int64', 'int64']])
    # Get the other dtypes and put together:
    df = pd.DataFrame(zip(df1.columns, df2.columns, df1.dtypes, df2.dtypes))
    df = pd.concat([index_dtype, df])
    # Get memory usages, difference, and put all together:
    df1_ram = df1.memory_usage(deep=True)
    df2_ram = df2.memory_usage(deep=True)
    ram_cols = ['raw_MB', 'prep_MB', 'diff_MB']
    df[ram_cols[0]] = df1_ram.values / 10 ** 6
    df[ram_cols[1]] = df2_ram.values / 10 ** 6
    df[ram_cols[2]] = (df1_ram.values - df2_ram.values) / 10 ** 6
    # Add the sum of memory columns:
    df.loc['Summe'] = [''] * 4 + list(map(sum, [df[x] for x in ram_cols]))
    df.columns = ['name_1', 'name_2', 'dtype_1', 'dtype_2'] + ram_cols

    return df


def compare_raw_and_prepared(df_raw, df_prepared, ):
    df = compare_df_memory(df_raw, df_prepared)

    new_columns = ['raw_name', 'new_name', 'raw_dtype', 'new_dtype']
    df.columns = new_columns + list(df.columns)[4:]

    for row in df.iterrows():
        post = '' if row[1].raw_name == '' else row[1].raw_name.split()[-1]
        if post != '' and len(post) < 3 and int(post) != 1:
            df = df.drop(row[0])

    return df

