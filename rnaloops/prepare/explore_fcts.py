"""Some functions for explorativ data analysis of RNALoops data"""

import os.path
import random
import webbrowser
from time import sleep

from IPython.display import SVG, display

from .data_loader import load_data
from ..obtain.getter_fcts import *
from ..obtain.helper_fcts import init_driver


def show_structure(indices=None, df=None, pdf=False, web=True, svg=False,
                   keep_files=False, n=1, scale=1, return_df=False,
                   max_n=10):
    """Show a structure representing a multiloop in RNALoops

    Parameters
    ----------
    indices : int or list or str None
        Index or indices of structure to show, if None opens random structure,
        if string tries to open all structures from that pdb_id.
    df : pd.DataFrame
        df to get data from, loaded from disk if not provided.
    pdf : bool
        Whether to open the pdf file corresponding to the structure
    web : bool
        Whether to open the web entry corresponding to the structure
    svg : bool
        Whether to download and open svg image corresponding to the structure
    keep_files : bool
        Whether to keep downloaded files
    n : int
        Number of structures to open if indices is None
    scale : int
        Scaling of webbrowser opened.
    return_df : bool
        Whether to return the rows of RNALoops df corresponding to structure.
    max_n : int
        Max number of structures to show

    Returns
    -------
    pd.DataFrame
        rows of RNALoops df corresponding to structure (if selected)

    """
    if return_df or indices is None or isinstance(indices, str):

        df = load_data(kind='_cleaned_L2') if df is None else df

        if indices is None:
            indices = random.sample(population=list(df.index), k=n)

        if isinstance(indices, str):
            if '|' in indices:
                indices = df[df.parts_seq == indices].index
            else:
                indices = df[df.home_structure == indices].index

    if not isinstance(indices, Iterable):
        indices = [indices]

    if len(indices) > max_n:
        indices = random.choices(indices, k=max_n)

    structures = pd.DataFrame(columns=df.columns) if return_df else None

    if web:
        urls = open_web_page(indices, scale=scale)
        for url in urls:
            print(f'Structure web url: {url}')

    for idx in indices:

        if return_df:
            structures.loc[idx] = df.loc[idx]

        if pdf:
            path = open_data_pdf(idx, keep_files=keep_files)
            print(f'Structure pdf file: {path}')
        if svg:
            path = open_svg_image(idx, keep_files=keep_files)
            print(f'Structure svg file: {path}')

    if return_df:
        return structures


def open_data_pdf(idx, keep_files=False):
    """Open a pdf file corresponding to structure of idx"""
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


def open_web_page(indices, scale=1):
    """Open webpages corresponding to structure of indices"""
    urls = [get_url(idx) for idx in indices]
    init_driver(urls=urls, headless=False, detached=True, scale=scale)
    return urls


def open_svg_image(idx, keep_files=False):
    """Download and open svg image corresponding to structure of idx"""
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
    """Compare the memory use of two dataframes"""
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


def compare_raw_and_prepared(df_raw, df_prepared):
    """Compare the memory use of raw and prepared RNALoops df"""
    df = compare_df_memory(df_raw, df_prepared)

    new_columns = ['raw_name', 'new_name', 'raw_dtype', 'new_dtype']
    df.columns = new_columns + list(df.columns)[4:]

    for row in df.iterrows():
        post = '' if row[1].raw_name == '' else row[1].raw_name.split()[-1]
        if post != '' and len(post) < 3 and int(post) != 1:
            df = df.drop(row[0])

    return df


def explore_duplicate_sequences(df):
    """Print multiloops with same whole_sequence, but different parts"""
    cols = [f"sequence_{idx}" for idx in range(1, 29)] + ["whole_sequence"]
    counts = pd.DataFrame(df[cols].value_counts())
    grouped = counts.groupby(level=0).count()
    duplicates = 0
    
    for c, entry in enumerate(grouped[grouped[0] > 1].index):

        if (all(s in counts.loc[entry].index.values[1] 
                for s in counts.loc[entry].index.values[0])):
            
            print("-----", entry)
            duplicates += sum(counts.loc[entry].iloc[1:, 0].values)
            print(*counts.loc[entry].index.values, sep="\n")

    return duplicates
