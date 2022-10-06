import math
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer
from random import shuffle

import joblib
import textract
from selenium.common import WebDriverException
from selenium import webdriver
from selenium.webdriver.common.by import By

from .helper_fcts import *


def init_driver(headless=True, detached=False, url=None):
    chrome_options = get_chrome_options(headless=headless, detached=detached)
    driver = webdriver.Chrome(options=chrome_options)
    if url is not None:
        driver.get(url)
    return driver


def get_structure(indices, verbose=False, pdf=True, svg=False):

    indices_checked, count, step = 0, 0, 100
    all_urls = [get_url(idx) for idx in indices]
    start = default_timer()

    while indices_checked < len(all_urls):

        if indices_checked + step < len(all_urls):
            urls = all_urls[indices_checked:indices_checked + step]
            indices_checked += step
        else:
            urls = all_urls[indices_checked:]
            indices_checked += step

        if not urls:
            continue

        driver = init_driver()

        for url in urls:

            try:
                driver.get(url)
            except WebDriverException as e:
                print(e)

            for element in driver.find_elements(By.CLASS_NAME,
                                                'MuiButtonBase-root'):

                attribute = element.get_attribute('aria-label')

                old_count = count
                if attribute == 'Download summary file' and pdf:
                    element.click()
                    count += 1
                    if verbose:
                        print(f'Downloaded {url} pdf (file {count})')

                if (attribute == 'download'
                        and svg):
                    element.click()
                    if count == old_count:
                        count += 1
                    if verbose:
                        print(f'Downloaded {url} svg (file {count})')

        el = default_timer() - start
        if len(indices) > 100:
            print(f'Worker scanned {indices_checked} urls '
                  f'after {el / 60:.1f} min')


def pdf_getter(n_pools=8, min_idx=100_000, max_idx=200_000):
    urls_per_pool = int((max_idx - min_idx) / n_pools)

    files = os.listdir(os.path.join("rnaloops", "data_getter"))
    files = [file for file in files if 'pdf' in file]
    prev_indices = [int(file.split('-')[-1][:-4]) for file in files]
    prev_indices_dict = {idx: False for idx in range(min_idx, max_idx)}
    for prev_idx in prev_indices:
        prev_indices_dict[prev_idx] = True
    joblib.dump(prev_indices_dict, 'prev_indices_dict')

    print(f"Found {len(prev_indices)} previous indices")

    indices = [i for i in range(min_idx, max_idx) if not prev_indices_dict[i]]
    shuffle(indices)

    print(f"Shuffled remaining {len(indices)} indices")

    params = [[indices[urls_per_pool * i:urls_per_pool * (i + 1)]]
              for i in range(n_pools)]

    print(f"Going to scan {len(indices)} urls...")

    with Pool(n_pools) as pool:
        pool.starmap(get_structure, params)
        pool.close()


def get_pdfs(n_pools, min_idx, max_idx):

    dir_path = os.path.join("rnaloops", "data_getter")
    n_files = len(os.listdir(dir_path))

    n_urls = max_idx - min_idx

    start = default_timer()
    pdf_getter(n_pools=n_pools, min_idx=min_idx, max_idx=max_idx)
    el = default_timer() - start

    n_files = len(os.listdir(dir_path)) - n_files

    print(f"{n_pools} Pools: {el / 60:.2f} min for {max_idx - min_idx} urls")
    ms = el * 1000
    print(f" -> {ms / n_urls:.2f} ms/url | {ms / n_files:.2f} ms/download")


def get_content(n=0):
    files = [x for x in os.listdir() if 'pdf' in x]
    contents, indices = [], []

    if n > 0:
        files = files[:n]

    for file in files:
        try:
            content = (textract.process(file).decode('utf-8').split("\n"))
            content = get_data_dict(content)
            contents.append(content)

            idx = int(file.split('-')[-1][:-4])
            indices.append(idx)

            joblib.dump(content, str(idx))

        except UnicodeDecodeError:
            print(f'Could not decode file {file}')

    return contents, indices


def batch_files(kind='pdf'):
    parent = 'data_pdfs' if kind == 'pdf' else 'data_files'
    Path(parent).mkdir(parents=True, exist_ok=True)

    for start in range(0, 300_000, 1000):

        stop = start + 1000

        subfolder = f"{start}-{stop}"
        folder = os.path.join(parent, subfolder)
        Path(folder).mkdir(parents=True, exist_ok=True)

        for file in os.listdir(parent):

            if not os.path.isdir(os.path.join(parent, file)):

                if kind == 'pdf':
                    idx = int(file.split("-")[-1][:-4])
                else:
                    idx = int(file)

                if start <= idx < stop:
                    os.rename(os.path.join(parent, file),
                              os.path.join(folder, file))

        if len(os.listdir(folder)) == 0:
            os.rmdir(folder)


def get_df():
    # make sure that data is in 'rnaloops/data/rnaloops_data.pkl'!
    path = os.path.join('rnaloops', 'data', 'rnaloops_data.pkl')
    # Load the data in a DataFrame in here:
    df = joblib.load(path)
    return df


def get_pdf(idx):

    lower_idx_1000 = math.floor(idx / 1000) * 1000
    idx_range = f'{lower_idx_1000}-{lower_idx_1000 + 1000}'
    folder = os.path.join('rnaloops', 'data_getter', 'data_pdfs', idx_range)

    for file in os.listdir(folder):
        if not os.path.isdir(file):
            if idx == int(file.split("-")[-1][:-4]):
                return os.path.join(folder, file)


def get_url(idx):
    return f'https://rnaloops.cs.put.poznan.pl/search/details/{idx}'


def prepare_df():
    # data_getter.getter_fcts.get_df will load initial df from
    # data/rnaloops_data.pkl:
    df = get_df()

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
            # Length in bps will never exceed 127:
            df[column] = df[column].astype('int8')
            # These columns show connection helix length. Rename accordingly:
            mapper = {column: f"helix_{column.split('_')[-1]}_bps"}
            df = df.rename(columns=mapper)
        if any(x in column for x in ['start', 'end', 'notation', 'whole']):
            # Use string[pyarrow] datatype to save memory (new in pandas 1.3.0):
            df[column] = df[column].astype('string[pyarrow]')
        if 'sequence' in column or 'home' in column:
            # We have ~1300 different sequences, here category is worth it:
            df[column] = df[column].astype('category')
        if 'euler' in column or 'planar' in column:
            # Angles are <=180° and measured with 3 decimals, so float32 is fine
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

    joblib.dump(df, 'rnaloops/data/rnaloops_data_prepared.pkl')

    return df
