"""Functions to obtain data from RNALoops web-database"""

import math
from multiprocessing import Pool
from pathlib import Path
from timeit import default_timer
from random import shuffle

import joblib
import textract
from selenium.common import WebDriverException
from selenium.webdriver.common.by import By

from .helper_fcts import *
from .helper_fcts import init_driver
from ..config.helper import mypath


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

        driver.close()
        driver.quit()

        el = default_timer() - start
        if len(indices) > 100:
            print(f'Worker scanned {indices_checked} urls '
                  f'after {el / 60:.1f} min')


def pdf_getter(n_pools=8, min_idx=100_000, max_idx=200_000):
    urls_per_pool = int((max_idx - min_idx) / n_pools)

    files = os.listdir(mypath("PDF_FILES"))
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

    n_files = len(os.listdir(mypath("PDF_FILES")))

    n_urls = max_idx - min_idx

    start = default_timer()
    pdf_getter(n_pools=n_pools, min_idx=min_idx, max_idx=max_idx)
    el = default_timer() - start

    n_files = len(os.listdir(mypath("PDF_FILES"))) - n_files

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

    parent = mypath("PDF_FILES") if kind == 'pdf' else mypath("PLAIN_FILES")

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


def get_pdf(idx):

    lower_idx_1000 = math.floor(idx / 1000) * 1000
    idx_range = f'{lower_idx_1000}-{lower_idx_1000 + 1000}'
    folder = mypath('PDF_FILES', idx_range)

    for file in os.listdir(folder):
        if not os.path.isdir(file):
            if idx == int(file.split("-")[-1][:-4]):
                return os.path.join(folder, file)


def get_url(idx):
    return f'https://rnaloops.cs.put.poznan.pl/search/details/{idx}'
