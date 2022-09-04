import asyncio
from multiprocessing import Pool
from timeit import default_timer
from random import shuffle

import joblib
import textract
from pyppeteer import launch, errors
from selenium.common import WebDriverException

from helper import *


async def get_pdfs_async():
    count, step, page = 0, 5, None
    range_start, range_stop = 100_050, 101_000

    files = os.listdir("/home/lukas/Documents/RNALoops/data")
    files = [file for file in files if 'pdf' in file]
    prev_indices = [int(file[-10:-4]) for file in files]

    start = default_timer()

    for start_idx in range(range_start, range_stop, step):

        urls = [get_url(idx) for idx in range(start_idx, start_idx + step)
                if idx not in prev_indices]

        if not urls:
            continue

        browser = await launch(headless=False)

        for url in urls:
            page = await browser.newPage()
            page.setDefaultNavigationTimeout(5000)
            try:
                await page.goto(url)
            except errors.TimeoutError:
                await browser.close()
                browser = await launch(headless=False)
                page = await browser.newPage()
                page.setDefaultNavigationTimeout(5000)
                try:
                    await page.goto(url)
                except errors.TimeoutError:
                    print(f"Could not open {url}")

            button = await page.J("[aria-label='Download summary file']")
            try:
                await button.click()
                count += 1
                print(f'Downloaded {url} (file {count})')
            except AttributeError:
                pass

        await page.waitFor(2000)
        await browser.close()

        el = default_timer() - start
        print(f'Scanned {start_idx}-{start_idx + step} after {el / 60:.1f} min')


def pdfs_async():
    asyncio.get_event_loop().run_until_complete(get_pdfs_async())


def get_pdfs_seq(indices, verbose=False):

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

        driver = webdriver.Chrome(options=get_chrome_options(headless=True))

        for url in urls:

            try:
                driver.get(url)
            except WebDriverException as e:
                print(e)

            for element in driver.find_elements(By.CLASS_NAME,
                                                'MuiButtonBase-root'):

                attribute = element.get_attribute('aria-label')

                if attribute == 'Download summary file':
                    element.click()
                    count += 1
                    if verbose:
                        print(f'Downloaded {url} (file {count})')

        el = default_timer() - start
        print(f'Worker scanned {indices_checked} urls after {el / 60:.1f} min')


def pdf_getter(n_pools=8, min_idx=100_000, max_idx=200_000):

    urls_per_pool = int((max_idx - min_idx) / n_pools)

    files = os.listdir("/home/lukas/Documents/RNALoops/data")
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
        pool.starmap(get_pdfs_seq, params)
        pool.close()


def get_pdfs(n_pools, min_idx, max_idx):

    dir_path = "/home/lukas/Documents/RNALoops/data"
    n_files = len(os.listdir(dir_path))

    n_urls = max_idx - min_idx

    start = default_timer()
    pdf_getter(n_pools=n_pools, min_idx=min_idx, max_idx=max_idx)
    el = default_timer() - start

    n_files = len(os.listdir(dir_path)) - n_files

    print(f"{n_pools} Pools: {el / 60:.2f} min for {max_idx - min_idx} urls")
    ms = el*1000
    print(f" -> {ms / n_urls:.2f} ms/url | {ms / n_files:.2f} ms/download")


def get_content(n=0):
    files = [x for x in os.listdir() if 'pdf' in x]

    if n > 0:
        files = files[:n]

    for file in files:
        try:
            content = (textract.process(file).decode('utf-8').split("\n"))
            result = get_data_dict(content)
            idx = file[-13:-3]
            joblib.dump(result, str(idx))
        except UnicodeDecodeError:
            print(f'Could not decode file {file}')
