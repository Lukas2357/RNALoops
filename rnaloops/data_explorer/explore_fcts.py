import random
import webbrowser

from ..data_getter.getter_fcts import get_df, get_pdf, get_url, init_driver


def show_structure(df=None, idx=None, log=False, pdf=False, web=False):

    df = get_df() if df is None else df
    idx = random.choice(df.index) if idx is None else idx
    structure = df.loc[idx]

    if log:
        print(structure)
    if pdf:
        path = open_data_pdf(idx)
        print(f'Structure pdf file: {path}')
    if web:
        url = open_web_page(idx)
        print(f'Structure web url: {url}')

    return structure


def open_data_pdf(idx):
    path = get_pdf(idx)
    webbrowser.open_new(path)
    return path


def open_web_page(idx):
    url = get_url(idx)
    init_driver(url=url, headless=False, detached=True)
    return url
