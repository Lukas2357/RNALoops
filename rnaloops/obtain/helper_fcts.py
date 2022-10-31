"""Helper functions to obtain data from RNALoops web-database"""

import os
from typing import Iterable

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from ..config.helper import mypath


def get_chrome_options(headless=True, detached=False, scale=1) -> Options:
    """Get options for chrome driver specified by attributes

    Parameters
    ----------
    headless : bool, default True
        Whether to run chrome in headless mode
    detached : bool, default False
        Whether to run chrome in detached mode
    scale : int, default 1
        Initial screen size of 1920x1080 will be scaled by this value

    Returns
    -------
    selenium.webdriver.chrome.options.Options
        The options for the driver to initialize

    """
    os.environ["PATH"] += os.pathsep + mypath('ROOT')

    chrome_path = mypath('CHROME_PATH', 'google-chrome')
    window_size = f"{int(1920 * scale)}, {int(1080 * scale)}"

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % window_size)
    chrome_options.add_experimental_option("detach", detached)
    chrome_options.binary_location = chrome_path

    return chrome_options


def get_data_dict(content: list[str]) -> dict:
    """Get data dictionary from data in standardized pdf file

    Takes a list of strings as input and returns a dictionary.
    Iterates through the list, splitting each string on ':', and then adds the
    key-value pairs to a dictionary. If there are multiple colons in any string,
    it splits them into separate keys with their values being empty strings.
    If there is more than one value for that key, it will split those values on
    '-' and add them to another dictionary.

    Parameters
    ----------
        content : list
            The list of lines in the pdf file as obtained by textract

    Returns
    -------
        A dictionary of the data in the file

    """
    result = {}

    for line in content:
        sep = line.find(':')
        if sep > -1:
            key = line[:sep]
            value = line[sep + 2:]
            # If there are multiple : in a line we need to split again:
            if ':' in value:
                new_keys = value.split(':')
                add_to_dict(result, new_keys[0], '-')  # default is '-'
                value = '-'
                # If space is parsed as new_key[0], we skip it:
                if len(new_keys) > 2:
                    result = add_to_dict(result, new_keys[1], new_keys[2])
            result = add_to_dict(result, key, value)

    # Parsing the sequence needs a little tweaking:
    result['Whole Sequence'] = content[content.index('Sequence:') - 1]
    result['Sequence'] = result['Sequence'][1:]

    # If value lists have only one element, we unpack it:
    for key, value in result.items():
        if len(value) == 1:
            result[key] = value[0]

    return result


def init_content_df() -> pd.DataFrame:
    """Initialize a df for the pdf file contents

    Creates a pandas dataframe with the following columns:
    Loop type, Home structure (PDB id), DB notation, Whole Sequence.

    Returns
    -------
        The initialized df

    """
    single_columns = ['Loop type', 'Home structure (PDB id)', 'DB notation',
                      'Whole Sequence']

    # Appear 14 times, for each strand/helix:
    multiple_columns = ['Length (bps)',
                        'Second Strand',
                        'First Strand',
                        'Length (nts)',
                        'End position',
                        'Euler Angle X',
                        'Start position',
                        'Euler Angle Y',
                        'Euler Angle Z',
                        'Planar Angle']

    # Appear 28 times, for each strand+helix
    twice_multiple_columns = ['Sequence']

    columns = single_columns

    for column in multiple_columns:
        columns += [column + f' {idx}' for idx in range(1, 15)]

    for column in twice_multiple_columns:
        columns += [column + f' {idx}' for idx in range(1, 29)]

    df = pd.DataFrame(columns=columns)

    return df


def insert_content_df(df, content: dict, idx: int) -> pd.DataFrame:
    """Insert content from dict to dataframe

    The insert_content_df function takes a df and a dictionary of content,
    and inserts the content into the df at index idx. If there are more than one
    entry for any given key in the dict, then multiple rows will be inserted.
    The function returns an updated version of df.

    Parameters
    ----------
        df : pd.DataFrame
            Initiale df
        content : dict
            Store the content of the pdf files
        idx : int
            Insert the content into the correct row of the dataframe

    Returns
    -------
        The dataframe with the new content inserted

    """
    flat_content = {}

    for key, value in content.items():
        if isinstance(value, list):
            for i, entry in enumerate(value):
                flat_content[key + f' {i + 1}'] = entry
        else:
            flat_content[key] = value

    for column in df.columns:
        if column not in flat_content.keys():
            flat_content[column] = None

    df.loc[idx, flat_content.keys()] = list(flat_content.values())

    return df


def add_to_dict(result: dict, key: str, value: str) -> dict:
    """Adds values to result dict

    Parameters
    ----------
        result : dict
            Initial dict
        key : str
            New key
        value : str
            New value

    Returns
    -------
        The dictionary with the new key and value pair

    """
    key, value = key.strip().replace('°', ''), value.strip().replace('°', '')

    if key in result.keys():
        result[key].append(value)
    else:
        result[key] = [value]

    return result


def init_driver(headless=True, detached=False, urls='',
                scale=1) -> webdriver.Chrome:
    """The init_driver function initializes a Chrome webdriver object.

    Parameters
    ----------
        headless=True
            Specify whether the browser should be run in headless mode
        detached=False
            Create a new window for the browser
        urls=''
            Open a list of urls in the same browser window
        scale=1
            Scale the browser window

    Returns
    -------
        A webdriver object

    """
    chrome_options = get_chrome_options(headless=headless, detached=detached,
                                        scale=scale)

    driver = webdriver.Chrome(options=chrome_options)
    if len(urls) != 0:
        if not isinstance(urls, Iterable):
            urls = [urls]
        driver.get(urls[0])
    if len(urls) > 1:
        for url in urls[1:]:
            driver.execute_script(f'''window.open("{url}","_blank");''')

    return driver
