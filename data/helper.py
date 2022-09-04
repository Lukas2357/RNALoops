import pandas as pd
from selenium.webdriver.chrome.options import Options
import os


def get_chrome_options(headless=True):
    os.environ["PATH"] += os.pathsep + '/home/lukas'

    chrome_path = '/usr/bin/google-chrome'
    window_size = "1920,1080"
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % window_size)
    chrome_options.binary_location = chrome_path

    return chrome_options


def get_data_dict(content):
    result = {}

    for line in content:
        sep = line.find(':')
        if sep > -1:
            key = line[:sep]
            value = line[sep + 2:]
            if ':' in value:
                new_keys = value.split(':')
                add_to_dict(result, new_keys[0], '-')
                value = '-'
                if len(new_keys) > 2:
                    result = add_to_dict(result, new_keys[1], new_keys[2])
            result = add_to_dict(result, key, value)

    result['Whole Sequence'] = content[content.index('Sequence:') - 1]
    result['Sequence'] = result['Sequence'][1:]

    for key, value in result.items():
        if len(value) == 1:
            result[key] = value[0]

    return result


def init_content_df():

    single_columns = ['Loop type', 'Home structure (PDB id)', 'DB notation',
                      'Whole Sequence']

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

    twice_multiple_columns = ['Sequence']

    columns = single_columns

    for column in multiple_columns:
        columns += [column + f' {idx}' for idx in range(1, 15)]

    for column in twice_multiple_columns:
        columns += [column + f' {idx}' for idx in range(1, 29)]

    df = pd.DataFrame(columns=columns)

    return df


def insert_content_df(df, content, idx):

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


def add_to_dict(result, key, value):
    key, value = key.strip().replace('°', ''), value.strip().replace('°', '')

    if key in result.keys():
        result[key].append(value)
    else:
        result[key] = [value]

    return result


def get_url(idx):
    return f'https://rnaloops.cs.put.poznan.pl/search/details/{idx}'
