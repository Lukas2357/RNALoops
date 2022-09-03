from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium import webdriver
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
            value = line[sep+2:]
            if value == 'Second Strand: -':
                if 'Second Strand' in result.keys():
                    result['Second Strand'].append('-')
                else:
                    result['Second Strand'] = ['-']
                value = '-'
            if key in result.keys():
                result[key].append(value)
            else:
                result[key] = [value]

    result['Whole Sequence'] = content[content.index('Sequence:')-1]

    for key, value in result.items():
        if len(value) == 1:
            result[key] = value[0]
            
    return result


def get_url(idx):
    return f'https://rnaloops.cs.put.poznan.pl/search/details/{idx}'