from os import sys, path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pymysql
import warnings
from matplotlib.colors import LogNorm, Normalize
from pandas.api.types import is_numeric_dtype
import pickle

from Bio.PDB import MMCIFParser, MMCIF2Dict

PARENT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from rnaloops.data_explorer.explore_fcts import *
from rnaloops.data_verifier.mmcif_parser import *

os.chdir(PARENT_DIR)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

warnings.filterwarnings('ignore')
