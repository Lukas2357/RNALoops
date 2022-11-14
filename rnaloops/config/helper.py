"""Helper functions for rnaloops"""

import os
import warnings

from .constants import PATHS
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def mypath(folder, file=None, create_if_missing=False, subfolder=None):
    """Returns path object to specified folder or file

    Parameters
    ----------
    folder : str
        Can be any string representing a folder on the local file system.
        If the string is found in config.constants.PATHS.keys(), the value of
        this key is used (you might add frequently used paths there).
    file : str or None, default None
        A file in that folder, if None only the folder is returned.
    create_if_missing : bool, default=False
        Create the folder path if it does not exist. Might create all parents.
    subfolder : str, default None
        A subfolder in the given folder to use

    Returns
    -------
    os.path.PATH
        The path object.

    Raises
    -------
    OSError:
        If the folder does not exist or can not be created, respectively.

    """

    constant_folder = folder in PATHS.keys()
    if not constant_folder and folder == folder.upper():
        warnings.warn('Uppercase folders should be defined in constants/PATHS!')
    folder = PATHS[folder] if constant_folder else folder

    folder = folder if subfolder is None else os.path.join(folder, subfolder)

    if not os.path.isdir(folder):
        if create_if_missing:
            os.makedirs(folder)
        else:
            m = "The chosen folder does not exist! \n If you want to" + \
                "create it call again with create_if_missing=True"
            raise OSError(m)

    path = folder if file is None else os.path.join(folder, file)

    return path


def save_data(df: pd.DataFrame, filename: str, formats=('csv', ),
              folder='DATA_PREP'):
    """Save the data in csv and/or xlsx format

    Parameters
    ----------
    df : pd.DataFrame
        The raw_data as pandas dataframe
    filename : str
        The name of the file to save
    formats : tuple[str], default ('csv', 'xlsx')
        The formats to save the raw_data in
    folder : string
        The folder to save data in

    """
    if 'csv' in formats:
        data_file = mypath(folder, filename + '.csv')
        df.to_csv(data_file)

    if 'xlsx' in formats:
        data_file = mypath(folder, filename + '.xlsx')
        df.to_excel(data_file)

    if 'pkl' in formats:
        data_file = mypath(folder, filename + '.pkl')
        df.to_pickle(data_file)


def save_figure(name=None, fig=None, tight=True, dpi='figure',
                fformat='png', recent=True, folder=None, save=True,
                create_if_missing=False):
    """Save current figure to proper location

    Parameters
    ----------
    name : str, default None -> set to current datetime
        The name of the figure file to create
    fig : plt.Figure, default None
        The figure handle, must be provided to set face color white
    tight : bool, default True
        Whether to save in tight layout
    dpi : int, default 'figure'
        Dots per inch, default uses the figures initial dpi
    fformat : str, default 'png'
        The file format to use
    recent : bool, default=True
        Whether to save the figure in recent
    folder : str, default None -> save only in recent
        The folder to save the figure in
    save : bool, default True
        Whether to do the saving
    create_if_missing : bool, default False
        create the folder if it does not exist
    
    """
    if not save:
        return

    if name is None:
        name = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f'))

    name = name + '.' + fformat
    path = mypath('RESULTS', name, subfolder=folder,
                  create_if_missing=create_if_missing)
    r_path = mypath('RESULTS_RECENT', name) if recent else ''

    if tight:
        plt.tight_layout()

    if fig is not None:
        fig.patch.set_facecolor('white')
        fc = fig.get_facecolor()
        if folder is not None:
            plt.savefig(path, dpi=dpi, facecolor=fc, format=fformat)
        if recent:
            plt.savefig(r_path, dpi=dpi, facecolor=fc, format=fformat)
    else:
        if folder is not None:
            plt.savefig(path, dpi=dpi, format=fformat)
        if recent:
            plt.savefig(r_path, dpi=dpi, format=fformat)
