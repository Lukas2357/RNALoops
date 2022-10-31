"""Some functions to verify the integrity of the RNALoops database"""

import os

import joblib
from matplotlib import pyplot as plt

from .mmcif_parser import check_structures
from ..config.helper import mypath


def get_qualities():
    """Get qualities of multiloops
        -> see mmcif_parser.PdbStructure.check_structure for details

    """
    pdb_ids = [x[:-4] for x in os.listdir('structures')]
    qualities = []
    categories = []
    if pdb_ids:
        result = check_structures(pdb_ids)
        for value in result.values():
            qualities.append(len(value))
            categories.append(value)

    # The number of indices in each quality class:
    joblib.dump(qualities, mypath('LOOP_DATA', 'quality_counts.csv'))
    # The qualities of each index as dict:
    joblib.dump(categories, mypath('LOOP_DATA', 'qualities.csv'))

    return categories


def plot_qualities(save=False):
    """Plot the number of indices in each quality class as histogram"""

    qualities = joblib.load("qualities.csv")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=250)
    ax.barh(range(len(qualities[-1])), qualities[-1][::-1])

    ax.set_yticklabels(
        [
            "No issues found",
            "Helix length missmatch found",
            "Strands with canonical bonds found",
            "Helix detached from strand found",
            "Disconnected strands found",
            "Less than 3 helices found (not a multiloop)",
            "Only 0 length strands (cant be analysed)",
            ""
        ][::-1]
    )
    ax.set_xlabel('number of multiloops')
    _ = ax.set_title(
        'Issues of multiloops in RNALoops in comparison to PDB mmcif')
    plt.tight_layout()

    if save:
        plt.savefig(mypath('RESULTS_PREP', 'ml_qualities.png'))

    return ax, fig


def load_qualities() -> dict:
    """Load the qualities from csv file and return as dict"""
    qualities = joblib.load(mypath('LOOP_DATA', 'qualities.csv'))

    # The dict with indices as keys and qualities as values:
    q_dict = {}
    for idx in range(7):
        for entry in qualities[idx]:
            q_dict[entry] = idx

    return q_dict
