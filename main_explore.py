"""Main exploration module"""
from matplotlib import pyplot as plt

from rnaloops.config.helper import save_figure
from rnaloops.explore.help_fcts import get_sequences_ordered_by_nstructures
from rnaloops.explore.plot_fcts import plot_angles
from rnaloops.prepare.data_loader import load_data

df = load_data('_cleaned_L2')

# Choose here all columns that should be checked. If the list entries are lists
# with one element, histograms are plotted for that angle feature, in case of
# two or three scatterplots in 2D/3D are shown, more is not possible.
cols = [['planar_1'], ['planar_2', 'planar_3']]

# Choose here the sequences to show. A sting will be interpreted as a sequence
# and must exist in the whole_sequence column. An integer n will be used to
# take the nth sequence when ordering them by number of structures descending.
sequences = [1, 3, 'GCC — GGCGAAAAG — CC']

# You might want to save results (ATTENTION: WILL OVERWRITE PREVIOUS!)
save = False

for sequence in sequences:
    for col in cols:
        plot_angles(col, save=save, df=df, sequence=sequence,
                    hue_col='parts_seq', legend=False, ms=10, show_other=False)

# If you want to see the n most often occurring sequences in one plot, use:
n = 0
feature = ['planar_3']

ns, _ = get_sequences_ordered_by_nstructures(df, n=n)
lns = len(ns)
_, ax = plt.subplots((lns - 1) // 2 + 1, 2, figsize=(16, 1 + lns), dpi=200)

for s, a in zip(ns, ax.ravel()):
    plot_angles(feature, save=False, legend=False, df=df, sequence=s, ax=a)

plt.tight_layout()
save_figure(f'{n}_most_often_occuring', folder='angles/' + '-'.join(feature),
            create_if_missing=True, save=True, recent=False)
