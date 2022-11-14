"""Main exploration module"""
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('TkAgg')

from rnaloops.config.helper import save_figure
from rnaloops.explore.get_similar import get_similar_seq
from rnaloops.explore.help_fcts import get_sequences_ordered_by_nstructures
from rnaloops.explore.plot_fcts import plot_angles
from rnaloops.prepare.data_loader import load_data

df = load_data('_cleaned_L2_complete_lower')

# Choose here all columns that should be checked. If the list entries are lists
# with one element, histograms are plotted for that angle feature, in case of
# two or three scatterplots in 2D/3D are shown, more is not possible.
cols = [['planar_1', 'planar_2'], ['planar_2', 'planar_3']]
# Or plot all relevant planar angles in a 2D automatically:
auto_cols = True
# Select the category of which all similar mls should be shown in one plot:
cat = 'chain_0_label'
# Choose whether to how legend:
show_legend = True

# Choose here the sequences to show. A string will be interpreted as a sequence
# and must exist in the whole_sequence column. An integer n will be used to
# take the nth sequence when ordering them by number of structures descending.
plot_sequences = ['U-A|UA|G-C||C-G||C-G||G-C|AAA']
# Or use this line to get sequences with most multiloops and select from it:
use_most_frequent = True

# Set here the hue and style columns to visually differentiate in the plot:
hue_col = 'whole_sequence'
marker_col = 'chain_0_organism'
# And whether to mar mean with dot and confidence ellipse:
plot_mean = False

# In case you want to plot similar sequences as well, use:
plot_similar = False
min_n = 10  # Minimum number of multiloop s in similar sequence
n_max = 20  # Max. number of similar sequences plotted (could be many otherwise)
# And provide z-score difference up to which similar sequences are considered:
z = 5

# If you want to see the n most often occurring sequences in one plot, use:
n = 0  # 0 to skip this type of plot
feature = ['planar_3']

# You might want to save results (ATTENTION: WILL OVERWRITE PREVIOUS!)
save = True
folder = 'micro/test'

# ---- END OF USER DEFINITION SECTION ---- #

s, counts = get_sequences_ordered_by_nstructures(df, n=None, cat=cat)
sequence_counts = {key: value for key, value in zip(s, counts)}
sequences = s if use_most_frequent else plot_sequences

for key in sequences:

    print(key)

    if plot_similar:
        sims = get_similar_seq(key, z=z, verbose=3)
        values = list(sims.similar_sequence.values)
        values = [x for x in values if sequence_counts[x] >= min_n]
        values = values[:n_max]
    else:
        values = [key]

    if n_max <= 10:
        colors = list(mcolors.TABLEAU_COLORS.values())
    else:
        colors = list(plt.cm.tab20.colors)

    if auto_cols:
        if cat == 'parts_seq':
            way = len(key.split('|'))//2
        else:
            parts_key = df.loc[df[cat] == key, 'parts_seq'].iloc[0]
            way = len(parts_key.split('|')) // 2
        cols = [[f'planar_{i}', f'planar_{i+1}']
                for i in range(1, way, 2)]
        if way % 2:
            cols.append([f'planar_{way-1}', f'planar_{way}'])

    rows = (len(cols)+1)//2
    _, ax = plt.subplots(rows, 2, figsize=(20, 9*rows), dpi=300)

    for sub, col in enumerate(cols):
        for i, sequence in enumerate(values):

            color = colors[i] if plot_similar else None
            c_ax = ax.ravel()[sub]
            if plot_similar:
                cut = 'std' if i == 0 else ''
                legend = values if sub == 0 else False
                ms = 12
            else:
                legend = show_legend if sub == 0 else False
                cut = 'min_max'
                ms = 20

            plot_angles(col,
                        save=False,
                        df=df,
                        sequence=sequence,
                        ax=c_ax,
                        ms=ms,
                        hue_col=hue_col,
                        marker_col=marker_col,
                        legend=legend,
                        show_other=False,
                        color=color,
                        cat=cat,
                        w_size=3 * z,
                        title=key,
                        cut=cut,
                        plot_mean=plot_mean)

    if save:
        save_figure(f'{key}_sims', folder=folder, create_if_missing=True)
    plt.close()

if n > 0:
    ns, _ = get_sequences_ordered_by_nstructures(df, n=n)
    lns = len(ns)
    _, ax = plt.subplots((lns - 1) // 2 + 1, 2, figsize=(16, 1 + lns), dpi=200)

    for s, a in zip(ns, ax.ravel()):
        plot_angles(feature, save=False, legend=False, df=df, sequence=s, ax=a)

    plt.tight_layout()
    if save:
        save_figure(f'{n}_most_often_occurring',
                    folder='angles/' + '-'.join(feature),
                    create_if_missing=True, save=True, recent=False)
