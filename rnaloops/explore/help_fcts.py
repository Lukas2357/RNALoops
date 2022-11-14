import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.patches import Ellipse


def get_frequent_sequences(df, min_n=100, category="parts_seq"):
    df_new = df.copy()
    df_new['id'] = df_new.index
    grouped = df_new.groupby(by=category).count().id
    df_new = df_new.set_index(category)
    mask = grouped.index[grouped >= min_n]
    df_new = df_new.loc[mask]

    return df_new


def get_sequences_ordered_by_nstructures(df, min_n=1, n=None,
                                         cat='whole_sequence'):

    frequent = get_frequent_sequences(df, min_n=min_n, category=cat)
    grouped = frequent.groupby(by=cat).count()
    sort = grouped.sort_values('loop_type', ascending=False)

    sequences = sort.index.values
    counts = sort.loop_type.values

    if n is not None:
        sequences = sequences[:n]
        counts = counts[:n]

    return sequences, counts


def get_sequence_mean_angles(df, category="parts_seq"):
    columns = [x for x in df.columns if 'planar_' in x]
    df_columns = columns if category == 'index' else columns + [category]

    df = df[df_columns]

    if category == 'index':
        seq_mean = df.groupby(level=0).mean()
    else:
        seq_mean = df.groupby(by=category).mean()

    seq_mean.columns = columns

    if category == 'index':
        seq_std = df.groupby(level=0).std()
    else:
        seq_std = df.groupby(by=category).std()

    seq_std.columns = [c + '_std' for c in columns]

    result = pd.concat([seq_mean, seq_std], axis=1)

    return result


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radius.

    facecolor : string, optional
        The face color to use for the ellipse

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    if np.sqrt(cov[0, 0] * cov[1, 1]) != 0:
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    else:
        pearson = 0
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)
