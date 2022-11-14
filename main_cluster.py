"""Main module to apply clustering on RNALoops data"""

from rnaloops.cluster.clustering import do_cluster
from rnaloops.engineer.add_features import load_agg_df
from rnaloops.prepare.data_loader import load_data

way = 3

# Select features for clustering:
feature = [f'planar_{i}_median' for i in range(1, way+1)]
# Load the data you want to cluster here:
df = load_agg_df(way=way)  # Aggregated df (way=None for all, else only way-x)
# df = load_data('_cleaned_L2')  # The full df (cleaned or not as you want)

# You might want to evaluate the clustering based on silhouette scores:
evaluate = False  # can be time-consuming for large data sets

# Set here the samples from all data you want to draw and cluster each:
samples = (len(df.index), )

# And the number of clusters to check, other parameters are set below:
n_clusters = (75, )

silhouette_scores = None  # only used if evaluate is True
for i, s in enumerate(samples):  # Clustering done for each sample size given
    output = do_cluster(
        df.sample(s), feature, n_clusters=n_clusters, silhouette=evaluate,
        alg='agglomerative',  # Algorithm (k-means|meanshift|agglomerative)
        dim=way,  # dim=len(features): all are clustered, else each combination
        scaler=None,  # Scaler for data (Standard|MinMax|None)
        n_bins=15,  # For dim=1 Histograms are shown, set nbins here
        s=4,  # For dim=2 Scatterplots are shown, set marker size here
        fontsize=7,  # Fontsize for plots
        dpi=300,  # Dots per inch for plots
        pca=False,  # Whether to perform PCA on all given features
        plot=True,  # Whether to plot histograms/scatterplots
        silhouette_plot=False,  # Whether to plot silhouette plots
        save=False,  # Whether to save results
        legend=True,  # Whether to display legend
        verbose=False
    )
    if evaluate:
        if silhouette_scores is None:
            silhouette_scores = output.reset_index(drop=True)
        else:
            silhouette_scores.loc[i] = output.iloc[0].values
    else:
        for idx, out in enumerate(output):
            out.to_csv(f'results/csv/clusters/{way}_{idx}.csv')

if evaluate:
    post = '-'.join(feature)
    silhouette_scores.to_csv(f'results/csv/silhouette/silhouette-{post}.csv')

import matplotlib.pyplot as plt
plt.show()
