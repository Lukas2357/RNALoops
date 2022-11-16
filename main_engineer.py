"""Main module to engineer features and analyse cluster"""
import timeit
from copy import deepcopy

import numpy as np

from rnaloops.engineer.add_features import load_agg_df, save_agg_df
from rnaloops.engineer.analyse_cluster import main_cluster_analysis, \
    main_clustering
from rnaloops.prepare.data_loader import load_data
from multiprocessing import Pool
import matplotlib as plt
plt.use('TkAgg')  # Avoid visual backend to save memory


# --- Parameter definition section ---

n_pools = 16  # Number of Pools to use (each will occupy a threat)
n_clusters = [10, 25, 50]  # Number of initial clusters
alg = 'agglomerative'  # The algorithm for initial clustering (string)
chunk_size = 1  # Size of chunks to evaluate in one Pool at a time
resets = 5  # Number of resets (to release memory)

# Select now if you want to cluster all, just rrna, trna or the rest:
kind = 'rest'

user_params = dict(
    save=True,  # Whether to save cluster id plots (bool)
    max_devs=[4, 8, 12, 16],  # Max angle deviation for cluster (Iterable)
    min_density=0,  # Minimum density to save cluster (Integer)
    min_n=5,  # Minimum number of sequences needed for cluster (Integer)
)


# --- Function call section ---

start_tot = timeit.default_timer()

df_cols = ["parts_seq", "db_notation", "main_organism", "chain_label",
           "loop_type"] + [f'planar_{i}' for i in range(1, 15)]

if kind == 'all':
    raw_df = load_data('_cleaned_L2_final')[df_cols]
else:
    raw_df = load_data(f'_cleaned_L2_{kind}')[df_cols]

features = [f'planar_{i}_mean' for i in range(1, 15)] + \
           [f'planar_{i}_median' for i in range(1, 15)]

try:
    agg_all = load_agg_df(level=f'L2_{kind}')[features]
except FileNotFoundError:
    save_agg_df(level=f'L2_{kind}')
    agg_all = load_agg_df(level=f'L2_{kind}')[features]

for way in range(3, 9):

    try:
        agg_df = load_agg_df(level=f'L2_{kind}', way=way)
    except FileNotFoundError:
        save_agg_df(level=f'L2_{kind}', way=way)
        agg_df = load_agg_df(level=f'L2_{kind}', way=way)

    cluster_files = []
    for n_cluster in n_clusters:
        files = main_clustering(agg_df, way, n_cluster, alg, chunk_size)
        cluster_files += files

    fix_params = dict(
        way=way,
        agg_df=agg_df,
        raw_df=raw_df,
        agg_all=agg_all
    )

    files_per_pool = int(np.ceil(len(cluster_files)/n_pools))
    params = [[cluster_files[i * files_per_pool:(i + 1) * files_per_pool]] +
              [x for x in fix_params.values()] +
              [x for x in user_params.values()]
              for i in range(n_pools)]

    files_per_pool = [len(x[0]) for x in params]

    if n_pools == 1:
        for param in params:
            main_cluster_analysis(*param)
    else:

        for reset in range(resets):

            c_params = deepcopy(params)
            for idx in range(len(params)):
                min_file = reset * files_per_pool[idx]//resets
                max_file = (reset + 1) * files_per_pool[idx]//resets
                c_params[idx][0] = c_params[idx][0][min_file:max_file]

            print(f'Analysing loop type {way} - Batch {reset + 1}/{resets} - '
                  f'Pools: {n_pools} - Files per Pool:', len(c_params[0][0]),
                  '- Clusters per File:', chunk_size)

            start = timeit.default_timer()

            with Pool(n_pools) as pool:
                pool.starmap(main_cluster_analysis, c_params)
                pool.terminate()

            print(f'Finished in {(timeit.default_timer()-start)/60:.2f} min\n')

print(f'\n----------------------------DONE-----------------------------------\n'
      f'---- Total time: {(timeit.default_timer()-start_tot)/60:.2f} min ----')
