"""Main module to get png of secondary structure of each multiloop"""
import os
import timeit
from multiprocessing import Pool

from rnaloops.config.helper import mypath
from rnaloops.engineer.analyse_cluster import get_seq_png
from rnaloops.prepare.data_loader import load_data

for _ in range(1000):

    start = timeit.default_timer()

    indices = load_data('_cleaned_L2_final').index
    missing = []
    for idx in indices:
        x = 1000
        subfolder = f'{int((int(idx) // x) * x)}-{int((int(idx) // x + 1) * x)}'
        path = mypath('SEQ_PNG', subfolder=subfolder, file=f'{idx}.png',
                      create_if_missing=True)
        if not os.path.isfile(path):
            missing.append(idx)

    print('Missing:', len(missing))

    if len(missing) == 0:
        break

    with Pool(16) as pool:
        pool.starmap(get_seq_png, [[missing, 20, False, False, i, True]
                                   for i in range(16)])
        pool.close()

    [os.remove(p) for p in os.listdir() if '.svg' in p]
    [os.remove(p) for p in os.listdir() if '.cif' in p]

    print(f'Elapsed: {timeit.default_timer()-start:.2f} sec')
