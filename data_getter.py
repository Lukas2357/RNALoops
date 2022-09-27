"""Main module to fetch data from web database"""

import joblib

from rnaloops.data_getter.helper_fcts import init_content_df, insert_content_df
from rnaloops.data_getter.getter_fcts import get_pdfs, get_content, batch_files

# Be careful with this module, it can download thousands of files in arbitrary
# many threads in parallel. Make sure to choose a reasonable number of threads
# and be aware that crawling took about 170ms per index on 8 threads.

min_idx = 1    # Min idx to check, last time 53808 was the smallest entry found
max_idx = 2    # Max idx to check, last time 201489 was the largest entry found
n_pools = 8    # Number of pools used for parallel download (8 was the fastest)

# This downloads the pdfs, see README.md and getter_fcts for details:
get_pdfs(n_pools, min_idx, max_idx)

# This fetches the relevant content from the pdfs and saves as plain files:
contents, indices = get_content()

# This inserts the contents in a pandas DataFrame:
df = init_content_df()
for content, idx in zip(contents, indices):
    df = insert_content_df(df, content, idx)

# This dumps the df into a serialized file with given name in this folder:
joblib.dump(df, 'rnaloops/data_getter/rnaloops.pkl')

# This moves files in folder data_pdfs & data_files with subfolders of <=1000
# entries organized by idx. 80k files at once might break down file explorer!
batch_files(kind='pdf')
batch_files(kind='plain')
