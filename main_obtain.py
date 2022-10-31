"""Main module to fetch data from web database"""

from rnaloops.obtain.getter_fcts import *

# Be careful with this module, it can download thousands of files in arbitrary
# many threads in parallel. Make sure to choose a reasonable number of threads
# and be aware that crawling took about 170ms per index on 8 threads.

# You might only select certain process steps in here (explanations below):
download_pdfs = False   # ~5 hours with 8 threads and ~200MBit download speed
parse_pdfs = False      # ~2 hours single threaded at around 5.1 GHz
create_batches = False  # <1 hour single threaded at around 5.1 GHz

min_idx = 1    # Min idx to check, last time 53808 was the smallest entry found
max_idx = 2    # Max idx to check, last time 201489 was the largest entry found
n_pools = 8    # Number of pools used for parallel download (8 was the fastest)

# This downloads the pdfs, see obtain/getter_fcts for details:
if download_pdfs:
    get_pdfs(n_pools, min_idx, max_idx)

# This fetches the relevant content from the pdfs and saves as plain files.
# After that it inserts the contents in a pandas DataFrame and dumps it:
if parse_pdfs:
    contents, indices = get_content()
    df = init_content_df()
    for content, idx in zip(contents, indices):
        df = insert_content_df(df, content, idx)
    joblib.dump(df, mypath('DATA_PREP', 'rnaloops_data.pkl'))

# This moves files in folder data_pdfs & data_files with subfolders of <=1000
# entries organized by idx. 80k files at once might break down file explorer!
# Attention: Files can not be parsed anymore after batching, do that before!
if batch_files:
    batch_files(kind='pdf')
    batch_files(kind='plain')
