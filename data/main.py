import joblib

from helper import init_content_df, insert_content_df
from getter import get_pdfs, get_content


min_idx = 1
max_idx = 1
n_pools = 8

# get_pdfs(n_pools, min_idx, max_idx)
contents, indices = get_content()

df = init_content_df()
for content, idx in zip(contents, indices):
    df = insert_content_df(df, content, idx)

joblib.dump(df, 'rnaloops')
