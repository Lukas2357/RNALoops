from rnaloops.engineer.add_features import load_agg_df
from rnaloops.explore.get_similar import get_similar_df, get_mean_diffs
from rnaloops.explore.help_fcts import get_sequences_ordered_by_nstructures
from rnaloops.prepare.data_loader import load_data

df = load_data('_cleaned_L2')
agg = load_agg_df()
sequences, counts = get_sequences_ordered_by_nstructures(df, n=None,
                                                         cat='parts_seq')

# get_similar_df(agg, sequences)
get_mean_diffs(sequences, counts)
