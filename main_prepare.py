"""Main module to prepare RNALoops data for analysis

Assuming main_obtain.py was successfully and completely executed, there
should be a file 'rnaloops_data.pkl' in the 'DATA_PREP' folder, which is
defined (as any other folder path) in config/constants.

We can then prepare this file in several steps for the analysis and save one
new file in each step as explained below.

Prepare package has also an explore_fcts module that contains functions
to be called from notebooks to explore the data.

"""
from rnaloops.prepare.data_preparer import prepare_df, save_clean_dfs, order_df

prepare = False  # Whether to create prepared df from initial
order = False    # Whether to create ordered df from prepared
clean = False    # Whether to create cleaned df from ordered

if prepare:
    # creates rnaloops_data_prepared.pkl from rnaloops_data.pkl:
    prepare_df(convert=False, save=True)  # use convert to make categorical

if order:
    # creates rnaloops_data_ordered.pkl from rnaloops_data_prepared.pkl:
    order_df(save=True, load=False)  # use load to just reorder ordered df

if clean:
    # creates rnaloops_data_cleaned_L0.pkl from rnaloops_data_prepared.pkl
    # creates rnaloops_data_cleaned_L1.pkl from rnaloops_data_prepared.pkl
    # creates rnaloops_data_cleaned_L2.pkl from rnaloops_data_prepared.pkl
    # creates rnaloops_data_cleaned_L3.pkl from rnaloops_data_prepared.pkl
    save_clean_dfs()
