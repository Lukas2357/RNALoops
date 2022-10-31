"""Main module to verify RNALoops data based on PDB data

Use the following workflow for the verification:

1) Download mmcif files for all pdb_ids from RNALoops using
PDB_DATA/batch_download.sh with the PDB_DATA/home_struct_ids file
See the instructions in the file for details.

2) Get PdbStructure objects for each file using
verify.mmcif_parser.get_full_structures_parallel

3) Get the qualities of each of these structures using
verify.verify_fcts.get_qualities
and eventually plot them using
verify.verify_fcts.plot_qualities

You may also want to compare RNALoops and RNAStrands. For this use
verify.rnastrands_con module as a starting point to connect the two databases.
You can then run queries on both or do any other stuff on th joined datasets.

"""
from rnaloops.verify.mmcif_parser import get_full_structures_parallel
from rnaloops.verify.verify_fcts import get_qualities, plot_qualities

#  Step 1)
#  Do manually in shell.

#  Step 2)
get_full_structures_parallel(n_pools=8)  # Might take several hours.

# Step 3)
get_qualities()  # Might take a minute
plot_qualities()
