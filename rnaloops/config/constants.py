"""Constants used for rnaloops"""

import os

# Paths to be used in config.mypath() for easy folder access.
# ROOT/PROJECT must be absolute paths, all others can be relative to PROJECT,
# since we assume that script are always and only executed from the project
# folder (or otherwise they must pretend to do so):

PATHS = dict(
    ROOT=os.path.join(os.sep, 'home', 'lukas'),
    PROJECT=os.path.join(os.sep, 'home', 'lukas', 'Projects', 'RNALoops'),
    DATA=os.path.join('rnaloops', 'data'),
    DATA_PREP=os.path.join('rnaloops', 'data', 'prepared'),
    CIF_FILES=os.path.join('rnaloops', 'data', 'pdb_data', 'cif'),
    PDB_FILES=os.path.join('rnaloops', 'data', 'pdb_data', 'pdb'),
    # CIF_FILES='backups/RNALoops_1.1.0_2022-10-27/rnaloops/data/pdb_data/cif',
    # PDB_FILES='backups/RNALoops_1.1.0_2022-10-27/rnaloops/data/pdb_data/pdb',
    PDF_FILES=os.path.join('rnaloops', 'data', 'rnaloops_data', 'data_pdfs'),
    PLAIN_FILES=os.path.join('rnaloops', 'data', 'rnaloops_data', 'data_files'),
    LOOP_DATA=os.path.join('rnaloops', 'data', 'rnaloops_data'),
    PDB_DATA=os.path.join('rnaloops', 'data', 'pdb_data'),
    STRAND_DATA=os.path.join('rnaloops', 'data', 'rnastrand_data'),
    STRUCTURES=os.path.join('rnaloops', 'data', 'structures'),
    RESULTS='results',
    CLUSTER=os.path.join('results', 'csv', 'cluster'),
    RESULTS_RECENT=os.path.join('results', 'recent'),
    RESULTS_PREP=os.path.join('results', 'prep'),
    CHROME_PATH=os.path.join(os.sep, 'usr', 'bin'),
    SEQ_PNG=os.path.join('rnaloops', 'data', 'rnaloops_data', 'seq_pngs')
)

PLANAR_STD_COLS = [f"planar_{i}_std" for i in range(1, 15)]
