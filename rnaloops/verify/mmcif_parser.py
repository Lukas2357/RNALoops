"""An extension for Biopythons mmcif parser to relate it to RNALoops"""

import os
import pickle
import warnings
import random
from multiprocessing import Pool

import pandas as pd
from Bio.PDB import MMCIF2Dict  # Bio is the Biopython package
from copy import deepcopy

from ..config.helper import mypath
from ..prepare.data_loader import load_data


class PdbStructure:
    """A class for representing PDB structure information from mmcif files

    Based on Biopythons MMCIF2Dict. Connects the retrieved information to the
    multiloop information from RNALoops database, to get detailed information
    on the bases and bonds from a particular multiloop structure in RNALoops.

    """

    def __init__(self, pdb_id, rnaloops_df=None, considered_bonds=None,
                 verbose=False):
        """Initialize a PdbStructure

        Parameters
        ----------
        pdb_id : str
            The PDB id (called 'home_structure' in RNALoops database)
        rnaloops_df : pd.DataFrame
            Summary all multiloop data from RNALoops. Check rnaloops.obtain
            for details. Loads the df created by that module by default.
        considered_bonds : tuple
            Bond types considered for helices of multiloops. Defaults to
            'WATSON-CRICK', 'G-C PAIR', 'TYPE_28_PAIR', 'C-G PAIR', 'A-U PAIR',
            'U-A PAIR', 'TYPE_10_PAIR', 'U-G MISPAIR'
        verbose : bool
            If verbose id off, no warnings will be logged.

        """
        self.pdb_id = pdb_id

        # These are the bonds considered for the helices, any other bond will
        # be neglected and helices are cut off:
        if considered_bonds is None:
            self.considered_bonds = ('WATSON-CRICK', 'G-C PAIR', 'TYPE_28_PAIR',
                                     'C-G PAIR', 'A-U PAIR', 'U-A PAIR',
                                     'TYPE_10_PAIR', 'U-G MISPAIR')
        else:
            self.considered_bonds = considered_bonds

        # Loads the full RNALoops dataframe with all multiloops and details:
        if rnaloops_df is None:
            rnaloops_df = load_data('_prepared')
        # We only store the part related to current home_structure:
        self.multiloops = rnaloops_df[rnaloops_df.home_structure == pdb_id]

        self.verbose = verbose
        if not self.verbose:
            warnings.filterwarnings("ignore")

        # Define some attributes storing issues with the analysed structures:
        init_dict = {key: [] for key in self.multiloops.index}
        self.ml_without_strands = []
        self.ml_less_than_three_helices = []
        self.disconnected_strands = deepcopy(init_dict)
        self.strands_with_canonical_bonds = deepcopy(init_dict)
        self.helix_length_missmatch = deepcopy(init_dict)
        self.detached_helices = deepcopy(init_dict)

        # All relevant infos about secondary structure are in these attributes:
        self.base_order = self.get_base_order()
        self.bonds_df = self.get_bonds_df()
        self.ml_strands = self.get_ml_strands()
        self.ml_helices = self.get_ml_helices()

    def get_cif_dict(self) -> dict:
        """Use Biopython and its MMCIF2Dict method to parse the mmcif file.

        Returns
        -------
        dict
            Contains exactly the same information as the mmcif file as a dict

        """
        # PDB id fetched from RNALoops download pdf files contains weird chars
        # ﬀ and ﬁ, which we need to replace (filenames can not contain those):
        pdb_id = self.pdb_id.replace('ﬀ', 'ff')
        pdb_id = pdb_id.replace('ﬁ', 'fi')
        pdb_id = pdb_id.replace('ﬂ', 'fl')

        # If cif files are stores elsewhere, change in rnaloops.config.constant
        file_path = mypath('CIF_FILES', f"{pdb_id}.cif")

        return MMCIF2Dict.MMCIF2Dict(file_path)

    def get_base_order(self) -> pd.DataFrame:
        """Get the order of bases in the sequence as pd.DataFrame

        Returns
        -------
        pd.DataFrame
            The order of bases in the structures sequence. Index is the base
            position label as <CHAIN>-<IDX>-<INSERT>. The columns are:
            base -> The residue type (C, G, A, U, or others)
            prev -> The previous base position label
            next -> The next base position label
            idx -> Sequential increment index
            chain -> The chain label
            chain_idx -> Sequential incrementing index per chain

            Different chains are not separated. To obtain a chain break check:
            df.at[label, 'next'].split('-')[0] != label.split('-')[0]

        """

        cif_dict = self.get_cif_dict()

        # We use the _pdbx_poly_seq_scheme table from mmcif for sequence info:
        prefix = '_pdbx_poly_seq_scheme.'
        # Bases are labeled like LABEL = <PDB-CHAIN-ID>-<AUTHOR-SEQ-ID>
        # It happens that bases are inserted and labeled like LABEL-A, LABEL-B,
        # etc. We need to construct these labels by appending these inserts:
        chains = cif_dict[prefix + 'pdb_strand_id']
        indices = cif_dict[prefix + 'pdb_seq_num']
        inserts = cif_dict[prefix + 'pdb_ins_code']
        inserts = ['-' + i for i in inserts]
        # Regular labels have inserts ., we need to remove that:
        inserts = [i.replace('-.', '') for i in inserts]
        # And put all together:
        base_order = [chain + '-' + idx + insert
                      for chain, idx, insert in zip(chains, indices, inserts)]

        df = pd.DataFrame(index=base_order)
        df['base'] = cif_dict[prefix + 'mon_id']
        # Remove duplicates, should not occur but in case of errors in mmcif:
        base_order_df = df[~df.index.duplicated(keep='first')]
        base_order = list(dict.fromkeys(base_order))
        prev_bases = [base_order[-1]] + base_order[:-1]
        next_bases = base_order[1:] + [base_order[0]]
        # Add prev and next bases:
        base_order_df['prev'] = prev_bases
        base_order_df['next'] = next_bases
        # Add index:
        base_order_df['idx'] = range(len(base_order))
        # Add chain_label:
        base_order_df['chain'] = base_order_df.index.str.split('-').str[0]
        # Add chain index:
        chain_idx = []
        for count in base_order_df.groupby('chain', sort=False).count()['idx']:
            chain_idx += list(range(count))
        base_order_df['chain_idx'] = chain_idx

        return base_order_df

    def get_order_dict(self) -> dict:
        """Get an ordering dict to be used as key in sorted(...)

        Returns
        -------
        dict
            dict with bases as keys and their index as values
            On a sequence x of bases from a structure s, call
            sorted(x, s.get_order_dict().get) to order bases

        """
        return {key: value for key, value in zip(self.base_order.index,
                                                 self.base_order.idx)}

    def get_bonds_df(self) -> pd.DataFrame:
        """Get pd.DataFrame containing information on all structure bonds

        The function simply parses the cif dict obtained from mmcif file.

        Returns
        -------
        pd.DataFrame

            The df contains all bonds in the structure with their details.
            It also contains non-bonded bases and assigns None to the bond
            type and kind, and '-' to the partner. This is to find unpaired
            bases from the base label easily.

            The df has a multiindex of 6 levels, that are:
            1. Position label (CHAIN-IDX-INSERT) of the first bond partner
            2. Position label (CHAIN-IDX-INSERT) of the second bond partner
            3. Atom of first partner that forms the bond
            4. Atom of second partner that forms the bond
            5. Residue type of first partner (e.g. C, G, U, A, but also other)
            6. Residue type of second partner (e.g. C, G, U, A, but also other)

            And two columns, which are:
            1. The type of bond (hydrog, metalc, covalent, ...)
            2. The kind of bond (e.g. WATSON-CRICK, GA-MISPAIR, ...

        """

        cif = self.get_cif_dict()
        bonds_df = pd.DataFrame()

        # Hydrogen, covalent, ...
        bonds_df["type"] = cif["_struct_conn.conn_type_id"]

        # We use the authors numbering scheme as it was also done in RNALoops:
        bonds_df["ptnr1"] = cif["_struct_conn.ptnr1_auth_comp_id"]
        bonds_df["ptnr2"] = cif["_struct_conn.ptnr2_auth_comp_id"]

        # Construction of labels as in get_base_order:
        inserts = cif["_struct_conn.pdbx_ptnr1_PDB_ins_code"]
        inserts = ['-' + i for i in inserts]
        # Regular labels have inserts ?, we need to remove that:
        inserts = [i.replace('-?', '') for i in inserts]
        # And put the label together:
        bonds_df["pos1"] = (pd.Series(cif["_struct_conn.ptnr1_auth_asym_id"])
                            + "-" +
                            pd.Series(cif["_struct_conn.ptnr1_auth_seq_id"])
                            +
                            pd.Series(inserts))
        # Same for partner 2:
        inserts = cif["_struct_conn.pdbx_ptnr2_PDB_ins_code"]
        inserts = ['-' + i for i in inserts]
        inserts = [i.replace('-?', '') for i in inserts]
        bonds_df["pos2"] = (pd.Series(cif["_struct_conn.ptnr2_auth_asym_id"])
                            + "-" +
                            pd.Series(cif["_struct_conn.ptnr2_auth_seq_id"])
                            +
                            pd.Series(inserts))

        # Atoms:
        bonds_df["atom1"] = cif["_struct_conn.ptnr1_label_atom_id"]
        bonds_df["atom2"] = cif["_struct_conn.ptnr2_label_atom_id"]
        bonds_df["kind"] = cif["_struct_conn.details"]

        # We sort according to the first position:
        bonds_df = bonds_df.sort_values('pos1')
        # We also add the inverted bonds. This is to find any bond by the label
        # of one of the partners quickly:
        idx_cols1 = ['pos1', 'pos2', 'atom1', 'atom2', 'ptnr1', 'ptnr2']
        idx_cols2 = ['pos2', 'pos1', 'atom2', 'atom1', 'ptnr2', 'ptnr1']
        bonds_df = pd.concat([bonds_df.set_index(idx_cols1),
                              bonds_df.set_index(idx_cols2)])

        # Unpaired bases have None bond:
        for base in self.base_order.index.values:
            if base not in bonds_df.index.get_level_values(level=0):
                bonds_df.loc[(base, '-', '-', '-', '-', '-'), :] = None

        try:
            # Set the index order to the base_order, so df is in sequence order:
            bonds_df = bonds_df.reindex(self.base_order.index.values, level=0)
        except ValueError:
            if self.verbose:
                print('Warning: Duplicate index in bonds_df... removing it!')
            # Residue labels should not be duplicated. If we need to remove
            # them here, it is an error in the mmcif file:
            bonds_df = bonds_df[~bonds_df.index.duplicated(keep='first')]
            bonds_df = bonds_df.reindex(self.base_order.index.values, level=0)
        except NotImplementedError:
            print(f'Warning: id {self.pdb_id} bonds_df reindexing impossible!')
            pass

        return bonds_df

    def get_ml_strands(self) -> pd.DataFrame:
        """Get entries from bonds_df corresponding to multiloop strands"""
        strands = []

        for ml in self.multiloops.iterrows():

            ml_strands = []

            for idx in range(1, 1 + int(ml[1].loop_type.split('-')[0])):
                start = ml[1][f'start_{idx}']
                end = ml[1][f'end_{idx}']
                if start.split('-')[0] != end.split('-')[0]:
                    if self.verbose:
                        print(f'Warning: id {ml[0]} strand {idx} '
                              f'({start}|{end}) seems disconnected!')
                    self.disconnected_strands[ml[0]].append(idx)
                try:
                    strand = self.bonds_df.loc[start:end]
                except pd.errors.UnsortedIndexError:
                    strand = self.bonds_df.loc[start:start]
                ml_strands.append(strand)
                if 'WATSON-CRICK' in strand.kind.values:
                    if self.verbose:
                        print(f'Warning: id {ml[0]} strand {idx} contains '
                              f'canonical bonds!')
                    self.strands_with_canonical_bonds[ml[0]].append(idx)

            keys = [f'strand{i}' for i in range(1, 1 + len(ml_strands))]
            strands.append(pd.concat(ml_strands, keys=keys))

        return pd.concat(strands, keys=self.multiloops.index)

    def get_ml_helices(self) -> pd.DataFrame:
        """Get entries from bonds_df corresponding to multiloop helices"""
        helices = []
        for multiloop in self.multiloops.index:
            ml_helices = self._get_single_ml_helices(multiloop)

            if len(ml_helices.index.get_level_values(0).unique()) < 3:
                if self.verbose:
                    print(f'Warning: id {multiloop} has less than 3 helices!')
                self.ml_less_than_three_helices.append(multiloop)

            helices.append(ml_helices)

        return pd.concat(helices, keys=self.multiloops.index)

    def _get_single_ml_helices(self, ml):
        """Get the helices for a single multiloop"""
        n = int(self.multiloops.loc[ml].loop_type.split('-')[0])
        keys = [f'helix{i}' for i in range(1, 1 + n)]
        ml_helices = []

        indices = [n] + list(range(1, n))

        for count, idx in enumerate(indices):
            ml_helix = self._get_ml_helix(ml, keys, count, idx, n)
            ml_helices.append(ml_helix)

        return pd.concat(ml_helices, keys=keys)

    def _get_ml_helix(self, ml, keys, count, idx, n):
        """Get the base pairs and bonds for a single helix"""
        h_idx = keys[count]
        end = self.multiloops.loc[ml, f'end_{idx}']
        expected_length = self.multiloops.loc[ml, f'helix_{count + 1}_bps']
        skipped = 0

        while end == '-':
            skipped += 1
            if skipped > n:
                if self.verbose:
                    print(f'Warning: id {ml} Cant get helix, no strand')
                self.ml_without_strands.append(ml)
                return self.bonds_df.iloc[0:0]
            idx = idx + 1 if idx < n else 1
            end = self.multiloops.loc[ml, f'start_{idx}']

        for _ in range(skipped):
            if end == '-':
                if self.verbose:
                    print(f'Warning: id {ml} Cant get helices, no basepair!')
                self.detached_helices[ml].append(h_idx)
                return self.bonds_df.iloc[0:0]
            prev = self.base_order.at[end, 'prev']
            end_bonds = self.bonds_df.loc[prev]
            considered = end_bonds[end_bonds.kind.isin(self.considered_bonds)]
            partner = considered.index.get_level_values(0).values
            if len(partner) == 0:
                if self.verbose:
                    print(f'Warning: id {ml} Cant get helices, no basepair!')
                self.detached_helices[ml].append(h_idx)
                return self.bonds_df.iloc[0:0]
            end = considered.index.get_level_values(0).values[0]

        if end == '-' or end not in self.base_order.index:
            if self.verbose:
                print(f'Warning: id {ml} Cant get helices, no basepair')
            self.detached_helices[ml].append(h_idx)
            return self.bonds_df.iloc[0:0]
        else:
            if skipped == 0:
                end = self.base_order.at[end, 'next']

        helix_bases = []
        if end in self.bonds_df.index:
            bond_kind = self.bonds_df.loc[end].kind.values
            while any(bond in bond_kind for bond in self.considered_bonds):
                helix_bases.append(end)
                next_base = self.base_order.at[end, 'next']
                if next_base == self.bonds_df.loc[end].index[0][0]:
                    break
                end = next_base
                try:
                    bond_kind = self.bonds_df.loc[end].kind.values
                except KeyError:
                    break

        bps_diff = len(helix_bases) - expected_length

        if bps_diff != 0:
            if self.verbose:
                print(f'Warning: id {ml} {h_idx} #bps found -',
                      f'#bps in RNALoops =', bps_diff)
            self.helix_length_missmatch[ml].append((h_idx, bps_diff))

        return self.bonds_df.loc[helix_bases]

    def get_dotbracket(self):
        """Get the dot-bracket notation from the bonds_df

        WARNING: Function is untested and might not work correctly!!!

        """
        order = self.get_base_order()
        df = self.bonds_df.copy()

        df[df.type != 'hydrog'] = None
        grouped = df.groupby(level=[0, 1]).count()

        db = ''
        for entry in grouped.groupby(level=0).idxmax().type.values:
            if entry[1] == '-':
                db += '.'
            elif entry[1] not in order.keys():
                continue
            elif (order.loc[entry[0], 'idx'].values[0] <
                  order.loc[entry[1], 'idx'].values[0]):
                db += '('
            else:
                db += ')'

        return db

    def check_structure(self):
        """ Check the structure and return quality of each multiloop:

            0 -> No issues found
            1 -> Helix length missmatch found
            2 -> Strands with canonical bonds found
            3 -> Helix detached from strand found
            4 -> Disconnected strands found
            5 -> Less than 3 helices found (appears not to be a multiloop)
            6 -> Only 0 length strands (multiloop can not be analysed)
            In case of multiple issues only most severe (highest) is reported

        """
        result = {i: set() for i in range(7)}
        for idx in self.multiloops.index:
            result[self._check_ml(idx)].add(idx)
        return result

    def _check_ml(self, idx):
        """Return the quality of a multiloop as an integer (less is better)"""
        if idx in self.ml_without_strands:
            return 6
        if idx in self.ml_less_than_three_helices:
            return 5
        elif len(self.disconnected_strands[idx]) > 0:
            return 4
        elif len(self.detached_helices[idx]) > 0:
            return 3
        elif len(self.strands_with_canonical_bonds[idx]) > 0:
            return 2
        elif len(self.helix_length_missmatch[idx]) > 0:
            return 1
        else:
            return 0


def get_full_structures(idx_min=0, idx_max=0, df=None):
    """Create and save a PdbStructure object for all given indices."""

    if df is None:
        df = load_data('_prepared')

    pdb_ids = df.home_structure.unique()

    for count, pdb_id in enumerate(pdb_ids[idx_min:idx_max + 1]):

        if f'{pdb_id}.pkl' in os.listdir(mypath('STRUCTURES')):
            continue

        print('\n--- Get home structure', pdb_id,
              f'(idx {idx_min + count}) ---')
        try:
            structure = PdbStructure(pdb_id, rnaloops_df=df)
            with open(mypath('STRUCTURES', f'{pdb_id}.pkl'), 'wb') as f:
                pickle.dump(structure, f)
        except Exception as e:
            print(f'WARNING: SKIPPING ID {pdb_id} DUE TO {e}')


def get_full_structures_parallel(idx_min=0, idx_max=1794, n_pools=8):
    """Create and save a PdbStructure object for all given indices
        using Pools in parallel.

    """
    n = idx_max - idx_min
    n_per_pool = n // n_pools + 1

    params = [(idx_min + x * n_per_pool,
               min(idx_min + (x + 1) * n_per_pool - 1, idx_max))
              for x in range(n_pools)]

    with Pool(n_pools) as pool:
        pool.starmap(get_full_structures, params)
        pool.close()


def load_full_structure(pdb_id=None):
    """Load a PdbStructure object with given pdb_id from disk"""
    if pdb_id is None:
        pdb_ids = [x[:-4] for x in os.listdir(mypath('STRUCTURES'))]
        pdb_id = random.choice(pdb_ids)
        print(f'No id specified... choosing random {pdb_id}')
    with open(mypath('STRUCTURES', f'{pdb_id}.pkl'), 'rb') as f:
        structure = pickle.load(f)
    return structure


def check_structures(pdb_ids):
    """Check all given pdb_ids for quality of there multiloops and return
       a dict with qualities as keys and all structure indices of that
       quality as values.

    """
    structure = load_full_structure(pdb_ids[0])
    result = structure.check_structure()
    for pdb_id in pdb_ids:
        structure = load_full_structure(pdb_id)
        current_result = structure.check_structure()
        for key in result.keys():
            result[key].update(current_result[key])
    return result
