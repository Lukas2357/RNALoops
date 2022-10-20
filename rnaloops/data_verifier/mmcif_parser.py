

class PdbStructure:
    
    def __init__(self, pdb_id, rnaloops_df=None):
        
        file_path = f"rnaloops/pdb_data/cif/{pdb_id}.cif"
        if rnaloops_df is None:
            rnaloops_df = get_prepared_df()
        
        self.cif_dict = MMCIF2Dict.MMCIF2Dict(file_path)
        self.order_df = self.get_order_df()
        self.bonds_df = self.get_bonds_df()
        self.multiloops = rnaloops_df[rnaloops_df.home_structure==pdb_id]
        self.ml_strands = self.get_ml_strands()
        self.strands_contain_canonical_bonds = ('WATSON-CRICK' in 
                                                self.ml_strands.kind.values)
        if self.strands_contain_canonical_bonds:
            print('Warning: Found canonical bonds in strands!')
            
        self.dotbracket = self.get_dotbracket()

    def get_bonds_df(self):

        cols = [
            "conn_type_id",
            "ptnr1_auth_comp_id",
            "ptnr2_auth_comp_id",
            "ptnr1_auth_asym_id",
            "ptnr1_auth_seq_id",
            "ptnr2_auth_asym_id",
            "ptnr2_auth_seq_id",
            "ptnr1_label_atom_id",
            "ptnr2_label_atom_id",
            "details",
        ]
        cols = ["_struct_conn." + x for x in cols]

        bonds_dict = {key: value for key, value in self.cif_dict.items() 
                      if key in cols}

        bonds_df = pd.DataFrame(bonds_dict)
        
        bonds_df["pos2"] = bonds_df[cols[5]] + "-" + bonds_df[cols[6]]
        bonds_df["type"] = bonds_df[cols[0]]
        bonds_df["ptnr1"] = bonds_df[cols[1]].str[-1]
        bonds_df["ptnr2"] = bonds_df[cols[2]].str[-1]
        bonds_df["pos1"] = bonds_df[cols[3]] + "-" + bonds_df[cols[4]]
        bonds_df["atom1"] = bonds_df[cols[7]]
        bonds_df["atom2"] = bonds_df[cols[8]]
        bonds_df["kind"] = bonds_df[cols[9]]
        bonds_df = bonds_df.drop(cols, axis=1)
        
        bonds_df = bonds_df.sort_values('pos1')
        idx_cols1 = ['pos1', 'pos2', 'atom1', 'atom2', 'ptnr1', 'ptnr2']
        idx_cols2 = ['pos2', 'pos1', 'atom2', 'atom1', 'ptnr2', 'ptnr1']
        bonds_df = pd.concat([bonds_df.set_index(idx_cols1),
                              bonds_df.set_index(idx_cols2)])
        
        base_order = self.get_base_order()
        
        for base in base_order.keys():
            if base not in bonds_df.index.get_level_values(level=0):
                bonds_df.loc[(base, '-', '-', '-', '-', '-'), :] = None
        
        try:
            bonds_df = bonds_df.reindex(base_order, level=0)
        except ValueError:
            print('Warning: Dublicate index in bonds_df... removing it!')
            bonds_df = bonds_df[~bonds_df.index.duplicated(keep='first')]
            bonds_df = bonds_df.reindex(base_order, level=0)

        return bonds_df

    def get_order_df(self):
        
        cols = [
            "pdbx_strand_id", 
            "pdbx_auth_seq_align_beg", 
            "pdbx_auth_seq_align_end",
        ]
        cols = ["_struct_ref_seq." + x for x in cols]
        
        order_dict = {key: value for key, value in self.cif_dict.items() 
                      if key in cols}
        
        order_df = pd.DataFrame(order_dict)
        
        return order_df   
    
    def get_base_order(self):
        base_order = [row[1].values[0] + '-' + str(idx) 
                      for row in self.order_df.iterrows()
                      for idx in range(int(row[1].values[1]), 
                                       int(row[1].values[2])+1)]
        base_order_dict = {b: idx 
                           for b, idx in zip(base_order, 
                                             range(len(base_order)))}
        return base_order_dict
    
    def get_ml_strands(self):
        
        strands = []
        
        for ml in self.multiloops.iterrows():
            
            ml_strands = []
            
            for idx in range(1, 1 + int(ml[1].loop_type.split('-')[0])):
                start = ml[1][f'start_{idx}']
                end = ml[1][f'end_{idx}']
                if start.split('-')[0] != end.split('-')[0]:
                    print(f'Warning: ID {ml[0]} strand {idx} ({start}|{end})',
                           'seems disconnected!')
                ml_strands.append(self.bonds_df.loc[start:end])
                
            strands.append(pd.concat(ml_strands, keys=[f'strand{i}' 
                                     for i in range(1, 1 + len(ml_strands))]))
                
        return pd.concat(strands, keys=self.multiloops.index)
    
    def get_dotbracket(self):
        
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
            elif order[entry[0]] < order[entry[1]]:
                db += '('
            else:
                db += ')'
        
        return db

    
def get_full_structures(df=None, idx_min=0, idx_max=0):
    
    if df is None:
        df = get_prepared_df()
        
    pdb_ids = df.home_structure.unique()
    structures = {}

    for pdb_id in pdb_ids[idx_min:idx_max+1]:
        print('\n---', pdb_id, '---')
        structures[pdb_id] = PdbStructure(pdb_id, rnaloops_df=df)

    with open(f'pdb_full_structures_{idx_min}-{idx_max}.pkl', 'wb') as f:
        pickle.dump(structures, f)
        
    return structures