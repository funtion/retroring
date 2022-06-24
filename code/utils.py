from rdkit import Chem

def demap(mols, stereo = True):
    if type(mols) == type((0, 0)):
        ss = []
        for mol in mols:
            [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, stereo))
            if mol == None:
                return None
            ss.append(Chem.MolToSmiles(mol))
        return '.'.join(sorted(ss))
    else:
        [atom.SetAtomMapNum(0) for atom in mols.GetAtoms()]
        return '.'.join(sorted(Chem.MolToSmiles(mols, stereo).split('.')))
