import multiprocessing
import os
from argparse import ArgumentParser
from functools import partial

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions
from rdkit.Chem.rdchem import ChiralType
from tqdm import tqdm

from utils import demap

RDLogger.DisableLog('rdApp.*')

chiral_type_map = {ChiralType.CHI_UNSPECIFIED : -1, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2}
chiral_type_map_inv = {v:k for k, v in chiral_type_map.items()}

a, b = 'a', 'b'

def get_idx_map(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    smiles = Chem.MolToSmiles(mol)
    num_map = {}
    for i, s in enumerate(smiles.split('.')):
        m = Chem.MolFromSmiles(s)
        for atom in m.GetAtoms():
            num_map[atom.GetAtomMapNum()] = atom.GetIdx()
    return num_map

def get_possible_map(pred_site, change_info):
    possible_maps = []
    if type(pred_site) == type(0):
        for edit_type, edits in change_info['edit_site'].items():
            if edit_type not in ['A', 'R']:
                continue
            for edit in edits:
                possible_maps.append({edit: pred_site})
    else:
        for edit_type, edits in change_info['edit_site'].items():
            if edit_type not in ['B', 'C']:
                continue
            for edit in edits:
                possible_maps.append({e:p for e, p in zip(edit, pred_site)})
    return possible_maps

def check_idx_match(mols, possible_maps):
    matched_maps = []
    found_map = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.HasProp('old_mapno') and atom.HasProp('react_atom_idx'):
                found_map[int(atom.GetProp('old_mapno'))] = int(atom.GetProp('react_atom_idx'))
    for possible_map in possible_maps:
        if possible_map.items() <= found_map.items():
            matched_maps.append(found_map)
    return matched_maps

def fix_aromatic(mol):
    for atom in mol.GetAtoms():
        if not atom.IsInRing() and atom.GetIsAromatic():
            atom.SetIsAromatic(False)
        
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            bond.SetIsAromatic(False) 
            if str(bond.GetBondType()) == 'AROMATIC':
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                
def validate_mols(mols):
    for mol in mols:
        if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) == None:
            return False
    return True

def fix_reactant_atoms(product, reactants, matched_map, change_info):
    H_change, C_change, S_change = change_info['change_H'], change_info['change_C'], change_info['change_S']
    fixed_mols = []
    for mol in reactants:
        for atom in mol.GetAtoms():
            if atom.HasProp('old_mapno'):
                mapno = int(atom.GetProp('old_mapno'))
                if mapno not in matched_map:
                    return None
                product_atom = product.GetAtomWithIdx(matched_map[mapno])
                H_before = product_atom.GetNumExplicitHs() + product_atom.GetNumImplicitHs()
                C_before = product_atom.GetFormalCharge()
                S_before = chiral_type_map[product_atom.GetChiralTag()]
                H_after = H_before + H_change[mapno]
                C_after = C_before + C_change[mapno]
                S_after = S_change[mapno]
                if H_after < 0:
                    return None
                atom.SetNumExplicitHs(H_after)
                atom.SetFormalCharge(C_after)
                if S_after != 0:
                    atom.SetChiralTag(chiral_type_map_inv[S_after])
        fix_aromatic(mol)
        fixed_mols.append(mol)
    if validate_mols(fixed_mols):
        return tuple(fixed_mols)
    else:
        return None

def decode_localtemplate(product, pred_site, template, template_info):
    if pred_site == None:
        return None
    possible_maps = get_possible_map(pred_site, template_info)
    reaction = rdChemReactions.ReactionFromSmarts(template)
    reactants = reaction.RunReactants([product])
    decodes = []
    for output in reactants:
        if output == None:
            continue
        matched_maps = check_idx_match(output, possible_maps)
        for matched_map in matched_maps:
            decoded = fix_reactant_atoms(product, output, matched_map, template_info)
            if decoded == None:
                continue
            else:
                return demap(decoded)
    return None

def get_edit_site(mol):
    A = [a for a in range(mol.GetNumAtoms())]
    B = []
    for bond in mol.GetBonds():
        s = bond.GetBeginAtom().GetIdx()
        t = bond.GetEndAtom().GetIdx()
        B.append((s,t))
        B.append((t,s))
    return A, B

def get_idx_map(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    smiles = Chem.MolToSmiles(mol)
    num_map = {}
    for i, s in enumerate(smiles.split('.')):
        m = Chem.MolFromSmiles(s)
        for atom in m.GetAtoms():
            num_map[atom.GetAtomMapNum()] = atom.GetIdx()
    return num_map

def read_prediction(smiles, prediction, atom_templates, bond_templates, template_infos, raw = False):
    mol = Chem.MolFromSmiles(smiles)
    if len(prediction) == 1:
        return mol, None, None, None, 0
    elif raw:
        edit_type, pred_site, pred_template_class, prediction_score = prediction # (edit_type, pred_site, pred_template_class)
    else:
        edit_type, pred_site, pred_template_class, prediction_score = eval(prediction) # (edit_type, pred_site, pred_template_class)
    atoms, bonds = get_edit_site(mol)
    idx_map = get_idx_map(mol)
    if edit_type == 'a':
        pred_site, template = atoms[pred_site], atom_templates[pred_template_class]
        if len(template.split('>>')[0].split('.')) > 1: 
            pred_site = idx_map[pred_site] 
    else:
        pred_site, template = bonds[pred_site], bond_templates[pred_template_class]
        if len(template.split('>>')[0].split('.')) > 1: 
            pred_site= (idx_map[pred_site[0]], idx_map[pred_site[1]])
    
    [atom.SetAtomMapNum(atom.GetIdx()) for atom in mol.GetAtoms()]
    return mol, pred_site, template, template_infos[template], prediction_score


def get_k_predictions(test_id, args):
    raw_prediction = args['raw_predictions'][test_id]
    all_prediction = []
    product = raw_prediction[0]
    predictions = raw_prediction[1:]
    for prediction in predictions:
        mol, pred_site, template, template_info, score = read_prediction(product, prediction, args['atom_templates'], args['bond_templates'], args['template_infos'])
        local_template = '>>'.join(['(%s)' % smarts for smarts in template.split('_')[0].split('>>')])
        try:
            decoded_smiles = decode_localtemplate(mol, pred_site, local_template, template_info)
            if decoded_smiles == None or str((decoded_smiles, score)) in all_prediction:
                continue
        except Exception as e:
            print (e)
            continue
        all_prediction.append(str((decoded_smiles, score)))

        if len (all_prediction) >= args['top_k']:
            break
    return (test_id, all_prediction)

def main(args):
    atom_templates = pd.read_csv('%s/atom_templates.csv' % args['data'])
    bond_templates = pd.read_csv('%s/bond_templates.csv' % args['data'])
    template_infos = pd.read_csv('%s/template_infos.csv' % args['data'])

    args['atom_templates'] = {atom_templates['Class'][i]: atom_templates['Template'][i] for i in atom_templates.index}
    args['bond_templates'] = {bond_templates['Class'][i]: bond_templates['Template'][i] for i in bond_templates.index}
    args['template_infos'] = {template_infos['Template'][i]: {'edit_site': eval(template_infos['edit_site'][i]), 'change_H': eval(template_infos['change_H'][i]), 'change_C': eval(template_infos['change_C'][i]), 'change_S': eval(template_infos['change_S'][i])} for i in template_infos.index}

    prediction_file = args['prediction_file']
    raw_predictions = {}
    with open(prediction_file, 'r') as f:
        for line in f.readlines():
            seps = line.split('\t')
            if seps[0] == 'Test_id':
                continue
            raw_predictions[int(seps[0])] = seps[1:]

    output_path = args['output_path']

    args['raw_predictions'] = raw_predictions
    # multi_processing
    result_dict = {}
    partial_func = partial(get_k_predictions, args = args)
    with multiprocessing.Pool(processes=8) as pool:
        tasks = range(len(raw_predictions))
        for result in tqdm(pool.imap_unordered(partial_func, tasks), total=len(tasks), desc='Decoding LocalRetro predictions'):
            result_dict[result[0]] = result[1]

    with open(output_path, 'w') as f1:
        for i in sorted(result_dict.keys()) :
            all_prediction = result_dict[i]
            f1.write('\t'.join([str(i)] + all_prediction) + '\n')
            print('Decoding LocalRetro predictions %d/%d' % (i, len(raw_predictions)), end='', flush=True)
    print()
       
if __name__ == '__main__':      
    parser = ArgumentParser('Decode Prediction')
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-k', '--top-k', default=50, help='Number of top predictions')
    parser.add_argument('-p', '--prediction-file', type=str)
    parser.add_argument('-o', '--output-path', type=str)
    args = parser.parse_args().__dict__
    print(args)
    main(args)
