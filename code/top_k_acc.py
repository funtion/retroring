import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

from utils import demap

def get_MaxFrag(smiles):
    return max(smiles.split('.'), key=len)

def get_isomers(smi):
    mol = Chem.MolFromSmiles(smi)
    isomers = tuple(EnumerateStereoisomers(mol))
    isomers_smi = [Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers]
    return isomers_smi

def isomer_match(preds, reac):
    reac_isomers = get_isomers(reac)
    for k, pred in enumerate(preds):
        try:
            pred_isomers = get_isomers(pred)
            if set(pred_isomers).issubset(set(reac_isomers)) or set(reac_isomers).issubset(set(pred_isomers)):
                return k+1
        except Exception as e:
            pass
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--result-file', type=str, required=True)
    
    args = parser.parse_args()

    test_file = pd.read_csv('%s/raw_test.csv' % args.data)
    rxn_ps = [rxn.split('>>')[1] for rxn in test_file['reactants>reagents>production']]

    ground_truth = [demap(Chem.MolFromSmiles(rxn.split('>>')[0])) for rxn in test_file['reactants>reagents>production']]
    ground_truth_MaxFrag = [get_MaxFrag(g) for g in ground_truth]

    results = {}       
    results_MaxFrag = {}
    with open(args.result_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.split('\n')[0]
            i = int(line.split('\t')[0])
            predictions = line.split('\t')[1:]
            MaxFrags = []
            results[i] = [eval(p)[0] for p in predictions]
            for p in results[i]:
                if p not in MaxFrags:
                    MaxFrags.append(get_MaxFrag(p))
            results_MaxFrag[i] = MaxFrags

    Exact_matches = []
    MaxFrag_matches = [] # Only compares the largest reactant fragment

    Exact_matches_multi = []
    MaxFrag_matches_multi = [] 
    for i in range(len(results)):
        match_exact = isomer_match(results[i], ground_truth[i])
        # match_exact = smiles_match(results[i], ground_truth[i])
        match_maxfrag = isomer_match(results_MaxFrag[i], ground_truth_MaxFrag[i])
        if len(rxn_ps[i].split('.')) > 1:
            Exact_matches_multi.append(match_exact)
            MaxFrag_matches_multi.append(match_maxfrag)
        Exact_matches.append(match_exact)
        MaxFrag_matches.append(match_maxfrag)
        if i % 100 == 0:
            print('Calculating accuracy... %s/%s' % (i, len(results)), end='', flush=True)

    ks = [1, 3, 5, 10, 50]
    exact_k = {k:0 for k in ks}
    MaxFrag_k = {k:0 for k in ks}

    print(len(Exact_matches))
    for i in range(len(Exact_matches)):
        for k in ks:
            if Exact_matches[i] <= k and Exact_matches[i] != -1:
                exact_k[k] += 1
            if MaxFrag_matches[i] <= k and MaxFrag_matches[i] != -1:
                MaxFrag_k[k] += 1

    for k in ks:
        print ('Top-%d Exact accuracy: %.3f, MaxFrag accuracy: %.3f' % (k, exact_k[k]/len(Exact_matches), MaxFrag_k[k]/len(MaxFrag_matches)))

if __name__ == '__main__':
    main()
