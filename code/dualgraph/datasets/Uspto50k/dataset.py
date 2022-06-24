from torch_geometric.data import InMemoryDataset
import pandas as pd
import shutil, os
import os.path as osp
import torch
from dualgraph.dataset import DGData
from dualgraph.mol import smiles2graphwithface
from rdkit import Chem


class USPTO50kDataset(InMemoryDataset):
    def __init__(self, raw_data_folder, root="/tmp/dataset/uspto50k", transform=None, pre_transform=None):
        self.raw_data_folder = raw_data_folder
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    def get_idx_split(self):
        path = os.path.join(self.root, "split", "split_dict.pt")
        return torch.load(path)

    @property
    def task_type(self):
        return 'rank'
    
    @property
    def eval_metric(self):
        return "loss"
    
    @property
    def num_tasks(self):
        return 1
    
    @property
    def raw_file_names(self):
        return ["preprocessed_train.csv", "preprocessed_val.csv", "preprocessed_test.csv"]
    
    @property
    def processed_file_names(self):
        return "uspto50k_processed.pt"
    
    def download(self):
        for idx, name in enumerate(['train', 'val', 'test']):
            shutil.copy(osp.join(self.raw_data_folder, f'preprocessed_{name}.csv'), self.raw_paths[idx])
    
    def process(self):
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        
        
        # df = pd.read_csv(self.raw_paths[0])
        # df = df[df['Mask']==1]
        # dfs = [df[df['Split']=='train'], df[df['Split']=='val'], df[df['Split']=='test']]
        train_df = pd.read_csv(self.raw_paths[0])
        val_df = pd.read_csv(self.raw_paths[1])
        test_df = pd.read_csv(self.raw_paths[2])
        dfs = [train_df, val_df, test_df]

        def get_preprpces_bonds(mol):
            B = []
            for atom in mol.GetAtoms():
                others = []
                bonds = atom.GetBonds()
                for bond in bonds:
                    atoms = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
                    other = [a for a in atoms if a != atom.GetIdx()][0]
                    others.append(other)
                b = [(atom.GetIdx(), other) for other in sorted(others)]
                B += b
            return B

        for insert_idx, df in zip([train_idx, valid_idx, test_idx], dfs):
            smiles_list = df["Products"].values.tolist()
            labels_list = df["Labels"].values.tolist()
            mask_list = (df["Frequency"]>1).values.tolist()
            assert len(smiles_list) == len(labels_list)

            for smiles, labels, mask in zip(smiles_list, labels_list, mask_list):
                data = DGData()
                mol = Chem.MolFromSmiles(smiles)
                graph = smiles2graphwithface(mol)

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_nodes__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

                atom_label = [0]*graph['n_nodes']
                bond_label = [0]*graph['n_edges']
                if mask:
                    preprocss_bonds = get_preprpces_bonds(mol)

                    for l in eval(labels):
                        label_type = l[0]
                        label_idx = l[1]
                        label_template = l[2]
                        if label_type == 'a':
                            atom_label[label_idx] = label_template
                        else:
                            s, t = preprocss_bonds[label_idx]
                            for edge_idx, (ss, tt) in enumerate(zip(*graph['edge_index'])):
                                if ss==s and tt==t:
                                    bond_label[edge_idx] = label_template
                                    break
                            else:
                                raise Exception("Edge not found")
                data.atom_label = torch.as_tensor(atom_label)
                data.bond_label = torch.as_tensor(bond_label)

                data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
                data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
                data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
                data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
                data.num_rings = int(graph["num_rings"])
                data.n_edges = int(graph["n_edges"])
                data.n_nodes = int(graph["n_nodes"])
                data.n_nfs = int(graph["n_nfs"])

                insert_idx.append(len(data_list))
                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

        os.makedirs(osp.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.as_tensor(train_idx, dtype=torch.long),
                "valid": torch.as_tensor(valid_idx, dtype=torch.long),
                "test": torch.as_tensor(test_idx, dtype=torch.long),
            },
            osp.join(self.root, "split", "split_dict.pt"),
        )

class USPTO50kTestDataset(InMemoryDataset):
    def __init__(self, root="/tmp/dataset/uspto50ktest", transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    def get_idx_split(self):
        path = os.path.join(self.root, "split", "split_dict.pt")
        return torch.load(path)

    @property
    def task_type(self):
        return 'classification'
    
    @property
    def eval_metric(self):
        return "loss"
    
    @property
    def num_tasks(self):
        return 1
    
    @property
    def raw_file_names(self):
        return ["raw_test.csv"]
    
    @property
    def processed_file_names(self):
        return "uspto50k_processed.pt"
    
    def download(self):
        shutil.copy('../data/USPTO_50K/raw_test.csv', self.raw_paths[0])
    
    def process(self):

        def canonicalize_rxn(rxn):
            canonicalized_smiles = []
            r, p = rxn.split('>>')
            for s in [r, p]:
                mol = Chem.MolFromSmiles(s)
                [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
                canonicalized_smiles.append(Chem.MolToSmiles(mol))
            return '>>'.join(canonicalized_smiles)
        
        data_list = []
        df = pd.read_csv(self.raw_paths[0])
        rxns = df['reactants>reagents>production'].tolist()
        rxns = [canonicalize_rxn(rxn) for rxn in rxns]
        smiles_list = [rxn.split('>>')[-1] for rxn in rxns]

        for smiles in smiles_list:
            data = DGData()
            mol = Chem.MolFromSmiles(smiles)
            graph = smiles2graphwithface(mol)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

            data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
            data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
            data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
            data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
            data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            data.n_nfs = int(graph["n_nfs"])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])
    