import os
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from dualgraph.gnn import GNN2, GNNwithvn
from tqdm import tqdm
import argparse
import numpy as np
import random
import pandas as pd
from rdkit import Chem

from dualgraph.datasets.Uspto50k.dataset import USPTO50kTestDataset

def get_id_template(a, class_n):
    class_n = class_n # no template
    edit_idx = a//class_n
    template = a%class_n
    return (edit_idx, template)

def output2edit(out, top_num):
    class_n = out.size(-1)
    readout = out.cpu().detach().numpy()
    readout = readout.reshape(-1)
    output_rank = np.flip(np.argsort(readout))
    output_rank = [r for r in output_rank if get_id_template(r, class_n)[1] != 0][:top_num]
    
    selected_edit = [get_id_template(a, class_n) for a in output_rank]
    selected_proba = [readout[a] for a in output_rank]
     
    return selected_edit, selected_proba
    
def combined_edit(graph, atom_out, bond_out, top_num):
    edit_id_a, edit_proba_a = output2edit(atom_out, top_num)
    edit_id_b, edit_proba_b = output2edit(bond_out, top_num)
    edit_id_c = edit_id_a + edit_id_b
    edit_type_c = ['a'] * top_num + ['b'] * top_num
    edit_proba_c = edit_proba_a + edit_proba_b
    edit_rank_c = np.flip(np.argsort(edit_proba_c))[:top_num]
    edit_type_c = [edit_type_c[r] for r in edit_rank_c]
    edit_id_c = [edit_id_c[r] for r in edit_rank_c]
    edit_proba_c = [edit_proba_c[r] for r in edit_rank_c]
    
    return edit_type_c, edit_id_c, edit_proba_c

def get_bg_partition(bg):
    # sg = bg.remove_self_loop()
    gs = bg.to_data_list()
    nodes_sep = [0]
    edges_sep = [0]
    for g in gs:
        assert not g.has_self_loops()
        nodes_sep.append(nodes_sep[-1] + g.num_nodes)
        edges_sep.append(edges_sep[-1] + g.num_edges)
    return gs, nodes_sep[1:], edges_sep[1:]

def main():
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument("--device", type=int, default=0, help="which gpu to use ")
    parser.add_argument("--gnn", type=str, default="dualgraph2")
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--face-reducer", type=str, default="sum")
    parser.add_argument("--graph-pooling", type=str, default="sum")
    parser.add_argument("--init-face", action="store_true", default=False)
    parser.add_argument("--dropedge-rate", type=float, default=0.1)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--ignore-face", action="store_true", default=False)
    parser.add_argument("--use-global", action="store_true", default=False)
    parser.add_argument("--train-subset", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-dir", type=str, default="", help="tensorboard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--save-test-dir", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropnet", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--use-outer", action="store_true", default=False)
    parser.add_argument("--lr-warmup", action="store_true", default=False)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--parallel", action="store_true", default=False)

    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--from-pretrained", type=str, default="")
    parser.add_argument("--layer-drop", type=float, default=0.0)
    parser.add_argument("--pooler-dropout", type=float, default=0.0)
    parser.add_argument("--use-vn", action="store_true", default=False)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--use-bn", action="store_true", default=False)
    parser.add_argument("--use-adamw", action="store_true", default=False)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--face-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradmultiply", type=float, default=-1)
    parser.add_argument("--ap-hid-size", type=int, default=None)
    parser.add_argument("--ap-mlp-layers", type=int, default=None)
    parser.add_argument("--save-ckt", action="store_true", default=False)
    parser.add_argument("--gra-layers", type=int, default=0)

    parser.add_argument("--test-ckpt", type=str, required=True)
    parser.add_argument("--result-path", type=str, required=True)
    parser.add_argument('--top_num', default=100, help='Num. of predictions to write')
    parser.add_argument("--raw-data-path", type=str)
    parser.add_argument('--rank-logit', action='store_true')
    parser.add_argument("--edge-rep", type=str, default="e") # n, e, f, u

    args = parser.parse_args()
    print(args)

    smiles_path = os.path.join(args.raw_data_path, 'preprocessed_test.csv')
    print("loading smiles from", smiles_path)
    df = pd.read_csv(smiles_path)
    prods = df['Products'].values.tolist()
    smiles_list = []
    for prod in prods:
        mol = Chem.MolFromSmiles(prod)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles_list.append(Chem.MolToSmiles(mol))
    print('load success')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    dataset = USPTO50kTestDataset(args.raw_data_path)

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    shared_params = {
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_layers": args.mlp_layers,
        "latent_size": args.latent_size,
        "use_layer_norm": args.use_layer_norm,
        "num_message_passing_steps": args.num_layers,
        "global_reducer": args.global_reducer,
        "node_reducer": args.node_reducer,
        "face_reducer": args.face_reducer,
        "dropedge_rate": args.dropedge_rate,
        "dropnode_rate": args.dropnode_rate,
        "ignore_globals": not args.use_global,
        "use_face": not args.ignore_face,
        "dropout": args.dropout,
        "dropnet": args.dropnet,
        "init_face": args.init_face,
        "graph_pooling": args.graph_pooling,
        "use_outer": args.use_outer,
        "residual": args.residual,
        "layernorm_before": args.layernorm_before,
        "parallel": args.parallel,
        "num_tasks": dataset.num_tasks,
        "layer_drop": args.layer_drop,
        "pooler_dropout": args.pooler_dropout,
        "encoder_dropout": args.encoder_dropout,
        "use_bn": args.use_bn,
        "node_attn": args.node_attn,
        "face_attn": args.face_attn,
        "global_attn": args.global_attn,
        "gradmultiply": args.gradmultiply,
        "ap_hid_size": args.ap_hid_size,
        "ap_mlp_layers": args.ap_mlp_layers,
        "gra_layers": args.gra_layers,
        "edge_rep": args.edge_rep,
    }
    if args.use_vn:
        model = GNNwithvn(**shared_params).to(device)
    else:
        model = GNN2(**shared_params).to(device)
        
    print(model)
    print('loading from', args.test_ckpt)
    checkpoint = torch.load(args.test_ckpt, map_location=torch.device("cpu"))["model_state_dict"]
    
    model.load_state_dict(checkpoint)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")

    model.eval()

    f = open(args.result_path, 'w')
    f.write('Test_id\tProduct\t%s\n' % '\t'.join(['Prediction %s' % (i+1) for i in range(args.top_num)]))
    for batch_id, batch in enumerate(tqdm(test_loader, desc="Eval")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                atom_pred, bond_pred = model(batch)
                if args.rank_logit:
                    batch_atom_prob = atom_pred
                    batch_bond_prob = bond_pred
                else:
                    batch_atom_prob = F.softmax(atom_pred, dim=-1)
                    batch_bond_prob = F.softmax(bond_pred, dim=-1) 

                graphs, nodes_sep, edges_sep = get_bg_partition(batch)
                start_node = 0
                start_edge = 0
                print('\rWriting test molecule batch %s/%s' % (batch_id, len(test_loader)), end='', flush=True)
                for single_id, (graph, end_node, end_edge) in enumerate(zip(graphs, nodes_sep, edges_sep)):
                    test_id = (batch_id * args.batch_size) + single_id
                    smiles = smiles_list[test_id]
                    pred_types, pred_sites, pred_scores = combined_edit(graph, batch_atom_prob[start_node:end_node], batch_bond_prob[start_edge:end_edge], args.top_num)
                    start_node = end_node
                    start_edge = end_edge
                    f.write('%s\t%s\t%s\n' % (test_id, smiles, '\t'.join(['(%s, %s, %s, %.3f)' % (pred_types[i], pred_sites[i][0], pred_sites[i][1], pred_scores[i]) for i in range(args.top_num)])))

    f.close()

if __name__ == "__main__":
    main()
