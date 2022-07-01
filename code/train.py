import shutil
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from dualgraph.gnn import GNN2, GNNwithvn
import os
from tqdm import tqdm
import argparse
import numpy as np
import random
import io
from dualgraph.utils import WarmCosine, WarmLinear
import json

use_aml = False
try:
    from azureml.core import Run
    use_aml = True
except:
    pass

from dualgraph.datasets.Uspto50k.dataset import USPTO50kDataset

def subsampled_cross_entropy(pred, label, ratio):
    zero_idx = (label == 0)
    templed_pred, template_label = pred[~zero_idx], label[~zero_idx]
    zero_pred, zero_label = pred[zero_idx], label[zero_idx]
    k = int(zero_pred.shape[0] * ratio)
    k = max(k, 1)
    _, zero_label_sample_idx = torch.topk(zero_pred[:, 0], k, largest=False, dim=0)
    if zero_label_sample_idx.nelement() == 0:
        return F.cross_entropy(pred, label, reduction='none')

    full_pred = torch.cat((templed_pred, zero_pred[zero_label_sample_idx]), dim=0)
    full_label = torch.cat((template_label, zero_label[zero_label_sample_idx]), dim=0)
    return F.cross_entropy(full_pred, full_label, reduction='none')


def train(model, device, loader, optimizer, task_type, scheduler, args, use_aml=False):
    model.train()
    loss_accum = 0

    pbar = tqdm(loader, desc="Train")
    for step, batch in enumerate(pbar):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            atom_pred, bond_pred = model(batch)
            optimizer.zero_grad()

            if args.neg_sample_ratio == 0:
                atom_loss = F.cross_entropy(atom_pred, batch.atom_label.long(), reduction='none')
                bond_loss = F.cross_entropy(bond_pred, batch.bond_label.long(), reduction='none')
            else:
                atom_loss = subsampled_cross_entropy(atom_pred, batch.atom_label.long(), args.neg_sample_ratio)
                bond_loss = subsampled_cross_entropy(bond_pred, batch.bond_label.long(), args.neg_sample_ratio)

            loss = torch.cat((atom_loss, bond_loss), dim=0).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_accum += loss.detach().item()
            if step % args.log_interval == 0:
                pbar.set_description(
                    "Iteration loss: {:6.4f} lr: {:.5e}".format(
                        loss_accum / (step + 1), scheduler.get_last_lr()[0]
                    )
                )

                if use_aml:
                    Run.get_context().log("iter loss", loss_accum / (step + 1))
                    Run.get_context().log('lr', scheduler.get_last_lr()[0])


def eval(model, device, loader, evaluator):
    model.eval()

    total = 0
    correct = 0
    loss = 0

    for step, batch in enumerate(tqdm(loader, desc="Eval")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                atom_pred, bond_pred = model(batch)
                total += atom_pred.shape[0] + bond_pred.shape[0]
                atom_pred_idx = atom_pred.argmax(-1)
                bond_pred_idx = bond_pred.argmax(-1)
                correct += (atom_pred_idx == batch.atom_label).sum().item() + (bond_pred_idx == batch.bond_label).sum().item()
                
                atom_loss = F.cross_entropy(atom_pred, batch.atom_label.long(), reduction='sum')
                bond_loss = F.cross_entropy(bond_pred, batch.bond_label.long(), reduction='sum')
                loss += atom_loss.item() + bond_loss.item()

    return {'acc': correct / total, 'loss': loss/total}


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
    parser.add_argument("--gra-layers", type=int, default=0)
    parser.add_argument("--ap-hid-size", type=int, default=None)
    parser.add_argument("--ap-mlp-layers", type=int, default=None)
    parser.add_argument("--save-ckt", action="store_true", default=False)
    parser.add_argument("--raw-data-path", type=str)
    parser.add_argument("--neg-sample-ratio", type=float, default=1.0)
    parser.add_argument("--edge-rep", type=str, default="e") # n, e, f, u

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # if args.dataset.startswith("dc-"):
    #     dataset = DCGraphPropPredDataset(args.dataset)
    # else:
    #     dataset = DGPygGraphPropPredDataset(name=args.dataset)
    dataset = USPTO50kDataset(args.raw_data_path)
    split_idx = dataset.get_idx_split()

    evaluator = None # Evaluator(args.dataset, dataset=dataset)

    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
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
    if args.from_pretrained:
        assert os.path.exists(args.from_pretrained)
        checkpoint = torch.load(args.from_pretrained, map_location=torch.device("cpu"))[
            "model_state_dict"
        ]
        keys_to_delete = []
        for k, v in checkpoint.items():
            if "decoder" in k or k.startswith("gnn_layers.11.face_model") or k.startswith("gnn_layers.11.global_model"):
                keys_to_delete.append(k)
        for k in keys_to_delete:
            print(f"delete {k} from pre-trained checkpoint...")
            del checkpoint[k]

        for k, v in model.state_dict().items():
            if k not in checkpoint:
                print(f"randomly init {k}...")
                checkpoint[k] = v
        model.load_state_dict(checkpoint)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")
    if args.use_adamw:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay
        )
    # if args.from_pretrained:
    #     lrscheduler = WarmLinear(
    #         tmax=len(train_loader) * args.period, warmup=len(train_loader) * args.period * 0.06
    #     )
    #     scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))
    # elif not args.lr_warmup:
    if not args.lr_warmup:
        lrscheduler = WarmLinear(
            tmax=len(train_loader) * args.epochs, warmup=len(train_loader) * args.epochs * 0.06
        )
    else:
        if args.dataset.startswith("dc-"):
            warmup_step = len(train_loader) * 4
        else:
            warmup_step = int(4e3)
        print('warmup step', warmup_step)
        lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=warmup_step)
    scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, optimizer, dataset.task_type, scheduler, args, use_aml)

        print("Evaluating...")
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)
        if args.checkpoint_dir:
            print(f"settings {os.path.basename(args.checkpoint_dir)}...")
        print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})

        if use_aml:
            Run.get_context().log('train/acc', train_perf['acc'])
            Run.get_context().log('train/loss', train_perf['loss'])
            Run.get_context().log('val/acc', valid_perf['acc'])
            Run.get_context().log('val/loss', valid_perf['loss'])
            Run.get_context().log('test/acc', test_perf['acc'])
            Run.get_context().log('test/loss', test_perf['loss'])


        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if args.checkpoint_dir:
            logs = {
                "Train": train_perf[dataset.eval_metric],
                "Validation": valid_perf[dataset.eval_metric],
                "Test": test_perf[dataset.eval_metric],
            }
            with io.open(
                os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)
            if args.save_ckt:
                print("Saving checkpoint to {}...".format(args.checkpoint_dir))
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                }
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))

    if "classification" in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print("Finished training!")
    print("Best validation score: {}".format(valid_curve[best_val_epoch]))
    print("Test score: {}".format(test_curve[best_val_epoch]))

    shutil.copy(os.path.join(args.checkpoint_dir, f"checkpoint_{best_val_epoch+1}.pt"), os.path.join(args.checkpoint_dir, f"checkpoint_best.pt"))

if __name__ == "__main__":
    main()
