"""
train_nc.py  –  MIL training with negative-control permutations.

Negative control types (--nc_type):
  gene_perm          : permute each gene's expression across all cells
  coord_perm         : randomly reassign spatial (X, Y) coordinates
  label_perm         : shuffle immune-cell infiltration labels across all cells
  intra_comp_shuffle : shuffle expression rows within each cell-type compartment

All other arguments are identical to train.py.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from model.dataset import BagsDataset, custom_collate_fn
from model.model import MIL, EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

NC_TYPES = ('gene_perm', 'coord_perm', 'label_perm', 'intra_comp_shuffle')


def load_all_genes(reference_gene_file):
    all_genes = pd.read_csv(reference_gene_file)
    return all_genes['Gene'].values.tolist()


def save_metrics(epoch, train_loss, val_loss, val_auroc, a, b, alpha, beta, output_dir):
    file_path = os.path.join(output_dir, 'training_metrics.csv')
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('Epoch,Train Loss,Val Loss,Val AUROC,a,b,alpha,beta\n')
    with open(file_path, 'a') as f:
        f.write(f'{epoch},{train_loss},{val_loss},{val_auroc},{a},{b},{alpha},{beta}\n')


def save_spacer_scores(epoch, all_genes, scores_before, scores_after, output_dir):
    df = pd.DataFrame({
        'Gene': all_genes,
        'SPACER Score Before Training': scores_before,
        'SPACER Score After Training': scores_after,
    })
    df['Difference'] = df['SPACER Score After Training'] - df['SPACER Score Before Training']
    df = df.sort_values(by='Difference', ascending=False)
    df.to_csv(os.path.join(output_dir, f'spacer_score_changes_epoch_{epoch+1}.csv'), index=False)


def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    else:
        print(f"Using device: {device}")
    print(f"Negative control type: {args.nc_type}")
    print("=====================================")

    all_genes = load_all_genes(args.reference_gene)
    print('Reference genes loaded:', len(all_genes))
    print("=====================================")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save run config for provenance
    with open(os.path.join(args.output_dir, 'run_config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    model = MIL(all_genes).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)

    dataset = BagsDataset(
        args.data,
        immune_cell=args.immune_cell,
        max_instances=args.max_instances,
        n_genes=args.n_genes,
        k=2,
        nc_type=args.nc_type,
    )

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    spacer_scores_before = [s.item() for s in model.immunogenicity.ig.clone().detach().cpu()]

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        all_outputs, all_labels = [], []

        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_data in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
                optimizer.zero_grad()

                distances_list, gene_expressions_list, labels_list, _, gene_names_list, _ = batch_data
                distances_list = [d.to(device) for d in distances_list]
                gene_expressions_list = [g.to(device) for g in gene_expressions_list]
                labels = torch.stack(labels_list).float().to(device)

                outputs = model(distances_list, gene_expressions_list, gene_names_list)
                if outputs is None or outputs.shape[0] != labels.shape[0]:
                    continue

                if args.selection == 'negative':
                    labels = 1 - labels
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                all_outputs.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        try:
            epoch_auc = roc_auc_score(all_labels, all_outputs)
        except ValueError:
            epoch_auc = float('nan')
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {train_loss:.4f}, AUROC: {epoch_auc:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        val_outputs, val_labels_all = [], []
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as vtepoch:
                for val_batch in vtepoch:
                    vd, vg, vl, _, vgn, _ = val_batch
                    vd = [d.to(device) for d in vd]
                    vg = [g.to(device) for g in vg]
                    vlabels = torch.stack(vl).float().to(device)

                    vout = model(vd, vg, vgn)
                    if vout is None or vout.shape[0] != vlabels.shape[0]:
                        continue

                    if args.selection == 'negative':
                        vlabels = 1 - vlabels
                    vloss = criterion(vout, vlabels)
                    val_loss += vloss.item()
                    vtepoch.set_postfix(val_loss=vloss.item())
                    val_outputs.extend(vout.detach().cpu().numpy())
                    val_labels_all.extend(vlabels.cpu().numpy())

            val_loss /= len(val_loader)
            try:
                val_auc = roc_auc_score(val_labels_all, val_outputs)
            except ValueError:
                val_auc = float('nan')
            print(f'Validation Loss: {val_loss:.4f}, Validation AUROC: {val_auc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss {val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))

        a = model.distance.a.clone().detach().cpu().numpy()
        b = model.gene_expression.b.clone().detach().cpu()
        alpha = model.alpha.clone().detach().cpu()
        beta = model.beta.clone().detach().cpu()
        save_metrics(epoch+1, train_loss, val_loss, val_auc, a, b, alpha, beta, args.output_dir)

        spacer_scores_after = [s.item() for s in model.immunogenicity.ig.clone().detach().cpu()]
        save_spacer_scores(epoch, all_genes, spacer_scores_before, spacer_scores_after, args.output_dir)

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

    # Write a one-line summary for easy collection
    try:
        metrics = pd.read_csv(os.path.join(args.output_dir, 'training_metrics.csv'))
        best_row = metrics.loc[metrics['Val Loss'].idxmin()]
        summary = {
            'nc_type': args.nc_type,
            'immune_cell': args.immune_cell,
            'best_epoch': int(best_row['Epoch']),
            'best_val_loss': float(best_row['Val Loss']),
            'best_val_auroc': float(best_row['Val AUROC']),
            'final_val_auroc': float(metrics['Val AUROC'].iloc[-1]),
        }
        pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
        print("\n=== Run Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Train MIL with a negative-control permutation to validate model specificity.'
    )
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--reference_gene', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--immune_cell', type=str, default='tcell')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--delta', type=float, default=0.001)
    parser.add_argument('--max_instances', type=int, default=None)
    parser.add_argument('--n_genes', type=int, default=10000)
    parser.add_argument('--selection', type=str, default='positive',
                        help="'positive' keeps labels as-is; 'negative' inverts them.")
    parser.add_argument('--nc_type', type=str, required=True, choices=list(NC_TYPES),
                        help='Negative control type: ' + ', '.join(NC_TYPES))

    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()
