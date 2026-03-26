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

def load_all_genes(reference_gene_file):
    all_genes = pd.read_csv(reference_gene_file)
    return all_genes['Gene'].values.tolist()

GLOBAL_PARAM_KEYS = ("immunogenicity.ig",)

def _get_global_state(model: torch.nn.Module) -> dict:
    sd = model.state_dict()
    return {k: sd[k].detach().clone() for k in GLOBAL_PARAM_KEYS if k in sd}

def _load_global_state_(model: torch.nn.Module, global_state: dict) -> None:
    model_sd = model.state_dict()
    for k, v in global_state.items():
        if k in model_sd:
            model_sd[k].copy_(v.to(model_sd[k].device))

def _fedavg_global_states(client_global_states: list[dict], client_weights: list[float]) -> dict:
    if len(client_global_states) == 0:
        raise ValueError("No client states provided for aggregation.")
    if len(client_global_states) != len(client_weights):
        raise ValueError("client_global_states and client_weights must have same length.")
    total_weight = float(sum(client_weights))
    if total_weight <= 0:
        raise ValueError("Sum of client_weights must be > 0.")

    out = {}
    for k in client_global_states[0].keys():
        acc = None
        for st, w in zip(client_global_states, client_weights):
            t = st[k].detach().float().cpu() * float(w)
            acc = t if acc is None else acc + t
        out[k] = (acc / total_weight).to(client_global_states[0][k].dtype)
    return out

def _local_train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    selection: str,
    fedprox_mu: float,
    global_ref_state: dict | None,
) -> float:
    model.train()
    running_loss = 0.0
    with tqdm(loader, unit="batch", leave=False) as tepoch:
        for batch_data in tepoch:
            optimizer.zero_grad()

            distances_list, gene_expressions_list, labels_list, core_idxs_list, gene_names_list, cell_ids_list = batch_data
            distances_list = [distances.to(device) for distances in distances_list]
            gene_expressions_list = [gene_exp.to(device) for gene_exp in gene_expressions_list]
            labels = torch.stack(labels_list).float().to(device)
            current_genes_list = gene_names_list

            outputs = model(distances_list, gene_expressions_list, current_genes_list)
            if outputs is None or outputs.shape[0] != labels.shape[0]:
                continue

            if selection == 'negative':
                labels = 1 - labels

            loss = criterion(outputs, labels)

            if fedprox_mu > 0 and global_ref_state is not None:
                prox = 0.0
                for k in GLOBAL_PARAM_KEYS:
                    if k not in global_ref_state:
                        continue
                    cur = model.state_dict()[k]
                    ref = global_ref_state[k].to(cur.device)
                    prox = prox + (cur - ref).pow(2).sum()
                loss = loss + 0.5 * float(fedprox_mu) * prox

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            tepoch.set_postfix(loss=float(loss.item()))
    return running_loss / max(1, len(loader))

def save_metrics(epoch, train_loss, val_loss, val_auroc, a, b, alpha, beta, output_dir):
    file_path = os.path.join(output_dir, 'training_metrics.csv')
    if not os.path.exists(file_path):
        # Create the CSV file with headers
        with open(file_path, 'w') as f:
            f.write('Epoch,Train Loss,Val Loss,Val AUROC,a,b,alpha,beta\n')
    
    # Append metrics for the current epoch
    with open(file_path, 'a') as f:
        f.write(f'{epoch},{train_loss},{val_loss},{val_auroc},{a},{b},{alpha},{beta}\n')

def save_spacer_scores(epoch, all_genes, spacer_scores_before_training, spacer_scores_after_training, output_dir):
    # Create a DataFrame with SPACER Scores before and after the current epoch
    spacer_score_data = {
        'Gene': all_genes,
        'SPACER Score Before Training': spacer_scores_before_training,
        'SPACER Score After Training': spacer_scores_after_training,
    }
    df = pd.DataFrame(spacer_score_data)
    
    # Calculate the difference and add it as a new column
    df['Difference'] = df['SPACER Score After Training'] - df['SPACER Score Before Training']
    df = df.sort_values(by='Difference', ascending=False)

    # Save to a CSV file for each epoch
    output_path = os.path.join(output_dir, f'spacer_score_changes_epoch_{epoch+1}.csv')
    df.to_csv(output_path, index=False)

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    else:
        print(f"Using device: {device}")
    print("=====================================")

    all_genes = load_all_genes(args.reference_gene)
    print('Reference genes loaded:', len(all_genes))
    print("=====================================")
    
    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.training_mode == "centralized":
        # Initialize the model, criterion, optimizer, and early stopping
        model = MIL(all_genes, gene_weighting=args.gene_weighting).to(device)
        criterion = nn.BCELoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)

        # Load dataset and create DataLoader
        dataset = BagsDataset(
            args.data,
            immune_cell=args.immune_cell,
            max_instances=args.max_instances,
            n_genes=args.n_genes,
            k=2
        )
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

        best_val_loss = float('inf')
        best_model_path = os.path.join(args.output_dir, 'best_model.pth')

        spacer_scores_before_training = model.immunogenicity.ig.clone().detach().cpu()
        spacer_scores_before_training = [score.item() for score in spacer_scores_before_training]

        for epoch in range(args.num_epochs):
            model.train()
            running_loss = 0.0

            all_outputs = []
            all_labels = []

            with tqdm(train_loader, unit="batch") as tepoch:
                for i, batch_data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
                    optimizer.zero_grad()

                    distances_list, gene_expressions_list, labels_list, core_idxs_list, gene_names_list, cell_ids_list = batch_data
                    distances_list = [distances.to(device) for distances in distances_list]
                    gene_expressions_list = [gene_exp.to(device) for gene_exp in gene_expressions_list]
                    labels = torch.stack(labels_list).float().to(device)
                    current_genes_list = gene_names_list

                    outputs = model(distances_list, gene_expressions_list, current_genes_list)
                    if outputs is None:
                        continue
                    if outputs.shape[0] != labels.shape[0]:
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

            model.eval()
            val_loss = 0.0
            val_all_outputs = []
            val_all_labels = []
            with torch.no_grad():
                with tqdm(val_loader, unit="batch") as vtepoch:
                    for val_batch_data in vtepoch:
                        val_distances_list, val_gene_expressions_list, val_labels_list, val_core_idxs_list, val_gene_names_list, val_cell_ids_list = val_batch_data
                        val_distances_list = [distances.to(device) for distances in val_distances_list]
                        val_gene_expressions_list = [gene_exp.to(device) for gene_exp in val_gene_expressions_list]
                        val_labels = torch.stack(val_labels_list).float().to(device)
                        val_current_genes_list = val_gene_names_list

                        val_outputs = model(val_distances_list, val_gene_expressions_list, val_current_genes_list)
                        if val_outputs is None:
                            continue
                        if val_outputs.shape[0] != val_labels.shape[0]:
                            continue

                        if args.selection == 'negative':
                            val_labels = 1 - val_labels
                        loss = criterion(val_outputs, val_labels)
                        val_loss += loss.item()
                        vtepoch.set_postfix(val_loss=loss.item())

                        val_all_outputs.extend(val_outputs.detach().cpu().numpy())
                        val_all_labels.extend(val_labels.cpu().numpy())

                val_loss /= len(val_loader)

                try:
                    val_epoch_auc = roc_auc_score(val_all_labels, val_all_outputs)
                except ValueError:
                    val_epoch_auc = float('nan')

                print(f'Validation Loss: {val_loss:.4f}, Validation AUROC: {val_epoch_auc:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with validation loss {val_loss:.4f}")

            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))

            a = model.distance.a.clone().detach().cpu().numpy()
            b = model.gene_expression.b.clone().detach().cpu()
            alpha = model.alpha.clone().detach().cpu()
            beta = model.beta.clone().detach().cpu()
            save_metrics(epoch+1, train_loss, val_loss, val_epoch_auc, a, b, alpha, beta, args.output_dir)

            spacer_scores_after_training = model.immunogenicity.ig.clone().detach().cpu()
            spacer_scores_after_training = [score.item() for score in spacer_scores_after_training]
            save_spacer_scores(epoch, all_genes, spacer_scores_before_training, spacer_scores_after_training, args.output_dir)

        torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
        return

    if args.training_mode == "federated":
        if args.client_data is None or len(args.client_data) == 0:
            raise ValueError("--client_data is required when --training_mode federated")

        criterion = nn.BCELoss().to(device)

        client_datasets = []
        client_loaders = []
        client_weights = []
        for client_path in args.client_data:
            ds = BagsDataset(
                client_path,
                immune_cell=args.immune_cell,
                max_instances=args.max_instances,
                n_genes=args.n_genes,
                k=2
            )
            loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
            client_datasets.append(ds)
            client_loaders.append(loader)
            client_weights.append(float(len(ds)))

        # Global model holds the authoritative global state (S).
        global_model = MIL(all_genes, gene_weighting=args.gene_weighting).to(device)
        global_state = _get_global_state(global_model)

        for rnd in range(args.comm_rounds):
            print(f"=== Communication round {rnd+1}/{args.comm_rounds} ===")

            client_global_updates = []
            for client_idx, loader in enumerate(client_loaders):
                # Each client gets its own private params (a,b,alpha,beta), but starts with current global S.
                client_model = MIL(all_genes, gene_weighting=args.gene_weighting).to(device)
                _load_global_state_(client_model, global_state)

                optimizer = optim.AdamW(client_model.parameters(), lr=args.learning_rate, weight_decay=0.01)

                # FedProx reference is the broadcast global state (detached).
                global_ref_state = {k: v.detach().clone().cpu() for k, v in global_state.items()}
                for local_ep in range(args.local_epochs):
                    _local_train_one_epoch(
                        model=client_model,
                        loader=loader,
                        device=device,
                        optimizer=optimizer,
                        criterion=criterion,
                        selection=args.selection,
                        fedprox_mu=float(args.fedprox_mu),
                        global_ref_state=global_ref_state,
                    )

                client_global_updates.append(_get_global_state(client_model))
                print(f"Client {client_idx+1}/{len(client_loaders)} done (bags={len(client_datasets[client_idx])}).")

            global_state = _fedavg_global_states(client_global_updates, client_weights)
            _load_global_state_(global_model, global_state)

            if args.save_global_each_round:
                torch.save(global_model.state_dict(), os.path.join(args.output_dir, f'global_model_round_{rnd+1}.pth'))

        torch.save(global_model.state_dict(), os.path.join(args.output_dir, 'final_global_model.pth'))
        return

    raise ValueError(f"Unknown training_mode: {args.training_mode}")

def main():
    parser = argparse.ArgumentParser(description='Train a MIL model for gene expression and immunogenicity.')
    parser.add_argument('--training_mode', type=str, default='centralized', choices=['centralized', 'federated'],
                        help='Training mode: centralized (single dataset) or federated (FedAvg + FedProx on global params).')
    parser.add_argument('--data', type=str, required=False, help='Path to the training data file (centralized mode).')
    parser.add_argument('--client_data', type=str, nargs='+', required=False,
                        help='List of client CSVs (one per node) for federated mode.')
    parser.add_argument('--reference_gene', type=str, required=True, help='Path to the reference gene CSV file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--immune_cell', type=str, default='tcell', help='Type of immune cell to consider.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train the model.')
    parser.add_argument('--comm_rounds', type=int, default=50, help='Number of communication rounds (federated mode).')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs per round (federated mode).')
    parser.add_argument('--fedprox_mu', type=float, default=0.0,
                        help='FedProx proximal strength (0 disables). Applies to global params only.')
    parser.add_argument('--save_global_each_round', action='store_true',
                        help='If set, saves `global_model_round_{t}.pth` each round (federated mode).')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--delta', type=float, default=0.001, help='Minimum change to qualify as an improvement.')
    parser.add_argument('--max_instances', type=int, default=None, help='Maximum instances for the dataset.')
    parser.add_argument('--n_genes', type=int, default=10000, help='Number of genes to consider.')
    parser.add_argument('--selection', type=str, default='positive', help='Selection of positive or negative samples.')
    parser.add_argument('--gene_weighting', type=str, default='softmax', choices=['softmax', 'sparsemax'],
                        help='How to normalize gene-expression weights across genes.')
    
    args = parser.parse_args()
    if args.training_mode == "centralized" and not args.data:
        parser.error("--data is required when --training_mode centralized")
    train_model(args)

if __name__ == '__main__':
    main()
