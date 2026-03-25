import argparse
import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import BagsDataset, custom_collate_fn
from model.model import MIL


def _infer_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_global_ig(local_model: MIL, global_ig: torch.Tensor) -> None:
    # Only broadcast the shared immunogenicity vector.
    with torch.no_grad():
        local_model.immunogenicity.ig.copy_(global_ig)


def _compute_bag_count(dataset) -> int:
    """
    FedAvg weighting uses the number of bags (instances) per node.
    In this codebase, a dataset item corresponds to a "batch" containing k bags.
    """
    if hasattr(dataset, "batches"):
        return sum(len(batch) for batch in dataset.batches)
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "batches"):
        # torch.utils.data.Subset
        base = dataset.dataset
        idxs = dataset.indices
        return sum(len(base.batches[i]) for i in idxs)
    raise TypeError(f"Unsupported dataset type for bag counting: {type(dataset)}")


def _proximal_term_ig(local_model: MIL, global_ig: torch.Tensor) -> torch.Tensor:
    # FedProx proximal term on shared parameters only.
    # mu * ||Sp - S||_2^2 = mu * sum_g (ig_p[g] - ig[g])^2
    return (local_model.immunogenicity.ig - global_ig).pow(2).sum()


def _make_loaders_for_node(
    data_csv: str,
    immune_cell: str,
    max_instances: int,
    n_genes: int,
    k: int,
    batch_size: int,
    num_workers: int,
):
    dataset = BagsDataset(
        data_csv,
        immune_cell=immune_cell,
        max_instances=max_instances,
        n_genes=n_genes,
        k=k,
    )
    bag_count = _compute_bag_count(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
    )
    return dataset, bag_count, loader


def main():
    parser = argparse.ArgumentParser(description="FedAvg + FedProx for SPACER MIL")
    parser.add_argument(
        "--nodes",
        type=str,
        required=True,
        help="Comma-separated list of per-node CSV paths (one SRT source per node).",
    )
    parser.add_argument("--reference_gene", type=str, required=True, help="Path to reference gene CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save federated checkpoints.")
    parser.add_argument("--immune_cell", type=str, default="tcell", help="Immune cell type to consider.")

    # Federated parameters
    parser.add_argument("--rounds", type=int, default=10, help="Number of communication rounds T.")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs U per round.")
    parser.add_argument("--mu", type=float, default=0.0, help="FedProx proximal strength on shared ig.")

    # Optim/training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Client learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Mini-batch size (bags are still inside a sample).")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--selection", type=str, default="positive", choices=["positive", "negative"])

    # Dataset/model parameters
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--n_genes", type=int, default=10000)
    parser.add_argument("--k", type=int, default=2, help="Number of bags per batch (see existing training code).")
    parser.add_argument("--gene_weighting", type=str, default="softmax", choices=["softmax", "sparsemax"], help="Gene weighting gate within each bag.")

    args = parser.parse_args()

    device = _infer_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # Existing `custom_collate_fn` assumes each DataLoader batch contains exactly one dataset item.
    if args.batch_size != 1:
        raise ValueError("--batch_size must be 1 to match `custom_collate_fn` and MIL bag batching.")

    node_csvs = [x.strip() for x in args.nodes.split(",") if x.strip()]
    if len(node_csvs) < 2:
        raise ValueError("Federated training expects at least 2 nodes (comma-separate multiple CSV paths).")

    # Load reference genes -> shared parameter size.
    import pandas as pd

    all_genes = pd.read_csv(args.reference_gene)["Gene"].values.tolist()

    # Server global model keeps only ig; private parameters are node-local.
    server_model = MIL(all_genes, gene_weighting=args.gene_weighting).to(device)
    global_ig = server_model.immunogenicity.ig.detach().clone().to(device)

    # Initialize persistent client models (private params persist across rounds).
    client_models: List[MIL] = []
    client_loaders: List[DataLoader] = []
    client_bag_counts: List[int] = []

    for node_idx, node_csv in enumerate(node_csvs):
        dataset, bag_count, loader = _make_loaders_for_node(
            data_csv=node_csv,
            immune_cell=args.immune_cell,
            max_instances=args.max_instances,
            n_genes=args.n_genes,
            k=args.k,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        # Each client gets its own model instance (private parameters persist).
        local_model = MIL(all_genes, gene_weighting=args.gene_weighting).to(device)
        _set_global_ig(local_model, global_ig)

        client_models.append(local_model)
        client_loaders.append(loader)
        client_bag_counts.append(bag_count)

        # Save config mapping for reproducibility.
        with open(os.path.join(args.output_dir, "federated_nodes.txt"), "a") as f:
            f.write(f"node_{node_idx}\t{node_csv}\tbag_count={bag_count}\n")

    total_bags = sum(client_bag_counts)
    if total_bags <= 0:
        raise ValueError("Total bag count across nodes is 0; check input datasets.")

    # Communication rounds
    for t in range(1, args.rounds + 1):
        print(f"\n=== Round {t}/{args.rounds} ===")

        # Broadcast shared global parameters S -> each client.
        global_ig_detached = global_ig.detach().clone()
        for local_model in client_models:
            _set_global_ig(local_model, global_ig_detached.to(device))

        client_ig_updates: List[torch.Tensor] = []

        for p, (local_model, loader) in enumerate(zip(client_models, client_loaders)):
            print(f"[Client {p}] local training...")
            # Use a snapshot of S for the proximal term within this round.
            global_ig_snapshot = global_ig_detached.to(device)

            # Reset optimizer each local epoch schedule; private params persist.
            # (If you want to persist optimizer state across rounds, say so.)
            local_model.train()
            optimizer = optim.AdamW(local_model.parameters(), lr=args.learning_rate, weight_decay=0.01)
            criterion = nn.BCELoss().to(device)

            for _ in range(args.local_epochs):
                for batch_data in tqdm(loader, unit="batch", leave=False):
                    (
                        distances_list,
                        gene_expressions_list,
                        labels_list,
                        _core_idxs_list,
                        gene_names_list,
                        _cell_ids_list,
                    ) = batch_data

                    distances_list = [distances.to(device) for distances in distances_list]
                    gene_expressions_list = [gene_exp.to(device) for gene_exp in gene_expressions_list]
                    labels = torch.stack(labels_list).float().to(device)
                    current_genes_list = gene_names_list

                    if args.selection == "negative":
                        labels = 1 - labels

                    outputs = local_model(distances_list, gene_expressions_list, current_genes_list)
                    if outputs is None:
                        continue
                    if outputs.shape[0] != labels.shape[0]:
                        continue

                    bce_loss = criterion(outputs, labels)
                    if args.mu > 0:
                        prox = _proximal_term_ig(local_model, global_ig_snapshot)
                        loss = bce_loss + args.mu * prox
                    else:
                        loss = bce_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            client_ig_updates.append(local_model.immunogenicity.ig.detach().clone())

        # Weighted aggregation: S <- sum_p n_p * S_p / sum_p n_p
        weighted_sum = torch.zeros_like(global_ig)
        for p, ig_p in enumerate(client_ig_updates):
            weighted_sum += client_bag_counts[p] * ig_p
        global_ig = (weighted_sum / float(total_bags)).to(device)

        # Save checkpoints
        torch.save(global_ig.detach().cpu(), os.path.join(args.output_dir, f"global_ig_round_{t}.pt"))
        if args.rounds == t:
            for p, local_model in enumerate(client_models):
                torch.save(
                    local_model.state_dict(),
                    os.path.join(args.output_dir, f"client_{p}_final_model.pth"),
                )

        print(f"Updated global_ig (L2 norm): {global_ig.norm().item():.4f}")


if __name__ == "__main__":
    main()

