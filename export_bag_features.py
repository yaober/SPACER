#!/usr/bin/env python3
"""
export_bag_features.py

Reads a BagsDataset, performs mean pooling on instances to create bag-level vectors,
maps them to a global reference gene space, and saves the result as a CSV.

Output CSV format (recommended):
sample_id | core_idx | cell_id | dataset | dataset_short | radius | resolution | label | Gene_A | ... | Gene_Z

Where:
- dataset = manifest CSV's 'adata' path (full path)
- dataset_short = parent directory name of adata path (e.g., HumanOvarianCancer)
- sample_id = f"{dataset_short}::{core_idx}"  (global-unique key)

Extra:
- After export, drop genes that are all-zero across ALL exported rows
  (i.e., not present in any dataset's selected gene set).
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset_benchmark import BagsDataset, custom_collate_fn


def ensure_dir(path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def load_all_genes(reference_gene_file: str) -> list:
    df = pd.read_csv(reference_gene_file)
    if "Gene" not in df.columns:
        raise ValueError(
            f"reference_gene_file must contain a column named 'Gene'. Found: {list(df.columns)}"
        )
    return df["Gene"].astype(str).tolist()


def bag_to_vector_mean(
    gene_expressions: torch.Tensor,
    gene_names_local: list,
    gene2idx: dict,
    n_features: int,
) -> np.ndarray:
    """
    Mean pool instance vectors -> Map to global reference vector.

    gene_expressions: (n_instances, n_genes_local)
    gene_names_local: list length n_genes_local
    """
    v_local = gene_expressions.float().mean(dim=0).detach().cpu().numpy()
    x = np.zeros((n_features,), dtype=np.float32)

    for j, g in enumerate(gene_names_local):
        idx = gene2idx.get(str(g), None)
        if idx is not None:
            x[idx] = v_local[j]
    return x


def drop_all_zero_genes(df: pd.DataFrame, gene_cols: list) -> pd.DataFrame:
    """
    Drop gene columns that are all ~0 across all rows.
    This corresponds to genes that never appeared in any bag's local gene list
    (i.e., not selected by any dataset's top genes / curated list).
    """
    if len(gene_cols) == 0:
        return df

    # Convert gene block to numpy for speed
    gene_mat = df[gene_cols].to_numpy(dtype=np.float32, copy=False)

    # keep if any value is non-zero (with float tolerance)
    keep_mask = ~np.all(np.isclose(gene_mat, 0.0), axis=0)
    kept_genes = [g for g, k in zip(gene_cols, keep_mask) if k]
    dropped_genes = [g for g, k in zip(gene_cols, keep_mask) if not k]

    print(f"[Info] Gene columns before filtering: {len(gene_cols)}")
    print(f"[Info] Dropping all-zero gene columns: {len(dropped_genes)}")
    print(f"[Info] Gene columns kept: {len(kept_genes)}")

    # Return df with only kept genes (preserve metadata columns)
    meta_cols = [c for c in df.columns if c not in gene_cols]
    return df[meta_cols + kept_genes]


def main():
    parser = argparse.ArgumentParser(
        description="Export aggregated bag-level tumor expression and labels to CSV."
    )
    parser.add_argument("--data", type=str, required=True, help="Path to the .h5ad or manifest CSV.")
    parser.add_argument("--reference_gene", type=str, required=True, help="CSV with 'Gene' column for global alignment.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV.")

    parser.add_argument("--immune_cell", type=str, default="tcell", help="Immune cell type for dataset filtering.")
    parser.add_argument("--max_instances", type=int, default=500, help="Max instances per bag.")
    parser.add_argument("--n_genes", type=int, default=500, help="Number of variable genes to load.")
    parser.add_argument(
        "--selection",
        type=str,
        default="positive",
        choices=["positive", "negative"],
        help="If 'negative', flips the label (0->1, 1->0).",
    )

    args = parser.parse_args()

    ensure_dir(args.output_csv)

    ref_genes = load_all_genes(args.reference_gene)
    gene2idx = {g: i for i, g in enumerate(ref_genes)}
    n_features = len(ref_genes)

    print(f"[Info] Reference genes loaded: {n_features}")
    print(f"[Info] Input data: {args.data}")

    dataset = BagsDataset(
        args.data,
        immune_cell=args.immune_cell,
        max_instances=args.max_instances,
        n_genes=args.n_genes,
        k=2,  # required by dataset init
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    rows = []
    print(f"[Info] Processing {len(dataset)} batches (each batch has k bags)...")

    for batch in tqdm(loader, desc="Aggregating bags"):
        (
            distances_list,
            gene_expressions_list,
            labels_list,
            core_idxs_list,
            gene_names_list,
            cell_ids_list,
            adata_paths_list,
            dataset_short_list,
            radii_list,
            resolutions_list,
        ) = batch

        n_items = len(gene_expressions_list)
        for i in range(n_items):
            gene_exp = gene_expressions_list[i]
            genes_local = gene_names_list[i]
            label_raw = labels_list[i]
            core_idx = core_idxs_list[i]
            cell_id = cell_ids_list[i]

            adata_path = adata_paths_list[i]
            dataset_short = dataset_short_list[i]
            radius = radii_list[i]
            resolution = resolutions_list[i]

            # dataset column: manifest 'adata' path
            dataset_name = adata_path

            # global unique id
            ds_key = dataset_short if dataset_short not in [None, ""] else "unknown_dataset"
            sample_id = f"{ds_key}::{core_idx}"

            # label
            label_val = float(label_raw.item() if hasattr(label_raw, "item") else label_raw)
            if args.selection == "negative":
                label_val = 1.0 - label_val
            label_final = int(round(label_val))

            # aggregate + map
            bag_vector = bag_to_vector_mean(gene_exp, genes_local, gene2idx, n_features)

            row = [
                sample_id,
                core_idx,
                cell_id,
                dataset_name,
                dataset_short,
                radius,
                resolution,
                label_final,
            ] + bag_vector.tolist()
            rows.append(row)

    print("[Info] Constructing DataFrame...")
    columns = [
        "sample_id",
        "core_idx",
        "cell_id",
        "dataset",
        "dataset_short",
        "radius",
        "resolution",
        "label",
    ] + ref_genes

    df = pd.DataFrame(rows, columns=columns)

    # ---- NEW: drop genes that are all-zero across all rows ----
    df = drop_all_zero_genes(df, ref_genes)

    print(f"[Info] Saving to {args.output_csv} ...")
    df.to_csv(args.output_csv, index=False)
    print(f"[Done] Saved shape: {df.shape}")


if __name__ == "__main__":
    main()
