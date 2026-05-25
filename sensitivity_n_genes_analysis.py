#!/usr/bin/env python3
"""
Top-G Genes Sensitivity Analysis for SPACER
============================================
Reviewer concern: the model relies on the top-G highly expressed genes
selected per dataset, but the choice G=500 is not empirically justified.

This script sweeps over a user-supplied list of G values (e.g. 50, 100,
200, 300, 500, 750, 1000), trains a fresh MIL model for each, and
measures how sensitive the gene-level SPACER recruitment scores are to
this hyperparameter.

Outputs (written to --output_dir):
  n_genes_metrics.csv                   -- per-run stability metrics
  n_genes_summary.csv                   -- mean ± SD per G value
  gene_spacer_scores_by_n_genes.csv     -- per-gene SPACER scores per condition
  run_gXXX_repY/spacer_scores.csv       -- raw scores per run
  run_gXXX_repY/auroc_history.csv       -- val AUROC per epoch per run
  n_genes_sensitivity.pdf/png           -- line plots of metrics vs G
  spacer_scatter.pdf/png                -- scatter: G vs reference SPACER
  auroc_curves.pdf/png                  -- val AUROC learning curves per G
"""

import argparse
import os
import random
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import issparse
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model.dataset import (
    BagsDataset, custom_collate_fn, preprocess_data, map_immune_cell,
)
from model.model import MIL, EarlyStopping


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset_for_n_genes(data_csv, immune_cell, n_genes,
                               radius=150, resolution='low',
                               max_instances=None, k=2):
    """
    Load manifest CSV, preprocess each AnnData with the given n_genes, and
    return a BagsDataset.  Bypasses BagsDataset.__init__ to supply already-
    preprocessed adata objects (same pattern as label_perturbation_analysis).
    """
    immune_col = map_immune_cell(immune_cell)
    manifest = pd.read_csv(data_csv)
    adata_radius_list = []

    for _, row in manifest.iterrows():
        adata_path = row['adata']
        res = row['resolution'] if 'resolution' in row and not pd.isna(row['resolution']) else resolution
        rad = row['radius'] if 'radius' in row and not pd.isna(row['radius']) else radius

        print(f"  Loading {os.path.basename(adata_path)} (n_genes={n_genes}) ...")
        adata = sc.read_h5ad(adata_path)
        adata.obs_names_make_unique()
        adata = preprocess_data(adata, immune_cell, n_genes, resolution=res)
        adata_radius_list.append((adata, int(rad), res))

    ds = object.__new__(BagsDataset)
    ds.immune_cell = immune_col
    ds.max_instances = max_instances
    ds.radius = int(radius)
    ds.resolution = resolution
    ds.n_genes = n_genes
    ds.k = k
    ds.nc_type = None
    ds.batches = ds.create_bags(adata_radius_list)
    return ds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(dataset, all_genes, device, learning_rate=0.05,
                 num_epochs=10, patience=5, delta=0.0001, seed=0):
    """
    Train a fresh MIL model on `dataset`.

    Returns:
        spacer_scores : np.ndarray of shape (n_genes,), sigmoid(ig) at best val loss
        auroc_history : list of float, one per epoch
    """
    set_seed(seed)
    model = MIL(all_genes).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    early_stopper = EarlyStopping(patience=patience, delta=delta)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Dataset too small to split (only {len(dataset)} batches)")

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=custom_collate_fn)

    auroc_history = []
    best_val_loss = float('inf')
    best_ig = None

    for epoch in range(num_epochs):
        model.train()
        for batch_data in train_loader:
            dist_l, gexp_l, lbl_l, _, gnames_l, _ = batch_data
            dist_l = [d.to(device) for d in dist_l]
            gexp_l = [g.to(device) for g in gexp_l]
            labels = torch.stack(lbl_l).float().to(device)
            optimizer.zero_grad()
            out = model(dist_l, gexp_l, gnames_l)
            if out is None or out.shape[0] != labels.shape[0]:
                continue
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_outs, val_lbls = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                dist_l, gexp_l, lbl_l, _, gnames_l, _ = batch_data
                dist_l = [d.to(device) for d in dist_l]
                gexp_l = [g.to(device) for g in gexp_l]
                labels_t = torch.stack(lbl_l).float().to(device)
                out = model(dist_l, gexp_l, gnames_l)
                if out is None or out.shape[0] != labels_t.shape[0]:
                    continue
                val_loss += criterion(out, labels_t).item()
                val_outs.extend(out.cpu().numpy())
                val_lbls.extend(labels_t.cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        try:
            auroc = roc_auc_score(val_lbls, val_outs)
        except ValueError:
            auroc = float('nan')
        auroc_history.append(auroc)
        print(f"    epoch {epoch+1:3d}  val_loss={val_loss:.4f}  auroc={auroc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ig = model.immunogenicity.ig.clone().detach().cpu().numpy()

        early_stopper(val_loss, model, epoch)
        if early_stopper.early_stop:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_ig is None:
        best_ig = model.immunogenicity.ig.clone().detach().cpu().numpy()

    spacer_scores = torch.sigmoid(torch.tensor(best_ig)).numpy()
    return spacer_scores, auroc_history


# ---------------------------------------------------------------------------
# Analysis and visualisation
# ---------------------------------------------------------------------------

def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def analyze_and_save(results_dict, all_genes, output_dir,
                     reference_n_genes, top_k=100):
    """
    Compute stability metrics for every (n_genes, replicate) run relative
    to the reference G value, then write CSVs and plots.

    results_dict keys : (n_genes: int, replicate: int)
    results_dict values: {'spacer': np.ndarray, 'auroc_history': list}
    """
    os.makedirs(output_dir, exist_ok=True)
    genes = np.array(all_genes)

    all_g = sorted({g for g, _ in results_dict})
    all_reps = sorted({r for _, r in results_dict})

    # Mean SPACER scores per G value (averaged over replicates)
    spacer_mean = {}
    spacer_all = {}
    for g in all_g:
        runs = [results_dict[(g, r)]['spacer']
                for r in all_reps if (g, r) in results_dict]
        spacer_mean[g] = np.mean(runs, axis=0)
        spacer_all[g] = runs

    # Reference: use reference_n_genes if available, else closest G
    if reference_n_genes in spacer_mean:
        ref_g = reference_n_genes
    else:
        ref_g = min(all_g, key=lambda g: abs(g - reference_n_genes))
        warnings.warn(
            f"reference_n_genes={reference_n_genes} not in results; "
            f"using G={ref_g} as reference."
        )
    baseline = spacer_mean[ref_g]
    baseline_rank = np.argsort(-baseline)
    top_k_baseline = set(genes[baseline_rank[:top_k]])

    # Per-run metrics table
    rows = []
    for g in all_g:
        for rep in all_reps:
            if (g, rep) not in results_dict:
                continue
            r = results_dict[(g, rep)]
            scores = r['spacer']
            auroc_hist = r['auroc_history']

            spear_r, spear_p = spearmanr(baseline, scores)
            top_k_now = set(genes[np.argsort(-scores)[:top_k]])
            jacc = jaccard(top_k_baseline, top_k_now)
            mad_all = np.mean(np.abs(baseline - scores))
            top_k_idx = baseline_rank[:top_k]
            mad_topk = np.mean(np.abs(baseline[top_k_idx] - scores[top_k_idx]))
            best_auroc = max(auroc_hist) if auroc_hist else float('nan')
            last_auroc = auroc_hist[-1] if auroc_hist else float('nan')

            rows.append({
                'n_genes': g,
                'replicate': rep,
                'spearman_r': spear_r,
                'spearman_p': spear_p,
                f'jaccard_top{top_k}': jacc,
                'mad_all_genes': mad_all,
                f'mad_top{top_k}_genes': mad_topk,
                'best_val_auroc': best_auroc,
                'last_val_auroc': last_auroc,
                'n_epochs_run': len(auroc_hist),
                'is_reference': (g == ref_g),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'n_genes_metrics.csv'), index=False)
    print("Saved n_genes_metrics.csv")

    summary = df.groupby('n_genes').agg(['mean', 'std']).round(4)
    summary.to_csv(os.path.join(output_dir, 'n_genes_summary.csv'))
    print("Saved n_genes_summary.csv")

    # Per-gene SPACER scores across G values
    gene_df = pd.DataFrame({'Gene': all_genes})
    for g in all_g:
        gene_df[f'spacer_G{g}_mean'] = spacer_mean[g]
        if len(spacer_all[g]) > 1:
            gene_df[f'spacer_G{g}_std'] = np.std(spacer_all[g], axis=0)
        else:
            gene_df[f'spacer_G{g}_std'] = 0.0
    gene_df['reference_rank'] = (
        pd.Series(baseline).rank(ascending=False).values.astype(int)
    )
    gene_df = gene_df.sort_values('reference_rank').reset_index(drop=True)
    gene_df.to_csv(
        os.path.join(output_dir, 'gene_spacer_scores_by_n_genes.csv'),
        index=False,
    )
    print("Saved gene_spacer_scores_by_n_genes.csv")

    # Figures
    _plot_sensitivity_lines(df, all_g, output_dir, top_k, ref_g)
    _plot_spacer_scatter(spacer_mean, all_g, output_dir, ref_g)
    _plot_auroc_curves(results_dict, all_g, all_reps, output_dir, ref_g)

    return df


def _plot_sensitivity_lines(df, all_g, output_dir, top_k, ref_g):
    """Line plots of stability metrics as a function of G."""
    metrics = [
        ('spearman_r',           f'Spearman ρ vs. G={ref_g} baseline',  '#4C72B0'),
        (f'jaccard_top{top_k}',  f'Jaccard overlap (top-{top_k} genes)', '#DD8452'),
        ('mad_all_genes',        'MAD of SPACER scores (all genes)',     '#C44E52'),
        ('best_val_auroc',       'Best validation AUROC',                '#55A868'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (col, title, color) in zip(axes, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        grp = df.groupby('n_genes')[col]
        means = grp.mean()
        stds = grp.std().fillna(0)
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    fmt='o-', color=color, capsize=5, linewidth=1.8,
                    markersize=6, elinewidth=1.2)
        ax.axvline(ref_g, color='gray', linestyle='--', linewidth=1,
                   label=f'reference G={ref_g}')
        ax.set_xlabel('Number of top genes (G)', fontsize=11)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(list(means.index))
        ax.legend(fontsize=8)
        for x, y, s in zip(means.index, means.values, stds.values):
            ax.annotate(f'{y:.3f}', (x, y + s + 0.003),
                        ha='center', va='bottom', fontsize=7)

    fig.suptitle('SPACER Sensitivity to Top-G Gene Selection',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'n_genes_sensitivity.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved n_genes_sensitivity.pdf/png")


def _plot_spacer_scatter(spacer_mean, all_g, output_dir, ref_g):
    """Scatter plots: SPACER score at G vs. SPACER score at reference G."""
    other_g = [g for g in all_g if g != ref_g]
    if not other_g:
        return
    ncols = min(3, len(other_g))
    nrows = (len(other_g) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    baseline = spacer_mean[ref_g]
    for k, g in enumerate(other_g):
        ax = axes[k // ncols][k % ncols]
        scores = spacer_mean[g]
        ax.scatter(baseline, scores, s=3, alpha=0.3, rasterized=True)
        lo = min(baseline.min(), scores.min()) - 0.02
        hi = max(baseline.max(), scores.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        rho, _ = spearmanr(baseline, scores)
        ax.set_title(f'G={g}  ρ={rho:.3f}', fontsize=11)
        ax.set_xlabel(f'SPACER score (G={ref_g})', fontsize=10)
        ax.set_ylabel(f'SPACER score (G={g})', fontsize=10)
    for k in range(len(other_g), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle(f'SPACER Score Stability: G vs. Reference G={ref_g}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'spacer_scatter.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved spacer_scatter.pdf/png")


def _plot_auroc_curves(results_dict, all_g, all_reps, output_dir, ref_g):
    """AUROC learning curves, one line per G, mean ± std over replicates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.cm.viridis(np.linspace(0.05, 0.9, len(all_g)))
    for color, g in zip(palette, all_g):
        curves = [results_dict[(g, r)]['auroc_history']
                  for r in all_reps if (g, r) in results_dict]
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [np.nan] * (max_len - len(c)) for c in curves])
        mean_c = np.nanmean(padded, axis=0)
        std_c = np.nanstd(padded, axis=0)
        epochs = np.arange(1, max_len + 1)
        lw = 2.5 if g == ref_g else 1.5
        ls = '-' if g == ref_g else '--'
        label = f'G={g} (n={len(curves)})' + (' [ref]' if g == ref_g else '')
        ax.plot(epochs, mean_c, color=color, label=label,
                linewidth=lw, linestyle=ls)
        ax.fill_between(epochs, mean_c - std_c, mean_c + std_c,
                        alpha=0.15, color=color)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation AUROC', fontsize=11)
    ax.set_title('Validation AUROC by Top-G Gene Count', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'auroc_curves.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved auroc_curves.pdf/png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SPACER sensitivity analysis for the top-G gene selection hyperparameter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data', required=True,
                        help='Training data manifest CSV (columns: adata, radius, resolution).')
    parser.add_argument('--reference_gene', required=True,
                        help='Reference gene CSV (column: Gene).')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for all output files.')
    parser.add_argument('--immune_cell', default='tcell',
                        choices=['tcell', 'bcell', 'macrophage',
                                 'neutrophil', 'fibroblast', 'endothelial'],
                        help='Immune cell type to model.')
    parser.add_argument('--n_genes_values', nargs='+', type=int,
                        default=[50, 100, 200, 300, 500, 750, 1000],
                        help='List of G values to sweep over.')
    parser.add_argument('--reference_n_genes', type=int, default=500,
                        help='G value treated as the reference for metric comparisons.')
    parser.add_argument('--n_replicates', type=int, default=3,
                        help='Independent replicates per G value.')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Top-K genes for Jaccard overlap metric.')
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--delta', type=float, default=0.0001)
    parser.add_argument('--max_instances', type=int, default=None,
                        help='Drop bags with more than this many instances.')
    parser.add_argument('--radius', type=int, default=150,
                        help='Default spatial radius (µm) if not in manifest.')
    parser.add_argument('--resolution', default='low', choices=['low', 'high'],
                        help='Default resolution if not in manifest.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device          : {device}")

    all_genes = pd.read_csv(args.reference_gene)['Gene'].values.tolist()
    print(f"Reference genes : {len(all_genes)}")
    print(f"G values        : {args.n_genes_values}")
    print(f"Reference G     : {args.reference_n_genes}")
    print(f"Replicates      : {args.n_replicates}")

    results_dict = {}

    for g in args.n_genes_values:
        for rep in range(args.n_replicates):
            seed = rep * 1000 + g
            tag = f"G={g}  rep={rep}  seed={seed}"
            print(f"\n{'='*65}\n  {tag}\n{'='*65}")

            try:
                dataset = build_dataset_for_n_genes(
                    data_csv=args.data,
                    immune_cell=args.immune_cell,
                    n_genes=g,
                    radius=args.radius,
                    resolution=args.resolution,
                    max_instances=args.max_instances,
                )
                if len(dataset) == 0:
                    warnings.warn(f"No bags created for {tag}. Skipping.")
                    continue

                spacer_scores, auroc_history = run_training(
                    dataset=dataset,
                    all_genes=all_genes,
                    device=device,
                    learning_rate=args.learning_rate,
                    num_epochs=args.num_epochs,
                    patience=args.patience,
                    delta=args.delta,
                    seed=seed,
                )
                results_dict[(g, rep)] = {
                    'spacer': spacer_scores,
                    'auroc_history': auroc_history,
                }

                run_dir = os.path.join(
                    args.output_dir,
                    f"run_g{g:04d}_rep{rep}",
                )
                os.makedirs(run_dir, exist_ok=True)
                pd.DataFrame({
                    'Gene': all_genes,
                    'spacer_score': spacer_scores,
                }).to_csv(os.path.join(run_dir, 'spacer_scores.csv'), index=False)
                pd.DataFrame({
                    'epoch': range(1, len(auroc_history) + 1),
                    'val_auroc': auroc_history,
                }).to_csv(os.path.join(run_dir, 'auroc_history.csv'), index=False)
                print(f"  Saved checkpoint to {run_dir}")

            except Exception as exc:
                warnings.warn(f"Run {tag} failed: {exc}")
                import traceback; traceback.print_exc()

    if not results_dict:
        print("No runs completed successfully. Exiting.")
        return

    print("\nAnalysing results ...")
    df_metrics = analyze_and_save(
        results_dict, all_genes, args.output_dir,
        reference_n_genes=args.reference_n_genes,
        top_k=args.top_k,
    )

    print("\n--- Summary (mean over replicates) ---")
    stable_cols = [
        'spearman_r',
        f'jaccard_top{args.top_k}',
        'mad_all_genes',
        'best_val_auroc',
    ]
    avail = [c for c in stable_cols if c in df_metrics.columns]
    print(df_metrics.groupby('n_genes')[avail].mean().round(4).to_string())
    print(f"\nAll outputs written to: {args.output_dir}")


if __name__ == '__main__':
    main()
