#!/usr/bin/env python3
"""
Label Perturbation Robustness Analysis for SPACER
==================================================
Reviewer concern: if T cells, tumor cells, fibroblasts, or macrophages are
systematically misannotated, SPACER scores could change substantially.

This script randomly replaces a given fraction of cell-type / immune-infiltration
labels (5%, 10%, 20%) and assesses whether SPACER scores and top-ranked genes
remain stable across n_replicates independent runs per perturbation level.

Outputs (written to --output_dir):
  perturbation_metrics.csv          -- per-run stability metrics
  perturbation_summary.csv          -- mean ± SD per perturbation level
  gene_spacer_scores_by_perturbation.csv  -- per-gene SPACER scores per condition
  run_fracXXX_repY/spacer_scores.csv      -- raw scores per run
  run_fracXXX_repY/auroc_history.csv      -- val AUROC per epoch per run
  stability_metrics.pdf/png          -- bar charts of spearman r, Jaccard, AUROC
  spacer_scatter.pdf/png             -- scatter: perturbed vs baseline SPACER
  auroc_curves.pdf/png               -- val AUROC learning curves per condition
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
# Perturbation utilities
# ---------------------------------------------------------------------------

def partial_label_perturbation(adata, immune_col, perturb_frac, seed=42):
    """
    Randomly flip `perturb_frac` fraction of binary immune-infiltration labels
    (0→1 or 1→0).  For continuous labels the same fraction of values are
    replaced by a random draw from the empirical distribution, preserving the
    overall marginal while breaking the spatial label assignment.
    """
    if perturb_frac == 0.0:
        return adata.copy()
    np.random.seed(seed)
    adata = adata.copy()
    vals = adata.obs[immune_col].values.copy().astype(float)
    n = len(vals)
    n_perturb = max(1, int(round(n * perturb_frac)))
    idx = np.random.choice(n, size=n_perturb, replace=False)
    unique_vals = np.unique(vals)
    if set(unique_vals).issubset({0.0, 1.0}):
        vals[idx] = 1.0 - vals[idx]          # flip binary labels
    else:
        vals[idx] = np.random.choice(vals, size=n_perturb, replace=True)
    adata.obs[immune_col] = vals
    return adata


def partial_celltype_perturbation(adata, perturb_frac, seed=42):
    """
    Randomly reassign `perturb_frac` fraction of cell_type codes to a
    different valid code drawn uniformly from the remaining observed types.
    This perturbs instance composition (which cells enter each bag) and
    which cells can act as bag centres.
    """
    if perturb_frac == 0.0:
        return adata.copy()
    np.random.seed(seed)
    adata = adata.copy()
    ct = adata.obs['cell_type'].values.copy().astype(int)
    n = len(ct)
    n_perturb = max(1, int(round(n * perturb_frac)))
    idx = np.random.choice(n, size=n_perturb, replace=False)
    unique_types = np.unique(ct)
    for i in idx:
        choices = unique_types[unique_types != ct[i]]
        if len(choices) > 0:
            ct[i] = np.random.choice(choices)
    adata.obs['cell_type'] = ct.astype(str)
    return adata


# ---------------------------------------------------------------------------
# Dataset construction with perturbation
# ---------------------------------------------------------------------------

def build_perturbed_dataset(data_csv, immune_cell, n_genes, perturb_frac,
                             perturb_type, seed, radius=150, resolution='low',
                             max_instances=None, k=2):
    """
    Load data from a CSV manifest, preprocess, apply perturbation, and
    return a BagsDataset whose batches are ready for DataLoader.

    perturb_type: 'immune_label' | 'cell_type' | 'both'
    """
    immune_col = map_immune_cell(immune_cell)
    manifest = pd.read_csv(data_csv)
    adata_radius_list = []

    for _, row in manifest.iterrows():
        adata_path = row['adata']
        res = row['resolution'] if 'resolution' in row and not pd.isna(row['resolution']) else resolution
        rad = row['radius'] if 'radius' in row and not pd.isna(row['radius']) else radius

        print(f"  Loading {os.path.basename(adata_path)} ...")
        adata = sc.read_h5ad(adata_path)
        adata.obs_names_make_unique()

        # Preprocess first so binarization uses the correct label column
        adata = preprocess_data(adata, immune_cell, n_genes, resolution=res)

        # Apply perturbation(s) to the preprocessed adata
        if perturb_type in ('immune_label', 'both'):
            adata = partial_label_perturbation(adata, immune_col, perturb_frac, seed=seed)
        if perturb_type in ('cell_type', 'both'):
            adata = partial_celltype_perturbation(adata, perturb_frac, seed=seed)

        adata_radius_list.append((adata, rad, res))

    # Bypass BagsDataset.__init__ so we can supply already-preprocessed data
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
# Training loop (mirrors train.py, returns SPACER scores + AUROC history)
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
        # ---- training ----
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

        # ---- validation ----
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


def analyze_and_save(results_dict, all_genes, output_dir, top_k=100):
    """
    Compute stability metrics for every (perturb_frac, replicate) run relative
    to the unperturbed baseline (perturb_frac = 0.0), then write CSVs and plots.

    results_dict keys : (perturb_frac: float, replicate: int)
    results_dict values: {'spacer': np.ndarray, 'auroc_history': list}
    """
    os.makedirs(output_dir, exist_ok=True)
    genes = np.array(all_genes)

    all_fracs = sorted({f for f, _ in results_dict})
    all_reps = sorted({r for _, r in results_dict})

    # Mean SPACER scores per perturbation level (averaged over replicates)
    spacer_mean = {}
    spacer_all = {}
    for frac in all_fracs:
        runs = [results_dict[(frac, r)]['spacer']
                for r in all_reps if (frac, r) in results_dict]
        spacer_mean[frac] = np.mean(runs, axis=0)
        spacer_all[frac] = runs

    baseline = spacer_mean.get(0.0, spacer_mean[all_fracs[0]])
    baseline_rank = np.argsort(-baseline)
    top_k_baseline = set(genes[baseline_rank[:top_k]])

    # Per-run metrics table
    rows = []
    for frac in all_fracs:
        for rep in all_reps:
            if (frac, rep) not in results_dict:
                continue
            r = results_dict[(frac, rep)]
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
                'perturb_frac': frac,
                'perturb_pct': int(frac * 100),
                'replicate': rep,
                'spearman_r': spear_r,
                'spearman_p': spear_p,
                f'jaccard_top{top_k}': jacc,
                'mad_all_genes': mad_all,
                f'mad_top{top_k}_genes': mad_topk,
                'best_val_auroc': best_auroc,
                'last_val_auroc': last_auroc,
                'n_epochs_run': len(auroc_hist),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'perturbation_metrics.csv'), index=False)
    print("Saved perturbation_metrics.csv")

    summary = df.groupby('perturb_frac').agg(['mean', 'std']).round(4)
    summary.to_csv(os.path.join(output_dir, 'perturbation_summary.csv'))
    print("Saved perturbation_summary.csv")

    # Per-gene SPACER scores across perturbation levels
    gene_df = pd.DataFrame({'Gene': all_genes})
    for frac in all_fracs:
        pct = int(frac * 100)
        gene_df[f'spacer_{pct}pct_mean'] = spacer_mean[frac]
        if len(spacer_all[frac]) > 1:
            gene_df[f'spacer_{pct}pct_std'] = np.std(spacer_all[frac], axis=0)
        else:
            gene_df[f'spacer_{pct}pct_std'] = 0.0
    gene_df['baseline_rank'] = (
        pd.Series(baseline).rank(ascending=False).values.astype(int)
    )
    gene_df = gene_df.sort_values('baseline_rank').reset_index(drop=True)
    gene_df.to_csv(
        os.path.join(output_dir, 'gene_spacer_scores_by_perturbation.csv'),
        index=False,
    )
    print("Saved gene_spacer_scores_by_perturbation.csv")

    # Figures
    _plot_stability(df, all_fracs, output_dir, top_k)
    _plot_spacer_scatter(spacer_mean, all_fracs, output_dir)
    _plot_auroc_curves(results_dict, all_fracs, all_reps, output_dir)

    return df


def _plot_stability(df, all_fracs, output_dir, top_k):
    fracs_pct = [int(f * 100) for f in all_fracs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ('spearman_r',        f'Spearman ρ (SPACER scores vs. baseline)', '#4C72B0'),
        (f'jaccard_top{top_k}', f'Jaccard overlap (top-{top_k} genes)',   '#DD8452'),
        ('best_val_auroc',    'Best validation AUROC',                     '#55A868'),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        if col not in df.columns:
            continue
        grp = df.groupby('perturb_pct')[col]
        means = grp.mean()
        stds = grp.std().fillna(0)
        ax.bar(means.index, means.values, yerr=stds.values,
               color=color, edgecolor='black', capsize=5, width=2.5)
        ax.set_xlabel('Label perturbation (%)', fontsize=11)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(list(means.index))
        # add value annotations
        for x, y in zip(means.index, means.values):
            ax.text(x, y + stds.loc[x] + 0.005, f'{y:.3f}',
                    ha='center', va='bottom', fontsize=8)
    fig.suptitle('SPACER Robustness to Label Perturbation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'stability_metrics.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved stability_metrics.pdf/png")


def _plot_spacer_scatter(spacer_mean, all_fracs, output_dir):
    nonzero = [f for f in all_fracs if f > 0]
    if not nonzero:
        return
    ncols = min(3, len(nonzero))
    nrows = (len(nonzero) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    baseline = spacer_mean.get(0.0, spacer_mean[min(spacer_mean)])
    for k, frac in enumerate(nonzero):
        ax = axes[k // ncols][k % ncols]
        scores = spacer_mean[frac]
        ax.scatter(baseline, scores, s=3, alpha=0.3, rasterized=True)
        lo = min(baseline.min(), scores.min()) - 0.02
        hi = max(baseline.max(), scores.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        rho, _ = spearmanr(baseline, scores)
        ax.set_title(f'{int(frac*100)}% perturbation  ρ={rho:.3f}', fontsize=11)
        ax.set_xlabel('SPACER score (baseline)', fontsize=10)
        ax.set_ylabel('SPACER score (perturbed)', fontsize=10)
    for k in range(len(nonzero), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle('SPACER Score Stability: Perturbed vs. Baseline',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'spacer_scatter.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved spacer_scatter.pdf/png")


def _plot_auroc_curves(results_dict, all_fracs, all_reps, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = plt.cm.viridis(np.linspace(0.05, 0.9, len(all_fracs)))
    for color, frac in zip(palette, all_fracs):
        curves = [results_dict[(frac, r)]['auroc_history']
                  for r in all_reps if (frac, r) in results_dict]
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [np.nan] * (max_len - len(c)) for c in curves])
        mean_c = np.nanmean(padded, axis=0)
        std_c = np.nanstd(padded, axis=0)
        epochs = np.arange(1, max_len + 1)
        label = f'{int(frac*100)}% perturbed (n={len(curves)})'
        ax.plot(epochs, mean_c, color=color, label=label, linewidth=1.8)
        ax.fill_between(epochs, mean_c - std_c, mean_c + std_c,
                        alpha=0.18, color=color)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation AUROC', fontsize=11)
    ax.set_title('Validation AUROC by Label Perturbation Level', fontsize=12)
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
        description='SPACER label perturbation robustness analysis.',
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
    parser.add_argument('--perturb_fracs', nargs='+', type=float,
                        default=[0.0, 0.05, 0.10, 0.20],
                        help='Fractions of labels to perturb (0.0 = baseline).')
    parser.add_argument('--perturb_type', default='immune_label',
                        choices=['immune_label', 'cell_type', 'both'],
                        help='Which label type to perturb.')
    parser.add_argument('--n_replicates', type=int, default=3,
                        help='Independent replicates per perturbation level.')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Top-K genes for Jaccard overlap metric.')
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--delta', type=float, default=0.0001)
    parser.add_argument('--n_genes', type=int, default=500,
                        help='Top N tumor genes to include per dataset.')
    parser.add_argument('--max_instances', type=int, default=None,
                        help='Drop bags with more than this many instances.')
    parser.add_argument('--radius', type=int, default=150,
                        help='Default spatial radius (µm) if not in manifest.')
    parser.add_argument('--resolution', default='low', choices=['low', 'high'],
                        help='Default resolution if not in manifest.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    all_genes = pd.read_csv(args.reference_gene)['Gene'].values.tolist()
    print(f"Reference genes : {len(all_genes)}")
    print(f"Perturbation fracs: {args.perturb_fracs}")
    print(f"Perturbation type : {args.perturb_type}")
    print(f"Replicates per level: {args.n_replicates}")

    results_dict = {}

    for frac in args.perturb_fracs:
        for rep in range(args.n_replicates):
            # Use deterministic seeds that vary across frac × rep combinations
            seed = rep * 1000 + int(round(frac * 100))
            tag = f"frac={frac:.2f}  rep={rep}  seed={seed}"
            print(f"\n{'='*65}\n  {tag}\n{'='*65}")

            try:
                dataset = build_perturbed_dataset(
                    data_csv=args.data,
                    immune_cell=args.immune_cell,
                    n_genes=args.n_genes,
                    perturb_frac=frac,
                    perturb_type=args.perturb_type,
                    seed=seed,
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
                results_dict[(frac, rep)] = {
                    'spacer': spacer_scores,
                    'auroc_history': auroc_history,
                }

                # Per-run checkpoint
                run_dir = os.path.join(
                    args.output_dir,
                    f"run_frac{int(frac*100):03d}_rep{rep}",
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
        results_dict, all_genes, args.output_dir, top_k=args.top_k,
    )

    print("\n--- Summary (mean over replicates) ---")
    stable_cols = [
        'spearman_r',
        f'jaccard_top{args.top_k}',
        'mad_all_genes',
        'best_val_auroc',
    ]
    avail = [c for c in stable_cols if c in df_metrics.columns]
    print(df_metrics.groupby('perturb_pct')[avail].mean().round(4).to_string())
    print(f"\nAll outputs written to: {args.output_dir}")


if __name__ == '__main__':
    main()
