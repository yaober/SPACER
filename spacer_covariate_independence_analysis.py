#!/usr/bin/env python3
"""
SPACER Covariate Independence Analysis
=======================================
Evaluates whether SPACER scores retain independent explanatory value after
controlling for:
  1. Gene expression level  (mean expression in tumor cells)
  2. Detection rate         (fraction of tumor cells expressing each gene)
  3. Gene-module collinearity (co-expression modules via PCA + hierarchical clustering)

Three analyses are run:

  A) Gene-level OLS regression
     Regress SPACER scores on simple gene covariates (log mean expression,
     detection rate, PCA gene-module loadings).  Low R² means SPACER captures
     signal that is not explained by these bulk statistics alone.

  B) Bag-level prediction comparison (AUROC, stratified k-fold CV)
     Six gene-weighting schemes produce a scalar bag signal, which is used as
     a single predictor in logistic regression:
       - equal_weight      : uniform weights
       - expr_weight       : proportional to global mean expression
       - det_weight        : proportional to global detection rate
       - pc_baseline       : multivariate – top-K PC module scores
       - spacer_residual   : SPACER residuals after regressing out expr + det
       - spacer_weight     : full SPACER scores
       - pc_plus_spacer    : PC module scores + SPACER signal (combined)
     Higher AUROC for SPACER than expression/detection/module baselines
     demonstrates independent predictive value.

  C) Gene-module collinearity (VIF for top SPACER genes)
     Genes are clustered into co-expression modules via hierarchical clustering
     of gene–gene Spearman correlations. Variance inflation factors (VIF) are
     computed for the top-K SPACER genes to quantify multicollinearity.

Outputs (written to --output_dir):
  gene_covariates.csv                  -- per-gene: SPACER, mean_expr, det_rate, PCA loadings
  ols_gene_regression.csv              -- OLS summary (coefs, R², partial R²)
  bag_features.csv                     -- per-bag signals under each weighting scheme
  bag_auroc_comparison.csv             -- AUROC mean ± SD for each weighting model
  gene_modules.csv                     -- gene-to-module assignment + SPACER score
  module_summary.csv                   -- per-module SPACER statistics
  module_vif.csv                       -- VIF for top-K SPACER genes
  ols_scatter.pdf/png                  -- SPACER vs covariate scatter plots
  auroc_comparison.pdf/png             -- bar chart of AUROC by weighting scheme
  module_analysis.pdf/png              -- SPACER distribution by module + VIF histogram
  covariate_independence_summary.pdf/png -- one-page summary figure
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.sparse import issparse
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn(
        "statsmodels not found; OLS partial-R² and VIF computations will be skipped. "
        "Install with: pip install statsmodels"
    )

from model.dataset import preprocess_data, map_immune_cell


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_adata_list(data_csv, immune_cell, n_genes, radius, resolution):
    """Load and preprocess all datasets from the manifest CSV."""
    immune_col = map_immune_cell(immune_cell)
    manifest = pd.read_csv(data_csv)
    adata_list = []

    for _, row in manifest.iterrows():
        adata_path = row['adata']
        res = row['resolution'] if ('resolution' in row and not pd.isna(row.get('resolution'))) else resolution
        rad = int(row['radius']) if ('radius' in row and not pd.isna(row.get('radius'))) else radius

        print(f"  Loading {os.path.basename(adata_path)} ...")
        adata = sc.read_h5ad(adata_path)
        adata.obs_names_make_unique()
        adata = preprocess_data(adata, immune_cell, n_genes, resolution=res)
        adata_list.append((adata, rad, res))

    return adata_list, immune_col


# ---------------------------------------------------------------------------
# Gene-level statistics
# ---------------------------------------------------------------------------

def compute_gene_stats(adata_list, gene_list):
    """
    Compute per-gene mean expression and detection rate across all tumor cells.
    Returns arrays of shape [n_genes] aligned to gene_list.
    """
    mean_expr_by_gene = {g: [] for g in gene_list}
    det_rate_by_gene = {g: [] for g in gene_list}

    for adata, _, _ in adata_list:
        tumor = adata[adata.obs['cell_type'].astype(int) == 1]
        X = tumor.X.toarray() if issparse(tumor.X) else np.array(tumor.X, dtype=np.float32)
        var_names = adata.var_names.tolist()

        for g in gene_list:
            if g not in var_names:
                continue
            col = X[:, var_names.index(g)]
            mean_expr_by_gene[g].append(float(col.mean()))
            det_rate_by_gene[g].append(float((col > 0).mean()))

    mean_expr = np.array([
        np.mean(mean_expr_by_gene[g]) if mean_expr_by_gene[g] else 0.0
        for g in gene_list
    ], dtype=np.float64)
    det_rate = np.array([
        np.mean(det_rate_by_gene[g]) if det_rate_by_gene[g] else 0.0
        for g in gene_list
    ], dtype=np.float64)

    return mean_expr, det_rate


def compute_gene_pca_loadings(adata_list, gene_list, n_components=10,
                               max_cells=5000, seed=42):
    """
    Fit PCA on pooled tumor-cell expression and return gene loadings.

    Returns:
        pca_loadings : ndarray [n_genes, n_components]  (gene × PC)
        pca          : fitted sklearn PCA object
    """
    np.random.seed(seed)
    X_blocks = []

    for adata, _, _ in adata_list:
        tumor = adata[adata.obs['cell_type'].astype(int) == 1]
        X = tumor.X.toarray() if issparse(tumor.X) else np.array(tumor.X, dtype=np.float32)
        var_names = adata.var_names.tolist()

        X_aligned = np.zeros((X.shape[0], len(gene_list)), dtype=np.float32)
        for j, g in enumerate(gene_list):
            if g in var_names:
                X_aligned[:, j] = X[:, var_names.index(g)]
        X_blocks.append(X_aligned)

    X_all = np.vstack(X_blocks)

    if X_all.shape[0] > max_cells:
        idx = np.random.choice(X_all.shape[0], max_cells, replace=False)
        X_all = X_all[idx]

    n_comp = min(n_components, X_all.shape[1], X_all.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(X_all)

    # components_ is [n_components, n_genes]; transpose to [n_genes, n_components]
    return pca.components_.T, pca


# ---------------------------------------------------------------------------
# Bag-level signal computation
# ---------------------------------------------------------------------------

def compute_bag_signals(adata_list, immune_col, gene_list, spacer_scores,
                         mean_expr, det_rate, pca, n_pcs):
    """
    For every bag center (cell_type != 0), compute its mean tumor-cell gene
    expression and project it through six weighting schemes.

    Also computes SPACER residuals by regressing SPACER on log_mean_expr and
    det_rate and using the leftover variation as weights.

    Returns (bag_df, spacer_residuals).
    """
    spacer_arr = np.array(spacer_scores, dtype=np.float64)

    # Normalise weight vectors so they are comparable scalars
    def _norm(w):
        s = w.sum()
        return w / s if s > 0 else w

    expr_w = _norm(mean_expr.copy())
    det_w = _norm(det_rate.copy())
    spacer_w = _norm(spacer_arr.copy())

    # SPACER residuals: regress out log_mean_expr and detection_rate
    X_cov = np.column_stack([np.log1p(mean_expr), det_rate])
    scaler_cov = StandardScaler()
    X_cov_s = scaler_cov.fit_transform(X_cov)
    reg = LinearRegression().fit(X_cov_s, spacer_arr)
    spacer_residuals = spacer_arr - reg.predict(X_cov_s)
    # Shift to non-negative so they can serve as weights
    res_w = spacer_residuals - spacer_residuals.min()
    res_w = _norm(res_w)

    n_actual_pcs = pca.n_components_

    rows = []

    for adata, rad, _ in adata_list:
        coords = np.column_stack([
            adata.obs['X'].astype(float).values,
            adata.obs['Y'].astype(float).values,
        ])
        X_raw = adata.X.toarray() if issparse(adata.X) else np.array(adata.X, dtype=np.float32)
        labels_arr = adata.obs[immune_col].values.astype(int)
        cell_types = adata.obs['cell_type'].astype(int).values
        var_names = adata.var_names.tolist()

        # Map gene_list → column indices in this adata (None if absent)
        gene_cols = [var_names.index(g) if g in var_names else None for g in gene_list]
        present_mask = np.array([c is not None for c in gene_cols])
        present_cols = np.array([c for c in gene_cols if c is not None], dtype=int)

        # Precompute per-PC gene weights aligned to gene_list
        pc_weights = []
        for k in range(min(n_pcs, n_actual_pcs)):
            w = pca.components_[k]     # [n_genes]
            pc_weights.append(w)       # still aligned to gene_list

        for i in range(len(coords)):
            if cell_types[i] == 0:
                continue  # skip cell_type==0; bag centers are other cell types

            dists = np.sqrt(((coords - coords[i]) ** 2).sum(axis=1))
            in_radius = np.where((dists <= rad) & (cell_types == 1))[0]
            if len(in_radius) == 0:
                continue

            # Mean gene expression across tumor-cell instances in bag
            X_bag = X_raw[np.ix_(in_radius, present_cols)]   # [n_inst, n_present]
            mean_bag_present = X_bag.mean(axis=0)              # [n_present]

            # Expand to full gene_list length (zeros for absent genes)
            mean_bag = np.zeros(len(gene_list), dtype=np.float64)
            mean_bag[present_mask] = mean_bag_present

            row = {
                'label': int(labels_arr[i]),
                'n_instances': len(in_radius),
                'equal_weight':       float(mean_bag.mean()),
                'expr_weight':        float((mean_bag * expr_w).sum()),
                'det_weight':         float((mean_bag * det_w).sum()),
                'spacer_residual':    float((mean_bag * res_w).sum()),
                'spacer_weight':      float((mean_bag * spacer_w).sum()),
            }

            # PC module scores (one per PC)
            for k, pw in enumerate(pc_weights):
                row[f'pc{k+1}_signal'] = float((mean_bag * pw).sum())

            rows.append(row)

    bag_df = pd.DataFrame(rows)
    return bag_df, spacer_residuals


# ---------------------------------------------------------------------------
# Analysis A: Gene-level OLS
# ---------------------------------------------------------------------------

def analysis_a_ols(spacer_scores, mean_expr, det_rate, pca_loadings,
                   gene_list, output_dir):
    """
    Regress SPACER scores on log_mean_expr, detection_rate, and PCA loadings.

    Restricted to expressed genes (mean_expr > 0) with non-initialised SPACER
    scores.  Genes absent from the training data stay at their initial value
    and would otherwise dominate the regression with a trivial expressed/
    unexpressed contrast.

    Returns (summary_df, r2, adj_r2, partial_r2_dict, n_expressed).
    """
    spacer_arr = np.array(spacer_scores, dtype=np.float64)
    init_val = float(np.median(spacer_arr[mean_expr == 0])) if (mean_expr == 0).any() else 0.5

    # Keep only genes that are expressed AND have non-initialised SPACER values
    expressed_mask = (mean_expr > 0) & (spacer_arr != init_val)
    n_expressed = int(expressed_mask.sum())
    n_total = len(spacer_arr)
    print(f"  OLS: using {n_expressed}/{n_total} expressed genes "
          f"(excluded {n_total - n_expressed} unexpressed/uninitialised)")

    y = spacer_arr[expressed_mask]
    me = mean_expr[expressed_mask]
    dr = det_rate[expressed_mask]
    pl = pca_loadings[expressed_mask]

    n_pcs = pl.shape[1]
    predictor_names = ['log_mean_expr', 'detection_rate'] + [f'pc{k+1}_loading' for k in range(n_pcs)]
    X_raw = np.column_stack(
        [np.log1p(me), dr] + [pl[:, k] for k in range(n_pcs)]
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    rows = []
    r2 = adj_r2 = float('nan')
    partial_r2 = {p: float('nan') for p in predictor_names}
    f_stat = f_pval = float('nan')

    if HAS_STATSMODELS:
        X_sm = sm.add_constant(X_scaled)
        model_full = sm.OLS(y, X_sm).fit()
        r2 = model_full.rsquared
        adj_r2 = model_full.rsquared_adj
        f_stat = model_full.fvalue
        f_pval = model_full.f_pvalue

        for j, pname in enumerate(predictor_names):
            other_idx = [i for i in range(len(predictor_names)) if i != j]
            X_red = sm.add_constant(X_scaled[:, other_idx])
            r2_red = sm.OLS(y, X_red).fit().rsquared
            partial_r2[pname] = r2 - r2_red

        for j, pname in enumerate(['intercept'] + predictor_names):
            rows.append({
                'predictor': pname,
                'coef':      model_full.params[j],
                'std_err':   model_full.bse[j],
                't_stat':    model_full.tvalues[j],
                'p_value':   model_full.pvalues[j],
                'partial_r2': partial_r2.get(pname, float('nan')),
            })

    else:
        reg = LinearRegression().fit(X_scaled, y)
        y_hat = reg.predict(X_scaled)
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        n, p = X_scaled.shape
        adj_r2 = 1.0 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else float('nan')

        for j, pname in enumerate(predictor_names):
            rows.append({
                'predictor': pname,
                'coef':      reg.coef_[j],
                'std_err':   float('nan'),
                't_stat':    float('nan'),
                'p_value':   float('nan'),
                'partial_r2': float('nan'),
            })

    summary_df = pd.DataFrame(rows)
    summary_df['r2_full'] = r2
    summary_df['adj_r2'] = adj_r2
    summary_df['f_stat'] = f_stat
    summary_df['f_pval'] = f_pval
    summary_df['n_genes_used'] = n_expressed
    summary_df.to_csv(os.path.join(output_dir, 'ols_gene_regression.csv'), index=False)

    print(f"  OLS R² = {r2:.4f}   adj-R² = {adj_r2:.4f}"
          + (f"   F = {f_stat:.2f}  p = {f_pval:.2e}" if not np.isnan(f_stat) else ""))

    return summary_df, r2, adj_r2, partial_r2, n_expressed


# ---------------------------------------------------------------------------
# Analysis B: Bag-level AUROC comparison
# ---------------------------------------------------------------------------

def analysis_b_auroc(bag_df, output_dir, n_folds=5, seed=42):
    """
    AUROC comparison across gene-weighting schemes.

    Scalar signals: direct rank-based AUROC (equivalent to Wilcoxon–Mann–Whitney).
    This is more robust than fitting a logistic regression when the signal is on
    a tiny scale (~0.002) with severe class imbalance (~5%): logistic regression
    can converge with a sign-flipped coefficient, yielding AUROC < 0.5 even when
    the raw Spearman correlation with the label is positive.

    PC baseline and combined model: stratified k-fold logistic regression, since
    they are multivariate and benefit from a fitted linear combination.
    """
    labels = bag_df['label'].values

    scalar_models = [
        ('equal_weight',    'Equal weights\n(unweighted)'),
        ('expr_weight',     'Expression-level\nweighted'),
        ('det_weight',      'Detection-rate\nweighted'),
        ('spacer_residual', 'SPACER residual\n(expr+det removed)'),
        ('spacer_weight',   'SPACER weighted'),
    ]

    pc_cols = sorted([c for c in bag_df.columns if c.startswith('pc') and c.endswith('_signal')])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []

    def _cv_auroc_multivariate(X, labels):
        """K-fold logistic regression AUROC for multivariate features."""
        fold_aurocs = []
        for tr, va in skf.split(X, labels):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = labels[tr], labels[va]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                continue
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_va_s = sc.transform(X_va)
            lr = LogisticRegression(max_iter=1000, random_state=seed,
                                    class_weight='balanced')
            lr.fit(X_tr_s, y_tr)
            fold_aurocs.append(roc_auc_score(y_va, lr.predict_proba(X_va_s)[:, 1]))
        return fold_aurocs

    # Scalar models — direct rank-based AUROC (no fitting required)
    for col, name in scalar_models:
        if col not in bag_df.columns:
            continue
        signal = bag_df[col].values
        auroc = roc_auc_score(labels, signal)
        rho, rho_p = spearmanr(signal, labels)
        rows.append({
            'model': col, 'name': name,
            'auroc_mean': float(auroc),
            'auroc_std':  float('nan'),     # no folds for direct AUROC
            'spearman_r': float(rho),
            'spearman_p': float(rho_p),
            'n_folds':    0,
            'method':     'direct_rank',
        })

    # PC baseline (multivariate logistic regression)
    if pc_cols:
        fold_aurocs = _cv_auroc_multivariate(bag_df[pc_cols].values, labels)
        if fold_aurocs:
            rows.append({
                'model': 'pc_baseline',
                'name':  f'Gene-module PCs\n(baseline, K={len(pc_cols)})',
                'auroc_mean': np.mean(fold_aurocs),
                'auroc_std':  np.std(fold_aurocs),
                'spearman_r': float('nan'),
                'spearman_p': float('nan'),
                'n_folds':    len(fold_aurocs),
                'method':     'logistic_cv',
            })

    # Combined: PC + SPACER
    if pc_cols and 'spacer_weight' in bag_df.columns:
        X_comb = np.column_stack([bag_df[pc_cols].values,
                                  bag_df['spacer_weight'].values])
        fold_aurocs = _cv_auroc_multivariate(X_comb, labels)
        if fold_aurocs:
            rows.append({
                'model': 'pc_plus_spacer',
                'name':  'PC modules + SPACER\n(combined)',
                'auroc_mean': np.mean(fold_aurocs),
                'auroc_std':  np.std(fold_aurocs),
                'spearman_r': float('nan'),
                'spearman_p': float('nan'),
                'n_folds':    len(fold_aurocs),
                'method':     'logistic_cv',
            })

    auroc_df = pd.DataFrame(rows)
    auroc_df.to_csv(os.path.join(output_dir, 'bag_auroc_comparison.csv'), index=False)

    print("  Bag-level AUROC comparison:")
    for _, r in auroc_df.iterrows():
        print(f"    {r['model']:30s}  {r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}")

    return auroc_df


# ---------------------------------------------------------------------------
# Analysis C: Gene module collinearity / VIF
# ---------------------------------------------------------------------------

def analysis_c_modules(adata_list, gene_list, spacer_scores, top_k=50,
                        n_modules=10, output_dir='.', seed=42, max_cells=3000):
    """
    1. Gene-gene Spearman correlation → hierarchical clustering → module labels.
    2. Per-module SPACER statistics (shows SPACER discriminates within modules).
    3. VIF for the top-K SPACER genes (quantifies multicollinearity).

    Restricted to expressed genes to avoid a prohibitively large correlation
    matrix when the reference gene list is much larger than the data.

    Returns (gene_module_df, module_df, vif_df).
    """
    np.random.seed(seed)
    spacer_arr = np.array(spacer_scores, dtype=np.float64)
    gene_arr = np.array(gene_list)

    # Restrict to expressed genes before building the correlation matrix
    init_val = float(np.median(spacer_arr[spacer_arr > 0.499])) if (spacer_arr > 0.499).any() else 0.5
    expressed_mask = spacer_arr != init_val
    if expressed_mask.sum() < 10:
        expressed_mask = np.ones(len(gene_list), dtype=bool)
    gene_list_expr = [g for g, m in zip(gene_list, expressed_mask) if m]
    spacer_arr_expr = spacer_arr[expressed_mask]
    gene_arr_expr = gene_arr[expressed_mask]
    print(f"  Module analysis: using {len(gene_list_expr)} expressed genes "
          f"(of {len(gene_list)} total)")

    # Collect pooled tumor-cell expression matrix
    X_blocks = []
    for adata, _, _ in adata_list:
        tumor = adata[adata.obs['cell_type'].astype(int) == 1]
        X = tumor.X.toarray() if issparse(tumor.X) else np.array(tumor.X, dtype=np.float32)
        var_names = adata.var_names.tolist()
        X_aligned = np.zeros((X.shape[0], len(gene_list_expr)), dtype=np.float32)
        for j, g in enumerate(gene_list_expr):
            if g in var_names:
                X_aligned[:, j] = X[:, var_names.index(g)]
        X_blocks.append(X_aligned)

    X_all = np.vstack(X_blocks).astype(np.float64)
    if X_all.shape[0] > max_cells:
        idx = np.random.choice(X_all.shape[0], max_cells, replace=False)
        X_all = X_all[idx]

    # Spearman correlation via Pearson-on-ranks (faster than scipy.stats.spearmanr)
    print(f"  Computing {len(gene_list_expr)}×{len(gene_list_expr)} gene–gene Spearman correlation matrix ...")
    X_ranked = np.apply_along_axis(rankdata, 0, X_all)
    corr_mat = np.corrcoef(X_ranked.T)          # [n_expr_genes, n_expr_genes]
    np.fill_diagonal(corr_mat, 1.0)

    # Hierarchical clustering (average linkage on 1 - |corr|)
    dist_mat = np.clip(1.0 - np.abs(corr_mat), 0, None)
    np.fill_diagonal(dist_mat, 0.0)
    link = linkage(squareform(dist_mat), method='average')
    module_labels = fcluster(link, t=n_modules, criterion='maxclust')

    # Per-module summary
    module_rows = []
    for m in range(1, n_modules + 1):
        mask = module_labels == m
        if not mask.any():
            continue
        ms = spacer_arr_expr[mask]
        top_idx = np.argmax(ms)
        module_rows.append({
            'module':        m,
            'n_genes':       int(mask.sum()),
            'mean_spacer':   float(ms.mean()),
            'std_spacer':    float(ms.std()),
            'max_spacer':    float(ms.max()),
            'min_spacer':    float(ms.min()),
            'spacer_range':  float(ms.max() - ms.min()),
            'top_gene':      gene_arr_expr[mask][top_idx],
        })
    module_df = pd.DataFrame(module_rows).sort_values('mean_spacer', ascending=False)

    # VIF for top-K SPACER genes (among expressed genes)
    top_k_idx = np.argsort(-spacer_arr_expr)[:top_k]
    X_top = X_all[:, top_k_idx]
    var_mask = X_top.var(axis=0) > 0          # drop zero-variance columns
    X_top_valid = X_top[:, var_mask]
    top_genes_valid = gene_arr_expr[top_k_idx][var_mask]
    top_spacer_valid = spacer_arr_expr[top_k_idx][var_mask]
    top_module_valid = module_labels[top_k_idx][var_mask]

    vif_rows = []
    if HAS_STATSMODELS and X_top_valid.shape[1] > 1:
        X_top_s = sm.add_constant(StandardScaler().fit_transform(X_top_valid))
        for j, gene in enumerate(top_genes_valid):
            try:
                vif = variance_inflation_factor(X_top_s, j + 1)
            except Exception:
                vif = float('nan')
            vif_rows.append({
                'gene':        gene,
                'spacer_score': float(top_spacer_valid[j]),
                'module':      int(top_module_valid[j]),
                'vif':         float(vif),
            })
    else:
        for j, gene in enumerate(top_genes_valid):
            vif_rows.append({
                'gene':        gene,
                'spacer_score': float(top_spacer_valid[j]),
                'module':      int(top_module_valid[j]),
                'vif':         float('nan'),
            })

    vif_df = pd.DataFrame(vif_rows)

    # Per-gene table (expressed genes only)
    gene_module_df = pd.DataFrame({
        'gene':        gene_list_expr,
        'spacer_score': spacer_arr_expr,
        'module':      module_labels,
    })

    # Save
    gene_module_df.to_csv(os.path.join(output_dir, 'gene_modules.csv'), index=False)
    module_df.to_csv(os.path.join(output_dir, 'module_summary.csv'), index=False)
    vif_df.to_csv(os.path.join(output_dir, 'module_vif.csv'), index=False)

    print(f"  Module analysis: {n_modules} modules, "
          f"top-{top_k} SPACER genes VIF computed.")
    if not vif_df['vif'].isna().all():
        v = vif_df['vif'].dropna()
        print(f"    VIF: median={v.median():.2f}  mean={v.mean():.2f}  "
              f"fraction>5: {(v > 5).mean()*100:.1f}%")

    return gene_module_df, module_df, vif_df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_ols_scatter(gene_df, output_dir):
    covs = [
        ('log_mean_expr',  'log(mean expression + 1)', '#4C72B0'),
        ('detection_rate', 'Detection rate',            '#DD8452'),
        ('pc1_loading',    'PC1 gene loading',          '#55A868'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, xlabel, color) in zip(axes, covs):
        if col not in gene_df.columns:
            ax.set_visible(False)
            continue
        x, y = gene_df[col].values, gene_df['spacer_score'].values
        ax.scatter(x, y, s=4, alpha=0.4, color=color, rasterized=True)
        rho, pval = spearmanr(x, y)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('SPACER score', fontsize=11)
        ax.set_title(f'Spearman ρ = {rho:.3f}  (p = {pval:.2e})', fontsize=10)
    fig.suptitle('SPACER Scores vs. Simple Gene Covariates',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'ols_scatter.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved ols_scatter.pdf/png")


def plot_auroc_comparison(auroc_df, output_dir):
    names = auroc_df['name'].tolist()
    means = auroc_df['auroc_mean'].values
    stds  = auroc_df['auroc_std'].values
    models = auroc_df['model'].tolist()

    colors = ['#e07b54' if 'spacer' in m else '#aec6cf' for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4), 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
           capsize=5, linewidth=0.8, width=0.65)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='random (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, ha='center')
    ax.set_ylabel('Validation AUROC (mean ± SD)', fontsize=11)
    ax.set_title('Bag-level Prediction AUROC by Gene Weighting Scheme', fontsize=12)
    ax.set_ylim(max(0.35, (means - stds).min() - 0.05), min(1.02, (means + stds).max() + 0.08))
    ax.legend(fontsize=9)
    for xi, (m, s) in zip(x, zip(means, stds)):
        ax.text(xi, m + s + 0.004, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    # Legend patch for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e07b54', edgecolor='black', label='SPACER-based'),
        Patch(facecolor='#aec6cf', edgecolor='black', label='Covariate baseline'),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color='gray', linestyle='--', label='random (0.5)')
    ], fontsize=9)

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'auroc_comparison.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved auroc_comparison.pdf/png")


def plot_module_analysis(gene_module_df, module_df, vif_df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: SPACER distribution per module (box plot)
    ax = axes[0]
    modules = sorted(gene_module_df['module'].unique())
    data_by_module = [
        gene_module_df[gene_module_df['module'] == m]['spacer_score'].values
        for m in modules
    ]
    bp = ax.boxplot(data_by_module, labels=[f'M{m}' for m in modules],
                    patch_artist=True, medianprops=dict(color='black'))
    for patch in bp['boxes']:
        patch.set_facecolor('#aec6cf')
    ax.set_xlabel('Co-expression module', fontsize=11)
    ax.set_ylabel('SPACER score', fontsize=11)
    ax.set_title('SPACER Score Distribution by Co-expression Module\n'
                 '(spread within modules shows SPACER discriminates beyond modules)',
                 fontsize=10)

    # Right: VIF histogram
    ax = axes[1]
    if not vif_df.empty and not vif_df['vif'].isna().all():
        vif_vals = vif_df['vif'].dropna()
        ax.hist(vif_vals, bins=20, color='#4C72B0', edgecolor='black', alpha=0.8)
        ax.axvline(5,  color='orange',  linestyle='--', linewidth=1.5, label='VIF = 5')
        ax.axvline(10, color='red',     linestyle='--', linewidth=1.5, label='VIF = 10')
        pct5  = (vif_vals > 5).mean() * 100
        pct10 = (vif_vals > 10).mean() * 100
        ax.set_xlabel(f'VIF (top-{len(vif_df)} SPACER genes)', fontsize=11)
        ax.set_ylabel('Number of genes', fontsize=11)
        ax.set_title(f'Collinearity: VIF Distribution\n'
                     f'{pct5:.0f}% > 5,  {pct10:.0f}% > 10', fontsize=10)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5,
                'statsmodels not installed\n(pip install statsmodels)',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('VIF Analysis (skipped)', fontsize=11)

    fig.suptitle('Gene Module & Collinearity Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'module_analysis.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved module_analysis.pdf/png")


def plot_summary(r2, adj_r2, auroc_df, vif_df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: OLS R²
    ax = axes[0]
    bar_colors = ['#4C72B0', '#aec6cf']
    ax.bar(['Full R²', 'Adj. R²'], [r2, adj_r2],
           color=bar_colors, edgecolor='black', width=0.45)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Coefficient of determination (R²)', fontsize=10)
    ax.set_title('A. Gene-level OLS\n'
                 'SPACER ~ log_expr + det_rate + PC loadings\n'
                 'Low R² → SPACER not redundant with covariates', fontsize=9)
    for xi, v in enumerate([r2, adj_r2]):
        if not np.isnan(v):
            ax.text(xi, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=11)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)

    # Panel B: AUROC comparison (key models only)
    ax = axes[1]
    key_models = ['equal_weight', 'pc_baseline', 'spacer_residual', 'spacer_weight']
    sub = auroc_df[auroc_df['model'].isin(key_models)].copy()
    sub['order'] = sub['model'].map({m: i for i, m in enumerate(key_models)})
    sub = sub.sort_values('order')

    names  = sub['name'].tolist()
    means  = sub['auroc_mean'].values
    stds   = sub['auroc_std'].values
    colors = ['#e07b54' if 'spacer' in m else '#aec6cf' for m in sub['model'].tolist()]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor='black', capsize=5, width=0.6)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ylim_lo = max(0.35, (means - stds).min() - 0.05)
    ylim_hi = min(1.02, (means + stds).max() + 0.08)
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.set_ylabel('AUROC (mean ± SD)', fontsize=10)
    ax.set_title('B. Bag-level Prediction AUROC\nHigher SPACER AUROC → independent\npredictive value', fontsize=9)
    for xi, (m, s) in zip(x, zip(means, stds)):
        ax.text(xi, m + s + 0.003, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    # Panel C: VIF summary
    ax = axes[2]
    if not vif_df.empty and not vif_df['vif'].isna().all():
        vif_vals = vif_df['vif'].dropna()
        n_low   = (vif_vals <= 5).sum()
        n_mid   = ((vif_vals > 5) & (vif_vals <= 10)).sum()
        n_high  = (vif_vals > 10).sum()
        ax.bar(['VIF ≤ 5\n(low)', 'VIF 5–10\n(moderate)', 'VIF > 10\n(high)'],
               [n_low, n_mid, n_high],
               color=['#55A868', '#f0a500', '#C44E52'], edgecolor='black')
        ax.set_ylabel('Number of top SPACER genes', fontsize=10)
        ax.set_title(f'C. Gene Collinearity (VIF)\nTop-{len(vif_df)} SPACER genes\nLow VIF → not merely collinear', fontsize=9)
        for xi, n in enumerate([n_low, n_mid, n_high]):
            ax.text(xi, n + 0.3, str(n), ha='center', va='bottom', fontsize=11)
    else:
        ax.text(0.5, 0.5, 'VIF skipped\n(install statsmodels)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('C. Gene Collinearity (VIF)', fontsize=9)

    fig.suptitle('SPACER Score Independence from Gene Expression Covariates',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'covariate_independence_summary.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved covariate_independence_summary.pdf/png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate whether SPACER scores have independent explanatory value '
            'after controlling for gene expression level, detection rate, and '
            'gene-module collinearity.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data', required=True,
                        help='Manifest CSV (columns: adata, radius, resolution).')
    parser.add_argument('--spacer_scores', required=True,
                        help='CSV with columns Gene and spacer_score.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for all outputs.')
    parser.add_argument('--immune_cell', default='tcell',
                        choices=['tcell', 'bcell', 'macrophage',
                                 'neutrophil', 'fibroblast', 'endothelial'])
    parser.add_argument('--n_genes', type=int, default=500,
                        help='Top-N tumor genes passed to preprocess_data.')
    parser.add_argument('--n_pcs', type=int, default=10,
                        help='Number of PCA components for gene-module loadings.')
    parser.add_argument('--n_modules', type=int, default=10,
                        help='Number of co-expression modules (hierarchical clustering).')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-K SPACER genes for VIF analysis.')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Folds for bag-level AUROC cross-validation.')
    parser.add_argument('--radius', type=int, default=150,
                        help='Default spatial radius (µm) if not in manifest.')
    parser.add_argument('--resolution', default='low', choices=['low', 'high'])
    parser.add_argument('--max_cells', type=int, default=5000,
                        help='Max tumour cells used for PCA / correlation matrix.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load SPACER scores
    spacer_df = pd.read_csv(args.spacer_scores)
    gene_list     = spacer_df['Gene'].tolist()
    spacer_scores = spacer_df['spacer_score'].values.astype(np.float64)
    print(f"Loaded {len(gene_list)} genes with SPACER scores.")

    # Load and preprocess data
    print("\nLoading and preprocessing datasets ...")
    adata_list, immune_col = load_adata_list(
        args.data, args.immune_cell, args.n_genes, args.radius, args.resolution
    )

    # Per-gene statistics
    print("\nComputing per-gene statistics (mean expression, detection rate) ...")
    mean_expr, det_rate = compute_gene_stats(adata_list, gene_list)

    # PCA gene modules
    print(f"\nFitting PCA on tumour-cell expression (n_components={args.n_pcs}) ...")
    pca_loadings, pca_model = compute_gene_pca_loadings(
        adata_list, gene_list,
        n_components=args.n_pcs,
        max_cells=args.max_cells,
        seed=args.seed,
    )

    # Save gene covariate table
    gene_df = pd.DataFrame({
        'gene':           gene_list,
        'spacer_score':   spacer_scores,
        'mean_expr':      mean_expr,
        'log_mean_expr':  np.log1p(mean_expr),
        'detection_rate': det_rate,
    })
    for k in range(pca_loadings.shape[1]):
        gene_df[f'pc{k+1}_loading'] = pca_loadings[:, k]
    gene_df.to_csv(os.path.join(args.output_dir, 'gene_covariates.csv'), index=False)
    print("Saved gene_covariates.csv")

    # ------------------------------------------------------------------
    # Analysis A: Gene-level OLS
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Analysis A: Gene-level OLS regression")
    print("=" * 60)
    ols_summary, r2, adj_r2, partial_r2, n_expressed = analysis_a_ols(
        spacer_scores, mean_expr, det_rate, pca_loadings,
        gene_list, args.output_dir,
    )

    # ------------------------------------------------------------------
    # Analysis B: Bag-level AUROC comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Analysis B: Bag-level prediction AUROC")
    print("=" * 60)
    print("Computing bag signals ...")
    bag_df, spacer_residuals = compute_bag_signals(
        adata_list, immune_col, gene_list, spacer_scores,
        mean_expr, det_rate, pca_model, n_pcs=args.n_pcs,
    )
    bag_df.to_csv(os.path.join(args.output_dir, 'bag_features.csv'), index=False)
    n_pos = int(bag_df['label'].sum())
    print(f"  {len(bag_df)} bags  ({n_pos} positive, {len(bag_df)-n_pos} negative)")

    auroc_df = analysis_b_auroc(
        bag_df, args.output_dir, n_folds=args.n_folds, seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Analysis C: Gene-module collinearity / VIF
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Analysis C: Gene module collinearity")
    print("=" * 60)
    gene_module_df, module_df, vif_df = analysis_c_modules(
        adata_list, gene_list, spacer_scores,
        top_k=args.top_k,
        n_modules=args.n_modules,
        output_dir=args.output_dir,
        seed=args.seed,
        max_cells=args.max_cells,
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nGenerating figures ...")
    plot_ols_scatter(gene_df, args.output_dir)
    plot_auroc_comparison(auroc_df, args.output_dir)
    plot_module_analysis(gene_module_df, module_df, vif_df, args.output_dir)
    plot_summary(r2, adj_r2, auroc_df, vif_df, args.output_dir)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COVARIATE INDEPENDENCE SUMMARY")
    print("=" * 60)

    print(f"\nA. Gene-level OLS  (SPACER ~ log_expr + det_rate + {args.n_pcs} PCs):")
    print(f"   Expressed genes used: {n_expressed} / {len(gene_list)}")
    print(f"   R²       = {r2:.4f}")
    print(f"   Adj. R²  = {adj_r2:.4f}")
    print("   Interpretation: low R² → SPACER captures signal not reducible")
    print("                   to mean expression, detection rate, or co-expression.")

    print(f"\nB. Bag-level AUROC (direct rank for scalar; CV logistic for PC models):")
    for _, row in auroc_df.iterrows():
        marker = ' <-- SPACER' if 'spacer' in row['model'] else ''
        std_str = f"± {row['auroc_std']:.4f}" if not np.isnan(row['auroc_std']) else "(direct)"
        print(f"   {row['model']:30s}  {row['auroc_mean']:.4f} {std_str}{marker}")

    spacer_row = auroc_df[auroc_df['model'] == 'spacer_weight']
    equal_row  = auroc_df[auroc_df['model'] == 'equal_weight']
    pc_row     = auroc_df[auroc_df['model'] == 'pc_baseline']
    if len(spacer_row) and len(equal_row):
        delta_eq = spacer_row['auroc_mean'].values[0] - equal_row['auroc_mean'].values[0]
        print(f"\n   ΔAUROC (SPACER vs. equal weights):   {delta_eq:+.4f}")
    if len(spacer_row) and len(pc_row):
        delta_pc = spacer_row['auroc_mean'].values[0] - pc_row['auroc_mean'].values[0]
        print(f"   ΔAUROC (SPACER vs. gene-module PCs): {delta_pc:+.4f}")

    print(f"\nC. Module collinearity (top-{args.top_k} SPACER genes):")
    if not vif_df['vif'].isna().all():
        v = vif_df['vif'].dropna()
        print(f"   Median VIF = {v.median():.2f}")
        print(f"   Fraction with VIF > 5  = {(v > 5).mean()*100:.1f}%")
        print(f"   Fraction with VIF > 10 = {(v > 10).mean()*100:.1f}%")
    else:
        print("   (install statsmodels to enable VIF)")

    print(f"\nAll outputs written to: {args.output_dir}")


if __name__ == '__main__':
    main()
