#!/usr/bin/env python3
"""
Expression-Matched Background Analysis for SPACER Scores
=========================================================
Addresses reviewer concern: Are SPACER top-ranked tumor genes enriched for
immune-related genes (antigen-presentation pathway, HLA genes, tumor antigens)
merely because those genes are already highly expressed, frequently detected,
long proteins, or well-annotated — rather than because SPACER captures spatial
predictive signal independent of those covariates?

This script compares the overlap of SPACER top-K genes with a combined
"immune-related" gene set against five controls:

  1. Top-K by mean expression          (expression-level baseline)
  2. Top-K by detection rate           (detection-rate baseline)
  3. Top-K by protein length           (annotation-richness proxy baseline)
  4. Expression-matched random sets    (N permutations; quantile-binned to
                                        match the expression distribution of
                                        the SPACER top-K set)
  5. Detection-matched random sets     (N permutations; quantile-binned on
                                        detection rate)
  6. Uniformly random sets             (N permutations; unmatched)

The "immune-related" gene set is the union of:
  - Curated antigen-presentation pathway genes (HLA-A/B/C, B2M, TAP1/2, etc.)
  - All HLA-prefixed genes detected in the pool
  - Curated tumor antigen genes

For expression-matched controls the script stratifies genes into
--n_bins expression quantile bins, counts how many SPACER top-K genes fall
in each bin, then for each permutation samples the same number of genes per
bin from the non-top-K pool.

Empirical p-value (SPACER vs. expression-matched background):
  fraction of matched-random draws whose overlap >= SPACER overlap.

Inputs:
  gene_covariates.csv   -- from spacer_covariate_independence_analysis.py
                           columns: gene, spacer_score, mean_expr, detection_rate
  tumor_antigens.csv    -- one-column Gene list
  idmapping.csv         -- Gene → UniProt Protein accession
  protein_lengths.csv   -- UniProt Protein → amino-acid Length
  gene_go_counts.csv    -- from download_go_annotations.py
                           columns: Gene, n_go_terms [, n_bp_terms, ...]

Outputs:
  enrichment_stats.csv           -- overlap counts, fold-enrichment, p-values
  permutation_distributions.csv  -- per-permutation overlap counts
  enrichment_comparison.pdf/png  -- bar chart: SPACER vs all baselines
  permutation_violin.pdf/png     -- violin plots of matched distributions
  enrichment_summary.pdf/png     -- one-page summary figure (2-row layout)
  decile_enrichment.pdf/png      -- enrichment by expression rank decile
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# ---------------------------------------------------------------------------
# Curated antigen-presentation / MHC pathway genes
# ---------------------------------------------------------------------------

ANTIGEN_PRESENTATION_GENES = frozenset({
    # MHC class I classical
    'HLA-A', 'HLA-B', 'HLA-C',
    # MHC class I non-classical
    'HLA-E', 'HLA-F', 'HLA-G',
    # MHC class II
    'HLA-DRA', 'HLA-DRB1', 'HLA-DRB3', 'HLA-DRB4', 'HLA-DRB5',
    'HLA-DQA1', 'HLA-DQA2', 'HLA-DQB1', 'HLA-DQB2',
    'HLA-DPA1', 'HLA-DPB1',
    'CD74',      # MHC-II invariant chain
    'CIITA',     # MHC-II transactivator
    # beta-2-microglobulin (MHC-I light chain)
    'B2M',
    # TAP complex
    'TAP1', 'TAP2', 'TAPBP', 'TAPBPL',
    # ER chaperones for peptide loading
    'CALR', 'CANX', 'PDIA3',
    # ER aminopeptidases
    'ERAP1', 'ERAP2',
    # Immunoproteasome catalytic subunits
    'PSMB8', 'PSMB9', 'PSMB10',
    # Proteasome activators
    'PSME1', 'PSME2', 'PSME3',
    # Cathepsins (lysosomal antigen processing)
    'CTSL', 'CTSS', 'CTSD', 'CTSB',
    # Co-stimulatory / immune checkpoint (tumor-expressed)
    'CD80', 'CD86', 'CD274', 'PDCD1LG2',
    # NKG2D ligands
    'MICA', 'MICB', 'ULBP1', 'ULBP2', 'ULBP3',
    # HSPs assisting peptide loading
    'HSP90AA1', 'HSP90AB1', 'HSPA1A', 'HSPA1B', 'HSPA5',
    # Antigen cross-presentation
    'SEC61A1', 'SEC61B', 'SEC61G',
    # Autophagy-mediated antigen presentation
    'BECN1', 'ATG5', 'ATG7',
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gene_covariates(path):
    df = pd.read_csv(path)
    if 'Gene' in df.columns and 'gene' not in df.columns:
        df = df.rename(columns={'Gene': 'gene'})
    required = {'gene', 'spacer_score', 'mean_expr', 'detection_rate'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gene_covariates.csv missing columns: {missing}")
    return df.drop_duplicates(subset='gene').reset_index(drop=True)


def load_gene_set(path):
    df = pd.read_csv(path)
    col = 'Gene' if 'Gene' in df.columns else df.columns[0]
    return frozenset(df[col].dropna().astype(str).str.strip())


def load_protein_lengths(idmapping_path, protein_lengths_path):
    """
    Join idmapping (Gene → UniProt) and protein_lengths (UniProt → Length)
    to produce a dict: gene_name -> protein_length (aa).
    Genes absent from idmapping or with no length entry are omitted.
    """
    id_df  = pd.read_csv(idmapping_path)
    id_df.columns = [c.lstrip('﻿') for c in id_df.columns]   # strip BOM
    len_df = pd.read_csv(protein_lengths_path)

    merged = id_df.merge(len_df, on='Protein', how='inner')
    gene_col = 'Gene' if 'Gene' in merged.columns else merged.columns[0]
    return dict(zip(merged[gene_col].astype(str).str.strip(),
                    merged['Length'].astype(float)))


def load_go_counts(go_counts_path):
    """
    Load gene_go_counts.csv (output of download_go_annotations.py).
    Returns a dict: gene_name -> n_go_terms (unique GO terms, all namespaces).
    """
    df = pd.read_csv(go_counts_path)
    gene_col = 'Gene' if 'Gene' in df.columns else df.columns[0]
    return dict(zip(df[gene_col].astype(str).str.strip(),
                    df['n_go_terms'].astype(float)))


# ---------------------------------------------------------------------------
# Interest set: single merged immune-related set
# ---------------------------------------------------------------------------

def build_immune_interest_set(gene_df, tumor_antigen_set):
    """
    Union of antigen-presentation pathway genes, HLA-prefixed genes,
    and tumor antigen genes — all intersected with the expressed gene pool.
    """
    pool = set(gene_df['gene'])
    hla_genes = {g for g in pool if g.upper().startswith('HLA')}
    ap_genes   = ANTIGEN_PRESENTATION_GENES & pool
    ta_genes   = tumor_antigen_set & pool
    immune_set = frozenset(hla_genes | ap_genes | ta_genes)

    print("  Immune-related gene set composition (in pool):")
    print(f"    Antigen-presentation pathway : {len(ap_genes)}")
    print(f"    HLA-prefixed genes           : {len(hla_genes)}")
    print(f"    Tumor antigens               : {len(ta_genes)}")
    print(f"    Union (immune-related)       : {len(immune_set)}")
    return immune_set


# ---------------------------------------------------------------------------
# Protein-length annotation
# ---------------------------------------------------------------------------

def attach_protein_length(gene_df, gene_to_length):
    """Add a protein_length column; genes without mapping receive NaN."""
    gene_df = gene_df.copy()
    gene_df['protein_length'] = gene_df['gene'].map(gene_to_length)
    n_mapped = gene_df['protein_length'].notna().sum()
    print(f"  Protein length mapped for {n_mapped}/{len(gene_df)} expressed genes.")
    return gene_df


def attach_go_counts(gene_df, gene_to_go):
    """Add an n_go_terms column; genes without GO mapping receive NaN."""
    gene_df = gene_df.copy()
    gene_df['n_go_terms'] = gene_df['gene'].map(gene_to_go)
    n_mapped = gene_df['n_go_terms'].notna().sum()
    print(f"  GO term count mapped for {n_mapped}/{len(gene_df)} expressed genes.")
    return gene_df


# ---------------------------------------------------------------------------
# Top-K selection helpers
# ---------------------------------------------------------------------------

def top_k_by(gene_df, column, top_k, require_nonnan=False):
    """Set of top-K gene names ranked by *column* descending."""
    df = gene_df.dropna(subset=[column]) if require_nonnan else gene_df
    return set(df.nlargest(top_k, column)['gene'])


# ---------------------------------------------------------------------------
# Expression / detection-matched random sampling
# ---------------------------------------------------------------------------

def _quantile_matched_sample(gene_df, reference_set, covariate_col, n_bins, rng):
    """
    Sample a gene set matching the quantile distribution of *reference_set*
    on *covariate_col* (see module docstring for details).
    """
    df = gene_df.copy()
    df['_bin'] = pd.qcut(df[covariate_col], q=n_bins, labels=False, duplicates='drop')

    ref_in_bin = (
        df[df['gene'].isin(reference_set)]
        .groupby('_bin', observed=True)
        .size()
    )

    non_ref = df[~df['gene'].isin(reference_set)]
    sampled = []
    for b, count in ref_in_bin.items():
        pool = non_ref[non_ref['_bin'] == b]['gene'].values
        n_sample = min(int(count), len(pool))
        if n_sample > 0:
            sampled.extend(rng.choice(pool, size=n_sample, replace=False))
    return set(sampled)


def expression_matched_sample(gene_df, reference_set, n_bins, rng):
    return _quantile_matched_sample(gene_df, reference_set, 'mean_expr', n_bins, rng)


def detection_matched_sample(gene_df, reference_set, n_bins, rng):
    return _quantile_matched_sample(gene_df, reference_set, 'detection_rate', n_bins, rng)


def uniform_random_sample(gene_df, reference_set, top_k, rng):
    non_ref = gene_df[~gene_df['gene'].isin(reference_set)]['gene'].values
    n_sample = min(top_k, len(non_ref))
    return set(rng.choice(non_ref, size=n_sample, replace=False))


# ---------------------------------------------------------------------------
# Enrichment statistics
# ---------------------------------------------------------------------------

def enrichment_for_group(gene_set, interest_set, pool_size, label):
    a = len(gene_set & interest_set)
    b = len(gene_set) - a
    c = len(interest_set) - a
    d = pool_size - a - b - c

    n_group  = len(gene_set)
    frac     = a / n_group if n_group > 0 else 0.0
    baseline = len(interest_set) / pool_size if pool_size > 0 else 0.0
    fold     = frac / baseline if baseline > 0 else float('nan')

    if n_group > 0:
        _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
    else:
        p = float('nan')

    return {
        'group':               label,
        'group_size':          n_group,
        'interest_size':       len(interest_set),
        'pool_size':           pool_size,
        'overlap':             a,
        'fraction_in_group':   round(frac, 6),
        'baseline_fraction':   round(baseline, 6),
        'fold_enrichment':     round(fold, 4) if not np.isnan(fold) else float('nan'),
        'fisher_p':            round(p,    6) if not np.isnan(p)    else float('nan'),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(gene_df, immune_set, top_k, n_permutations, n_bins, seed=42):
    """
    Run enrichment analysis for all fixed groups and all permutation controls.

    Fixed groups  : SPACER, Expression, Detection-rate, Protein-length,
                    Annotation-density top-K
    Permutations  : expression-matched, detection-matched, uniform random
    """
    rng = np.random.default_rng(seed)
    pool_size = len(gene_df)

    spacer_top  = top_k_by(gene_df, 'spacer_score',   top_k)
    expr_top    = top_k_by(gene_df, 'mean_expr',       top_k)
    det_top     = top_k_by(gene_df, 'detection_rate',  top_k)
    protlen_top = top_k_by(gene_df, 'protein_length',  top_k, require_nonnan=True)
    annot_top   = top_k_by(gene_df, 'n_go_terms',      top_k, require_nonnan=True)

    enrichment_rows = []
    perm_rows       = []

    # --- Fixed comparators ---
    for label, gset in [
        ('SPACER top-K',             spacer_top),
        ('Expression top-K',         expr_top),
        ('Detection-rate top-K',     det_top),
        ('Protein-length top-K',     protlen_top),
        ('Annotation-density top-K', annot_top),
    ]:
        row = enrichment_for_group(gset, immune_set, pool_size, label)
        enrichment_rows.append(row)

    # --- Permutation backgrounds ---
    for i in range(n_permutations):
        expr_bg  = expression_matched_sample(gene_df, spacer_top, n_bins, rng)
        det_bg   = detection_matched_sample(gene_df,  spacer_top, n_bins, rng)
        unif_bg  = uniform_random_sample(gene_df,     spacer_top, top_k,  rng)

        for bg_label, bg_set in [
            ('Expr-matched random', expr_bg),
            ('Det-matched random',  det_bg),
            ('Uniform random',      unif_bg),
        ]:
            ov   = len(bg_set & immune_set)
            frac = ov / len(bg_set) if bg_set else 0.0
            perm_rows.append({
                'background_type': bg_label,
                'permutation':     i,
                'bg_size':         len(bg_set),
                'overlap':         ov,
                'fraction':        frac,
            })

    return enrichment_rows, perm_rows


def add_empirical_pvalue(enrichment_rows, perm_rows):
    """Empirical p-value for each group vs. expression-matched background."""
    perm_df = pd.DataFrame(perm_rows)
    expr_bg = perm_df[perm_df['background_type'] == 'Expr-matched random']['overlap'].values

    spacer_ov = next(
        r['overlap'] for r in enrichment_rows if r['group'] == 'SPACER top-K'
    )

    result = []
    for row in enrichment_rows:
        row = dict(row)
        row['empirical_p_vs_expr_matched'] = (
            float(np.mean(expr_bg >= spacer_ov)) if len(expr_bg) > 0 else float('nan')
        )
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

_FIXED_GROUPS  = ['SPACER top-K', 'Expression top-K',
                   'Detection-rate top-K', 'Protein-length top-K',
                   'Annotation-density top-K']
_FIXED_COLORS  = ['#e07b54', '#4C72B0', '#55A868', '#9b59b6', '#e6a817']
_FIXED_XLABELS = ['SPACER\ntop-K', 'Expression\ntop-K',
                   'Det-rate\ntop-K', 'Protein-len\ntop-K',
                   'Annot-density\ntop-K']

_BG_TYPES  = ['Expr-matched random', 'Det-matched random', 'Uniform random']
_BG_COLORS = ['#aec6cf', '#b5e2b5', '#d9d9d9']
_BG_LABELS = ['Expression-\nmatched', 'Detection-\nmatched', 'Uniform\nrandom']


def _bar_panel(ax, enrich_df, perm_df, top_k, title='', show_legend=True):
    fracs = [
        float(enrich_df[enrich_df['group'] == g]['fraction_in_group'].values[0])
        if len(enrich_df[enrich_df['group'] == g]) else 0.0
        for g in _FIXED_GROUPS
    ]

    x = np.arange(len(_FIXED_GROUPS))
    ax.bar(x, fracs, color=_FIXED_COLORS, edgecolor='black',
           width=0.55, alpha=0.85, zorder=3)

    # Expression-matched permutation band
    expr_perm = perm_df[perm_df['background_type'] == 'Expr-matched random']['fraction']
    if len(expr_perm):
        pm, ps = expr_perm.mean(), expr_perm.std()
        ax.axhline(pm, color='gray', linestyle='--', linewidth=1.5, zorder=4,
                   label=f'Expr-matched mean ({pm:.3f})')
        ax.fill_between([-0.5, len(_FIXED_GROUPS) - 0.5],
                        pm - ps, pm + ps, color='gray', alpha=0.12, zorder=2)

    y_top = max(fracs + [0.001])
    for xi, g in enumerate(_FIXED_GROUPS):
        row = enrich_df[enrich_df['group'] == g]
        if not len(row):
            continue
        ov   = int(row['overlap'].values[0])
        frac = fracs[xi]
        ann  = f'{ov}/{top_k}'
        if g == 'SPACER top-K':
            emp_p = row['empirical_p_vs_expr_matched'].values[0]
            ann  += f'\np={emp_p:.3f}'
        ax.text(xi, frac + y_top * 0.04, ann,
                ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(_FIXED_XLABELS, fontsize=9)
    ax.set_ylabel('Fraction of top-K genes in immune set', fontsize=9)
    ax.set_ylim(0, y_top * 1.55)
    ax.set_title(title, fontsize=10, fontweight='bold')
    if show_legend:
        ax.legend(fontsize=7.5)


def _violin_panel(ax, enrich_df, perm_df, top_k, title=''):
    data_per_bg = [
        perm_df[perm_df['background_type'] == bg]['overlap'].values
        for bg in _BG_TYPES
    ]

    non_empty = [(i, d) for i, d in enumerate(data_per_bg) if len(d) > 0]
    if non_empty:
        parts = ax.violinplot(
            [d for _, d in non_empty],
            positions=[i for i, _ in non_empty],
            showmeans=True, showmedians=False, showextrema=True,
        )
        for pc, (idx, _) in zip(parts['bodies'], non_empty):
            pc.set_facecolor(_BG_COLORS[idx])
            pc.set_alpha(0.75)
        for key in ('cmeans', 'cmins', 'cmaxes', 'cbars'):
            parts[key].set_color('black')

    spacer_row = enrich_df[enrich_df['group'] == 'SPACER top-K']
    if len(spacer_row):
        spacer_ov = int(spacer_row['overlap'].values[0])
        emp_p     = spacer_row['empirical_p_vs_expr_matched'].values[0]
        ax.axhline(spacer_ov, color='#e07b54', linestyle='--', linewidth=2.5,
                   zorder=5,
                   label=f'SPACER top-{top_k} ({spacer_ov} genes, p={emp_p:.3f})')

    ax.set_xticks(range(len(_BG_TYPES)))
    ax.set_xticklabels(_BG_LABELS, fontsize=8)
    ax.set_ylabel(f'Overlap count (of top-{top_k})', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right')


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_enrichment_comparison(enrich_df, perm_df, output_dir, top_k):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _bar_panel(ax, enrich_df, perm_df, top_k,
               title='Immune-related gene set', show_legend=True)
    fig.suptitle(
        f'SPACER Top-{top_k} Gene Enrichment vs. Expression / Detection / Protein-length Baselines\n'
        '(p: empirical, SPACER vs. expression-matched random background)',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'enrichment_comparison.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved enrichment_comparison.pdf/png")


def plot_permutation_violin(enrich_df, perm_df, output_dir, top_k):
    fig, ax = plt.subplots(figsize=(7, 5))
    _violin_panel(ax, enrich_df, perm_df, top_k,
                  title='Immune-related gene set')
    n_perm = perm_df['permutation'].nunique()
    fig.suptitle(
        f'SPACER Overlap vs. Matched Random Backgrounds (N={n_perm} permutations)',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'permutation_violin.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved permutation_violin.pdf/png")


def plot_summary(enrich_df, perm_df, output_dir, top_k):
    """2-row summary: top = bar chart, bottom = violin."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    _bar_panel(axes[0], enrich_df, perm_df, top_k,
               title='A. Immune-related gene enrichment fraction', show_legend=True)
    _violin_panel(axes[1], enrich_df, perm_df, top_k,
                  title='B. Permutation background distribution')
    fig.suptitle(
        f'SPACER Top-{top_k} Gene Enrichment vs. Expression / Detection / Protein-length Controls\n'
        'Empirical p-value: SPACER overlap vs. expression-matched random background',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'enrichment_summary.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved enrichment_summary.pdf/png")


def plot_decile_enrichment(gene_df, immune_set, output_dir, top_k):
    """
    Within each expression rank decile, compare the fraction of all genes
    vs. SPACER top-K genes that belong to the immune-related set.
    """
    df = gene_df.sort_values('mean_expr', ascending=False).reset_index(drop=True)
    n  = len(df)
    df['spacer_rank'] = df['spacer_score'].rank(ascending=False, method='first')
    df['_in_set']     = df['gene'].isin(immune_set).astype(int)
    df['_spacer_top'] = (df['spacer_rank'] <= top_k).astype(int)

    n_deciles    = 10
    decile_size  = n // n_deciles
    bg_fracs, sp_fracs, labels = [], [], []

    for d in range(n_deciles):
        lo    = d * decile_size
        hi    = lo + decile_size if d < n_deciles - 1 else n
        chunk = df.iloc[lo:hi]
        bg_fracs.append(chunk['_in_set'].mean())
        sp_chunk = chunk[chunk['_spacer_top'] == 1]
        sp_fracs.append(sp_chunk['_in_set'].mean() if len(sp_chunk) else float('nan'))
        labels.append(f'D{d+1}')

    x     = np.arange(n_deciles)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width/2, bg_fracs, width, label='All genes in decile',
           color='#aec6cf', edgecolor='black', alpha=0.85)
    ax.bar(x + width/2, sp_fracs, width, label='SPACER top-K in decile',
           color='#e07b54', edgecolor='black', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('Expression rank decile (D1 = highest expression)', fontsize=10)
    ax.set_ylabel('Fraction in immune-related set', fontsize=10)
    ax.set_title(
        f'SPACER Top-{top_k} Enrichment Within Expression Rank Deciles\n'
        '(shows enrichment is not confined to the highest-expression stratum)',
        fontsize=11, fontweight='bold',
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(output_dir, f'decile_enrichment.{ext}'),
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print("Saved decile_enrichment.pdf/png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Test whether SPACER top-K genes are enriched for a combined immune-related '
            'gene set (antigen-presentation + HLA + tumor antigens) beyond expression-, '
            'detection-rate-, protein-length-, and annotation-density-matched baselines.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--gene_covariates', required=True,
                        help='gene_covariates.csv from spacer_covariate_independence_analysis.py.')
    parser.add_argument('--tumor_antigens', required=True,
                        help='tumor_antigens CSV with a Gene column.')
    parser.add_argument('--idmapping', required=True,
                        help='Gene → UniProt protein accession CSV (columns: Gene, Protein).')
    parser.add_argument('--protein_lengths', required=True,
                        help='UniProt accession → length CSV (columns: Protein, Length).')
    parser.add_argument('--go_counts', required=True,
                        help='gene_go_counts.csv from download_go_annotations.py '
                             '(columns: Gene, n_go_terms).')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--n_permutations', type=int, default=1000)
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Quantile bins for expression/detection matching.')
    parser.add_argument('--min_mean_expr', type=float, default=0.0)
    parser.add_argument('--min_detection_rate', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load gene covariates ---
    print("Loading gene covariates ...")
    gene_df  = load_gene_covariates(args.gene_covariates)
    n_before = len(gene_df)
    gene_df  = gene_df[
        (gene_df['mean_expr']      > args.min_mean_expr) &
        (gene_df['detection_rate'] > args.min_detection_rate)
    ].reset_index(drop=True)
    print(f"  {n_before} total → {len(gene_df)} expressed genes "
          f"(min_expr>{args.min_mean_expr}, min_det>{args.min_detection_rate})")

    if len(gene_df) < args.top_k:
        raise ValueError(
            f"Only {len(gene_df)} genes after filtering but top_k={args.top_k}.")

    # --- Attach protein lengths ---
    print("\nLoading protein lengths ...")
    gene_to_length = load_protein_lengths(args.idmapping, args.protein_lengths)
    gene_df = attach_protein_length(gene_df, gene_to_length)

    # --- Attach GO annotation counts ---
    print("\nLoading GO annotation counts ...")
    gene_to_go = load_go_counts(args.go_counts)
    gene_df = attach_go_counts(gene_df, gene_to_go)

    # --- Build immune interest set ---
    print("\nLoading tumor antigen gene set ...")
    tumor_antigen_set = load_gene_set(args.tumor_antigens)
    print(f"  {len(tumor_antigen_set)} tumor antigen genes loaded.")

    print("\nBuilding immune-related interest set ...")
    immune_set = build_immune_interest_set(gene_df, tumor_antigen_set)

    # --- Overview of top-K sets ---
    spacer_top  = top_k_by(gene_df, 'spacer_score',  args.top_k)
    expr_top    = top_k_by(gene_df, 'mean_expr',      args.top_k)
    det_top     = top_k_by(gene_df, 'detection_rate', args.top_k)
    protlen_top = top_k_by(gene_df, 'protein_length', args.top_k, require_nonnan=True)
    annot_top   = top_k_by(gene_df, 'n_go_terms',     args.top_k, require_nonnan=True)
    print(f"\nTop-{args.top_k} set overlaps:")
    print(f"  SPACER ∩ expr-top       = {len(spacer_top & expr_top)}")
    print(f"  SPACER ∩ det-top        = {len(spacer_top & det_top)}")
    print(f"  SPACER ∩ protlen-top    = {len(spacer_top & protlen_top)}")
    print(f"  SPACER ∩ annot-top      = {len(spacer_top & annot_top)}")
    print(f"  expr-top ∩ annot-top    = {len(expr_top & annot_top)}")

    # --- Run permutation analysis ---
    print(f"\nRunning analysis (top_k={args.top_k}, "
          f"n_permutations={args.n_permutations}, n_bins={args.n_bins}) ...")
    enrichment_rows, perm_rows = run_analysis(
        gene_df, immune_set,
        top_k=args.top_k,
        n_permutations=args.n_permutations,
        n_bins=args.n_bins,
        seed=args.seed,
    )
    enrichment_rows = add_empirical_pvalue(enrichment_rows, perm_rows)

    enrich_df = pd.DataFrame(enrichment_rows)
    perm_df   = pd.DataFrame(perm_rows)

    enrich_df.to_csv(os.path.join(args.output_dir, 'enrichment_stats.csv'), index=False)
    perm_df.to_csv(os.path.join(args.output_dir, 'permutation_distributions.csv'), index=False)
    print("Saved enrichment_stats.csv and permutation_distributions.csv")

    # --- Figures ---
    print("\nGenerating figures ...")
    plot_enrichment_comparison(enrich_df, perm_df, args.output_dir, args.top_k)
    plot_permutation_violin(enrich_df, perm_df, args.output_dir, args.top_k)
    plot_summary(enrich_df, perm_df, args.output_dir, args.top_k)
    plot_decile_enrichment(gene_df, immune_set, args.output_dir, args.top_k)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("ENRICHMENT SUMMARY  (immune-related gene set)")
    print("=" * 70)
    cols = ['group', 'overlap', 'fraction_in_group', 'fold_enrichment',
            'fisher_p', 'empirical_p_vs_expr_matched']
    print(enrich_df[cols].to_string(index=False))
    print(f"\nAll outputs written to: {args.output_dir}")


if __name__ == '__main__':
    main()
