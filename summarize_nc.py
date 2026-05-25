"""
summarize_nc.py  –  Collect and compare negative-control results.

Usage:
    python summarize_nc.py [--base_dir sensitivity] [--out_dir sensitivity]

For every sub-directory that contains a training_metrics.csv the script:
  1. Extracts the per-epoch train loss, val loss, and val AUROC curve
  2. Identifies the best epoch (lowest val loss)
  3. Prints a comparison table and writes:
       sensitivity/nc_summary_table.csv    – one row per run
       sensitivity/nc_auroc_curves.csv     – epoch-level AUROC for all runs
       sensitivity/nc_summary.txt          – human-readable report
"""

import argparse
import os
import sys
import pandas as pd

NC_ORDER = [
    'tcell_original',
    'nc_gene_perm_tcell',
    'nc_coord_perm_tcell',
    'nc_label_perm_tcell',
    'nc_intra_comp_tcell',
]

NC_LABELS = {
    'tcell_original':       'Original (no permutation)',
    'nc_gene_perm_tcell':   'Gene permutation',
    'nc_coord_perm_tcell':  'Spatial coord permutation',
    'nc_label_perm_tcell':  'Cell-type label permutation',
    'nc_intra_comp_tcell':  'Within-compartment expr shuffle',
}


def load_run(run_dir):
    metrics_path = os.path.join(run_dir, 'training_metrics.csv')
    if not os.path.isfile(metrics_path):
        return None
    try:
        df = pd.read_csv(metrics_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"  [warn] Could not read {metrics_path}: {e}", file=sys.stderr)
        return None


def run_summary(name, df):
    best_idx = df['Val Loss'].idxmin()
    best = df.loc[best_idx]
    final = df.iloc[-1]
    auroc_vals = df['Val AUROC'].dropna()

    return {
        'run': name,
        'label': NC_LABELS.get(name, name),
        'n_epochs': len(df),
        'best_epoch': int(best['Epoch']),
        'best_val_loss': round(float(best['Val Loss']), 5),
        'best_val_auroc': round(float(best['Val AUROC']), 4),
        'final_val_auroc': round(float(final['Val AUROC']), 4),
        'mean_val_auroc': round(float(auroc_vals.mean()), 4),
        'train_loss_epoch1': round(float(df.iloc[0]['Train Loss']), 5),
        'train_loss_final': round(float(final['Train Loss']), 5),
    }


def print_table(rows, f=None):
    header = (
        f"{'Run':<35} {'Label':<38} {'Best Ep':>7} "
        f"{'Best VLoss':>11} {'Best AUROC':>11} {'Final AUROC':>12}"
    )
    sep = '-' * len(header)

    def emit(line):
        print(line)
        if f:
            f.write(line + '\n')

    emit(sep)
    emit(header)
    emit(sep)
    for r in rows:
        line = (
            f"{r['run']:<35} {r['label']:<38} {r['best_epoch']:>7} "
            f"{r['best_val_loss']:>11.5f} {r['best_val_auroc']:>11.4f} {r['final_val_auroc']:>12.4f}"
        )
        emit(line)
    emit(sep)


def interpretation(rows):
    lines = []
    orig = next((r for r in rows if r['run'] == 'tcell_original'), None)
    lines.append("\n=== Interpretation ===")
    if orig is None:
        lines.append("  Original run not found; add sensitivity/tcell_original for baseline comparison.")
    else:
        base = orig['best_val_auroc']
        lines.append(f"  Baseline (original) best AUROC: {base:.4f}")
        for r in rows:
            if r['run'] == 'tcell_original':
                continue
            delta = r['best_val_auroc'] - base
            flag = "OK  – drops as expected" if r['best_val_auroc'] < base - 0.05 else "WARN – unexpectedly close to baseline"
            lines.append(f"  {r['label']:<38}  AUROC {r['best_val_auroc']:.4f}  (Δ {delta:+.4f})  {flag}")
    lines.append("")
    lines.append("  Under any true negative control the model should achieve AUROC ≈ 0.5.")
    lines.append("  AUROC well above 0.5 in a permuted run suggests the model is learning")
    lines.append("  spurious structure (tissue compartment bias, abundance bias, etc.).")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Summarise negative-control training runs.')
    parser.add_argument('--base_dir', type=str,
                        default='sensitivity_perm_gene_coords_label_compart',
                        help='Root directory containing run sub-directories.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for summary files (default: same as base_dir).')
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir or base_dir
    os.makedirs(out_dir, exist_ok=True)

    # Discover all run directories
    candidates = sorted(os.listdir(base_dir)) if os.path.isdir(base_dir) else []
    found = {
        name: os.path.join(base_dir, name)
        for name in candidates
        if os.path.isfile(os.path.join(base_dir, name, 'training_metrics.csv'))
    }

    if not found:
        print(f"No completed runs found under '{base_dir}'. Ensure training_metrics.csv exists in each sub-directory.")
        sys.exit(0)

    # Preferred ordering; append any extra runs alphabetically
    ordered_names = [n for n in NC_ORDER if n in found]
    ordered_names += sorted(n for n in found if n not in NC_ORDER)

    rows = []
    curve_frames = []
    for name in ordered_names:
        df = load_run(found[name])
        if df is None:
            continue
        summary = run_summary(name, df)
        rows.append(summary)
        # Build curve frame
        curve = df[['Epoch', 'Val AUROC', 'Val Loss', 'Train Loss']].copy()
        curve.insert(0, 'run', name)
        curve.insert(1, 'label', NC_LABELS.get(name, name))
        curve_frames.append(curve)
        print(f"  Loaded: {name} ({len(df)} epochs)")

    # --- Summary table CSV ---
    summary_csv = os.path.join(out_dir, 'nc_summary_table.csv')
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"\nSummary table saved → {summary_csv}")

    # --- AUROC curves CSV ---
    if curve_frames:
        curves_csv = os.path.join(out_dir, 'nc_auroc_curves.csv')
        pd.concat(curve_frames, ignore_index=True).to_csv(curves_csv, index=False)
        print(f"AUROC curves saved  → {curves_csv}")

    # --- Human-readable report ---
    report_path = os.path.join(out_dir, 'nc_summary.txt')
    with open(report_path, 'w') as f:
        f.write("SPACER – Negative Control Summary\n")
        f.write("=" * 80 + '\n\n')
        f.write("Run directories scanned: " + base_dir + '\n')
        f.write(f"Runs found: {len(rows)}\n\n")
        print_table(rows, f=f)
        interp = interpretation(rows)
        print(interp)
        f.write(interp + '\n')

    print(f"Report saved          → {report_path}")

    # Print table to console as well
    print()
    print_table(rows)


if __name__ == '__main__':
    main()
