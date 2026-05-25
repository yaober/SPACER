#!/usr/bin/env python3
"""
Download and parse GO annotations for human genes (UniProt-GOA).
=====================================================================
Downloads the human Gene Association File (GAF 2.2) from EBI and counts
the number of unique GO terms annotated to each gene symbol.

Source: https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz

GAF 2.x column layout (tab-separated, 0-indexed):
  0  DB                  (e.g. UniProtKB)
  1  DB_Object_ID        (UniProt accession)
  2  DB_Object_Symbol    (gene symbol  <-- used here)
  4  GO_ID
  6  Evidence_Code
  8  Aspect              P = Biological Process (BP)
                         F = Molecular Function (MF)
                         C = Cellular Component (CC)

Output (--output_dir/gene_go_counts.csv):
  Gene          -- HGNC gene symbol
  n_go_terms    -- number of unique GO terms (all namespaces)
  n_bp_terms    -- unique Biological Process terms
  n_mf_terms    -- unique Molecular Function terms
  n_cc_terms    -- unique Cellular Component terms
"""

import argparse
import gzip
import os
import sys
import urllib.request
from collections import defaultdict

import pandas as pd

GOA_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz"


def download_gaf(url, dest_path):
    print(f"Downloading GO annotation file from EBI ...")
    print(f"  URL : {url}")
    print(f"  Dest: {dest_path}")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  {pct:3d}%", end='', flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
    print()
    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"  Downloaded {size_mb:.1f} MB")


def parse_gaf(gaf_gz_path, evidence_codes=None):
    """
    Count unique GO terms per gene symbol.

    Parameters
    ----------
    gaf_gz_path   : str   path to .gaf.gz file
    evidence_codes: set | None   restrict to these codes; None = all

    Returns
    -------
    dict: gene_symbol -> {'all': set, 'BP': set, 'MF': set, 'CC': set}
    """
    aspect_map = {'P': 'BP', 'F': 'MF', 'C': 'CC'}
    gene_go = defaultdict(lambda: {'all': set(), 'BP': set(), 'MF': set(), 'CC': set()})

    n_lines = n_skipped = 0
    print("Parsing GAF file ...")

    with gzip.open(gaf_gz_path, 'rt') as fh:
        for line in fh:
            if line.startswith('!'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                n_skipped += 1
                continue

            gene   = parts[2].strip()
            go_id  = parts[4].strip()
            evcode = parts[6].strip()
            aspect = aspect_map.get(parts[8].strip())

            if evidence_codes and evcode not in evidence_codes:
                n_skipped += 1
                continue
            if not gene or not go_id:
                n_skipped += 1
                continue

            gene_go[gene]['all'].add(go_id)
            if aspect:
                gene_go[gene][aspect].add(go_id)
            n_lines += 1

    print(f"  Parsed {n_lines:,} annotations across {len(gene_go):,} genes "
          f"({n_skipped:,} lines skipped).")
    return gene_go


def main():
    parser = argparse.ArgumentParser(
        description='Download GO human annotations and count terms per gene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--output_dir', default='.',
                        help='Directory to write gene_go_counts.csv (and cached .gaf.gz).')
    parser.add_argument('--gaf_path', default=None,
                        help='Path to a pre-downloaded goa_human.gaf.gz. '
                             'If omitted, the file is downloaded from EBI and '
                             'cached in --output_dir.')
    parser.add_argument('--evidence_codes', nargs='*', default=None,
                        help='Restrict to these GO evidence codes '
                             '(e.g. EXP IDA IMP IPI IGI IEP). '
                             'Default: all codes including IEA (electronic).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve / download GAF file
    gaf_path = args.gaf_path
    if gaf_path is None:
        gaf_path = os.path.join(args.output_dir, 'goa_human.gaf.gz')
        if os.path.exists(gaf_path):
            size_mb = os.path.getsize(gaf_path) / 1e6
            print(f"Using cached GAF file ({size_mb:.1f} MB): {gaf_path}")
        else:
            download_gaf(GOA_URL, gaf_path)

    ev_set = set(args.evidence_codes) if args.evidence_codes else None
    if ev_set:
        print(f"Filtering to evidence codes: {sorted(ev_set)}")

    gene_go = parse_gaf(gaf_path, evidence_codes=ev_set)

    # Build output dataframe
    rows = [
        {
            'Gene':       gene,
            'n_go_terms': len(sets['all']),
            'n_bp_terms': len(sets['BP']),
            'n_mf_terms': len(sets['MF']),
            'n_cc_terms': len(sets['CC']),
        }
        for gene, sets in gene_go.items()
    ]
    df = (pd.DataFrame(rows)
            .sort_values('n_go_terms', ascending=False)
            .reset_index(drop=True))

    out_path = os.path.join(args.output_dir, 'gene_go_counts.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} genes → {out_path}")

    print("\nTop 10 most annotated genes:")
    print(df.head(10)[['Gene', 'n_go_terms', 'n_bp_terms',
                        'n_mf_terms', 'n_cc_terms']].to_string(index=False))
    print(f"\nAnnotation count distribution:")
    print(df['n_go_terms'].describe().round(1).to_string())


if __name__ == '__main__':
    main()
