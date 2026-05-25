"""
create_example_data.py
Generates synthetic spatial transcriptomics data for train_show.ipynb and predict_show.ipynb.

Outputs
-------
data/example_genes.csv        – 300-gene reference list (use in place of human_filtered.csv)
data/example_spatial.h5ad     – AnnData: 600 cells × 97 genes, with spatial layout
data/example_sample.csv       – multi-sample CSV pointing at example_spatial.h5ad
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# ── 1. Reference gene list ───────────────────────────────────────────────────
TUMOR_GENES = [
    "TAP2","IFI6","TOP2A","PBK","TPX2","PRAME","MUC1","EPCAM",
    "PMEL","MLANA","HORMAD1","KRT8","KRT18","KRT19","ERBB2",
    "MAGEA3","MAGEA4","MAGEA10","AFP","SOX2","WT1","HOXB9",
    "MYC","CCND1","CDK4","CDK6","AURKA","BCL2","BIRC5","MCL1",
    "CD274","MMP2","MMP9","VEGFA","HIF1A","LDHA","VIM","SNAI1",
    "EGFR","CXCL8","STAT3","KRAS","TP53",
]
HLA_GENES = [
    "HLA-A","HLA-B","HLA-C","HLA-E","HLA-F",
    "HLA-DRA","HLA-DRB1","HLA-DRB5","HLA-DQA1","HLA-DQB1",
    "HLA-DPA1","HLA-DPB1",
]
SIGNAL_GENES = [
    # T-cell signal
    "CXCL9","CXCL10","GZMB","PRF1","CD8A","CD8B",
    "IFNG","TNF","IL2","FOXP3","CTLA4","PDCD1",
    # Macrophage signal
    "CD163","MRC1","CSF1R","FCGR3A","CD14",
    "IL10","TGFB1","CCL2","CCL7","MMP12","VEGFB",
]
FILLER = [f"GENE{i:04d}" for i in range(1, 300)]

ALL_GENES = list(dict.fromkeys(TUMOR_GENES + HLA_GENES + SIGNAL_GENES + FILLER))
pd.DataFrame({"Gene": ALL_GENES}).to_csv("data/example_genes.csv", index=False)
print(f"[1] Gene reference list: {len(ALL_GENES)} genes  →  data/example_genes.csv")

# ── 2. Spatial layout ────────────────────────────────────────────────────────
# 20×20 grid of tumor cells (cell_type=1) at 20-px spacing → coords 0..380
GRID, STEP = 20, 20
gx = np.tile(np.arange(GRID) * STEP, GRID).astype(float)
gy = np.repeat(np.arange(GRID) * STEP, GRID).astype(float)
n_tumor = len(gx)  # 400

# 200 non-tumor / stromal / immune cells (cell_type=0)
n_other = 200
ox = np.random.uniform(0, GRID * STEP, n_other)
oy = np.random.uniform(0, GRID * STEP, n_other)

X_coord = np.concatenate([gx, ox])
Y_coord = np.concatenate([gy, oy])
n_cells  = n_tumor + n_other
cell_type = np.array([1]*n_tumor + [0]*n_other, dtype=int)

# ── 3. Labels ────────────────────────────────────────────────────────────────
# T-cell infiltration:  tumor cells in the upper half (Y < half of tissue)
T_lbl   = ((cell_type == 1) & (Y_coord < GRID * STEP / 2)).astype(int)
# Macrophage infiltration: tumor cells in the right half (X > half of tissue)
Mac_lbl = ((cell_type == 1) & (X_coord > GRID * STEP / 2)).astype(int)
# B-cell: bottom-right quadrant
B_lbl   = ((cell_type == 1) & (Y_coord > GRID * STEP * 0.6) & (X_coord > GRID * STEP * 0.6)).astype(int)

# ── 4. Gene expression ───────────────────────────────────────────────────────
VAR_GENES = list(dict.fromkeys(TUMOR_GENES + HLA_GENES + SIGNAL_GENES + FILLER[:20]))
n_genes = len(VAR_GENES)

# Sparse log-normal base expression (~60 % zeros)
X_expr = np.random.lognormal(0.0, 1.0, (n_cells, n_genes)).astype(np.float32)
X_expr = np.clip(X_expr - 1.5, 0.0, None)

# Boost T-cell markers in T-infiltrated spots
t_mask = np.where(T_lbl == 1)[0]
t_idx  = [VAR_GENES.index(g) for g in ["CXCL9","CXCL10","GZMB","CD8A","IFNG"] if g in VAR_GENES]
X_expr[np.ix_(t_mask, t_idx)] += np.random.lognormal(1.5, 0.5, (len(t_mask), len(t_idx)))

# Boost Macrophage markers in macrophage-infiltrated spots
m_mask = np.where(Mac_lbl == 1)[0]
m_idx  = [VAR_GENES.index(g) for g in ["CD163","MRC1","CSF1R","CCL2","IL10"] if g in VAR_GENES]
X_expr[np.ix_(m_mask, m_idx)] += np.random.lognormal(1.5, 0.5, (len(m_mask), len(m_idx)))

X_sparse = sp.csr_matrix(X_expr)

# ── 5. Build obs DataFrame ───────────────────────────────────────────────────
barcodes = [f"CELL_{i:05d}-1" for i in range(n_cells)]
obs = pd.DataFrame({
    "X":          X_coord,
    "Y":          Y_coord,
    "cell_type":  cell_type,
    "T":          T_lbl,
    "Macrophage": Mac_lbl,
    "B":          B_lbl,
    "Endothelial": np.zeros(n_cells, int),
    "Fibroblast":  np.zeros(n_cells, int),
    "in_tissue":  np.ones(n_cells, int),
    "array_row":  (Y_coord / STEP).astype(int),
    "array_col":  (X_coord / STEP).astype(int),
    "n_genes":    (X_expr > 0).sum(axis=1).astype(int),
    "leiden":     np.random.randint(0, 8, n_cells),
    # dummy gene-signature scores
    "tumor_gene_signature":          np.random.uniform(0, 20, n_cells),
    "stromal_immune_gene_signature": np.random.uniform(0, 150, n_cells),
    "t_gene_signature":              T_lbl.astype(float) * np.random.uniform(0, 10, n_cells),
    "b_gene_signature":              B_lbl.astype(float) * np.random.uniform(0, 8, n_cells),
    "endothelial_gene_signature":    np.random.uniform(0, 5, n_cells),
    "fibroblast_gene_signature":     np.random.uniform(0, 15, n_cells),
}, index=barcodes)

var = pd.DataFrame(index=VAR_GENES)
adata = ad.AnnData(X=X_sparse, obs=obs, var=var)

out_h5 = "data/example_spatial.h5ad"
adata.write_h5ad(out_h5)
print(f"[2] AnnData {adata.shape}  →  {out_h5}")
print(f"    T+          (tumor): {T_lbl.sum():3d} / {(cell_type==1).sum()} tumor cells")
print(f"    Macrophage+ (tumor): {Mac_lbl.sum():3d} / {(cell_type==1).sum()} tumor cells")
print(f"    B+          (tumor): {B_lbl.sum():3d} / {(cell_type==1).sum()} tumor cells")

# ── 6. Multi-sample CSV ──────────────────────────────────────────────────────
pd.DataFrame({
    "adata":      [out_h5, out_h5],
    "radius":     [50,     50],
    "resolution": ["low",  "low"],
}).to_csv("data/example_sample.csv", index=False)
print("[3] Sample CSV  →  data/example_sample.csv")
print("\nDone. Use these files in the notebooks as shown in the example cells.")
