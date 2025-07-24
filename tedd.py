import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = 'validation_tumor/tumor_antigen_developmental_stage/tedd_finish'
rank_files = {
    'T cell': pd.read_csv('validation_tumor/tumor_antigen_developmental_stage/tcell_ranked.csv'),
    'B cell': pd.read_csv('validation_tumor/tumor_antigen_developmental_stage/bcell_ranked.csv'),
    'Macrophage': pd.read_csv('validation_tumor/tumor_antigen_developmental_stage/macrophage_ranked.csv'),
    'Fibroblast': pd.read_csv('validation_tumor/tumor_antigen_developmental_stage/fibroblast_ranked.csv'),
    'Endothelial': pd.read_csv('validation_tumor/tumor_antigen_developmental_stage/endothelial_ranked.csv')
}

top_n_list = [20, 50, 100, 150, 200, 250, 300]

def analyze_mean_log_expr(rank_df, label, fetal_log, adult_log, top_n):
    top_genes = rank_df['Gene'].head(top_n).values
    top_genes = [g for g in top_genes if g in fetal_log.index and g in adult_log.index]
    return pd.DataFrame({
        'gene': top_genes,
        'fetal_log_expr': fetal_log[top_genes].values,
        'adult_log_expr': adult_log[top_genes].values,
        'cell_type': label
    })

def process_organ_folder(folder_path):
    organ_name = os.path.basename(folder_path)
    print(f"Processing: {organ_name}")

    count_path = os.path.join(folder_path, 'counts.csv')
    if not os.path.exists(count_path):
        count_path += '.gz'
        if not os.path.exists(count_path):
            print(f"Skipped {organ_name}: No count file found")
            return

    count = pd.read_csv(count_path)
    count = count.T

    meta_path = os.path.join(folder_path, 'meta.csv')
    if not os.path.exists(meta_path):
        print(f"Skipped {organ_name}: No meta file")
        return
    meta = pd.read_csv(meta_path, index_col=0)

    count = count.merge(meta, left_index=True, right_index=True, how='left')

    fetal = count[count['Timepoint'].str.startswith('GW')].drop(columns=['Tissue', 'Celltype', 'Timepoint', 'Sex', 'UMAP_1', 'UMAP_2'])
    adult = count[~count['Timepoint'].str.startswith('GW')].drop(columns=['Tissue', 'Celltype', 'Timepoint', 'Sex', 'UMAP_1', 'UMAP_2'])

    fetal_mean_log = fetal.mean(axis=0)
    adult_mean_log = adult.mean(axis=0)

    for top_n in top_n_list:
        results = []
        for label, rank_df in rank_files.items():
            res = analyze_mean_log_expr(rank_df, label, fetal_mean_log, adult_mean_log, top_n)
            results.append(res)

        combined = pd.concat(results, ignore_index=True)
        combined['fetal_minus_adult'] = combined['fetal_log_expr'] - combined['adult_log_expr']

        # 保存 CSV
        out_csv = os.path.join(folder_path, f'combined_log_expr_top{top_n}.csv')
        combined.to_csv(out_csv, index=False)

        # 保存图像
        try:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=combined, x='cell_type', y='fetal_minus_adult')
            plt.axhline(0, color='gray', linestyle='--')
            plt.title(f"{organ_name} | Top {top_n} Cell-Type-Specific Genes: Fetal - Adult Log Expression")
            plt.ylabel("Fetal - Adult (log-transformed expression)")
            plt.xlabel("Cell Type")
            plt.xticks(rotation=45)
            plt.tight_layout()

            fig_path = os.path.join(folder_path, f'fetal_minus_adult_boxplot_top{top_n}.png')
            plt.savefig(fig_path)
            plt.close()
        except Exception as e:
            print(f"Plotting failed for top_n={top_n}: {e}")

for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if os.path.isdir(subpath):
        process_organ_folder(subpath)
