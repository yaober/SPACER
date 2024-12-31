from torch.utils.data import Dataset
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from tqdm import trange
from scipy.sparse import issparse


def preprocess_data(adata, immune_cell, n_genes, resolution):
    # Read the data

    # Ensure adata is not a view
    adata = adata.copy()
    adata.var_names_make_unique()  # Ensure unique gene names

    # Filter the tumor and non-tumor cells
    print(adata.obs['cell_type'].unique())
    tumor_cells = adata[adata.obs['cell_type'].astype(int) == 1].copy()
    non_tumor_cells = adata[adata.obs['cell_type'].astype(int) != 1].copy()

    # Debug: Check tumor cells
    print(f"Tumor cells shape after filtering: {tumor_cells.shape}")
    print(f"Non-tumor cells shape after filtering: {non_tumor_cells.shape}")
    if tumor_cells.shape[0] == 0:
        print("Warning: No tumor cells found after filtering.")
        return None  # Stop processing if no tumor cells

    # Calculate mean expression for tumor and non-tumor cells
    if issparse(tumor_cells.X):
        mean_expression_tumor = np.asarray(tumor_cells.X.mean(axis=0)).ravel()
    else:
        mean_expression_tumor = tumor_cells.X.mean(axis=0)

    if issparse(non_tumor_cells.X):
        mean_expression_non_tumor = np.asarray(non_tumor_cells.X.mean(axis=0)).ravel()
    else:
        mean_expression_non_tumor = non_tumor_cells.X.mean(axis=0)

    # Avoid division by zero by adding a small epsilon
    mean_expression_non_tumor += 1e-10

    # Calculate relative values
    relative_values = mean_expression_tumor - mean_expression_non_tumor

    # Get gene names
    gene_names = tumor_cells.var_names

    # Select top n genes based on relative values
    print(f"Selecting top {n_genes} genes based on relative values")
    if n_genes > len(gene_names):
        n_genes = int(len(gene_names) * 0.2)
    top_n_gene_indices = relative_values.argsort()[-n_genes:][::-1]
    top_n_gene_names = gene_names[top_n_gene_indices]

    # Include additional tumor-related genes and filter out unwanted ones
    tumor_genes = [
        # Possible tumor antigens or genes that promote tumor antigen presentation
        'TAP2', 'IFI6', 'TOP2A', 'PBK', 'TPX2', 'PRAME', 'MUC1', 'MUC12', 'CEACAM1', 'EPCAM', 'PMEL', 'MLANA',
        'LAGE3', 'HORMAD1', 'CTAG1B', 'KRT8', 'KRT18', 'KRT19', 'ERBB2', 'MAGEA3', 'MAGEA4', 'MAGEA10', 'AFP',
        'CEACAM5', 'SOX2', 'SLC45A2', 'WT1'
    ]
    hla_genes = list(adata.var_names[adata.var_names.str.startswith("HLA")])
    select_genes = tumor_genes + hla_genes + list(top_n_gene_names)
    existing_genes = [gene for gene in select_genes if gene in adata.var_names]

    genes_to_exclude = ["CD68", "STAT1", "MMP13", "EPDR1", "CLCA1", "FBLN1", "C9orf16", "ADGRF1", "LINGO2"]
    existing_genes = [gene for gene in existing_genes if gene not in genes_to_exclude]

    # Subset adata using selected genes
    adata = adata[:, existing_genes].copy()

    adata.obs[immune_cell] = adata.obs[immune_cell].astype(float)
    tumor_cells.obs[immune_cell] = tumor_cells.obs[immune_cell].astype(float)

    # Binarize the immune cell column based on the percentile value if resolution is not 'high'
    if resolution != 'high':
        if tumor_cells.obs[immune_cell].empty:
            print(f"Error: tumor_cells.obs[{immune_cell}] is empty.")
        else:
            unique_values = tumor_cells.obs[immune_cell].unique()
            if set(unique_values).issubset({0, 1}):
                print(f"tumor_cells.obs[{immune_cell}] is already binary. Skipping binarization.")
            else:
                percentile_value = np.percentile(tumor_cells.obs[immune_cell], 75)
                print(f"Percentile value: {percentile_value}")
                adata.obs[immune_cell] = np.where(adata.obs[immune_cell] > percentile_value, 1, 0)
                print(f"adata.obs[{immune_cell}] after binarization: {adata.obs[immune_cell].head()}")

    return adata



class BagsDataset(Dataset):
    def __init__(self, input_data, immune_cell, max_instances=None, radius=200, resolution='low', n_genes=500, k=2):
        self.immune_cell = map_immune_cell(immune_cell)
        self.max_instances = max_instances
        self.radius = radius
        self.resolution = resolution
        self.n_genes = n_genes
        self.k = k  # Number of bags per batch
        if isinstance(input_data, str):
            self.batches = self.create_bags_from_csv(input_data)
        elif isinstance(input_data, sc.AnnData):
            input_data = preprocess_data(input_data, immune_cell, n_genes, self.resolution)
            print(f"Preprocessed data: {input_data.X.shape}")
            self.batches = self.create_bags_from_adata(input_data)
        else:
            raise ValueError("input_data must be either a path to a CSV file or an AnnData object")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]
        # batch is a list of bags
        batch_data = []
        for bag in batch:
            distances = bag['distances']
            gene_expression = bag['gene_expression']
            label = bag['label']
            core_idx = bag['core_idx']
            gene_names = bag['gene_names']
            cell_id = bag['cell_id']
            bag_dict = {
                'distances': distances,
                'gene_expression': gene_expression,
                'label': label,
                'core_idx': core_idx,
                'gene_names': gene_names,
                'cell_id': cell_id
            }
            batch_data.append(bag_dict)
        return batch_data

    def create_bags_from_csv(self, csv_file):
        data = pd.read_csv(csv_file)
        adata_radius_list = []
        for _, row in data.iterrows():
            adata_path = row['adata']
            print(f"Reading adata from {adata_path}")
            resolution = row['resolution'] if 'resolution' in row and not pd.isna(row['resolution']) else self.resolution
            adata = sc.read_h5ad(adata_path)
            adata.obs_names_make_unique()
            adata = preprocess_data(adata, self.immune_cell, self.n_genes, resolution=resolution)
            radius = row['radius'] if 'radius' in row and not pd.isna(row['radius']) else self.radius
            adata_radius_list.append((adata, radius, resolution))
            print(f"Processing: adata={adata_path.split('/')[-1]}, radius={radius}, resolution={resolution}")
        return self.create_bags(adata_radius_list)

    def create_bags_from_adata(self, adata):
        adata_radius_list = [(adata, self.radius, self.resolution)]
        return self.create_bags(adata_radius_list)

    def create_bags(self, adata_radius_list):
        all_batches = []
        for adata, radius, resolution in adata_radius_list:
            # Collect positive and negative bags per adata
            positive_bags = []
            negative_bags = []
            spatial_coords_x = adata.obs['X'].astype(float)
            spatial_coords_y = adata.obs['Y'].astype(float)
            spatial_coords = np.array(list(zip(spatial_coords_x, spatial_coords_y)))
            gene_expression = adata.X
            labels = adata.obs[self.immune_cell].values.astype(int)  
            adata.obs['cell_type'] = adata.obs['cell_type'].astype(int)
            cell_types = adata.obs['cell_type'].values
            barcodes = adata.obs.index.values  
            gene_names = adata.var_names.tolist()

            for i in trange(len(spatial_coords), desc=f"Creating Bags with radius {radius}", ncols=100):
                if cell_types[i] == 0:
                    continue
                dist_matrix_row = cdist([spatial_coords[i]], spatial_coords, metric='euclidean')[0]
                in_circle = np.where(dist_matrix_row <= radius)[0]
                in_circle = [idx for idx in in_circle if cell_types[idx] == 1]
                num_tumor_cells = len(in_circle)
                if resolution == 'high' and num_tumor_cells < 10:
                    continue
                if resolution == 'high':
                    in_circle = [idx for idx in in_circle if idx != i]
                if len(in_circle) == 0:
                    continue
                if self.max_instances is not None and len(in_circle) > self.max_instances:
                    continue

                gene_data = gene_expression[in_circle]
                distances = np.asmatrix(dist_matrix_row[in_circle].reshape(-1, 1), dtype=np.float32)

                bag = {
                    'distances': distances,
                    'gene_expression': gene_data,
                    'label': labels[i],
                    'core_idx': i,
                    'gene_names': gene_names,
                    'cell_id': barcodes[i]
                }

                if labels[i] == 1:
                    positive_bags.append(bag)
                else:
                    negative_bags.append(bag)

            num_negative_per_batch = self.k - 1
            if len(negative_bags) < num_negative_per_batch:
                print(f"Not enough negative bags in this adata to create batches. Dropping extra positive bags.")
                num_batches = len(negative_bags) // num_negative_per_batch
                if num_batches == 0:
                    continue 
                if len(positive_bags) > num_batches:
                    positive_bags = positive_bags[:num_batches]
            else:
                num_batches = min(len(positive_bags), len(negative_bags) // num_negative_per_batch)
                if len(positive_bags) > num_batches:
                    positive_bags = positive_bags[:num_batches]
                if len(negative_bags) > num_batches * num_negative_per_batch:
                    negative_bags = negative_bags[:num_batches * num_negative_per_batch]
        
            np.random.shuffle(negative_bags)

            for i in range(num_batches):
                batch = [positive_bags[i]] + negative_bags[i * num_negative_per_batch: (i + 1) * num_negative_per_batch]
                all_batches.append(batch)

        total_batches = len(all_batches)
        print(f"Total batches created: {total_batches}")
        return all_batches



def custom_collate_fn(batch):
    
    batch_bags = batch[0]
    distances_list = []
    gene_expressions_list = []
    labels_list = []
    core_idxs_list = []
    gene_names_list = []
    cell_ids_list = []
    for bag_data in batch_bags:
        distances = torch.tensor(bag_data['distances'], dtype=torch.float32)
        gene_expression = bag_data['gene_expression']
        if sp.issparse(gene_expression):
            gene_expression = torch.tensor(gene_expression.todense(), dtype=torch.float32)
        else:
            gene_expression = torch.tensor(gene_expression, dtype=torch.float32)
        label = torch.tensor(bag_data['label'], dtype=torch.float32)
        core_idx = bag_data['core_idx']
        gene_names = bag_data['gene_names']
        cell_id = bag_data['cell_id']
        distances_list.append(distances)
        gene_expressions_list.append(gene_expression)
        labels_list.append(label)
        core_idxs_list.append(core_idx)
        gene_names_list.append(gene_names)
        cell_ids_list.append(cell_id)
    return distances_list, gene_expressions_list, labels_list, core_idxs_list, gene_names_list, cell_ids_list



def map_immune_cell(immune_cell):
    mapping = {
        'tcell': 'T',
        'bcell': 'B',
        'macrophage': 'Macrophage',
        'neutrophil': 'Neutrophil',
        'fibroblast': 'Fibroblast',
        'endothelial': 'Endothelial',
    }
    if immune_cell in mapping:
        return mapping[immune_cell]
    else:
        raise ValueError('Invalid immune cell type')
