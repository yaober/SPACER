from torch.utils.data import Dataset
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from tqdm import trange

from torch.utils.data import Dataset
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from tqdm import trange
from scipy.sparse import issparse

def preprocess_data(adata, immune_cell, n_genes):
    # Read the data
    if immune_cell == 'tcell':
        immune_cell = 'T'
    elif immune_cell == 'bcell':
        immune_cell = 'B'
    else:
        raise ValueError('Invalid immune cell type')

    # Ensure adata is not a view
    adata = adata.copy()

    # Filter the tumor cells
    tumor_cells = adata[adata.obs['cell_type'].astype(int) == 1].copy()

    # Check if the expression matrix is sparse and convert to dense if necessary
    if issparse(tumor_cells.X):
        tumor_cells_X_dense = tumor_cells.X.toarray()
    else:
        tumor_cells_X_dense = tumor_cells.X

    # Calculate mean expression
    mean_expression = tumor_cells_X_dense.mean(axis=0)

    # Select top n genes
    top_n_genes = mean_expression.argsort()[-n_genes:][::-1]
    adata = adata[:, top_n_genes].copy()
<<<<<<< HEAD
    adata.obs[immune_cell] = adata.obs[immune_cell].astype(float)
    tumor_cells.obs[immune_cell] = tumor_cells.obs[immune_cell].astype(float)
    # Calculate the 50th percentile of the immune cell column
    percentile_value = np.percentile(tumor_cells.obs[immune_cell], 50)
=======
    adata.obs[immune_cell] = tumor_cells.obs[immune_cell].astype(float)
    # Calculate the 50th percentile of the immune cell column
    percentile_value = np.percentile(adata.obs[immune_cell], 50)
>>>>>>> 891b0aff2698f7f173096783187843475284a3c7

    # Binarize the immune cell column based on the percentile value
    adata.obs[immune_cell] = np.where(adata.obs[immune_cell] >= percentile_value, 1, 0)

    return adata

class BagsDataset(Dataset):
    def __init__(self, input_data, immune_cell, max_instances=None, radius=200, resolution='low',n_genes=500):
        self.immune_cell = immune_cell
        self.max_instances = max_instances
        self.radius = radius
        self.resolution = resolution
        self.n_genes = n_genes
        if isinstance(input_data, str):
            self.bags = self.create_bags_from_csv(input_data)
        elif isinstance(input_data, sc.AnnData):
            input_data = preprocess_data(input_data, immune_cell,n_genes)
            print(f"Preprocessed data: {input_data.X.shape}")
            self.bags = self.create_bags_from_adata(input_data)
        else:
            raise ValueError("input_data must be either a path to a CSV file or an AnnData object")

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        distances = torch.tensor(bag['distances'], dtype=torch.float32)
        gene_expression = bag['gene_expression']
        if sp.issparse(gene_expression):
            gene_expression = torch.tensor(gene_expression.todense(), dtype=torch.float32)
        else:
            gene_expression = torch.tensor(gene_expression, dtype=torch.float32)
        label = torch.tensor(bag['label'], dtype=torch.float32)
        core_idx = bag['core_idx']
        gene_names = bag['gene_names']
        return distances, gene_expression, label, core_idx, gene_names

    def create_bags_from_csv(self, csv_file):
        data = pd.read_csv(csv_file)
        adata_radius_list = []
        for _, row in data.iterrows():
            adata_path = row['adata']
            adata = sc.read_h5ad(adata_path)
            adata = preprocess_data(adata, self.immune_cell, self.n_genes)
            
            radius = row['radius'] if 'radius' in row and not pd.isna(row['radius']) else self.radius
            resolution = row['resolution'] if 'resolution' in row and not pd.isna(row['resolution']) else self.resolution
            adata_radius_list.append((adata, radius, resolution))
            print(f"Processing: adata={adata_path.split('/')[-1]}, radius={radius}, resolution={resolution}")
        return self.create_bags(adata_radius_list)

    def create_bags_from_adata(self, adata):
        adata_radius_list = [(adata, self.radius, self.resolution)]
        return self.create_bags(adata_radius_list)

    def create_bags(self, adata_radius_list):
        bags = {}
        bag_id = 0

        for adata, radius, resolution in adata_radius_list:
            spatial_coords_x = adata.obs['X'].astype(float)
            spatial_coords_y = adata.obs['Y'].astype(float)
            spatial_coords = np.array(list(zip(spatial_coords_x, spatial_coords_y)))
            gene_expression = adata.X
            if self.immune_cell == 'tcell':
                labels = adata.obs['T'].values
            elif self.immune_cell == 'bcell':
                labels = adata.obs['B'].values
            else:
                raise ValueError("immune_cell must be either 'tcell' or 'bcell'")
            adata.obs['cell_type'] = adata.obs['cell_type'].astype(int)
            cell_types = adata.obs['cell_type'].values
            barcodes = adata.obs.index.values
            gene_names = adata.var_names.tolist()
            
            for i in trange(len(spatial_coords), desc=f"Creating Bags with radius {radius}", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
                if cell_types[i] == 0:
                    continue

                dist_matrix_row = cdist([spatial_coords[i]], spatial_coords, metric='euclidean')[0]
                in_circle = np.where(dist_matrix_row <= radius)[0]
                in_circle = [idx for idx in in_circle if cell_types[idx] != 0]

                if resolution == 'high':
                    if cell_types[i] == 1:
                        in_circle.append(i)
                    else:
                        in_circle = [idx for idx in in_circle if idx != i]

                if len(in_circle) == 0:
                    continue

                if self.max_instances is not None and len(in_circle) > self.max_instances:
                    continue

                gene_data = gene_expression[in_circle]
                distances = np.asmatrix(dist_matrix_row[in_circle].reshape(-1, 1), dtype=np.float32)

                bags[bag_id] = {
                    'distances': distances,
                    'gene_expression': gene_data,
                    'label': labels[i],
                    'core_idx': i,
                    'gene_names': gene_names
                }

                bag_id += 1

        total_bags = len(bags)
        avg_instances_per_bag = sum(bags[i]['gene_expression'].shape[0] for i in bags) / total_bags if total_bags > 0 else 0
        print(f"Total bags created: {total_bags}")
        print(f"Average instances per bag: {avg_instances_per_bag:.0f}")

        return bags

def custom_collate_fn(batch):
    distances, gene_expressions, labels, core_idxs, gene_names_list = zip(*batch)
    distances = [torch.tensor(np.array(d), dtype=torch.float32) for d in distances]
    gene_expressions_tensors = []
    for g in gene_expressions:
        if sp.issparse(g):
            gene_expressions_tensors.append(torch.tensor(g.todense(), dtype=torch.float32))
        else:
            gene_expressions_tensors.append(g.clone().detach().float())
    labels = torch.tensor(labels, dtype=torch.float32)
    core_idxs = torch.tensor(core_idxs, dtype=torch.long)
    gene_names = gene_names_list
    return distances, gene_expressions_tensors, labels, core_idxs, gene_names
