from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
from scipy.spatial.distance import cdist
import scipy.sparse as sp
from tqdm import trange

class BagsDataset(Dataset):
    def __init__(self, adata_radius_input, immune_cell, radius=None, max_instances=None, resolution='high'):
        self.immune_cell = immune_cell
        self.max_instances = max_instances
        self.resolution = resolution
        if isinstance(adata_radius_input, list):
            self.bags = self.create_bags(adata_radius_input)
        else:
            assert radius is not None, "When a single adata is provided, radius must also be provided."
            self.bags = self.create_bags([(adata_radius_input, radius)])

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
        core_idx = bag['core_idx']  # Add core_idx to the output
        return distances, gene_expression, label, core_idx

    def create_bags(self, adata_radius_list):
        bags = {}
        bag_id = 0  # Initialize bag_id to ensure continuous IDs starting from 0

        for adata, radius in adata_radius_list:
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
            cell_types = adata.obs['cell_type'].values
            barcodes = adata.obs.index.values

            for i in trange(len(spatial_coords), desc=f"Creating Bags with radius {radius}", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
                if cell_types[i] == 0:
                    continue  # Skip if the cell type is 0

                dist_matrix_row = cdist([spatial_coords[i]], spatial_coords, metric='euclidean')[0]
                in_circle = np.where(dist_matrix_row <= radius)[0]
                in_circle = [idx for idx in in_circle if cell_types[idx] != 0]  # Filter based on cell type

                if self.resolution == 'high':
                    # Only include the core instance if its cell type is 1
                    if cell_types[i] == 1:
                        in_circle.append(i)
                    else:
                        in_circle = [idx for idx in in_circle if idx != i]

                if len(in_circle) == 0:
                    continue  # Skip if no instances meet the criteria

                # Drop the bag if the number of instances exceeds max_instances
                if self.max_instances is not None and len(in_circle) > self.max_instances:
                    continue

                gene_data = gene_expression[in_circle]
                distances = np.asmatrix(dist_matrix_row[in_circle].reshape(-1, 1), dtype=np.float32)

                bags[bag_id] = {
                    'distances': distances,
                    'gene_expression': gene_data,
                    'label': labels[i],
                    'core_idx': i  # Store the core index
                }

                bag_id += 1  # Increment bag_id for the next bag

        total_bags = len(bags)
        avg_instances_per_bag = sum(bags[i]['gene_expression'].shape[0] for i in bags) / total_bags if total_bags > 0 else 0
        print(f"Total bags created: {total_bags}")
        print(f"Average instances per bag: {avg_instances_per_bag:.0f}")

        return bags


def custom_collate_fn(batch):
    distances, gene_expressions, labels, core_idxs = zip(*batch)
    distances = [torch.tensor(np.array(d), dtype=torch.float32) for d in distances]
    gene_expressions_tensors = []
    for g in gene_expressions:
        if sp.issparse(g):
            gene_expressions_tensors.append(torch.tensor(g.todense(), dtype=torch.float32))
        else:
            gene_expressions_tensors.append(g.clone().detach().float())
    labels = torch.tensor(labels, dtype=torch.float32)
    core_idxs = torch.tensor(core_idxs, dtype=torch.long)  # Ensure core_idxs are tensors of type long
    return distances, gene_expressions_tensors, labels, core_idxs
