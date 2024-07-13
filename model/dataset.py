from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
from scipy.spatial.distance import cdist
import scipy.sparse as sp
from tqdm import trange

class BagsDataset(Dataset):
    def __init__(self, adata, radius=50, output_csv='bags.csv'):
        self.bags = self.create_bags(adata, radius, output_csv)

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
        return distances, gene_expression, label

    def create_bags(self, adata, radius, output_csv):
        spatial_coords_x = adata.obs['X'].astype(float)
        spatial_coords_y = adata.obs['Y'].astype(float)
        spatial_coords = np.array(list(zip(spatial_coords_x, spatial_coords_y)))
        gene_expression = adata.X
        labels = adata.obs['tcr'].values
        cell_types = adata.obs['cell_type'].values
        barcodes = adata.obs.index.values

        bags = {}
        csv_data = []
        filtered_count = 0
        no_neighbors_count = 0
        bag_id = 0  # Initialize bag_id to ensure continuous IDs starting from 0

        for i in trange(len(spatial_coords), desc="Creating Bags", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
            if cell_types[i] == 0:
                continue  # Skip if the cell type is 0

            filtered_count += 1
            dist_matrix_row = cdist([spatial_coords[i]], spatial_coords, metric='euclidean')[0]
            in_circle = np.where(dist_matrix_row <= radius)[0]
            in_circle = [idx for idx in in_circle if cell_types[idx] != 0]  # Filter based on cell type
            if len(in_circle) == 0:
                no_neighbors_count += 1
                continue  # Skip if no instances meet the criteria

            gene_data = gene_expression[in_circle]
            distances = np.asmatrix(dist_matrix_row[in_circle].reshape(-1, 1), dtype=np.float32)

            bags[bag_id] = {
                'distances': distances,
                'gene_expression': gene_data,
                'label': labels[i]
            }

            bag_barcodes = barcodes[in_circle]
            for barcode in bag_barcodes:
                csv_data.append([bag_id, barcode, labels[i]])

            bag_id += 1  # Increment bag_id for the next bag

        total_bags = len(bags)
        avg_instances_per_bag = sum(len(bags[i]['gene_expression']) for i in bags) / total_bags
        print(f"Total bags created: {total_bags}")
        print(f"Average instances per bag: {avg_instances_per_bag:.0f}")

        return bags

def custom_collate_fn(batch):
    distances, gene_expressions, labels = zip(*batch)
    distances = [torch.tensor(np.array(d), dtype=torch.float32) for d in distances]
    gene_expressions_tensors = []
    for g in gene_expressions:
        if sp.issparse(g):
            gene_expressions_tensors.append(torch.tensor(g.todense(), dtype=torch.float32))
        else:
            gene_expressions_tensors.append(g.clone().detach().float())
    labels = torch.tensor(labels, dtype=torch.float32)
    return distances, gene_expressions_tensors, labels
