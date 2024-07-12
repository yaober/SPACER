import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import cdist
import scanpy as sc
import torch

class BagsDataset(Dataset):
    def __init__(self, adata, radius=50, output_csv='bags.csv'):
        self.bags = self.create_bags(adata, radius, output_csv)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        distances = torch.tensor(bag['distances'], dtype=torch.float32)
        gene_expression = torch.tensor(bag['gene_expression'], dtype=torch.float32)
        label = torch.tensor(bag['label'], dtype=torch.float32)
        return distances, gene_expression, label

    def create_bags(self, adata, radius, output_csv):
        spatial_coords_x = adata.obs['X']
        spatial_coords_y = adata.obs['Y']
        spatial_coords = np.array(list(zip(spatial_coords_x, spatial_coords_y)))
        gene_expression = adata.X
        labels = adata.obs['tcr'].values
        barcodes = adata.obs.index.values
        bags = {}
        dist_matrix = cdist(spatial_coords, spatial_coords, metric='euclidean')
        csv_data = []

        for i in range(len(spatial_coords)):
            in_circle = np.where(dist_matrix[i] <= radius)[0]
            gene_data = gene_expression[in_circle].todense()
            distances = np.asmatrix(dist_matrix[i][in_circle].reshape(-1, 1), dtype=np.float32)

            if i == 0: 
                print(f"Checking data for bag {i}:")
                print(f"Number of cells in this bag: {len(in_circle)}")
                print(f"Sample of in_circle indices: {in_circle[:5]}")
                print(f"Shape of gene_data: {gene_data.shape}")

            bags[i] = {
                'distances': distances,
                'gene_expression': gene_data,
                'label': labels[i]
            }

            bag_barcodes = barcodes[in_circle]
            for barcode in bag_barcodes:
                csv_data.append([i, barcode, labels[i]])

            print(f"Bag {i} has {gene_data.shape[0]} instances")

        df = pd.DataFrame(csv_data, columns=['bag_id', 'barcode', 'label'])
        df.to_csv(output_csv, index=False)
        return bags

def custom_collate_fn(batch):
    distances, gene_expressions, labels = zip(*batch)
    distances = [torch.tensor(np.array(d), dtype=torch.float32) for d in distances]
    gene_expressions = [torch.tensor(np.array(g), dtype=torch.float32) for g in gene_expressions]
    labels = torch.tensor(labels, dtype=torch.float32)
    return distances, gene_expressions, labels
