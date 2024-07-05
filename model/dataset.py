import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import cdist

class BagsDataset(Dataset):
    """
    without flatten for check the data
    
    """
    def __init__(self, adata, binding_aff, radius=50, output_csv='bags.csv'):
        self.bags = self.create_bags(adata, binding_aff, radius, output_csv)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        distances = bag['distances']
        gene_expression = bag['gene_expression']
        affinity_data = bag['affinity_data']
        label = bag['label']
        return distances, gene_expression, affinity_data, label

    def create_bags(self, adata, binding_aff, radius, output_csv):
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
            gene_data = gene_expression[in_circle].todense()  # Convert sparse matrix to dense
             # Use barcodes to get the correct affinity data
            circle_barcodes = barcodes[in_circle]
            affinity_data = np.asmatrix(binding_aff.loc[circle_barcodes].values, dtype=np.float32)
            distances = np.asmatrix(dist_matrix[i][in_circle].reshape(-1, 1), dtype=np.float32)
 
            if i == 0: 
                print(f"Checking data for bag {i}:")
                print(f"Number of cells in this bag: {len(circle_barcodes)}")
                print(f"Sample of circle_barcodes: {circle_barcodes[:5]}")
                print(f"Shape of affinity_data: {affinity_data.shape}")

            
            bags[i] = {
                'distances': distances,
                'gene_expression': gene_data,
                'binding_affinity': affinity_data,
                'label': labels[i]
            }

            bag_barcodes = barcodes[in_circle]
            for barcode in bag_barcodes:
                csv_data.append([i, barcode, labels[i]])

            print(f"Bag {i} has {gene_data.shape[0]} instances")

        df = pd.DataFrame(csv_data, columns=['bag_id', 'barcode', 'label'])
        df.to_csv(output_csv, index=False)
        return bags

# Rest of the code remains the same


def custom_collate_fn(batch):
    # Custom collate function to handle bags with variable number of instances
    distances, gene_expressions, affinity_data, labels = zip(*batch)
    distances = [torch.tensor(d, dtype=torch.float32) for d in distances]
    gene_expressions = [torch.tensor(g, dtype=torch.float32) for g in gene_expressions]
    affinity_data = [torch.tensor(a, dtype=torch.float32) for a in affinity_data]
    return distances, gene_expressions, affinity_data, torch.tensor(labels, dtype=torch.float32).view(-1)

