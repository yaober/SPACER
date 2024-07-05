import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import cdist

class BagsDataset(Dataset):
    def __init__(self, adata, radius=50, output_csv='bags.csv'):
        self.bags = self.create_bags(adata, radius, output_csv)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        instances = bag['instances']
        label = bag['label']
        return instances, label

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

            bag_data = gene_expression[in_circle].todense()  # Convert sparse matrix to dense
            bag_label = labels[i]
            bag_barcodes = barcodes[in_circle]

            bags[i] = {
                'instances': bag_data,
                'label': bag_label
            }

            for barcode in bag_barcodes:
                csv_data.append([i, barcode, bag_label])

        df = pd.DataFrame(csv_data, columns=['bag_id', 'barcode', 'label'])
        df.to_csv(output_csv, index=False)

        return bags

def custom_collate_fn(batch):
    # Custom collate function to handle bags with variable number of instances
    instances, labels = zip(*batch)
    return instances, torch.tensor(labels[0], dtype=torch.float32).view(1)

