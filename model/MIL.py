import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial.distance import cdist

class BagsDataset(Dataset):
    def __init__(self, adata, radius=50):
        self.bags = self.create_bags(adata, radius)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        instances = bag['instances']
        label = bag['label']
        return instances, label

    def create_bags(self, adata, radius):
        spatial_coords_x = adata.obs['X']
        spatial_coords_y = adata.obs['Y']
        spatial_coords = np.array(list(zip(spatial_coords_x, spatial_coords_y)))

        gene_expression = adata.X
        labels = adata.obs['tcr'].values
        bags = {}

        dist_matrix = cdist(spatial_coords, spatial_coords, metric='euclidean')

        for i in range(len(spatial_coords)):
            in_circle = np.where(dist_matrix[i] <= radius)[0]

            bag_data = gene_expression[in_circle].todense()  # Convert sparse matrix to dense
            bag_label = labels[i]

            bags[i] = {
                'instances': bag_data,
                'label': bag_label
            }
            print(f"Bag {i} contains {len(in_circle)} instances")

        return bags

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=0)
        return weights

class MIL(nn.Module):
    def __init__(self, input_dim):
        super(MIL, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class MILAggregator(nn.Module):
    def __init__(self, input_dim):
        super(MILAggregator, self).__init__()
        self.mil = MIL(input_dim)
        self.attention = Attention(1)  # Attention module input is the output dimension of MIL, which is 1
    
    def forward(self, bag):
        # Process each instance in the bag
        instance_outputs = torch.stack([self.mil(instance) for instance in bag])
        
        # Calculate attention weights
        weights = self.attention(instance_outputs)
        
        # Aggregate the instance outputs using attention weights
        bag_output = torch.sum(weights * instance_outputs, dim=0)
        
        return bag_output.view(1)  # Make sure the output shape is compatible with the label shape


def custom_collate_fn(batch):
    # Custom collate function to handle bags with variable number of instances
    instances, labels = zip(*batch)
    return instances, torch.tensor(labels[0], dtype=torch.float32).view(1)

def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for instances, label in dataloader:
            instances = [torch.tensor(np.array(instance), dtype=torch.float32).to('cuda') for instance in instances[0]]
            output = model(instances)
            predictions.append(output.item())
    return predictions


