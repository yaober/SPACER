# %%
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scanpy as sc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import csv
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from tqdm import trange

from torch.utils.data import Dataset


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
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


def preprocess_data(adata, immune_cell, n_genes, resolution):
    # Read the data
    if immune_cell == 'tcell':
        immune_cell = 'T'
    elif immune_cell == 'bcell':
        immune_cell = 'B'
    else:
        raise ValueError('Invalid immune cell type')

    # Ensure adata is not a view
    adata = adata.copy()
    adata.var_names_make_unique()  # Ensure unique gene names
    

    # Filter the tumor cells
    tumor_cells = adata[adata.obs['cell_type'].astype(int) == 1].copy()

    # Debug: Check tumor cells
    print(f"Tumor cells shape after filtering: {tumor_cells.shape}")
    if tumor_cells.shape[0] == 0:
        print("Warning: No tumor cells found after filtering.")

    # Calculate mean expression
    if issparse(tumor_cells.X):
        # Convert to dense and compute mean expression
        mean_expression = np.asarray(tumor_cells.X.mean(axis=0)).ravel()
    else:
        mean_expression = tumor_cells.X.mean(axis=0)

    # Get gene names
    gene_names = tumor_cells.var_names
    

    # Select top n genes
    print(f"Selecting top {n_genes} genes based on mean expression")
    top_n_gene_indices = mean_expression.argsort()[-n_genes:][::-1]
    top_n_gene_names = gene_names[top_n_gene_indices]
    print(f"Top {n_genes} genes: {top_n_gene_names}")

    # Subset adata using gene names to keep indices consistent
    adata = adata[:, top_n_gene_names].copy()

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
                percentile_value = np.percentile(tumor_cells.obs[immune_cell], 50)
                print(f"Percentile value: {percentile_value}")
                adata.obs[immune_cell] = np.where(adata.obs[immune_cell] > percentile_value, 1, 0)
                print(f"adata.obs[{immune_cell}] after binarization: {adata.obs[immune_cell].head()}")

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
            input_data = preprocess_data(input_data, immune_cell,n_genes,self.resolution)
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
            resolution = row['resolution'] if 'resolution' in row and not pd.isna(row['resolution']) else self.resolution
            adata = sc.read_h5ad(adata_path)
            adata.obs_names_make_unique()
            adata
            adata = preprocess_data(adata, self.immune_cell, self.n_genes,resolution=resolution)
            radius = row['radius'] if 'radius' in row and not pd.isna(row['radius']) else self.radius
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
            barcodes = adata.obs.index.values  # Get cell IDs
            gene_names = adata.var_names.tolist()

            for i in trange(len(spatial_coords), desc=f"Creating Bags with radius {radius}", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
                if cell_types[i] == 0:
                    continue

                dist_matrix_row = cdist([spatial_coords[i]], spatial_coords, metric='euclidean')[0]
                in_circle = np.where(dist_matrix_row <= radius)[0]
                in_circle = [idx for idx in in_circle if cell_types[idx] == 1]

                if resolution == 'high':
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
                    'gene_names': gene_names,
                    'cell_id': barcodes[i]  # Store cell ID
                }

                bag_id += 1

        total_bags = len(bags)
        avg_instances_per_bag = sum(bags[i]['gene_expression'].shape[0] for i in bags) / total_bags if total_bags > 0 else 0
        print(f"Total bags created: {total_bags}")
        print(f"Average instances per bag: {avg_instances_per_bag:.0f}")

        return bags

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
        cell_id = bag['cell_id']  # Include cell ID here
        return distances, gene_expression, label, core_idx, gene_names, cell_id

# Modify the 'custom_collate_fn' to include 'cell_id':

def custom_collate_fn(batch):
    distances, gene_expressions, labels, core_idxs, gene_names_list, cell_ids = zip(*batch)
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
    return distances, gene_expressions_tensors, labels, core_idxs, gene_names, cell_ids

# %%
input_dir = '/project/DPDS/Wang_lab/shared/spatial_TCR/data/train_validate/VisiumHD/HumanLungCancer'
adata = sc.read_h5ad(os.path.join(input_dir, 'T_cell.h5ad'))
output_dir = os.path.join('pretrained_models', input_dir.split('/')[-1])
os.makedirs(output_dir, exist_ok=True)

# %%
adata.var_names_make_unique

# %%
dataset = BagsDataset(adata, immune_cell='tcell', radius=150, n_genes=10000, resolution='high')
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#dataset = load_datasets(data_dir='/project/DPDS/Wang_lab/s439765/spatial_tcr/HumanLungCancerFFPE', immune_cell='tcell',radius=200,)

# %%
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)



# %%
distances, gene_expressions, labels, core_index, gene_names, cell_ids = next(iter(dataloader))


# %%
class Distance(nn.Module):
    def __init__(self):
        super(Distance, self).__init__()
        self.a = nn.Parameter(torch.tensor(-3.0),requires_grad=True)
        #self.sparsemax = Sparsemax(dim=0)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        #print(x)
        a = self.a
        x = self.softmax(-torch.exp(a) * x)
        return x

# %%
model = Distance()
output = model(distances[0])
print(output)

# %%
class Gene_expression(nn.Module):
    def __init__(self):
        super(Gene_expression, self).__init__()
        self.b = nn.Parameter(torch.tensor(-1.0),requires_grad=True)
        #self.sparsemax = Sparsemax(dim=-1) 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        b = self.b
        x = self.softmax(torch.exp(b) * x)
        return x

# %%
gene_expressions[0]

# %%
model = Gene_expression()
input_tensor = gene_expressions[0]
output = model(input_tensor)
print(output)

# %%
class Immunogenicity(nn.Module):
    def __init__(self, all_genes):
        super(Immunogenicity, self).__init__()
        self.all_genes = all_genes
        self.gene_to_index = {gene: idx for idx, gene in enumerate(all_genes)}
        self.ig = nn.Parameter(torch.full((len(all_genes),), -1.0), requires_grad=True)
    
    def forward(self, current_genes):
        # Filter genes to include only those present in all_genes
        #print(current_genes)
        filtered_genes = [gene for gene in current_genes if gene in self.gene_to_index]
        indices = [self.gene_to_index[gene] for gene in filtered_genes]
        ig = torch.sigmoid(self.ig[indices])
        return ig, filtered_genes

# %%
all_genes = pd.read_csv('data/human.csv')
all_genes = all_genes['Gene'].values.tolist()

# %%
model = Immunogenicity(all_genes)

# %%
output, filtered_genes = model(list(gene_names[0]))
print(len(output))
print(filtered_genes)
df = pd.DataFrame({'Gene': filtered_genes})


# %%
class MIL(nn.Module):
    def __init__(self, all_genes):
        super(MIL, self).__init__()
        self.distance = Distance()
        self.gene_expression = Gene_expression()
        self.immunogenicity = Immunogenicity(all_genes)
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    
    def forward(self, distances, gene_expressions, current_genes):
        bag_outputs = []
        for distance, gene_expression in zip(distances, gene_expressions):
            distance = self.distance(distance)
            gene_expression = self.gene_expression(gene_expression)
            immunogenicity, filtered_genes = self.immunogenicity(current_genes)
            alpha = self.alpha
            beta = self.beta
        
            if len(filtered_genes) == 0:
                continue  # Skip if no overlapping genes
        
        # Print debug information
            #print(f"gene_expression shape: {gene_expression.shape}")
            #print(f"current_genes length: {len(current_genes)}")
            #print(f"filtered_genes length: {len(filtered_genes)}")
        
        
            gene_to_index = {gene: idx for idx, gene in enumerate(current_genes)}
        
            gene_indices = [gene_to_index[gene] for gene in filtered_genes if gene in gene_to_index]
            gene_expression = gene_expression[:, gene_indices]
        
            #print(f"Filtered gene_expression shape: {gene_expression.shape}")
            #print(f"Immunogenicity shape: {immunogenicity.shape}")
        
            z = gene_expression @ immunogenicity
            #print(f"z shape: {z.shape}")
            z = z.unsqueeze(1)
            #print(f"z shape: {z.shape}")
            #print(f"distance shape: {distance}")
            bag_output = distance * z
            bag_output = torch.sum(bag_output, dim=0)
            bag_output = torch.exp(alpha) * bag_output + beta
            bag_output = torch.sigmoid(bag_output)
            #print(bag_output)
            bag_outputs.append(bag_output)
            #df = pd.DataFrame(bag_outputs)
            #df.to_csv('output.csv')
    
        
        return torch.stack(bag_outputs).squeeze(dim=1)

# %%
model = MIL(all_genes)

# %%
output = model(distances, gene_expressions, list(gene_names[0]))
output

# %%

labels[0]


# %%
ig_scores_before_training = torch.sigmoid(model.immunogenicity.ig).detach().numpy()

# %%
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), 'final_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch

# %%

model = MIL(all_genes).to(device)
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
num_epochs = 1000
patience = 5
early_stopping = EarlyStopping(patience=patience, delta=0.00001)

# %%
current_genes = gene_names[0]

# %%
for epoch in range(num_epochs):
    model.train() 
    running_loss = 0.0
    
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, (distances, gene_expressions, label, core_idx, gene_names, cell_ids) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            optimizer.zero_grad()

            # Convert distances and gene expressions to tensors
            distances = [d.to(device) for d in distances]
            gene_expressions = [g.to(device) for g in gene_expressions]
            label = label.clone().detach().float().to(device)
            current_genes = gene_names[0]  # Since batch_size=1

            output = model(distances, gene_expressions, current_genes)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())


    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    

    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        for val_distances, val_gene_expressions, val_label, val_core_idx, val_gene_names, val_cell_ids in val_dataloader:
            val_distances = [d.to(device) for d in val_distances]
            val_gene_expressions = [g.to(device) for g in val_gene_expressions]
            val_label = val_label.clone().detach().float().to(device)
            val_current_genes = val_gene_names[0]  # Since batch_size=1

            val_output = model(val_distances, val_gene_expressions, val_current_genes)
            val_loss += criterion(val_output, val_label).item()
            val_predictions.extend(val_output.cpu().numpy())
            val_labels.extend(val_label.cpu().numpy())

        val_loss /= len(val_dataloader)
        val_auroc = roc_auc_score(val_labels, val_predictions)
        print(f'Validation Loss: {val_loss:.4f}, Validation AUROC: {val_auroc:.4f}')

    
   
    early_stopping(val_loss, model, epoch)
    if early_stopping.early_stop:
        print(f'Early stopping at epoch {epoch+1}')
        break
   

# %%
ig_scores_after_training = torch.sigmoid(model.immunogenicity.ig)
alpha = model.state_dict()['alpha'].item()
beta = model.state_dict()['beta'].item()
a = model.state_dict()['distance.a'].item()
b = model.state_dict()['gene_expression.b'].item()  
with open(os.path.join(output_dir, 'ig_score_changes.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['alpha', alpha])
    writer.writerow(['beta', beta])
    writer.writerow(['a', a])
    writer.writerow(['b', b])
    writer.writerow(['Gene', 'IG Score Before Training', 'IG Score After Training'])
    for gene, before, after in zip(all_genes, ig_scores_before_training, ig_scores_after_training):
        writer.writerow([gene, before.item(), after.item()])
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))


# %%
output_dir

# %%
model.state_dict()

# %%


# %%



