# %%
# Import necessary libraries
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scanpy as sc
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from model.dataset import BagsDataset, custom_collate_fn
from model.model import MIL, EarlyStopping


# %%
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Using device: {device} ({gpu_name})")
else:
    print(f"Using device: {device}")
print("=====================================")


# %%

# Functions to load gene lists
def load_all_genes(reference_gene_file):
    
    all_genes = pd.read_csv(reference_gene_file)
    all_genes = all_genes['Gene'].values.tolist()
    
    return all_genes


# %%
# Set parameters (replace these with your own paths and settings)
# Paths to data and model
data_path = 'data/training_all.csv'
reference_gene_path = 'data/tumor_antigen_8000.csv'
pretrained_gene_path = 'data/human.csv'  # Pre-trained gene list
output_dir = 'fine_tuned_model/all_data_human2antigen'
model_path = 'test/all_cpu_revised_human_0.1_10000/final_model.pth'  # Set to None if training from scratch


# %%

# Training parameters
immune_cell = 'tcell'       # Type of immune cell to consider
learning_rate = 0.1      # Learning rate for the optimizer
num_epochs = 1000           # Number of epochs to train the model
patience = 5                # Patience for early stopping
delta = 0.001               # Minimum change to qualify as an improvement
max_instances = None        # Maximum instances for the dataset
n_genes = 10000             # Number of genes to consider


# %%

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load fine-tuning gene list
all_genes = load_all_genes(reference_gene_path)
print('Reference genes loaded:', len(all_genes))
print("=====================================")


# %%

# Load pre-trained gene list
pretrained_genes = load_all_genes(pretrained_gene_path)
print('Pre-trained genes loaded:', len(pretrained_genes))


# %%
common_genes = list(set(pretrained_genes) & set(all_genes))
print(f'Number of common genes: {len(common_genes)}')

# %%

# Create gene name to index mappings
pretrained_gene_to_index = {gene: idx for idx, gene in enumerate(pretrained_genes)}
fine_tuning_gene_to_index = {gene: idx for idx, gene in enumerate(all_genes)}


# %%

# Initialize the model
model = MIL(all_genes).to(device)


# %%

# Initialize a new tensor for immunogenicity.ig
new_ig_tensor = model.immunogenicity.ig.data.clone()


# %%

# Load pre-trained model's state dict
pretrained_state_dict = torch.load(model_path)


# %%

# Get the pre-trained immunogenicity.ig tensor
pretrained_ig_tensor = pretrained_state_dict['immunogenicity.ig']


# %%

# Copy over the values for common genes
for gene in common_genes:
    pretrained_idx = pretrained_gene_to_index[gene]
    fine_tuning_idx = fine_tuning_gene_to_index[gene]
    new_ig_tensor[fine_tuning_idx] = pretrained_ig_tensor[pretrained_idx]

# Assign the new tensor to the model
with torch.no_grad():
    model.immunogenicity.ig.copy_(new_ig_tensor)

print("Copied immunogenicity.ig values for common genes.")


# %%

# Remove immunogenicity.ig from the pre-trained state dict
pretrained_state_dict.pop('immunogenicity.ig', None)


# %%

# Load other compatible parameters
model_state_dict = model.state_dict()
common_keys = [k for k in pretrained_state_dict.keys()
               if k in model_state_dict and pretrained_state_dict[k].size() == model_state_dict[k].size()]
filtered_pretrained_state_dict = {k: pretrained_state_dict[k] for k in common_keys}
model_state_dict.update(filtered_pretrained_state_dict)
model.load_state_dict(model_state_dict)

print(f"Loaded matching model weights from {model_path} (excluding immunogenicity.ig).")


# %%

# Optionally freeze pre-trained parameters (excluding immunogenicity.ig)
# for name, param in model.named_parameters():
#     if name in filtered_pretrained_state_dict:
#         param.requires_grad = False

# Define loss criterion and optimizer
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# %%

# Load dataset
# Replace 'BagsDataset' and 'custom_collate_fn' with your data loading functions
#adata = sc.read(data_path)
dataset = BagsDataset(data_path, immune_cell=immune_cell, max_instances=max_instances, n_genes=n_genes, radius=300, resolution='low')
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)


# %%

# Initialize early stopping
early_stopping = EarlyStopping(patience=patience, delta=delta)

# Store IG scores before training
ig_scores_before_training = torch.sigmoid(model.immunogenicity.ig.detach().cpu())


# %%

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (distances, gene_expressions, label, core_idx, current_genes) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            optimizer.zero_grad()

            distances = torch.stack(distances).to(device)
            gene_expressions = torch.stack(gene_expressions).to(device)
            label = label.clone().detach().float().to(device)

            output = model(distances, gene_expressions, list(current_genes[0]))

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Validation
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for val_distances, val_gene_expressions, val_label, _, val_current_genes in val_loader:
            val_distances = torch.stack(val_distances).to(device)
            val_gene_expressions = torch.stack(val_gene_expressions).to(device)
            val_label = val_label.clone().detach().float().to(device)
            val_output = model(val_distances, val_gene_expressions, list(val_current_genes[0]))
            val_loss += criterion(val_output, val_label).item()
            val_predictions.extend(val_output.cpu().numpy())
            val_labels.extend(val_label.cpu().numpy())

    val_loss /= len(val_loader)
    val_auroc = roc_auc_score(val_labels, val_predictions)
    print(f'Validation Loss: {val_loss:.4f}, Validation AUROC: {val_auroc:.4f}')

    # Early stopping
    early_stopping(val_loss, model, epoch)
    if early_stopping.early_stop:
        print(f'Early stopping at epoch {epoch+1}')
        breaks


# %%

# Store IG scores after training
ig_scores_after_training = torch.sigmoid(model.immunogenicity.ig.detach().cpu())

# Create DataFrame for IG scores
ig_score = {
    'Gene': all_genes,
    'IG Score Before Training': ig_scores_before_training.numpy(),
    'IG Score After Training': ig_scores_after_training.numpy()
}
df = pd.DataFrame(ig_score)

# Calculate the difference and add it as a new column
df['Difference'] = df['IG Score After Training'] - df['IG Score Before Training']

# Sort the DataFrame by the Difference column in descending order
df = df.sort_values(by='Difference', ascending=False)

# Write the sorted DataFrame to a CSV file
output_path = os.path.join(output_dir, 'ig_score_changes.csv')
df.to_csv(output_path, index=False)

# Save the final model
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))

print("Training complete. Model and IG scores saved.")

# %%



