# %%
# Import necessary libraries
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse

import scanpy as sc
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from model.dataset import BagsDataset, custom_collate_fn
from model.model import MIL, EarlyStopping


# %%
# Argument parsing setup
parser = argparse.ArgumentParser(description="Train a MIL model for immunogenicity prediction")

# Add arguments
parser.add_argument("--immune_cell", type=str, default='tcell', help="Type of immune cell to consider")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")
parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train the model")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
parser.add_argument("--delta", type=float, default=0.001, help="Minimum change to qualify as an improvement")
parser.add_argument("--radius", type=int, default=None, help="Radius for the neighborhood")
parser.add_argument("--n_genes", type=int, default=10000, help="Number of genes to consider")
parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
parser.add_argument("--reference_gene_path", type=str, required=True, help="Path to reference gene list")
parser.add_argument("--pretrained_gene_path", type=str, required=True, help="Path to pre-trained gene list")
parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model (optional)")
parser.add_argument("--output_dir", type=str, default='fine_tuned_model', help="Directory to save the trained model")
parser.add_argument("--resolution", type=str, default='low', choices=['high', 'low'], help="Resolution for the dataset")
parser.add_argument("--gene_weighting", type=str, default='softmax', choices=['softmax', 'sparsemax'],
                    help="How to normalize gene-expression weights across genes.")

# Parse the arguments
args = parser.parse_args()

# Assign arguments to variables
immune_cell = args.immune_cell
learning_rate = args.learning_rate
num_epochs = args.num_epochs
patience = args.patience
delta = args.delta
max_instances = None
radius = args.radius
n_genes = args.n_genes
data_path = args.data_path
reference_gene_path = args.reference_gene_path
pretrained_gene_path = args.pretrained_gene_path
model_path = args.model_path
output_dir = args.output_dir
resolution = args.resolution
gene_weighting = args.gene_weighting


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
# Find common genes
common_genes = list(set(pretrained_genes) & set(all_genes))
print(f'Number of common genes: {len(common_genes)}')


# %%
# Create gene name to index mappings
pretrained_gene_to_index = {gene: idx for idx, gene in enumerate(pretrained_genes)}
fine_tuning_gene_to_index = {gene: idx for idx, gene in enumerate(all_genes)}


# %%
# Initialize the model
model = MIL(all_genes, gene_weighting=gene_weighting).to(device)


# %%
# Initialize a new tensor for immunogenicity.ig
new_ig_tensor = model.immunogenicity.ig.data.clone()


# %%
# Load pre-trained model's state dict
if model_path:
    pretrained_state_dict = torch.load(model_path, map_location=device)

    # Get the pre-trained immunogenicity.ig tensor
    pretrained_ig_tensor = pretrained_state_dict['immunogenicity.ig']

    # Copy over the values for common genes
    for gene in common_genes:
        pretrained_idx = pretrained_gene_to_index[gene]
        fine_tuning_idx = fine_tuning_gene_to_index[gene]
        new_ig_tensor[fine_tuning_idx] = pretrained_ig_tensor[pretrained_idx]

    # Assign the new tensor to the model
    with torch.no_grad():
        model.immunogenicity.ig.copy_(new_ig_tensor)

    print("Copied immunogenicity.ig values for common genes.")

    # Remove immunogenicity.ig from the pre-trained state dict
    pretrained_state_dict.pop('immunogenicity.ig', None)

    # Load other compatible parameters
    model_state_dict = model.state_dict()
    common_keys = [k for k in pretrained_state_dict.keys()
                   if k in model_state_dict and pretrained_state_dict[k].size() == model_state_dict[k].size()]
    filtered_pretrained_state_dict = {k: pretrained_state_dict[k] for k in common_keys}
    model_state_dict.update(filtered_pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    print(f"Loaded matching model weights from {model_path} (excluding immunogenicity.ig).")

ig_scores_before_training = model.immunogenicity.ig.clone().detach().cpu()
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
adata = sc.read(data_path)
dataset = BagsDataset(adata, immune_cell=immune_cell, max_instances=max_instances, n_genes=n_genes, radius=radius, resolution=resolution)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

"""train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
"""
from torch.utils.data import Subset
import random

def balance_dataset(dataset, label_ratio=5):
    # Separate bags by labels
    label_0_indices = []
    label_1_indices = []

    for idx in range(len(dataset)):
        _, _, label, _, _, _ = dataset[idx]
        if label.item() == 0:
            label_0_indices.append(idx)
        else:
            label_1_indices.append(idx)

    # Calculate the number of label=0 bags to keep
    num_label_1 = len(label_1_indices)
    num_label_0_to_keep = min(len(label_0_indices), label_ratio * num_label_1)

    # Randomly sample label=0 indices
    sampled_label_0_indices = random.sample(label_0_indices, num_label_0_to_keep)

    # Combine sampled label=0 indices and all label=1 indices
    balanced_indices = sampled_label_0_indices + label_1_indices

    # Shuffle the combined indices
    random.shuffle(balanced_indices)

    # Return a Subset of the original dataset
    return Subset(dataset, balanced_indices)

# Balance train and validation datasets
balanced_train_dataset = balance_dataset(train_dataset, label_ratio=5)
balanced_val_dataset = balance_dataset(val_dataset, label_ratio=5)

# Create new DataLoaders with balanced datasets
train_loader = DataLoader(balanced_train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(balanced_val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

# Print the new label distribution
train_labels_count = sum(1 for _, _, label, _, _, _ in train_loader if label.item() == 1)
train_label_0_count = sum(1 for _, _, label, _, _, _ in train_loader if label.item() == 0)
val_labels_count = sum(1 for _, _, label, _, _, _ in val_loader if label.item() == 1)
val_label_0_count = sum(1 for _, _, label, _, _, _ in val_loader if label.item() == 0)

print(f"Train dataset - Negative: {train_label_0_count}, Positive: {train_labels_count}")
print(f"Validation dataset - Negative: {val_label_0_count}, Postive: {val_labels_count}")

# %%
# Initialize early stopping and variables to track the best model
early_stopping = EarlyStopping(patience=patience, delta=delta)
best_auroc = 0.0
best_model_path = os.path.join(output_dir, 'best_model.pth')

# List to store metrics for logging
metrics_log = []


# %%
# Training loop
# List to store metrics and IG scores for logging
metrics_log = []
ig_score_log = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (distances, gene_expressions, label, core_idx, gene_names, cell_ids) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            optimizer.zero_grad()

            distances = torch.stack(distances).to(device)
            gene_expressions = torch.stack(gene_expressions).to(device)
            label = label.clone().detach().float().to(device)

            output = model(distances, gene_expressions, list(gene_names[0]))

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
        for val_distances, val_gene_expressions, val_label, _, val_gene_names, val_cell_ids in val_loader:
            val_distances = torch.stack(val_distances).to(device)
            val_gene_expressions = torch.stack(val_gene_expressions).to(device)
            val_label = val_label.clone().detach().float().to(device)
            val_output = model(val_distances, val_gene_expressions, list(val_gene_names[0]))
            val_loss += criterion(val_output, val_label).item()
            val_predictions.extend(val_output.cpu().numpy())
            val_labels.extend(val_label.cpu().numpy())

    val_loss /= len(val_loader)
    val_auroc = roc_auc_score(val_labels, val_predictions)
    print(f'Validation Loss: {val_loss:.4f}, Validation AUROC: {val_auroc:.4f}')

    # Save model if it achieves the best AUROC
    if val_auroc > best_auroc:
        best_auroc = val_auroc
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with AUROC: {best_auroc:.4f}")

    # Log metrics and IG scores
    ig_scores_current = model.immunogenicity.ig.clone().detach().cpu().numpy()
    metrics_log.append({
        'epoch': epoch + 1,
        'training_loss': epoch_loss,
        'validation_loss': val_loss,
        'validation_auroc': val_auroc
    })
    ig_score_log.append(ig_scores_current)

    # Save IG scores for this epoch
    ig_df = pd.DataFrame({
        'Gene': all_genes,
        'IG Score': ig_scores_current
    })
    ig_epoch_path = os.path.join(output_dir, f'ig_scores_epoch_{epoch+1}.csv')
    ig_df.to_csv(ig_epoch_path, index=False)

    # Early stopping
    early_stopping(val_loss, model, epoch)
    if early_stopping.early_stop:
        print(f'Early stopping at epoch {epoch+1}')
        break

# %%
# Save the metrics log to a CSV file
metrics_df = pd.DataFrame(metrics_log)
metrics_output_path = os.path.join(output_dir, 'training_metrics.csv')
metrics_df.to_csv(metrics_output_path, index=False)

# Store IG scores after training
ig_scores_after_training = model.immunogenicity.ig.clone().detach().cpu()

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
final_model_path = os.path.join(output_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)

print("Training complete. Best model, final model, and metrics log saved.")
