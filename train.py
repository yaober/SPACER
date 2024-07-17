import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model.dataset import BagsDataset, custom_collate_fn
from model.model import MIL
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train MIL model for gene expression and immunogenicity')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--radius', type=float, default=100, help='Radius for spatial coordinates')
    parser.add_argument('--max_instances', type=int, default=None, help='Maximum instances per bag')
    parser.add_argument('--immune_cell', type=str, choices=['tcell', 'bcell'], default='tcell', help='Immune cell type')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing h5ad data files')
    return parser.parse_args()

def load_datasets(data_dir, immune_cell, radius, max_instances):
    h5ad_files = [f for f in os.listdir(data_dir) if f.endswith('.h5ad')]
    datasets = []
    for h5ad_file in h5ad_files:
        adata = sc.read_h5ad(os.path.join(data_dir, h5ad_file))
        dataset = BagsDataset(adata, immune_cell=immune_cell, radius=radius, max_instances=max_instances)
        data
        datasets.append(dataset)
    return datasets



def main():
    args = parse_arguments()
    
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    
    # Load and concatenate datasets
    datasets = load_datasets(args.data_dir, args.immune_cell, args.radius, args.max_instances)

    # Initialize model
    model = MIL(gene_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for distances, gene_expressions, labels, core_idxs, current_genes in dataloader:
            distances = [d.to(device) for d in distances]
            gene_expressions = [g.to(device) for g in gene_expressions]
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(distances, gene_expressions, current_genes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), 'mil_model.pth')

    # Evaluation function (example, can be modified based on specific requirements)
    def evaluate(model, dataloader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for distances, gene_expressions, labels, core_idxs, current_genes in dataloader:
                distances = [d.to(device) for d in distances]
                gene_expressions = [g.to(device) for g in gene_expressions]
                labels = labels.to(device)

                outputs = model(distances, gene_expressions, current_genes)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')

    # Evaluate the model
    evaluate(model, dataloader)

if __name__ == '__main__':
    main()
