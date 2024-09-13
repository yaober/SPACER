import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from model.dataset import BagsDataset, custom_collate_fn
from model.model import MIL, EarlyStopping
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def load_all_genes(reference_gene_file):
    all_genes = []
    with open(reference_gene_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            all_genes.append(row['Gene'])
    return all_genes

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"Using device: {device} ({gpu_name})")
    else:
        print(f"Using device: {device}")
    print("=====================================")
    all_genes = load_all_genes(args.reference_gene)
    print('Reference genes loaded:', len(all_genes))
    print("=====================================")
    
    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    model = MIL(all_genes).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    # Load dataset
    dataset = BagsDataset(args.data, immune_cell=args.immune_cell, max_instances=args.max_instances, n_genes=args.n_genes)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)
    
    ig_scores_before_training = torch.sigmoid(model.immunogenicity.ig)
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (distances, gene_expressions, label, core_idx, current_genes) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}/{args.num_epochs}")

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
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {epoch_loss:.4f}')

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

        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    ig_scores_after_training = torch.sigmoid(model.immunogenicity.ig)
    ig_score = {
    'Gene': all_genes,
    'IG Score Before Training': [score.item() for score in ig_scores_before_training],
    'IG Score After Training': [score.item() for score in ig_scores_after_training]
}   
    df = pd.DataFrame(ig_score)

    # Calculate the difference and add it as a new column
    df['Difference'] = df['IG Score After Training'] - df['IG Score Before Training']

    # Sort the DataFrame by the Difference column in descending order
    df = df.sort_values(by='Difference', ascending=False)

    # Write the sorted DataFrame to a CSV file
    output_path = os.path.join(args.output_dir, 'ig_score_changes.csv')
    df.to_csv(output_path, index=False)
    
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))

def main():
    parser = argparse.ArgumentParser(description='Train a MIL model for gene expression and immunogenicity.')
    parser.add_argument('--data', type=str, required=True, help='Path to the training data file.')
    parser.add_argument('--reference_gene', type=str, required=True, help='Path to the reference gene CSV file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files.')
    parser.add_argument('--immune_cell', type=str, default='tcell', help='Type of immune cell to consider.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train the model.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--delta', type=float, default=0.001, help='Minimum change to qualify as an improvement.')
    parser.add_argument('--max_instances', type=int, default=None, help='Maximum instances for the dataset.')
    parser.add_argument('--n_genes', type=int, default=10000, help='Number of genes to consider.')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == '__main__':
    main()
