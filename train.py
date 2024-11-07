import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from model.dataset import BagsDataset, custom_collate_fn
from model.model import MIL, EarlyStopping

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_all_genes(reference_gene_file):
    all_genes = pd.read_csv(reference_gene_file)
    return all_genes['Gene'].values.tolist()

def save_metrics(epoch, train_loss, val_loss, val_auroc, output_dir):
    file_path = os.path.join(output_dir, 'training_metrics.csv')
    if not os.path.exists(file_path):
        # Create the CSV file with headers
        with open(file_path, 'w') as f:
            f.write('Epoch,Train Loss,Val Loss,Val AUROC\n')
    
    # Append metrics for the current epoch
    with open(file_path, 'a') as f:
        f.write(f'{epoch},{train_loss},{val_loss},{val_auroc}\n')

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
    else:
        print(f"Using device: {device}")
    print("=====================================")

    all_genes = load_all_genes(args.reference_gene)
    print('Reference genes loaded:', len(all_genes))
    print("=====================================")
    
    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model, criterion, optimizer, and early stopping
    model = MIL(all_genes).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)
    
    # Load dataset and create DataLoader
    dataset = BagsDataset(args.data, immune_cell=args.immune_cell, max_instances=args.max_instances, n_genes=args.n_genes)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    # Save IG scores before training
    ig_scores_before_training = model.immunogenicity.ig.clone().detach().cpu()
    ig_scores_before_training = [score.item() for score in ig_scores_before_training]  # Ensure it's a list of floats
    #print(f"IG Scores Before Training: {ig_scores_before_training}")

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, (distances, gene_expressions, label, core_idx, gene_names, cell_ids) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
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

        train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {train_loss:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for val_distances, val_gene_expressions, val_label, val_core_idx, val_gene_names, val_cell_ids in val_loader:
                val_distances = [d.to(device) for d in val_distances]
                val_gene_expressions = [g.to(device) for g in val_gene_expressions]
                val_label = val_label.clone().detach().float().to(device)
                val_current_genes = val_gene_names[0]  # Since batch_size=1

                val_output = model(val_distances, val_gene_expressions, val_current_genes)
                val_loss += criterion(val_output, val_label).item()
                val_predictions.extend(val_output.cpu().numpy())
                val_labels.extend(val_label.cpu().numpy())

        val_loss /= len(val_loader)
        val_auroc = roc_auc_score(val_labels, val_predictions)
        print(f'Validation Loss: {val_loss:.4f}, Validation AUROC: {val_auroc:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss {val_loss:.4f}")

        # Save metrics
        save_metrics(epoch+1, train_loss, val_loss, val_auroc, args.output_dir)

        # Early stopping
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Save IG scores after training
    ig_scores_after_training = model.immunogenicity.ig.clone().detach().cpu()
    ig_scores_after_training = [score.item() for score in ig_scores_after_training]  # Ensure it's a list of floats
    #print(f"IG Scores After Training: {ig_scores_after_training}")

    # Save IG score changes to a CSV file
    ig_score_data = {
        'Gene': all_genes,
        'IG Score Before Training': ig_scores_before_training,
        'IG Score After Training': ig_scores_after_training
    }
    df = pd.DataFrame(ig_score_data)

    # Calculate the difference and add it as a new column
    df['Difference'] = df['IG Score After Training'] - df['IG Score Before Training']
    df = df.sort_values(by='Difference', ascending=False)

    # Write the sorted DataFrame to a CSV file
    output_path = os.path.join(args.output_dir, 'ig_score_changes.csv')
    df.to_csv(output_path, index=False)

    # Save the final model
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
