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

def save_metrics(epoch, train_loss, val_loss, val_auroc, a, b, alpha, beta, output_dir):
    file_path = os.path.join(output_dir, 'training_metrics.csv')
    if not os.path.exists(file_path):
        # Create the CSV file with headers
        with open(file_path, 'w') as f:
            f.write('Epoch,Train Loss,Val Loss,Val AUROC,a,b,alpha,beta\n')
    
    # Append metrics for the current epoch
    with open(file_path, 'a') as f:
        f.write(f'{epoch},{train_loss},{val_loss},{val_auroc},{a},{b},{alpha},{beta}\n')

def save_spacer_scores(epoch, all_genes, spacer_scores_before_training, spacer_scores_after_training, output_dir):
    # Create a DataFrame with SPACER Scores before and after the current epoch
    spacer_score_data = {
        'Gene': all_genes,
        'SPACER Score Before Training': spacer_scores_before_training,
        'SPACER Score After Training': spacer_scores_after_training,
    }
    df = pd.DataFrame(spacer_score_data)
    
    # Calculate the difference and add it as a new column
    df['Difference'] = df['SPACER Score After Training'] - df['SPACER Score Before Training']
    df = df.sort_values(by='Difference', ascending=False)

    # Save to a CSV file for each epoch
    output_path = os.path.join(output_dir, f'spacer_score_changes_epoch_{epoch+1}.csv')
    df.to_csv(output_path, index=False)

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
    model = MIL(all_genes, gene_weighting=args.gene_weighting).to(device)  # Adjust 'k' as needed
    criterion = nn.BCELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)
    
    # Load dataset and create DataLoader
    dataset = BagsDataset(
        args.data,
        immune_cell=args.immune_cell,
        max_instances=args.max_instances,
        n_genes=args.n_genes,
        k=2  # Ensure 'k' matches the number of bags per batch
    )
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    # Save SPACER Scores before training
    spacer_scores_before_training = model.immunogenicity.ig.clone().detach().cpu()
    spacer_scores_before_training = [score.item() for score in spacer_scores_before_training]

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        
        # Lists to store outputs and labels for AUROC calculation
        all_outputs = []
        all_labels = []
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, batch_data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}/{args.num_epochs}")
                optimizer.zero_grad()

                # Unpack the batch data
                distances_list, gene_expressions_list, labels_list, core_idxs_list, gene_names_list, cell_ids_list = batch_data
                
                # Move data to device and prepare labels
                distances_list = [distances.to(device) for distances in distances_list]
                gene_expressions_list = [gene_exp.to(device) for gene_exp in gene_expressions_list]
                labels = torch.stack(labels_list).float().to(device)
                current_genes_list = gene_names_list  # List of gene names for each bag

                # Forward pass
                outputs = model(distances_list, gene_expressions_list, current_genes_list)
                
                if outputs is None:
                    continue  # Skip this batch if the model returns None
                
                if outputs.shape[0] != labels.shape[0]:
                    # Handle mismatch in batch sizes if necessary
                    continue
                
                # Compute BCE loss
                if args.selection == 'negative':
                    labels = 1 - labels
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                
                # Accumulate outputs and labels for AUROC calculation
                all_outputs.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        
        # Compute Training AUROC
        try:
            epoch_auc = roc_auc_score(all_labels, all_outputs)
        except ValueError:
            epoch_auc = float('nan')  # Handle case where AUROC can't be computed
        
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {train_loss:.4f}, AUROC: {epoch_auc:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_all_outputs = []
        val_all_labels = []
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as vtepoch:
                for val_batch_data in vtepoch:
                    # Unpack validation batch data
                    val_distances_list, val_gene_expressions_list, val_labels_list, val_core_idxs_list, val_gene_names_list, val_cell_ids_list = val_batch_data
                    
                    # Move data to device and prepare labels
                    val_distances_list = [distances.to(device) for distances in val_distances_list]
                    val_gene_expressions_list = [gene_exp.to(device) for gene_exp in val_gene_expressions_list]
                    val_labels = torch.stack(val_labels_list).float().to(device)
                    val_current_genes_list = val_gene_names_list  # List of gene names for each bag
                    
                    # Forward pass
                    val_outputs = model(val_distances_list, val_gene_expressions_list, val_current_genes_list)
                    
                    if val_outputs is None:
                        continue  # Skip this batch if the model returns None
                    
                    if val_outputs.shape[0] != val_labels.shape[0]:
                        # Handle mismatch in batch sizes if necessary
                        continue
                    
                    # Compute BCE loss
                    if args.selection == 'negative':
                        val_labels = 1 - val_labels
                    loss = criterion(val_outputs, val_labels)
                    val_loss += loss.item()
                    vtepoch.set_postfix(val_loss=loss.item())
                    
                    # Accumulate outputs and labels for AUROC calculation
                    val_all_outputs.extend(val_outputs.detach().cpu().numpy())
                    val_all_labels.extend(val_labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            # Compute Validation AUROC
            try:
                val_epoch_auc = roc_auc_score(val_all_labels, val_all_outputs)
            except ValueError:
                val_epoch_auc = float('nan')  # Handle case where AUROC can't be computed
            
            print(f'Validation Loss: {val_loss:.4f}, Validation AUROC: {val_epoch_auc:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss {val_loss:.4f}")
            
        torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))
        
        a = model.distance.a.clone().detach().cpu().numpy()
        b = model.gene_expression.b.clone().detach().cpu()
        alpha = model.alpha.clone().detach().cpu()
        beta = model.beta.clone().detach().cpu()
        # Save metrics
        save_metrics(epoch+1, train_loss, val_loss, val_epoch_auc,a,b,alpha,beta, args.output_dir)

        # Save SPACER Scores after each epoch
        spacer_scores_after_training = model.immunogenicity.ig.clone().detach().cpu()
        spacer_scores_after_training = [score.item() for score in spacer_scores_after_training]
        save_spacer_scores(epoch, all_genes, spacer_scores_before_training, spacer_scores_after_training, args.output_dir)
    
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
    parser.add_argument('--selection', type=str, default='positive', help='Selection of positive or negative samples.')
    parser.add_argument('--gene_weighting', type=str, default='softmax', choices=['softmax', 'sparsemax'],
                        help='How to normalize gene-expression weights across genes.')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == '__main__':
    main()
