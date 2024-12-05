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

def save_metrics(epoch, train_loss, val_loss, val_auroc,a,b,alpha,beta, output_dir):
    file_path = os.path.join(output_dir, 'training_metrics.csv')
    if not os.path.exists(file_path):
        # Create the CSV file with headers
        with open(file_path, 'w') as f:
            f.write('Epoch,Train Loss,Val Loss,Val AUROC\n')
    
    # Append metrics for the current epoch
    with open(file_path, 'a') as f:
        f.write(f'{epoch},{train_loss},{val_loss},{val_auroc},{a},{b},{alpha},{beta}\n')

def save_ig_scores(epoch, all_genes, ig_scores_before_training, ig_scores_after_training, output_dir):
    # Create a DataFrame with IG scores before and after the current epoch
    ig_score_data = {
        'Gene': all_genes,
        'IG Score Before Training': ig_scores_before_training,
        'IG Score After Training': ig_scores_after_training,
    }
    df = pd.DataFrame(ig_score_data)
    
    df['Difference'] = df['IG Score After Training'] - df['IG Score Before Training']
    df = df.sort_values(by='Difference', ascending=False)

    output_path = os.path.join(output_dir, f'ig_score_changes_epoch_{epoch+1}.csv')
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

    # Initialize the model, optimizer, and early stopping
    model = MIL(all_genes).to(device)  # Adjust 'k' as needed
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)
    
    # Load dataset and create DataLoader
    dataset = BagsDataset(
        args.data,
        immune_cell=args.immune_cell,
        max_instances=args.max_instances,
        n_genes=args.n_genes,
        k=4  # Ensure 'k' matches the number of negative bags per batch
    )
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    # Save IG scores before training
    ig_scores_before_training = model.immunogenicity.ig.clone().detach().cpu()
    ig_scores_before_training = [score.item() for score in ig_scores_before_training]

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

                distances_list, gene_expressions_list, labels_list, core_idxs_list, gene_names_list, cell_ids_list = batch_data
                
                distances_list = [distances.to(device) for distances in distances_list]
                gene_expressions_list = [gene_exp.to(device) for gene_exp in gene_expressions_list]
                labels = torch.stack(labels_list).float().to(device)
                current_genes_list = gene_names_list  
                
                outputs = model(distances_list, gene_expressions_list, current_genes_list)
                
                if outputs is None:
                    continue  # Skip this batch if the model returns None
                
                if outputs.shape[0] != labels.shape[0]:
                    # Handle mismatch in batch sizes if necessary
                    continue
                
                # Compute custom loss
                # Identify positive and negative outputs
                positive_idxs = (labels == 1).nonzero(as_tuple=True)[0]
                negative_idxs = (labels == 0).nonzero(as_tuple=True)[0]
                
                if positive_idxs.numel() == 0 or negative_idxs.numel() == 0:
                    continue  # Skip batch if no positive or negative bags
                
                positive_output = outputs[positive_idxs]
                negative_outputs = outputs[negative_idxs]
                
                # Compute mean of negative outputs and loss
                mean_negative_output = negative_outputs.mean()
                positive_output = positive_output.mean() 
                
                loss = torch.relu(mean_negative_output - positive_output+0.1)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                
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
                 
                    val_distances_list, val_gene_expressions_list, val_labels_list, val_core_idxs_list, val_gene_names_list, val_cell_ids_list = val_batch_data
                    
                    
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
                    
                    # Compute custom loss
                    # Identify positive and negative outputs
                    positive_idxs = (val_labels == 1).nonzero(as_tuple=True)[0]
                    negative_idxs = (val_labels == 0).nonzero(as_tuple=True)[0]
                    
                    if positive_idxs.numel() == 0 or negative_idxs.numel() == 0:
                        continue 
                    
                    positive_output = val_outputs[positive_idxs]
                    negative_outputs = val_outputs[negative_idxs]
                    
                    # Compute mean of negative outputs and loss
                    mean_negative_output = negative_outputs.mean()
                    positive_output = positive_output.mean()
                    
                    loss = torch.relu(mean_negative_output - positive_output+0.1)
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
        
        a = model.distance.a.clone().detach().cpu().item()
        b = model.gene_expression.b.clone().detach().cpu().item()
        alpha =model.alpha.clone().detach().cpu().item()
        beta = model.beta.clone().detach().cpu().item()
        
        # Save metrics
        save_metrics(epoch+1, train_loss, val_loss, val_epoch_auc,
                     a, b, alpha, beta,
                     args.output_dir)

        # Save IG scores after each epoch
        ig_scores_after_training = model.immunogenicity.ig.clone().detach().cpu()
        ig_scores_after_training = [score.item() for score in ig_scores_after_training]
        save_ig_scores(epoch, all_genes, ig_scores_before_training, ig_scores_after_training, args.output_dir)
    
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
